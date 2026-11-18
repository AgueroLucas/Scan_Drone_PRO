"""Implementación de ByteTrack (Kalman + asociación en dos etapas).

Referencia: Zhang et al., "ByteTrack: Multi-Object Tracking by Associating
Every Detection Box", ECCV 2022.

Diferencias deliberadas respecto al paper:
    * Los tracks nuevos se confirman de inmediato para mostrar el ID y contar
      a la persona desde el primer frame (menor latencia percibida).
    * No se usa ReID: ByteTrack se basa únicamente en posición (IoU) y score.
      Esto mantiene el tracking rápido en CPU y GPU.

La clase pública es :class:`ByteTrackTracker`; el resto son helpers internos.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Iterable, Sequence

import numpy as np

from config.settings import TrackingSettings
from core.types import Detection, TrackView, bbox_iou, tlwh_to_xyah, xyah_to_tlwh

try:  # SciPy acelera la asignación; si no está, se usa el fallback puro Python.
    from scipy.optimize import linear_sum_assignment as _scipy_assignment
except ImportError:  # pragma: no cover - dependencia opcional
    _scipy_assignment = None


def _hungarian_assignment(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Implementación O(n³) del algoritmo húngaro para matrices rectangulares.

    Se usa únicamente cuando SciPy no está disponible. Es determinista y no
    añade dependencias: con decenas de personas su coste es insignificante.
    """
    matrix = np.asarray(cost, dtype=np.float64)
    rows, columns = matrix.shape
    if rows == 0 or columns == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    if rows > columns:
        # En la traspuesta, la primera salida son columnas originales y la
        # segunda filas originales: se intercambian para conservar el contrato.
        transposed_columns, transposed_rows = _hungarian_assignment(matrix.T)
        return transposed_rows, transposed_columns

    potentials_u = np.zeros(rows + 1, dtype=np.float64)
    potentials_v = np.zeros(columns + 1, dtype=np.float64)
    assignments = np.zeros(columns + 1, dtype=int)
    way = np.zeros(columns + 1, dtype=int)

    for row in range(1, rows + 1):
        assignments[0] = row
        current_column = 0
        minimums = np.full(columns + 1, np.inf)
        used = np.zeros(columns + 1, dtype=bool)
        while True:
            used[current_column] = True
            current_row = assignments[current_column]
            delta = np.inf
            next_column = 0
            for column in range(1, columns + 1):
                if used[column]:
                    continue
                value = (
                    matrix[current_row - 1, column - 1]
                    - potentials_u[current_row]
                    - potentials_v[column]
                )
                if value < minimums[column]:
                    minimums[column] = value
                    way[column] = current_column
                if minimums[column] < delta:
                    delta = minimums[column]
                    next_column = column
            for column in range(columns + 1):
                if used[column]:
                    potentials_u[assignments[column]] += delta
                    potentials_v[column] -= delta
                else:
                    minimums[column] -= delta
            current_column = next_column
            if assignments[current_column] == 0:
                break
        while True:
            previous_column = way[current_column]
            assignments[current_column] = assignments[previous_column]
            current_column = previous_column
            if current_column == 0:
                break

    selected_rows: list[int] = []
    selected_columns: list[int] = []
    for column in range(1, columns + 1):
        if assignments[column] > 0:
            selected_rows.append(assignments[column] - 1)
            selected_columns.append(column - 1)
    return np.array(selected_rows, dtype=int), np.array(selected_columns, dtype=int)


def linear_sum_assignment(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resuelve la asignación de mínimo coste con SciPy o fallback interno."""
    if _scipy_assignment is not None:
        return _scipy_assignment(cost)
    return _hungarian_assignment(cost)


# ---------------------------------------------------------------------------
# Helper de asociación
# ---------------------------------------------------------------------------
def iou_cost(detections: Sequence[Detection], tracks: Sequence["Track"]) -> np.ndarray:
    """Matriz ``1 - IoU`` entre detecciones y tracks."""
    if not detections:
        return np.zeros((0, len(tracks)), dtype=np.float64)
    if not tracks:
        return np.zeros((len(detections), 0), dtype=np.float64)
    detection_boxes = np.asarray([detection.tlwh for detection in detections], dtype=np.float64)
    track_boxes = np.asarray([track.tlwh for track in tracks], dtype=np.float64)
    return 1.0 - bbox_iou(detection_boxes, track_boxes)


def fuse_score(cost: np.ndarray, detections: Sequence[Detection]) -> np.ndarray:
    """Combina IoU y confianza: ``cost = 1 - IoU * score``."""
    if cost.size == 0 or not detections:
        return cost
    scores = np.asarray([detection.confidence for detection in detections], dtype=np.float64)
    similarity = np.clip(1.0 - cost, 0.0, 1.0) * scores[:, None]
    return np.clip(1.0 - similarity, 0.0, 1.0)


def linear_assignment(cost: np.ndarray, threshold: float) -> tuple[list[list[int]], list[int], list[int]]:
    """Hungarian con umbral: devuelve matches, detecciones y tracks sin emparejar."""
    if cost.size == 0:
        rows, columns = cost.shape
        return [], list(range(rows)), list(range(columns))

    row_indices, column_indices = linear_sum_assignment(cost)
    matches: list[list[int]] = []
    matched_rows: set[int] = set()
    matched_columns: set[int] = set()
    for row, column in zip(row_indices, column_indices):
        if cost[row, column] >= threshold:
            continue
        matches.append([int(row), int(column)])
        matched_rows.add(int(row))
        matched_columns.add(int(column))

    unmatched_detections = [i for i in range(cost.shape[0]) if i not in matched_rows]
    unmatched_tracks = [i for i in range(cost.shape[1]) if i not in matched_columns]
    return matches, unmatched_detections, unmatched_tracks


# ---------------------------------------------------------------------------
# Filtro de Kalman (xyah)
# ---------------------------------------------------------------------------
class KalmanFilter:
    """Filtro de Kalman lineal para cajas ``[cx, cy, aspect, height]``."""

    def __init__(self) -> None:
        position_weight = 1.0 / 20.0
        velocity_weight = 1.0 / 160.0
        self._position_weight = position_weight
        self._velocity_weight = velocity_weight

        # Matriz de transición: posición += velocidad; velocidad constante.
        self._transition = np.eye(8, dtype=np.float64)
        for index in range(4):
            self._transition[index, index + 4] = 1.0
        self._update_matrix = np.eye(4, 8, dtype=np.float64)

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Crea estado y covarianza a partir de la primera observación."""
        mean = np.concatenate([measurement, np.zeros(4, dtype=np.float64)])
        width, height = max(abs(measurement[2]), 1e-3), max(abs(measurement[3]), 1e-3)
        std = np.array(
            [
                self._position_weight * width,
                self._position_weight * width,
                self._position_weight * width,
                self._position_weight * height,
                self._velocity_weight * width,
                self._velocity_weight * width,
                self._velocity_weight * width,
                self._velocity_weight * height,
            ],
            dtype=np.float64,
        )
        return mean, np.diag(np.square(std))

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predice el estado del siguiente frame."""
        predicted_mean = self._transition @ mean
        predicted_covariance = self._transition @ covariance @ self._transition.T
        aspect = max(abs(predicted_mean[2]), 1e-3)
        height = max(abs(predicted_mean[3]), 1e-3)
        std = np.array(
            [
                self._position_weight * aspect,
                self._position_weight * aspect,
                self._position_weight * aspect,
                self._position_weight * height,
                self._velocity_weight * aspect,
                self._velocity_weight * aspect,
                self._velocity_weight * aspect,
                self._velocity_weight * height,
            ],
            dtype=np.float64,
        )
        return predicted_mean, predicted_covariance + np.diag(np.square(std))

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Proyecta el estado al espacio de observación (4D)."""
        aspect = max(abs(mean[2]), 1e-3)
        height = max(abs(mean[3]), 1e-3)
        std = np.array(
            [
                self._position_weight * aspect,
                self._position_weight * aspect,
                self._position_weight * aspect,
                self._position_weight * height,
            ],
            dtype=np.float64,
        )
        projected_mean = self._update_matrix @ mean
        projected_covariance = (
            self._update_matrix @ covariance @ self._update_matrix.T + np.diag(np.square(std))
        )
        return projected_mean, projected_covariance

    def update(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Corrige la predicción con una detección asociada."""
        projected_mean, projected_covariance = self.project(mean, covariance)
        regularized = projected_covariance + np.eye(4, dtype=np.float64) * 1e-6
        cross_covariance = covariance @ self._update_matrix.T
        kalman_gain = np.linalg.solve(regularized, cross_covariance.T).T
        innovation = measurement - projected_mean
        updated_mean = mean + innovation @ kalman_gain.T
        # Covarianza_post = C - K * S * K^T  (K: 8x4, S: 4x4 -> 8x8).
        updated_covariance = covariance - kalman_gain @ projected_covariance @ kalman_gain.T
        return updated_mean, updated_covariance


# ---------------------------------------------------------------------------
# Track
# ---------------------------------------------------------------------------
class TrackState(Enum):
    """Ciclo de vida interno de un track."""

    NEW = auto()
    TRACKED = auto()
    LOST = auto()
    REMOVED = auto()


class Track:
    """Trayectoria de una persona identificada."""

    def __init__(
        self,
        detection: Detection,
        track_id: int,
        max_lost: int,
        kalman: KalmanFilter,
    ) -> None:
        self.track_id = int(track_id)
        self.score = float(detection.confidence)
        self.class_id = int(detection.class_id)
        self.mean, self.covariance = kalman.initiate(detection.xyah)
        self.state = TrackState.TRACKED  # Confirmación inmediata (ver docstring).
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        self.max_lost = int(max_lost)
        self.history: list[np.ndarray] = []

    @property
    def tlwh(self) -> np.ndarray:
        """Caja predicha/actualizada en formato ``[x, y, w, h]``."""
        return xyah_to_tlwh(self.mean[:4])

    @property
    def is_alive(self) -> bool:
        """Indica si el track sigue en la lista activa."""
        return self.state != TrackState.REMOVED

    @property
    def is_tracked(self) -> bool:
        """Indica si este frame tuvo una asociación válida."""
        return self.state == TrackState.TRACKED and self.time_since_update == 0

    def predict(self, kalman: KalmanFilter) -> None:
        """Avanza el filtro un frame y envejece el track."""
        self.mean, self.covariance = kalman.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1
        self.history.append(self.tlwh)

    def update(self, detection: Detection, kalman: KalmanFilter) -> None:
        """Asocia una detección y corrige la predicción."""
        self.mean, self.covariance = kalman.update(self.mean, self.covariance, detection.xyah)
        self.score = float(detection.confidence)
        self.time_since_update = 0
        self.hits += 1
        self.state = TrackState.TRACKED

    def mark_lost(self) -> None:
        """Marca el track como perdido (sigue proyectándose unos frames)."""
        self.state = TrackState.LOST

    def mark_removed(self) -> None:
        """Elimina definitivamente el track de la sesión."""
        self.state = TrackState.REMOVED

    def to_view(self) -> TrackView:
        """Instantánea inmutable para contador y render."""
        return TrackView(
            track_id=self.track_id,
            tlwh=self.tlwh.copy(),
            confidence=self.score,
            age=self.age,
            state=self.state.name,
            class_id=self.class_id,
        )


# ---------------------------------------------------------------------------
# Tracker público
# ---------------------------------------------------------------------------
class ByteTrackTracker:
    """Tracker multi-persona thread-safe (se usa desde un único worker)."""

    def __init__(self, settings: TrackingSettings) -> None:
        self._match_threshold = float(settings.match_threshold)
        self._low_match_threshold = float(settings.low_match_threshold)
        self._max_lost = int(settings.max_lost)
        # El umbral "alto" se inyecta por sesión porque depende de la config
        # de detección; por defecto se usa un valor razonable.
        self._score_threshold = 0.50
        self._kalman = KalmanFilter()
        self._tracks: list[Track] = []
        self._next_id = 1
        self.frame_id = 0

    def configure_score_threshold(self, threshold: float) -> None:
        """Establece la frontera entre detecciones altas y bajas."""
        self._score_threshold = max(0.01, min(0.99, float(threshold)))

    def reset(self) -> None:
        """Reinicia todos los tracks (nueva sesión de conteo)."""
        self._tracks.clear()
        self._next_id = 1
        self.frame_id = 0

    def update(self, detections: Iterable[Detection]) -> list[TrackView]:
        """ Ejecuta un paso de ByteTrack y devuelve los tracks visibles. """
        self.frame_id += 1
        detection_list = list(detections)

        high = [d for d in detection_list if d.confidence >= self._score_threshold]
        low = [d for d in detection_list if d.confidence < self._score_threshold]

        # 1) Predicción de todos los tracks vivos.
        for track in self._tracks:
            track.predict(self._kalman)
        tracks = self._tracks

        # 2) Primera asociación: detecciones de alta confianza.
        matches: list[list[int]] = []
        unmatched_detections: list[int] = list(range(len(high)))
        unmatched_tracks: list[int] = list(range(len(tracks)))

        if high and tracks:
            cost = fuse_score(iou_cost(high, tracks), high)
            matches, unmatched_detections, unmatched_tracks = linear_assignment(
                cost, self._match_threshold
            )
            for detection_index, track_index in matches:
                tracks[track_index].update(high[detection_index], self._kalman)

        # 3) Segunda asociación: detecciones de baja confianza con tracks ya
        #    seguidos. Es la clave de ByteTrack para recuperar oclusiones.
        low_matched_tracks: set[int] = set()
        if low and unmatched_tracks:
            candidates = [
                (global_index, tracks[global_index])
                for global_index in unmatched_tracks
                if tracks[global_index].state == TrackState.TRACKED
            ]
            if candidates:
                global_indices = [index for index, _ in candidates]
                candidate_tracks = [track for _, track in candidates]
                low_cost = iou_cost(low, candidate_tracks)
                low_matches, _, _ = linear_assignment(low_cost, self._low_match_threshold)
                for detection_index, candidate_index in low_matches:
                    global_index = global_indices[candidate_index]
                    tracks[global_index].update(low[detection_index], self._kalman)
                    low_matched_tracks.add(global_index)

        # 4) Detecciones altas sin asociar -> personas nuevas.
        for detection_index in unmatched_detections:
            self._tracks.append(
                Track(high[detection_index], self._next_id, self._max_lost, self._kalman)
            )
            self._next_id += 1

        # 5) Tracks sin asociar -> perdidos (o borrados si eran nuevos).
        for track_index in unmatched_tracks:
            if track_index in low_matched_tracks:
                continue
            track = tracks[track_index]
            if track.state == TrackState.NEW:
                track.mark_removed()
            else:
                track.mark_lost()

        # 6) Limpieza de tracks perdidos durante demasiados frames.
        for track in self._tracks:
            if track.state == TrackState.LOST and track.time_since_update > self._max_lost:
                track.mark_removed()
        self._tracks = [track for track in self._tracks if track.is_alive]

        return sorted(
            (track.to_view() for track in self._tracks if track.is_tracked),
            key=lambda view: view.track_id,
        )

    @property
    def track_count(self) -> int:
        """Número de tracks vivos (incluyendo perdidos temporalmente)."""
        return len(self._tracks)
