"""Thin wrapper around Ultralytics YOLO returning plain detections.

Only this module knows about Ultralytics; everything downstream works with the
small :class:`Detection` value object.  That keeps the seating logic testable
and makes swapping detectors (a fine-tuned model, a different runtime) local.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .config import (
    BELONGING_CLASS_LABELS,
    CONTEXT_CLASS_LABELS,
    PERSON_CLASS_ID,
    SEAT_CLASS_LABELS,
    TRACKED_CLASS_IDS,
    DetectorConfig,
)
from .geometry import Box

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ultralytics import YOLO

_LABELS: dict[int, str] = {
    PERSON_CLASS_ID: "person",
    **SEAT_CLASS_LABELS,
    **BELONGING_CLASS_LABELS,
    **CONTEXT_CLASS_LABELS,
}


@dataclass(frozen=True)
class Detection:
    """A single object detection in pixel coordinates."""

    class_id: int
    label: str
    confidence: float
    box: Box
    track_id: int | None = None

    @property
    def is_person(self) -> bool:
        return self.class_id == PERSON_CLASS_ID


class Detector:
    """Runs a YOLO model on frames, with optional multi-object tracking."""

    def __init__(self, cfg: DetectorConfig) -> None:
        from ultralytics import YOLO  # heavy import, kept local

        self.cfg = cfg
        self._model: YOLO = YOLO(cfg.weights)

    @property
    def model(self) -> YOLO:
        return self._model

    def reset_tracking(self) -> None:
        """Forget tracker state so track ids do not leak between videos."""
        self._model.predictor = None

    def infer(self, frame: np.ndarray) -> list[Detection]:
        """Detect tracked objects of interest in a BGR frame."""
        kwargs: dict[str, object] = {
            "imgsz": self.cfg.imgsz,
            "conf": min(self.cfg.conf, self.cfg.belonging_conf, self.cfg.table_conf),
            "iou": self.cfg.iou,
            "max_det": self.cfg.max_det,
            "classes": list(TRACKED_CLASS_IDS),
            "verbose": False,
        }
        if self.cfg.device:
            kwargs["device"] = self.cfg.device

        if self.cfg.track:
            results = self._model.track(
                frame, persist=True, tracker=self.cfg.tracker, **kwargs
            )
        else:
            results = self._model.predict(frame, **kwargs)

        detections = _parse(results[0])
        return [d for d in detections if d.confidence >= _threshold(d, self.cfg)]


def _threshold(detection: Detection, cfg: DetectorConfig) -> float:
    if detection.class_id in BELONGING_CLASS_LABELS:
        return cfg.belonging_conf
    if detection.class_id in CONTEXT_CLASS_LABELS:
        return cfg.table_conf
    return cfg.conf


def _parse(result: object) -> list[Detection]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)

    ids = getattr(boxes, "id", None)
    track_ids = ids.cpu().numpy().astype(int) if ids is not None else None

    detections: list[Detection] = []
    for index, ((x1, y1, x2, y2), conf, cls) in enumerate(
        zip(xyxy, confs, classes)
    ):
        class_id = int(cls)
        track_id = int(track_ids[index]) if track_ids is not None else None
        detections.append(
            Detection(
                class_id=class_id,
                label=_LABELS.get(class_id, str(class_id)),
                confidence=float(conf),
                box=(float(x1), float(y1), float(x2), float(y2)),
                track_id=track_id,
            )
        )
    return detections


def split_by_role(
    detections: list[Detection],
) -> tuple[list[Detection], list[Detection], list[Detection]]:
    """Split detections into ``(persons, furniture, belongings)``."""
    persons: list[Detection] = []
    furniture: list[Detection] = []
    belongings: list[Detection] = []
    for det in detections:
        if det.class_id == PERSON_CLASS_ID:
            persons.append(det)
        elif det.class_id in SEAT_CLASS_LABELS:
            furniture.append(det)
        elif det.class_id in BELONGING_CLASS_LABELS:
            belongings.append(det)
    return persons, furniture, belongings
