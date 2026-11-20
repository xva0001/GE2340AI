"""Configuration objects and COCO class mappings for the seat finder.

The prototype uses a YOLO model trained on COCO, so seat and belonging classes
come from that label set.  Keeping the mapping here makes it straightforward to
swap in a fine-tuned model later: only this table needs to change.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --- COCO class ids used by the pipeline -------------------------------------
PERSON_CLASS_ID = 0

SEAT_CLASS_LABELS: dict[int, str] = {
    13: "bench",
    56: "chair",
    57: "couch",
}

#: Personal items that can be left on a seat.  ``book`` is deliberately absent:
#: the laminated signs taped to chair backs in study spaces are reliably
#: misdetected as books and would make almost every seat look occupied.
BELONGING_CLASS_LABELS: dict[int, str] = {
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    28: "suitcase",
    39: "bottle",
    41: "cup",
    63: "laptop",
    64: "mouse",
    66: "keyboard",
    67: "cell phone",
}

# Context detections that are never seats but help place a person at a desk.
CONTEXT_CLASS_LABELS: dict[int, str] = {
    60: "dining table",
}

TRACKED_CLASS_IDS: tuple[int, ...] = (
    PERSON_CLASS_ID,
    *SEAT_CLASS_LABELS,
    *BELONGING_CLASS_LABELS,
    *CONTEXT_CLASS_LABELS,
)


@dataclass(frozen=True)
class DetectorConfig:
    """Settings for the YOLO wrapper."""

    weights: str = "yolov10s.pt"
    imgsz: int = 1280
    conf: float = 0.25
    #: Belongings are small and rare, so they get a lower confidence floor.
    belonging_conf: float = 0.15
    #: Tables are only used to tell "on the table" from "on the seat"; a very
    #: low floor is fine because they never become seats themselves.
    table_conf: float = 0.08
    iou: float = 0.50
    max_det: int = 300
    device: str | None = None
    track: bool = True
    tracker: str = "bytetrack.yaml"


@dataclass(frozen=True)
class SeatConfig:
    """Thresholds for turning furniture detections into tracked seats."""

    seat_conf: float = 0.30
    #: Two furniture boxes overlapping more than this are the same seat.
    duplicate_iou: float = 0.45
    #: A box mostly contained in another furniture box is the same seat.
    duplicate_containment: float = 0.80
    #: Frames a furniture detection must persist before it becomes a seat.
    confirm_hits: int = 3
    #: IoU needed to carry a seat's id across frames.
    match_iou: float = 0.25
    #: Frames a seat survives after it was last detected.
    ttl_frames: int = 60
    #: A seat keeps being drawn (with its last state) for this many frames
    #: after the detector loses it, so boxes and labels do not blink.  Kept
    #: short so a fast camera pan does not leave boxes trailing behind.
    paint_grace_frames: int = 4
    #: Margin added to a seat cell when testing whether a person is on it.
    attach_margin_x: float = 0.25
    attach_margin_y: float = 0.35
    #: Minimum (IoU + centre bonus) for a person to count as seated on a cell.
    person_min_score: float = 0.15
    #: A belonging must lie this far inside a seat cell to count as left on it.
    #: High enough to reject items on a table that a loose couch box overlaps.
    belonging_inside_ratio: float = 0.70
    #: Relative depth (0 = top of the seat box) an object's centre must reach.
    #: A chair box includes the backrest, so a belonging actually left on the
    #: seat sits in its lower half; items higher up are usually background or a
    #: loose couch box covering the table beside it.
    belonging_min_depth: float = 0.45
    #: A belonging covering more of the seat than this is a bad detection.
    belonging_max_area_ratio: float = 0.45
    #: An object this far inside a detected table is on the table, not a seat.
    #: Loose couch boxes sometimes cover the coffee table next to them.
    belonging_table_ratio: float = 0.80
    #: Belongings overlapping a person this much are being carried, not left.
    belonging_held_iou: float = 0.20
    #: Nominal person width, as a fraction of furniture box width, used to
    #: estimate how many people a couch or bench can seat.
    person_width_fraction: float = 0.55
    #: A person box this far inside a larger person box is the same person
    #: tracked twice (full body vs upper body); the smaller one is dropped.
    person_duplicate_containment: float = 0.80


@dataclass(frozen=True)
class InferenceSeatConfig:
    """Seats inferred from a stationary, seated-looking person.

    COCO models frequently miss the chair a person is sitting on, so an
    occupied chair would otherwise never appear in the seat inventory.  When a
    person is still for ``min_frames``, looks seated, and sits next to
    furniture, the pipeline registers a synthetic seat for them.
    """

    enabled: bool = True
    min_conf: float = 0.35
    #: Seconds (converted to frames by the pipeline) a person must stay put.
    min_still_seconds: float = 0.6
    #: Max height/width ratio; standing people are around 2.5-4.
    max_aspect: float = 2.30
    #: Max distance to the nearest furniture, in person heights.
    max_furniture_distance: float = 1.60
    #: Allowed drift while "still", as a fraction of frame width.
    still_fraction: float = 0.025
    #: How far a person may move from where their inferred seat was created
    #: before the seat is released, as a fraction of frame width.
    anchor_fraction: float = 0.06
    #: Frames a synthetic seat survives after the person disappears.
    ttl_frames: int = 45


@dataclass(frozen=True)
class TemporalConfig:
    """Hysteresis applied to every seat state change."""

    #: Consecutive frames a new state must be observed before switching to it.
    confirm_frames: int = 8
    #: Objects are static, so "occupied with objects" confirms faster.
    ghost_confirm_frames: int = 5
    #: Consecutive frames needed to leave occupied / ghost states.
    release_frames: int = 15
    #: Frames a detected belonging stays "on the seat" while the detector
    #: flickers.  Small items are detected intermittently; without this the
    #: ghost count would never reach its confirmation threshold.
    object_memory_frames: int = 4


@dataclass(frozen=True)
class AppConfig:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    seat: SeatConfig = field(default_factory=SeatConfig)
    inferred: InferenceSeatConfig = field(default_factory=InferenceSeatConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)


@dataclass(frozen=True)
class RenderConfig:
    """Visual style of the annotated video."""

    seat_thickness: int = 2
    person_thickness: int = 1
    object_thickness: int = 1
    show_persons: bool = True
    show_objects: bool = True
    show_hud: bool = True
    #: Boxes smaller than this many pixels are not labelled (keeps video clean).
    min_label_height: int = 18
    #: Person boxes stay on screen this many frames after they are last seen,
    #: so a single missed detection does not make the label blink.
    person_grace_frames: int = 2
