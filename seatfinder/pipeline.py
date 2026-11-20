"""Per-video pipeline: detect -> track seats -> associate -> smooth -> render."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .association import CellObservation, associate
from .config import AppConfig, RenderConfig
from .detector import Detection, Detector, split_by_role
from .geometry import (
    Box,
    area,
    bottom_center,
    containment,
    distance,
    height,
    point_inside,
    width,
)
from .renderer import HudInfo, Renderer
from .state import CellStatus, SeatState, SeatStateMachine, raw_state
from .tracking import GlobalMotion, PersonHistory, SeatTracker
from .video_io import CsvRecorder, VideoInfo, VideoReader, VideoWriter


@dataclass
class TimelineEntry:
    seat_id: int
    cell: int
    kind: str
    state: str
    frame: int
    time_s: float


@dataclass
class VideoResult:
    video: Path
    info: VideoInfo
    output_video: Path | None
    timelines: list[TimelineEntry] = field(default_factory=list)
    frame_totals: list[dict] = field(default_factory=list)
    seat_cell_count: int = 0
    frames_processed: int = 0
    fps_processed: float = 0.0


class VideoPipeline:
    """Processes one video file end to end."""

    def __init__(
        self,
        cfg: AppConfig,
        detector: Detector,
        render_cfg: RenderConfig | None = None,
        render: bool = True,
        stride: int = 1,
        max_frames: int | None = None,
        progress: bool = True,
        stills: int = 0,
    ) -> None:
        self.cfg = cfg
        self.detector = detector
        self.renderer = Renderer(render_cfg or RenderConfig())
        self.render = render
        self.stride = max(1, stride)
        self.max_frames = max_frames
        self.progress = progress
        self.stills = max(0, stills)
        self._persons: list[Detection] = []
        self._belongings: list[Detection] = []
        self._timelines: list[TimelineEntry] = []
        self._fps: float = 30.0
        self._object_memory: dict[tuple[int, int], tuple[int, tuple[str, ...]]] = {}
        self._recent_persons: dict[int, tuple[int, Detection]] = {}
        self._synthetic_anchor: dict[int, tuple[float, float]] = {}

    def run(
        self,
        video_path: Path,
        output_dir: Path,
        csv: CsvRecorder | None = None,
        stills_dir: Path | None = None,
    ) -> VideoResult:
        self.detector.reset_tracking()
        seats = SeatTracker(self.cfg.seat)
        history = PersonHistory()
        motion = GlobalMotion()
        machine = SeatStateMachine(self.cfg.temporal)
        self._timelines = []
        self._object_memory = {}
        self._recent_persons = {}
        self._synthetic_anchor = {}

        with VideoReader(video_path) as reader:
            info = reader.info
            self._fps = info.fps if info.fps else 30.0
            still_pixels = self.cfg.inferred.still_fraction * max(info.width, 1)
            min_still = max(
                3, int(round(self.cfg.inferred.min_still_seconds * info.fps))
            )
            output_video = (
                output_dir / f"{video_path.stem}_annotated.mp4" if self.render else None
            )
            even_size = (
                info.width - info.width % 2,
                info.height - info.height % 2,
            )
            writer = (
                VideoWriter(
                    output_video, fps=info.fps / self.stride, size=even_size
                )
                if output_video is not None
                else None
            )

            result = VideoResult(
                video=video_path, info=info, output_video=output_video
            )
            started = time.perf_counter()
            still_frames: set[int] = set()
            if self.stills and stills_dir is not None and info.frame_count > 0:
                still_frames = {
                    int(info.frame_count * (i + 0.5) / self.stills)
                    for i in range(self.stills)
                }

            try:
                for index, frame in reader:
                    if index % self.stride:
                        continue
                    if (
                        self.max_frames is not None
                        and result.frames_processed >= self.max_frames
                    ):
                        break

                    statuses = self._process_frame(
                        frame, index, seats, history, motion, machine,
                        still_pixels, min_still,
                    )

                    time_s = index / info.fps if info.fps else 0.0
                    counts = _count(statuses)
                    if csv is not None:
                        self._record_csv(
                            csv, video_path.name, index, time_s, seats, statuses
                        )
                    result.frame_totals.append(
                        {
                            "frame": index,
                            "time_s": round(time_s, 3),
                            **counts,
                        }
                    )

                    if writer is not None:
                        elapsed = time.perf_counter() - started
                        fps = (
                            (result.frames_processed + 1) / elapsed
                            if elapsed
                            else None
                        )
                        hud = HudInfo(
                            frame_index=index,
                            time_s=time_s,
                            empty=counts["empty"],
                            occupied=counts["occupied"],
                            ghost=counts["objects"],
                            fps=fps,
                            total_frames=info.frame_count,
                        )
                        annotated = self.renderer.draw(
                            frame, seats.all_seats(), statuses,
                            self._persons, self._belongings, hud,
                        )
                        writer.write(
                            annotated[: even_size[1], : even_size[0]]
                        )
                        if index in still_frames and stills_dir is not None:
                            stills_dir.mkdir(parents=True, exist_ok=True)
                            cv2.imwrite(
                                str(stills_dir / f"{video_path.stem}_{index:05d}.jpg"),
                                annotated,
                                [cv2.IMWRITE_JPEG_QUALITY, 92],
                            )

                    result.frames_processed += 1
                    if self.progress and result.frames_processed % 50 == 0:
                        print(
                            f"    {video_path.name}: {result.frames_processed} frames",
                            flush=True,
                        )
            finally:
                if writer is not None:
                    writer.close()

            elapsed = time.perf_counter() - started
            result.fps_processed = (
                result.frames_processed / elapsed if elapsed > 0 else 0.0
            )
            result.timelines = list(self._timelines)
            result.seat_cell_count = len(
                {seat_id for seat_id, _ in machine.active_keys()}
            )
            return result

    # -- one frame ---------------------------------------------------------
    def _process_frame(
        self,
        frame: np.ndarray,
        index: int,
        seats: SeatTracker,
        history: PersonHistory,
        motion: GlobalMotion,
        machine: SeatStateMachine,
        still_pixels: float,
        min_still: int,
    ) -> dict[tuple[int, int], CellStatus]:
        detections = self.detector.infer(frame)
        persons, furniture, belongings = split_by_role(detections)
        persons = self._dedupe_persons(persons)
        self._persons = self._remember_persons(persons, index)
        self._belongings = belongings

        motion.update(persons, frame.shape[1])
        history.update(index, persons, motion.offset)

        seats.reset_frame_flags()
        person_widths = [width(p.box) for p in persons if width(p.box) > 1]
        live = seats.update(furniture, index, person_widths)
        table_boxes = [d.box for d in detections if d.label == "dining table"]
        furniture_boxes = [s.box for s in live] + table_boxes

        result = associate(
            [s for s in live if not s.synthetic],
            persons,
            belongings,
            self.cfg.seat,
            table_boxes,
        )

        active_synthetic: set[int] = set()
        anchor_radius = self.cfg.inferred.anchor_fraction * frame.shape[1]
        offset = motion.offset
        for person in result.unmatched_persons:
            if person.track_id is None:
                continue
            feet = bottom_center(person.box)
            compensated = (feet[0] - offset[0], feet[1] - offset[1])
            anchor = self._synthetic_anchor.get(person.track_id)
            near_anchor = (
                anchor is not None
                and distance(compensated, anchor) <= anchor_radius
                and seats.synthetic_for(person.track_id) is not None
            )
            if not near_anchor and not self._is_seated_candidate(
                person, history, still_pixels, min_still,
                furniture_boxes, result.cells,
            ):
                continue
            seat = seats.upsert_synthetic(
                person.track_id, person.box, index, person.confidence
            )
            self._synthetic_anchor.setdefault(person.track_id, compensated)
            active_synthetic.add(person.track_id)
            result.cells[(seat.seat_id, 0)] = CellObservation(
                seat_id=seat.seat_id,
                cell_index=0,
                box=person.box,
                person_id=person.track_id,
                score=1.0,
            )

        seats.prune_synthetic(active_synthetic, index)
        for track_id in list(self._synthetic_anchor):
            if seats.synthetic_for(track_id) is None:
                del self._synthetic_anchor[track_id]

        statuses: dict[tuple[int, int], CellStatus] = {}
        for seat in seats.all_seats():
            if not seat.observed:
                continue
            for cell_index, _ in enumerate(seat.cells):
                key = (seat.seat_id, cell_index)
                observation = result.cells.get(key)
                person_id = None if observation is None else observation.person_id
                objects = () if observation is None else tuple(observation.objects)
                objects = self._remember_objects(key, objects, index)
                status = machine.update(
                    key, raw_state(person_id is not None, bool(objects)),
                    index, person_id, objects,
                )
                statuses[key] = status
                if status.since_frame == index:
                    self._timelines.append(
                        TimelineEntry(
                            seat_id=seat.seat_id,
                            cell=cell_index,
                            kind=seat.kind,
                            state=status.state.value,
                            frame=index,
                            time_s=index / self._fps,
                        )
                    )

        machine.prune({s.seat_id for s in seats.all_seats()}, index)
        return self._paint_recent(statuses, seats, machine, index)

    def _paint_recent(
        self,
        statuses: dict[tuple[int, int], CellStatus],
        seats: SeatTracker,
        machine: SeatStateMachine,
        index: int,
    ) -> dict[tuple[int, int], CellStatus]:
        """Keep recently seen seats on screen through a few missed detections.

        The detector drops a chair for a frame or two, and especially an
        inferred seat disappears whenever its person fails the "is still" test
        for one frame.  Painting the last known state for a short grace period
        removes that blinking without changing the state machine.
        """
        grace = self.cfg.seat.paint_grace_frames
        for seat in seats.all_seats():
            for cell_index, _ in enumerate(seat.cells):
                key = (seat.seat_id, cell_index)
                if key in statuses:
                    continue
                status = machine.get(key)
                if status is not None and index - status.seen_frame <= grace:
                    statuses[key] = status
        return statuses

    def _dedupe_persons(self, persons: list[Detection]) -> list[Detection]:
        """Drop a person box that is really the same person tracked twice.

        ByteTrack sometimes keeps a full-body box and an upper-body box for one
        person; both would otherwise get their own inferred seat.
        """
        kept: list[Detection] = []
        for person in sorted(persons, key=lambda d: area(d.box), reverse=True):
            if any(
                containment(person.box, other.box)
                >= self.cfg.seat.person_duplicate_containment
                for other in kept
            ):
                continue
            kept.append(person)
        return kept

    def _remember_persons(
        self, persons: list[Detection], index: int
    ) -> list[Detection]:
        """Keep a person box visible for a couple of frames after a miss."""
        grace = self.renderer.cfg.person_grace_frames
        for person in persons:
            if person.track_id is not None:
                self._recent_persons[person.track_id] = (index, person)
        stale = [
            track_id
            for track_id, (seen, _) in self._recent_persons.items()
            if index - seen > grace
        ]
        for track_id in stale:
            del self._recent_persons[track_id]
        current = {p.track_id for p in persons if p.track_id is not None}
        return list(persons) + [
            detection
            for track_id, (_, detection) in self._recent_persons.items()
            if track_id not in current
        ]

    def _remember_objects(
        self,
        key: tuple[int, int],
        objects: tuple[str, ...],
        index: int,
    ) -> tuple[str, ...]:
        """Keep a belonging visible for a few frames after the detector drops it."""
        memory = self.cfg.temporal.object_memory_frames
        if objects:
            self._object_memory[key] = (index, objects)
            return objects
        remembered = self._object_memory.get(key)
        if remembered is None:
            return objects
        last_frame, labels = remembered
        if index - last_frame <= memory:
            return labels
        del self._object_memory[key]
        return objects

    def _is_seated_candidate(
        self,
        person: Detection,
        history: PersonHistory,
        still_pixels: float,
        min_still: int,
        furniture_boxes: list[Box],
        cells: dict[tuple[int, int], CellObservation],
    ) -> bool:
        cfg = self.cfg.inferred
        if not cfg.enabled:
            return False
        if person.confidence < cfg.min_conf:
            return False
        box = person.box
        w = width(box)
        h = height(box)
        if w <= 1 or h <= 1:
            return False
        if h / w > cfg.max_aspect:
            return False
        feet = bottom_center(box)
        # A person whose feet fall inside a known seat cell is sitting on
        # detected furniture; never invent a second seat for them.
        if any(point_inside(obs.box, feet) for obs in cells.values()):
            return False
        if history.samples(person.track_id) < min_still:
            return False
        still = history.stillness(person.track_id)
        if still is None or still > still_pixels:
            return False
        if not furniture_boxes:
            return False
        limit = cfg.max_furniture_distance * h
        nearest = min(
            distance(feet, _closest_point(box_, feet)) for box_ in furniture_boxes
        )
        return nearest <= limit

    def _record_csv(
        self,
        csv: CsvRecorder,
        name: str,
        index: int,
        time_s: float,
        seats: SeatTracker,
        statuses: dict[tuple[int, int], CellStatus],
    ) -> None:
        counts = _count(statuses)
        for seat in seats.all_seats():
            for cell_index, box in enumerate(seat.cells):
                status = statuses.get((seat.seat_id, cell_index))
                if status is None:
                    continue
                csv.add_seat(
                    video=name,
                    frame=index,
                    time_s=time_s,
                    seat_id=seat.seat_id,
                    cell=cell_index,
                    kind=seat.kind,
                    state=status.state.value,
                    person_id=status.person_id,
                    objects=list(status.objects),
                    box=box,
                )
        csv.add_frame(
            name, index, time_s, counts["seats"], counts["empty"],
            counts["occupied"], counts["objects"],
        )


def _count(statuses: dict[tuple[int, int], CellStatus]) -> dict[str, int]:
    counts = {"seats": 0, "empty": 0, "occupied": 0, "objects": 0}
    for status in statuses.values():
        counts["seats"] += 1
        if status.state is SeatState.EMPTY:
            counts["empty"] += 1
        elif status.state is SeatState.OCCUPIED:
            counts["occupied"] += 1
        else:
            counts["objects"] += 1
    return counts


def _closest_point(box: Box, point: tuple[float, float]) -> tuple[float, float]:
    x = min(max(point[0], box[0]), box[2])
    y = min(max(point[1], box[1]), box[3])
    return (x, y)
