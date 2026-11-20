"""Drawing the annotated video: seats, people, belongings and the HUD."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import RenderConfig
from .detector import Detection
from .geometry import Box, height
from .state import CellStatus, SeatState
from .tracking import Seat

COLORS = {
    SeatState.EMPTY: (90, 200, 90),
    SeatState.OCCUPIED: (60, 60, 235),
    SeatState.GHOST: (0, 165, 255),
}
PERSON_COLOR = (255, 150, 60)
OBJECT_COLOR = (255, 255, 0)
PANEL_COLOR = (28, 28, 28)

STATE_LABEL = {
    SeatState.EMPTY: "EMPTY",
    SeatState.OCCUPIED: "OCCUPIED",
    SeatState.GHOST: "OBJECTS",
}


def _overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    """True when two pixel rectangles intersect."""
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


@dataclass
class HudInfo:
    frame_index: int
    time_s: float
    empty: int
    occupied: int
    ghost: int
    fps: float | None = None
    total_frames: int | None = None

    @property
    def seats(self) -> int:
        return self.empty + self.occupied + self.ghost


class Renderer:
    """Draws the seat overlay.  Stateless apart from its config."""

    def __init__(self, cfg: RenderConfig) -> None:
        self.cfg = cfg
        self._labels: list[tuple[int, int, int, int]] = []

    def draw(
        self,
        frame: np.ndarray,
        seats: list[Seat],
        statuses: dict[tuple[int, int], CellStatus],
        persons: list[Detection],
        belongings: list[Detection],
        hud: HudInfo,
    ) -> np.ndarray:
        self._labels = []
        # HUD text is sized in real pixels with a floor, so a 480p clip stays
        # legible instead of shrinking with the frame.
        height_px = frame.shape[0]
        hud_scale = float(np.clip(height_px / 740.0, 0.62, 1.15))
        label_scale = max(0.40, hud_scale * 0.58)
        if self.cfg.show_persons:
            self._draw_detections(frame, persons, PERSON_COLOR, label_scale, prefix="P")
        if self.cfg.show_objects:
            self._draw_detections(frame, belongings, OBJECT_COLOR, label_scale, prefix="")
        for seat in seats:
            self._draw_seat(frame, seat, statuses, label_scale)
        if self.cfg.show_hud:
            self._draw_hud(frame, hud, hud_scale)
        return frame

    # -- internals ---------------------------------------------------------
    def _draw_seat(
        self,
        frame: np.ndarray,
        seat: Seat,
        statuses: dict[tuple[int, int], CellStatus],
        label_scale: float,
    ) -> None:
        cells = seat.cells
        for index, box in enumerate(cells):
            status = statuses.get((seat.seat_id, index))
            if status is None:
                continue
            color = COLORS[status.state]
            if status.state is not SeatState.EMPTY:
                self._fill_alpha(frame, box, color, 0.16)

            thickness = self.cfg.seat_thickness
            if seat.synthetic:
                thickness = 1
            self._rect(frame, box, color, thickness)

            if height(box) < self.cfg.min_label_height:
                continue
            label = self._seat_label(seat, index, status)
            self._label(frame, box, label, color, label_scale, above=True)

    def _seat_label(self, seat: Seat, index: int, status: CellStatus) -> str:
        if seat.synthetic:
            name = f"~P{status.person_id if status.person_id is not None else '?'}"
        else:
            name = f"S{seat.seat_id}"
            if seat.capacity > 1:
                name = f"{name}.{index + 1}"
        text = f"{name} {STATE_LABEL[status.state]}"
        if status.state is SeatState.GHOST and status.objects:
            text = f"{name} {', '.join(sorted(set(status.objects)))}"
        return text

    def _draw_detections(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        color: tuple[int, int, int],
        label_scale: float,
        prefix: str,
    ) -> None:
        for det in detections:
            self._rect(frame, det.box, color, self.cfg.person_thickness)
            if height(det.box) < 36:
                continue
            name = det.label if not prefix else f"{prefix}{det.track_id or '?'}"
            self._label(frame, det.box, name, color, label_scale * 0.85, above=False)

    def _draw_hud(self, frame: np.ndarray, hud: HudInfo, fs: float) -> None:
        font = cv2.FONT_HERSHEY_DUPLEX
        pad = max(7, int(round(9 * fs)))
        gap = max(4, int(round(6 * fs)))
        chip = max(9, int(round(10 * fs)))
        margin = max(6, int(round(8 * fs)))
        bold = 2 if fs >= 0.9 else 1

        title = "SEAT FINDER"
        (title_w, title_h), title_b = cv2.getTextSize(title, font, fs, bold)
        counter = f"frame {hud.frame_index}"
        if hud.total_frames:
            counter += f"/{hud.total_frames}"
        (count_w, _), _ = cv2.getTextSize(counter, font, fs * 0.8, 1)
        line1_h = title_h + title_b
        line1_w = title_w + gap * 2 + count_w

        counts = (
            (SeatState.EMPTY, hud.empty, "empty"),
            (SeatState.OCCUPIED, hud.occupied, "occupied"),
            (SeatState.GHOST, hud.ghost, "objects"),
        )
        parts: list[tuple[tuple[int, int, int], str, int, int]] = []
        for state, number, label in counts:
            text = f"{number} {label}"
            (text_w, text_h), text_b = cv2.getTextSize(text, font, fs * 0.85, 1)
            parts.append((COLORS[state], text, text_w, text_h + text_b))
        line2_h = max(part[3] for part in parts)
        line2_w = sum(chip + gap // 2 + part[2] for part in parts) + gap * (len(parts) - 1)

        time_text = f"{hud.time_s:.2f} s"
        if hud.fps:
            time_text += f"   {hud.fps:.1f} fps"
        (time_w, time_h), time_b = cv2.getTextSize(time_text, font, fs * 0.75, 1)

        content_w = max(line1_w, line2_w, time_w)
        panel_w = content_w + pad * 2
        panel_h = pad * 2 + line1_h + gap + line2_h + gap + time_h + time_b
        x0, y0 = margin, margin

        self._fill_alpha(frame, (x0, y0, x0 + panel_w, y0 + panel_h), PANEL_COLOR, 0.78)
        cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h),
                      (95, 95, 95), 1, cv2.LINE_AA)

        title_y = y0 + pad + title_h
        cv2.putText(frame, title, (x0 + pad, title_y), font, fs,
                    (255, 255, 255), bold, cv2.LINE_AA)
        cv2.putText(frame, counter, (x0 + panel_w - pad - count_w, title_y), font,
                    fs * 0.8, (185, 185, 185), 1, cv2.LINE_AA)

        counts_y = y0 + pad + line1_h + gap
        x = x0 + pad
        chip_y = counts_y + (line2_h - chip) // 2
        for color, text, text_w, text_h in parts:
            cv2.rectangle(frame, (x, chip_y), (x + chip, chip_y + chip), color, -1)
            cv2.rectangle(frame, (x, chip_y), (x + chip, chip_y + chip),
                          (25, 25, 25), 1, cv2.LINE_AA)
            cv2.putText(frame, text, (x + chip + gap // 2, counts_y + text_h - 3),
                        font, fs * 0.85, (240, 240, 240), 1, cv2.LINE_AA)
            x += chip + gap // 2 + text_w + gap

        time_y = counts_y + line2_h + gap + time_h
        cv2.putText(frame, time_text, (x0 + pad, time_y), font, fs * 0.75,
                    (190, 190, 190), 1, cv2.LINE_AA)

    # -- primitives --------------------------------------------------------
    @staticmethod
    def _rect(frame: np.ndarray, box: Box, color: tuple[int, int, int],
              thickness: int) -> None:
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    @staticmethod
    def _fill_alpha(frame: np.ndarray, box: Box, color: tuple[int, int, int],
                    alpha: float) -> None:
        h, w = frame.shape[:2]
        x1 = max(0, int(round(box[0])))
        y1 = max(0, int(round(box[1])))
        x2 = min(w, int(round(box[2])))
        y2 = min(h, int(round(box[3])))
        if x2 <= x1 or y2 <= y1:
            return
        roi = frame[y1:y2, x1:x2]
        overlay = np.empty_like(roi)
        overlay[:] = color
        cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0.0, roi)

    def _label(self, frame: np.ndarray, box: Box, text: str,
               color: tuple[int, int, int], font_scale: float,
               above: bool) -> None:
        font = cv2.FONT_HERSHEY_DUPLEX
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        h, w = frame.shape[:2]
        x = int(round(box[0]))
        if above:
            top = int(round(box[1])) - th - baseline - 4
        else:
            top = int(round(box[1])) + 3
        top = max(0, min(top, h - th - baseline - 3))
        x = max(0, min(x, w - tw - 8))

        # Nudge a label upwards while it would cover one already drawn, so
        # neighbouring seats stay readable instead of printing over each other.
        step = th + baseline + 3
        for _ in range(4):
            rect = (x, top, x + tw + 6, top + th + baseline + 2)
            if not any(_overlaps(rect, other) for other in self._labels):
                break
            if top - step < 0:
                break
            top -= step
        self._labels.append(rect)
        cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), color, -1)
        cv2.putText(frame, text, (x + 3, top + th + 1), font, font_scale,
                    (15, 15, 15), thickness, cv2.LINE_AA)
