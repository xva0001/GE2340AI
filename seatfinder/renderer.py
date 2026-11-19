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
TEXT_COLOR = (245, 245, 245)
PANEL_COLOR = (28, 28, 28)

STATE_LABEL = {
    SeatState.EMPTY: "EMPTY",
    SeatState.OCCUPIED: "OCCUPIED",
    SeatState.GHOST: "OBJECTS",
}


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

    def draw(
        self,
        frame: np.ndarray,
        seats: list[Seat],
        statuses: dict[tuple[int, int], CellStatus],
        persons: list[Detection],
        belongings: list[Detection],
        hud: HudInfo,
    ) -> np.ndarray:
        scale = float(np.clip(frame.shape[0] / 720.0, 0.5, 1.7))
        if self.cfg.show_persons:
            self._draw_detections(frame, persons, PERSON_COLOR, scale, prefix="P")
        if self.cfg.show_objects:
            self._draw_detections(frame, belongings, OBJECT_COLOR, scale, prefix="")
        for seat in seats:
            self._draw_seat(frame, seat, statuses, scale)
        if self.cfg.show_hud:
            self._draw_hud(frame, hud, scale)
        return frame

    # -- internals ---------------------------------------------------------
    def _draw_seat(
        self,
        frame: np.ndarray,
        seat: Seat,
        statuses: dict[tuple[int, int], CellStatus],
        scale: float,
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
            self._label(frame, box, label, color, scale * 0.42, above=True)

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
        scale: float,
        prefix: str,
    ) -> None:
        for det in detections:
            self._rect(frame, det.box, color, self.cfg.person_thickness)
            if height(det.box) < 36:
                continue
            name = det.label if not prefix else f"{prefix}{det.track_id or '?'}"
            self._label(frame, det.box, name, color, scale * 0.34, above=False)

    def _draw_hud(self, frame: np.ndarray, hud: HudInfo, scale: float) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        line_h = int(22 * scale)
        pad = int(10 * scale)
        width = int(330 * scale)
        height_px = int(96 * scale) if self.cfg.show_legend else int(52 * scale)
        x0, y0 = pad, pad

        self._fill_alpha_panel(frame, (x0, y0, x0 + width, y0 + height_px), 0.55)

        title = "Seat Finder"
        if hud.total_frames:
            title += f"  frame {hud.frame_index}/{hud.total_frames}"
        else:
            title += f"  frame {hud.frame_index}"
        cv2.putText(frame, title, (x0 + 10, y0 + line_h),
                    font, scale * 0.45, TEXT_COLOR, 1, cv2.LINE_AA)

        summary = (
            f"seats {hud.seats}   empty {hud.empty}   "
            f"occupied {hud.occupied}   objects {hud.ghost}"
        )
        cv2.putText(frame, summary, (x0 + 10, y0 + line_h * 2),
                    font, scale * 0.42, TEXT_COLOR, 1, cv2.LINE_AA)

        time_text = f"t={hud.time_s:5.2f}s"
        if hud.fps:
            time_text += f"   {hud.fps:4.1f} fps"
        cv2.putText(frame, time_text, (x0 + 10, y0 + line_h * 3),
                    font, scale * 0.40, (200, 200, 200), 1, cv2.LINE_AA)

        if not self.cfg.show_legend:
            return
        entries = [
            (SeatState.EMPTY, "empty seat"),
            (SeatState.OCCUPIED, "person seated"),
            (SeatState.GHOST, "belongings only"),
        ]
        y = y0 + line_h * 3 + int(14 * scale)
        for state, text in entries:
            cv2.rectangle(frame, (x0 + 12, y - int(7 * scale)),
                          (x0 + 12 + int(10 * scale), y + int(3 * scale)),
                          COLORS[state], -1)
            cv2.putText(frame, text, (x0 + 12 + int(18 * scale), y + int(2 * scale)),
                        font, scale * 0.36, TEXT_COLOR, 1, cv2.LINE_AA)
            y += int(15 * scale)

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

    @staticmethod
    def _fill_alpha_panel(frame: np.ndarray, box: Box,
                          alpha: float) -> None:
        Renderer._fill_alpha(frame, box, PANEL_COLOR, alpha)

    @staticmethod
    def _label(frame: np.ndarray, box: Box, text: str,
               color: tuple[int, int, int], font_scale: float,
               above: bool) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        h, w = frame.shape[:2]
        x1 = int(round(box[0]))
        if above:
            y = max(0, int(round(box[1])) - baseline - 2)
            x = x1
        else:
            y = min(h - 2, int(round(box[1])) + th + 2)
            x = x1
        x = max(0, min(x, w - tw - 4))
        cv2.rectangle(frame, (x, y - th - baseline), (x + tw + 4, y + baseline),
                      color, -1)
        cv2.putText(frame, text, (x + 2, y - baseline // 2), font, font_scale,
                    (20, 20, 20), thickness, cv2.LINE_AA)
