"""Video reading/writing and CSV result recording."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0


class VideoReader:
    """Iterates frames of a video file."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise OSError(f"cannot open video: {self.path}")
        fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.info = VideoInfo(
            path=self.path,
            fps=fps if fps > 0 else 30.0,
            frame_count=int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        )

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        index = 0
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            yield index, frame
            index += 1

    def close(self) -> None:
        self._cap.release()

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class VideoWriter:
    """Writes an mp4 file, accommodating codecs that need even dimensions."""

    def __init__(self, path: Path | str, fps: float, size: tuple[int, int]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        width, height = size
        self._writer = cv2.VideoWriter(
            str(self.path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps if fps > 0 else 30.0,
            (width, height),
        )
        if not self._writer.isOpened():
            raise OSError(f"cannot open video writer for {self.path}")

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def close(self) -> None:
        self._writer.release()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class CsvRecorder:
    """Streams per-seat rows and per-frame totals to CSV files."""

    SEAT_FIELDS = [
        "video",
        "frame",
        "time_s",
        "seat_id",
        "cell",
        "kind",
        "state",
        "person_id",
        "objects",
        "x1",
        "y1",
        "x2",
        "y2",
    ]
    FRAME_FIELDS = ["video", "frame", "time_s", "seats", "empty", "occupied", "objects"]

    def __init__(self, seat_path: Path, frame_path: Path) -> None:
        seat_path.parent.mkdir(parents=True, exist_ok=True)
        self._seat_file = seat_path.open("w", newline="", encoding="utf-8")
        self._frame_file = frame_path.open("w", newline="", encoding="utf-8")
        self._seat_writer = csv.DictWriter(self._seat_file, fieldnames=self.SEAT_FIELDS)
        self._frame_writer = csv.DictWriter(self._frame_file, fieldnames=self.FRAME_FIELDS)
        self._seat_writer.writeheader()
        self._frame_writer.writeheader()

    def add_seat(
        self,
        video: str,
        frame: int,
        time_s: float,
        seat_id: int,
        cell: int,
        kind: str,
        state: str,
        person_id: int | None,
        objects: list[str],
        box: tuple[float, float, float, float],
    ) -> None:
        self._seat_writer.writerow(
            {
                "video": video,
                "frame": frame,
                "time_s": f"{time_s:.3f}",
                "seat_id": seat_id,
                "cell": cell,
                "kind": kind,
                "state": state,
                "person_id": "" if person_id is None else person_id,
                "objects": "|".join(objects),
                "x1": round(box[0], 1),
                "y1": round(box[1], 1),
                "x2": round(box[2], 1),
                "y2": round(box[3], 1),
            }
        )

    def add_frame(
        self,
        video: str,
        frame: int,
        time_s: float,
        seats: int,
        empty: int,
        occupied: int,
        objects: int,
    ) -> None:
        self._frame_writer.writerow(
            {
                "video": video,
                "frame": frame,
                "time_s": f"{time_s:.3f}",
                "seats": seats,
                "empty": empty,
                "occupied": occupied,
                "objects": objects,
            }
        )

    def close(self) -> None:
        self._seat_file.close()
        self._frame_file.close()

    def __enter__(self) -> "CsvRecorder":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def find_videos(path: Path) -> list[Path]:
    """Returns the video files at ``path`` or, for a directory, inside it."""
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"no such input: {path}")
    return sorted(
        p for p in path.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES
    )
