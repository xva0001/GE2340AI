"""Turning pipeline results into JSON/Markdown summaries and console output."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .pipeline import TimelineEntry, VideoResult

STATES = ("empty", "occupied", "ghost")


def summarise(result: VideoResult) -> dict:
    """Aggregate one video's timeline and per-frame counts."""
    fps = result.info.fps or 30.0
    frames = result.frame_totals
    last_frame = frames[-1]["frame"] if frames else 0
    total_frames = max(1, len(frames))

    mean = {
        state: round(sum(f[state] for f in frames) / total_frames, 2)
        for state in ("seats", "empty", "occupied", "objects")
    }
    peak = {
        state: max((f[state] for f in frames), default=0)
        for state in ("seats", "empty", "occupied", "objects")
    }

    cells: dict[tuple[int, int], list[TimelineEntry]] = defaultdict(list)
    for entry in result.timelines:
        cells[(entry.seat_id, entry.cell)].append(entry)

    seat_rows = []
    for (seat_id, cell), entries in sorted(cells.items()):
        entries.sort(key=lambda e: e.frame)
        durations = {state: 0.0 for state in STATES}
        for current, following in zip(entries, entries[1:]):
            span = (following.frame - current.frame) / fps
            durations[current.state] = durations.get(current.state, 0.0) + span
        tail = (last_frame - entries[-1].frame) / fps
        durations[entries[-1].state] = durations.get(entries[-1].state, 0.0) + tail
        seat_rows.append(
            {
                "seat_id": seat_id,
                "cell": cell,
                "kind": entries[0].kind,
                "first_frame": entries[0].frame,
                "seconds": {k: round(v, 2) for k, v in durations.items()},
                "transitions": [
                    [round(e.time_s, 2), e.state] for e in entries
                ],
            }
        )

    occupancy_rate = (
        round(mean["occupied"] / mean["seats"], 3) if mean["seats"] else 0.0
    )
    return {
        "video": result.video.name,
        "annotated_video": (
            str(result.output_video) if result.output_video is not None else None
        ),
        "duration_s": round(result.info.frame_count / fps, 2),
        "frames_analysed": result.frames_processed,
        "processing_fps": round(result.fps_processed, 1),
        "seats": {
            "mean_tracked": mean["seats"],
            "peak_tracked": peak["seats"],
            "unique_cells": len(cells),
        },
        "occupancy": {
            "mean_empty": mean["empty"],
            "mean_occupied": mean["occupied"],
            "mean_objects": mean["objects"],
            "peak_occupied": peak["occupied"],
            "occupancy_rate": occupancy_rate,
        },
        "seat_states": seat_rows,
    }


def write_video_summary(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def write_run_summary(
    summaries: list[dict], path: Path, meta: dict
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"run": meta, "videos": summaries}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def format_table(rows: list[dict]) -> str:
    """Fixed-width console table for the end of a run."""
    header = (
        f"{'video':34s} {'frames':>6s} {'seats':>6s} {'empty':>6s} "
        f"{'occ':>6s} {'obj':>6s} {'occ%':>6s} {'fps':>6s}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        name = row["video"]
        if len(name) > 34:
            name = name[:31] + "..."
        occupancy = row["occupancy"]
        lines.append(
            f"{name:34s} {row['frames_analysed']:6d} "
            f"{row['seats']['mean_tracked']:6.1f} "
            f"{occupancy['mean_empty']:6.1f} "
            f"{occupancy['mean_occupied']:6.1f} "
            f"{occupancy['mean_objects']:6.1f} "
            f"{occupancy['occupancy_rate'] * 100:5.0f}% "
            f"{row['processing_fps']:6.1f}"
        )
    return "\n".join(lines)
