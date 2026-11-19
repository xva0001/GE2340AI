"""Command line interface: ``python -m seatfinder --input data``."""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

from .config import AppConfig, DetectorConfig, InferenceSeatConfig
from .detector import Detector
from .pipeline import VideoPipeline
from .summary import format_table, summarise, write_run_summary, write_video_summary
from .video_io import CsvRecorder, find_videos


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="seatfinder",
        description=(
            "Detect people, chairs and personal belongings in a video and label "
            "every seat as empty, occupied, or occupied-with-objects."
        ),
    )
    parser.add_argument("--input", required=True,
                        help="video file or directory of videos")
    parser.add_argument("--output", default="runs/latest",
                        help="output directory (default: runs/latest)")
    parser.add_argument("--weights", default="yolov10s.pt",
                        help="YOLO weights file or name (default: yolov10s.pt)")
    parser.add_argument("--imgsz", type=int, default=1280,
                        help="inference resolution (default: 1280)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="detection confidence threshold (default: 0.25)")
    parser.add_argument("--belonging-conf", type=float, default=0.15,
                        help="lower threshold for bags/laptops/cups (default: 0.15)")
    parser.add_argument("--device", default=None,
                        help="torch device, e.g. cpu or 0 (default: auto)")
    parser.add_argument("--stride", type=int, default=1,
                        help="process every Nth frame (default: 1)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="stop each video after N processed frames")
    parser.add_argument("--limit", type=int, default=None,
                        help="only process the first N videos")
    parser.add_argument("--no-video", action="store_true",
                        help="skip the annotated video, only write CSV/JSON")
    parser.add_argument("--stills", type=int, default=4,
                        help="annotated stills per video (default: 4, 0 = none)")
    parser.add_argument("--no-inferred", action="store_true",
                        help="do not invent seats from seated people")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress per-video progress output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    videos = find_videos(Path(args.input))
    if args.limit is not None:
        videos = videos[: args.limit]
    if not videos:
        print(f"no videos found in {args.input}", file=sys.stderr)
        return 1

    output = Path(args.output)
    annotated_dir = output / "annotated"
    json_dir = output / "json"
    stills_dir = output / "stills"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    stills_dir.mkdir(parents=True, exist_ok=True)

    cfg = AppConfig(
        detector=DetectorConfig(
            weights=args.weights,
            imgsz=args.imgsz,
            conf=args.conf,
            belonging_conf=args.belonging_conf,
            device=args.device,
        ),
        inferred=InferenceSeatConfig(enabled=not args.no_inferred),
    )

    print(f"loading {cfg.detector.weights} (imgsz={cfg.detector.imgsz}) ...")
    started = time.perf_counter()
    detector = Detector(cfg.detector)
    pipeline = VideoPipeline(
        cfg,
        detector,
        render=not args.no_video,
        stride=args.stride,
        max_frames=args.max_frames,
        progress=not args.quiet,
        stills=args.stills,
    )

    summaries = []
    with CsvRecorder(output / "seats.csv", output / "frames.csv") as csv:
        for position, video in enumerate(videos, start=1):
            print(f"[{position}/{len(videos)}] {video.name}", flush=True)
            result = pipeline.run(video, annotated_dir, csv, stills_dir)
            summary = summarise(result)
            write_video_summary(summary, json_dir / f"{video.stem}.json")
            summaries.append(summary)
            oc = summary["occupancy"]
            print(
                f"    seats~{summary['seats']['mean_tracked']:.1f} "
                f"empty~{oc['mean_empty']:.1f} occupied~{oc['mean_occupied']:.1f} "
                f"objects~{oc['mean_objects']:.1f} "
                f"({result.fps_processed:.1f} fps)",
                flush=True,
            )

    meta = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "weights": cfg.detector.weights,
        "imgsz": cfg.detector.imgsz,
        "conf": cfg.detector.conf,
        "device": args.device or "auto",
        "stride": args.stride,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "elapsed_s": round(time.perf_counter() - started, 1),
    }
    write_run_summary(summaries, output / "summary.json", meta)

    print()
    print(format_table(summaries))
    print()
    print(f"outputs written to {output.resolve()}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
