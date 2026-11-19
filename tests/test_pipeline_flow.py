"""End-to-end pipeline test with a scripted detector (no model, no GPU).

A chair is detected in every frame; a person sits on it for a while and then
leaves a backpack behind.  The pipeline must report occupied, then ghost.
"""

import shutil
import unittest
from pathlib import Path

import cv2
import numpy as np

from seatfinder.config import AppConfig, DetectorConfig
from seatfinder.detector import Detection
from seatfinder.pipeline import VideoPipeline

HERE = Path(__file__).resolve().parent
WORK = HERE / "_artifacts"

CHAIR = Detection(56, "chair", 0.9, (100, 100, 200, 300))
SITTING = Detection(0, "person", 0.9, (110, 60, 190, 290), track_id=1)
BAG = Detection(24, "backpack", 0.6, (120, 200, 170, 280))


class ScriptedDetector:
    """Returns canned detections based on how many frames were processed."""

    def __init__(self) -> None:
        self.frame = 0

    def reset_tracking(self) -> None:
        self.frame = 0

    def infer(self, frame: np.ndarray) -> list[Detection]:
        index = self.frame
        self.frame += 1
        detections = [CHAIR]
        if 10 <= index < 30:
            detections.append(SITTING)
        if index >= 35:
            detections.append(BAG)
        return detections


def write_blank_video(path: Path, frames: int = 60) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 240)
    )
    for _ in range(frames):
        writer.write(np.full((240, 320, 3), 90, dtype=np.uint8))
    writer.release()


class PipelineFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        WORK.mkdir(parents=True, exist_ok=True)
        cls.video = WORK / "blank.mp4"
        write_blank_video(cls.video, frames=60)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(WORK, ignore_errors=True)

    def _run(self):
        pipeline = VideoPipeline(
            AppConfig(detector=DetectorConfig(weights="unused")),
            ScriptedDetector(),
            render=False,
            progress=False,
        )
        return pipeline.run(self.video, WORK)

    def test_states_progress_empty_occupied_ghost(self):
        result = self._run()
        states = [entry.state for entry in result.timelines]
        self.assertIn("occupied", states)
        self.assertIn("ghost", states)
        self.assertLess(states.index("occupied"), states.index("ghost"))

    def test_ghost_is_visible_in_frame_counts(self):
        result = self._run()
        last = result.frame_totals[-1]
        self.assertGreaterEqual(last["objects"], 1)
        self.assertEqual(last["occupied"], 0)

    def test_every_frame_is_processed(self):
        result = self._run()
        self.assertEqual(result.frames_processed, 60)


if __name__ == "__main__":
    unittest.main()
