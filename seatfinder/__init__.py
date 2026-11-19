"""Seat Finder - detect seat occupancy (including belongings-only) in video."""

from .config import AppConfig, DetectorConfig, InferenceSeatConfig, SeatConfig, TemporalConfig
from .state import SeatState

__all__ = [
    "AppConfig",
    "DetectorConfig",
    "InferenceSeatConfig",
    "SeatConfig",
    "SeatState",
    "TemporalConfig",
]

__version__ = "0.1.0"
