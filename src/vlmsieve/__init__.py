"""vlm-sieve: Spatial-temporal video gatekeeper and storyboard packer for VLMs."""
from __future__ import annotations

from vlmsieve.gatekeeper import (
    MotionGatekeeper,
    is_camera_disturbance,
    merge_episodes_with_bridges,
    select_keyframes,
)
from vlmsieve.models import (
    BoundingBox,
    DisturbanceRecord,
    FramePacket,
    SieveConfig,
    TokenSavingsReport,
    VisualEvent,
)
from vlmsieve.roi import compute_event_roi, crop_packet_to_roi
from vlmsieve.sieve import VideoSieve
from vlmsieve.storyboard import StoryboardPacker

__version__ = "0.2.1"

__all__ = [
    "__version__",
    "BoundingBox",
    "DisturbanceRecord",
    "FramePacket",
    "MotionGatekeeper",
    "SieveConfig",
    "StoryboardPacker",
    "TokenSavingsReport",
    "VideoSieve",
    "VisualEvent",
    "compute_event_roi",
    "crop_packet_to_roi",
    "is_camera_disturbance",
    "merge_episodes_with_bridges",
    "select_keyframes",
]
