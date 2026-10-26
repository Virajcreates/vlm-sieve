"""vlm-sieve: Event-driven video frame gatekeeper, active ROI cropper, and temporal storyboard packer."""

from vlmsieve.gatekeeper import MotionGatekeeper
from vlmsieve.models import (
    BoundingBox,
    FramePacket,
    SieveConfig,
    TokenSavingsReport,
    VisualEvent,
)
from vlmsieve.roi import compute_event_roi, crop_packet_to_roi
from vlmsieve.sampler import select_keyframes
from vlmsieve.sieve import VideoSieve
from vlmsieve.storyboard import StoryboardPacker

__version__ = "0.1.0"

__all__ = [
    "BoundingBox",
    "FramePacket",
    "MotionGatekeeper",
    "SieveConfig",
    "StoryboardPacker",
    "TokenSavingsReport",
    "VideoSieve",
    "VisualEvent",
    "__version__",
    "compute_event_roi",
    "crop_packet_to_roi",
    "select_keyframes",
]
