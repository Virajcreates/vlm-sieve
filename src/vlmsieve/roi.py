"""Stage 3: Active Region-of-Interest (ROI) union and context-padded cropping."""
from __future__ import annotations

import numpy as np

from vlmsieve.models import BoundingBox, FramePacket, SieveConfig


def compute_event_roi(
    packets: list[FramePacket],
    frame_width: int,
    frame_height: int,
    config: SieveConfig | None = None,
) -> BoundingBox:
    """Compute union of motion bounding boxes across the episode with generous context padding."""
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame_width and frame_height must be positive integers.")

    cfg = config or SieveConfig()
    full_box = BoundingBox(x=0, y=0, width=frame_width, height=frame_height)
    if not cfg.auto_crop_roi or not packets:
        return full_box

    boxes = [
        p.motion_bbox
        for p in packets
        if p.motion_bbox is not None and p.motion_bbox.area > 0
    ]
    if not boxes:
        return full_box

    union_box = boxes[0]
    for box in boxes[1:]:
        union_box = union_box.union(box)

    padded = union_box.pad(cfg.roi_padding_ratio, frame_width, frame_height)
    target_w = max(padded.width, min(cfg.min_roi_size, frame_width))
    target_h = max(padded.height, min(cfg.min_roi_size, frame_height))

    cx = padded.x + padded.width // 2
    cy = padded.y + padded.height // 2

    nx = max(0, min(frame_width - target_w, cx - target_w // 2))
    ny = max(0, min(frame_height - target_h, cy - target_h // 2))
    return BoundingBox(x=nx, y=ny, width=target_w, height=target_h)


def crop_packet_to_roi(packet: FramePacket, roi: BoundingBox) -> np.ndarray:
    """Crop a FramePacket's RGB array to the given ROI bounding box."""
    if roi.width <= 0 or roi.height <= 0:
        raise ValueError("ROI width and height must be positive.")
    rgb = packet.rgb_array
    h, w = rgb.shape[:2]
    x1 = max(0, min(w, roi.x))
    y1 = max(0, min(h, roi.y))
    x2 = max(x1, min(w, roi.x2))
    y2 = max(y1, min(h, roi.y2))
    return np.ascontiguousarray(rgb[y1:y2, x1:x2])
