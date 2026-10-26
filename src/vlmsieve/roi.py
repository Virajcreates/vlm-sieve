"""Stage 3 Active ROI Auto-Cropper for dynamic motion focus."""

from __future__ import annotations

import numpy as np

from vlmsieve.models import BoundingBox, FramePacket, SieveConfig


def compute_event_roi(
    packets: list[FramePacket],
    frame_width: int,
    frame_height: int,
    config: SieveConfig | None = None,
) -> BoundingBox:
    """Compute a stationary Region of Interest enclosing motion across all event keyframes.

    Aggregates active motion bounding boxes across the entire action trajectory,
    applies contextual padding, and enforces a minimum ROI dimension with
    edge-aware clamping so no visual context is lost near video boundaries.

    Args:
        packets: Chronological frame packets in an event.
        frame_width: Full video frame width in pixels.
        frame_height: Full video frame height in pixels.
        config: SieveConfig runtime hyperparameters. Uses default config if None.

    Returns:
        BoundingBox defining the global event crop area.
    """
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError(
            f"frame_width and frame_height must be positive, got {frame_width}x{frame_height}"
        )

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

    # Union all active bounding boxes across the action duration
    union_box = boxes[0]
    for b in boxes[1:]:
        union_box = union_box.union(b)

    # Apply symmetric contextual padding
    padded = union_box.pad(cfg.roi_padding_ratio, frame_width, frame_height)

    # Enforce minimum dimension while respecting frame limits
    target_w = max(padded.width, min(cfg.min_roi_size, frame_width))
    target_h = max(padded.height, min(cfg.min_roi_size, frame_height))

    # Center target box around the motion center
    cx = padded.x + padded.width // 2
    cy = padded.y + padded.height // 2

    new_x = cx - target_w // 2
    new_y = cy - target_h // 2

    # Inward shift clamping so boundary boxes preserve target size
    new_x = max(0, min(frame_width - target_w, new_x))
    new_y = max(0, min(frame_height - target_h, new_y))

    return BoundingBox(x=new_x, y=new_y, width=target_w, height=target_h)


def crop_packet_to_roi(packet: FramePacket, roi: BoundingBox) -> np.ndarray:
    """Crop a FramePacket's RGB array to the specified bounding box.

    Applies safe boundary clamping against array bounds and returns a
    contiguous C-order numpy array.

    Args:
        packet: FramePacket to crop.
        roi: Target BoundingBox.

    Returns:
        Cropped RGB numpy array with shape (roi.height, roi.width, 3).
    """
    h, w = packet.rgb_array.shape[:2]
    x1 = max(0, min(w, roi.x))
    y1 = max(0, min(h, roi.y))
    x2 = max(x1, min(w, roi.x2))
    y2 = max(y1, min(h, roi.y2))
    cropped = packet.rgb_array[y1:y2, x1:x2]
    return np.ascontiguousarray(cropped)
