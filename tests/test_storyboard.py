"""Unit tests for Stage 4 Structured Storyboard Packer."""

from __future__ import annotations

import numpy as np
import pytest

from vlmsieve import (
    BoundingBox,
    FramePacket,
    SieveConfig,
    StoryboardPacker,
)


def _create_synthetic_packet(
    frame_index: int,
    timestamp_sec: float,
    shape: tuple[int, int] = (720, 1280),
    rect_box: tuple[int, int, int, int] = (200, 200, 100, 100),
    rect_color: tuple[int, int, int] = (255, 128, 0),
) -> FramePacket:
    """Create a synthetic 720p/1080p frame packet with an embedded moving shape."""
    img = np.zeros((*shape, 3), dtype=np.uint8)
    # Background subtle tone
    img[:, :] = (30, 40, 50)
    rx, ry, rw, rh = rect_box
    img[ry : ry + rh, rx : rx + rw] = rect_color
    return FramePacket(
        frame_index=frame_index,
        timestamp_sec=timestamp_sec,
        rgb_array=img,
        motion_score=0.15,
        motion_bbox=BoundingBox(x=rx, y=ry, width=rw, height=rh),
    )


def test_storyboard_exact_geometry_and_proportional_downscaling() -> None:
    """Verify exact canvas geometry with tile downscaling: 800x400 -> 640x320 in a 2x3 grid."""
    packer = StoryboardPacker()
    roi = BoundingBox(x=100, y=100, width=800, height=400)
    keyframes = [
        _create_synthetic_packet(
            i * 5,
            timestamp_sec=i * 0.25,
            rect_box=(150 + i * 50, 150, 80, 80),
        )
        for i in range(6)
    ]

    cfg = SieveConfig(
        grid_layout=(2, 3),
        tile_max_width=640,
        burn_timestamps=False,
        spatial_grid_overlay=False,
    )

    storyboard = packer.pack(keyframes, roi=roi, config=cfg)

    # Geometry verification:
    # tile_w = 640, tile_h = round(400 * (640 / 800)) = 320
    # canvas_w = 3 * 640 + 4 * 4 = 1920 + 16 = 1936
    # canvas_h = 2 * 320 + 3 * 4 = 640 + 12 = 652
    assert storyboard.mode == "RGB"
    assert storyboard.size == (1936, 652)


def test_storyboard_overlays_timestamps_and_spatial_grid() -> None:
    """Verify that timestamp badge and 4x4 spatial grid modify tile pixels predictably."""
    packer = StoryboardPacker()
    roi = BoundingBox(x=50, y=50, width=400, height=300)
    keyframes = [
        _create_synthetic_packet(0, timestamp_sec=0.0),
        _create_synthetic_packet(10, timestamp_sec=1.5),
    ]

    cfg_plain = SieveConfig(
        grid_layout=(1, 2),
        tile_max_width=400,
        burn_timestamps=False,
        spatial_grid_overlay=False,
    )
    storyboard_plain = packer.pack(keyframes, roi=roi, config=cfg_plain)

    cfg_overlays = SieveConfig(
        grid_layout=(1, 2),
        tile_max_width=400,
        burn_timestamps=True,
        spatial_grid_overlay=True,
    )
    storyboard_overlays = packer.pack(keyframes, roi=roi, config=cfg_overlays)

    assert storyboard_overlays.size == storyboard_plain.size
    assert storyboard_overlays.mode == "RGB"

    arr_plain = np.array(storyboard_plain)
    arr_overlays = np.array(storyboard_overlays)

    # Top-left region of Tile 1 (around x=4+10, y=4+10) where timestamp badge is burned
    badge_slice_plain = arr_plain[10:30, 10:80]
    badge_slice_overlay = arr_overlays[10:30, 10:80]
    assert not np.array_equal(badge_slice_plain, badge_slice_overlay)

    # Spatial grid line check at x ~ 4 + 100 (quarter of 400px tile width)
    line_x = 4 + 100
    assert not np.array_equal(arr_plain[150:250, line_x], arr_overlays[150:250, line_x])


def test_storyboard_partial_grid_slot_watermarking() -> None:
    """Verify empty slots render EMPTY_TILE_RGB and maintain complete canvas dimensions."""
    packer = StoryboardPacker()
    roi = BoundingBox(x=0, y=0, width=320, height=240)
    # Only 4 keyframes provided for a 2x3 (6 slots) grid
    keyframes = [_create_synthetic_packet(i, timestamp_sec=float(i)) for i in range(4)]

    cfg = SieveConfig(grid_layout=(2, 3), tile_max_width=320)
    storyboard = packer.pack(keyframes, roi=roi, config=cfg)

    # Expected dimensions: 3 * 320 + 4 * 4 = 976, 2 * 240 + 3 * 4 = 492
    assert storyboard.size == (976, 492)

    # Slot 5 (row 1, col 2 -> last slot):
    # x = 4 + 2 * (320 + 4) = 652
    # y = 4 + 1 * (240 + 4) = 248
    # Top-left corner of empty slot (away from centered "NO FRAME" watermark)
    empty_pixel = storyboard.getpixel((652 + 5, 248 + 5))
    assert empty_pixel == StoryboardPacker.EMPTY_TILE_RGB


def test_storyboard_validation_errors() -> None:
    """Verify proper ValueError exceptions on empty keyframes or invalid grid layout."""
    packer = StoryboardPacker()
    roi = BoundingBox(x=0, y=0, width=200, height=200)

    # 1. Empty keyframes
    with pytest.raises(ValueError, match="keyframes cannot be empty"):
        packer.pack([], roi=roi)

    # 2. Invalid grid layout
    dummy_packet = _create_synthetic_packet(0, timestamp_sec=0.0)
    with pytest.raises(ValueError, match="grid_layout"):
        packer.pack([dummy_packet], roi=roi, config=SieveConfig(grid_layout=(0, 3)))

    with pytest.raises(ValueError, match="grid_layout"):
        packer.pack([dummy_packet], roi=roi, config=SieveConfig(grid_layout=(2, -1)))
