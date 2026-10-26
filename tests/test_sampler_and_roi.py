"""Unit tests for Stage 2 Keyframe Sampler and Stage 3 Active ROI Auto-Cropper."""

from __future__ import annotations

import numpy as np
import pytest

from vlmsieve import (
    BoundingBox,
    FramePacket,
    SieveConfig,
    compute_event_roi,
    crop_packet_to_roi,
    select_keyframes,
)


def _make_packet(
    frame_index: int,
    motion_score: float = 0.05,
    rgb_color: tuple[int, int, int] = (0, 0, 0),
    shape: tuple[int, int] = (64, 64),
    bbox: BoundingBox | None = None,
) -> FramePacket:
    """Create a synthetic FramePacket for testing."""
    img = np.zeros((*shape, 3), dtype=np.uint8)
    img[:, :] = rgb_color
    return FramePacket(
        frame_index=frame_index,
        timestamp_sec=frame_index / 30.0,
        rgb_array=img,
        motion_score=motion_score,
        motion_bbox=bbox,
    )


def test_select_keyframes_count_and_ordering() -> None:
    """Verify select_keyframes returns exact target_count strictly ordered by frame_index."""
    packets = [
        _make_packet(
            i,
            motion_score=0.01 + (i % 7) * 0.05,
            rgb_color=((i * 5) % 256, (i * 13) % 256, (i * 29) % 256),
        )
        for i in range(50)
    ]

    selected = select_keyframes(packets, target_count=6)
    assert len(selected) == 6

    # Verify strictly increasing frame_index
    for a, b in zip(selected[:-1], selected[1:]):
        assert a.frame_index < b.frame_index

    # Onset and termination frames must be pinned
    assert selected[0].frame_index == packets[0].frame_index
    assert selected[-1].frame_index == packets[-1].frame_index


def test_select_keyframes_peak_motion_and_diversity() -> None:
    """Verify that within intermediate buckets, peak motion and visual diversity are chosen."""
    # 10 frames total: onset=0, bucket1=1..4, bucket2=5..8, termination=9
    packets = [
        _make_packet(0, motion_score=0.1, rgb_color=(10, 10, 10)),  # Onset
        _make_packet(1, motion_score=0.02, rgb_color=(10, 10, 10)),
        _make_packet(2, motion_score=0.95, rgb_color=(255, 0, 0)),  # Red peak
        _make_packet(3, motion_score=0.03, rgb_color=(10, 10, 10)),
        _make_packet(4, motion_score=0.01, rgb_color=(10, 10, 10)),
        _make_packet(5, motion_score=0.02, rgb_color=(10, 10, 10)),
        _make_packet(6, motion_score=0.01, rgb_color=(10, 10, 10)),
        _make_packet(7, motion_score=0.90, rgb_color=(0, 0, 255)),  # Blue peak
        _make_packet(8, motion_score=0.02, rgb_color=(10, 10, 10)),
        _make_packet(9, motion_score=0.05, rgb_color=(10, 10, 10)),  # Termination
    ]

    selected = select_keyframes(packets, target_count=4)
    assert len(selected) == 4
    assert selected[0].frame_index == 0
    assert selected[1].frame_index == 2  # Picked peak red
    assert selected[2].frame_index == 7  # Picked peak blue
    assert selected[3].frame_index == 9


def test_select_keyframes_edge_cases() -> None:
    """Verify boundary conditions for select_keyframes."""
    # Empty packets list
    assert select_keyframes([], target_count=6) == []

    # Invalid target_count <= 0
    with pytest.raises(ValueError, match="target_count must be >= 1"):
        select_keyframes([_make_packet(0)], target_count=0)
    with pytest.raises(ValueError, match="target_count must be >= 1"):
        select_keyframes([_make_packet(0)], target_count=-3)

    packets_10 = [_make_packet(i, motion_score=i * 0.1) for i in range(10)]

    # len(packets) <= target_count returns copy of sorted packets
    res_few = select_keyframes(packets_10[:3], target_count=6)
    assert [p.frame_index for p in res_few] == [0, 1, 2]

    # target_count == 1 returns highest motion_score packet
    res_one = select_keyframes(packets_10, target_count=1)
    assert len(res_one) == 1
    assert res_one[0].frame_index == 9  # highest motion score

    # target_count == 2 returns first and last
    res_two = select_keyframes(packets_10, target_count=2)
    assert len(res_two) == 2
    assert res_two[0].frame_index == 0
    assert res_two[1].frame_index == 9


def test_compute_event_roi_union_padding_and_min_size() -> None:
    """Verify compute_event_roi union aggregation, padding, and edge-aware min_size expansion."""
    cfg = SieveConfig(min_roi_size=224, roi_padding_ratio=0.15)

    # 1. Standard motion trajectories within bounds
    packets = [
        _make_packet(0, bbox=BoundingBox(x=100, y=100, width=50, height=40)),
        _make_packet(1, bbox=BoundingBox(x=130, y=120, width=60, height=50)),
    ]
    roi = compute_event_roi(packets, frame_width=640, frame_height=360, config=cfg)
    assert roi.width == 224
    assert roi.height == 224
    assert 0 <= roi.x <= 640 - 224
    assert 0 <= roi.y <= 360 - 224

    # 2. Extreme edge case: tiny 20x20 box near bottom-right corner (630, 350)
    corner_box = BoundingBox(x=620, y=340, width=20, height=20)
    corner_packets = [_make_packet(0, bbox=corner_box)]
    corner_roi = compute_event_roi(
        corner_packets, frame_width=640, frame_height=360, config=cfg
    )

    # Must expand to min_roi_size 224x224 and shift inward cleanly
    assert corner_roi.width == 224
    assert corner_roi.height == 224
    assert corner_roi.x == 640 - 224  # 416
    assert corner_roi.y == 360 - 224  # 136
    assert corner_roi.x2 == 640
    assert corner_roi.y2 == 360


def test_compute_event_roi_fallbacks_and_validation() -> None:
    """Verify compute_event_roi fallbacks on auto_crop_roi=False, missing bboxes, and invalid dims."""
    # Invalid frame dimensions
    with pytest.raises(ValueError, match="positive"):
        compute_event_roi([], frame_width=0, frame_height=360)
    with pytest.raises(ValueError, match="positive"):
        compute_event_roi([], frame_width=640, frame_height=-10)

    packets_with_box = [
        _make_packet(0, bbox=BoundingBox(x=10, y=10, width=50, height=50))
    ]

    # Fallback when auto_crop_roi=False
    no_crop_cfg = SieveConfig(auto_crop_roi=False)
    roi_no_crop = compute_event_roi(
        packets_with_box, frame_width=640, frame_height=360, config=no_crop_cfg
    )
    assert roi_no_crop == BoundingBox(x=0, y=0, width=640, height=360)

    # Fallback when packets have no motion_bbox
    packets_no_box = [_make_packet(0, bbox=None), _make_packet(1, bbox=None)]
    roi_no_boxes = compute_event_roi(packets_no_box, frame_width=640, frame_height=360)
    assert roi_no_boxes == BoundingBox(x=0, y=0, width=640, height=360)

    # Fallback when packets list is empty
    roi_empty = compute_event_roi([], frame_width=640, frame_height=360)
    assert roi_empty == BoundingBox(x=0, y=0, width=640, height=360)


def test_crop_packet_to_roi() -> None:
    """Verify crop_packet_to_roi extracts the expected slice and maintains C-contiguity."""
    full_frame_packet = _make_packet(
        0,
        shape=(360, 640),
        rgb_color=(42, 84, 126),
        bbox=BoundingBox(x=50, y=60, width=120, height=100),
    )
    roi = BoundingBox(x=40, y=50, width=224, height=224)

    cropped = crop_packet_to_roi(full_frame_packet, roi)
    assert cropped.shape == (224, 224, 3)
    assert cropped.flags.c_contiguous
    assert np.all(cropped == (42, 84, 126))

    # Out-of-bounds ROI clamp test
    out_of_bounds_roi = BoundingBox(x=-20, y=-30, width=200, height=200)
    safe_cropped = crop_packet_to_roi(full_frame_packet, out_of_bounds_roi)
    assert safe_cropped.flags.c_contiguous
    # Clamped bounds x: [0, 180], y: [0, 170]
    assert safe_cropped.shape == (170, 180, 3)
