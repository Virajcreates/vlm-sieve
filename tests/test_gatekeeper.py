"""Unit tests for vlm-sieve models and Stage 1 CPU MotionGatekeeper."""

from __future__ import annotations

import base64
import time
from pathlib import Path

import numpy as np
from PIL import Image

from vlmsieve import (
    BoundingBox,
    FramePacket,
    MotionGatekeeper,
    SieveConfig,
    TokenSavingsReport,
    VisualEvent,
)


def test_bounding_box_properties_and_union() -> None:
    """Verify BoundingBox properties (x2, y2, area) and union calculation."""
    b1 = BoundingBox(x=10, y=20, width=50, height=60)
    assert b1.x2 == 60
    assert b1.y2 == 80
    assert b1.area == 3000

    b2 = BoundingBox(x=40, y=50, width=70, height=80)
    assert b2.x2 == 110
    assert b2.y2 == 130
    assert b2.area == 5600

    union_b = b1.union(b2)
    assert union_b.x == 10
    assert union_b.y == 20
    assert union_b.x2 == 110
    assert union_b.y2 == 130
    assert union_b.width == 100
    assert union_b.height == 110
    assert union_b.area == 11000


def test_bounding_box_padding_and_clamping() -> None:
    """Verify BoundingBox symmetric padding and strict boundary clamping."""
    box = BoundingBox(x=10, y=20, width=100, height=100)
    # 15% of 100 is 15px pad per side
    padded = box.pad(padding_ratio=0.15, max_w=200, max_h=200)

    # Left: 10 - 15 = -5 -> clamped to 0
    assert padded.x == 0
    # Top: 20 - 15 = 5
    assert padded.y == 5
    # Right: 110 + 15 = 125 <= 200
    assert padded.x2 == 125
    # Bottom: 120 + 15 = 135 <= 200
    assert padded.y2 == 135
    assert padded.width == 125
    assert padded.height == 130

    # Test clamping against max_w and max_h boundaries
    corner_box = BoundingBox(x=180, y=180, width=30, height=30)
    corner_padded = corner_box.pad(padding_ratio=0.5, max_w=200, max_h=200)
    assert corner_padded.x2 == 200
    assert corner_padded.y2 == 200
    assert corner_padded.x >= 0
    assert corner_padded.y >= 0


def test_visual_event_serialization(tmp_path: Path) -> None:
    """Verify VisualEvent serialization methods: to_bytes, to_base64, to_data_uri, save."""
    test_img = Image.new("RGB", (320, 240), color=(100, 150, 200))
    report = TokenSavingsReport(
        raw_frames_inspected=60,
        frames_dropped=54,
        storyboards_emitted=1,
        estimated_naive_tokens=67200,
        estimated_sieve_tokens=1120,
        compression_ratio=60.0,
        tokens_saved_pct=98.33,
    )
    event = VisualEvent(
        event_id=1,
        start_frame=10,
        end_frame=30,
        start_time_sec=0.33,
        end_time_sec=1.0,
        raw_frame_count=21,
        keyframes=[],
        roi_bbox=BoundingBox(x=0, y=0, width=320, height=240),
        storyboard_pil=test_img,
        savings=report,
    )

    # 1. to_bytes
    jpeg_bytes = event.to_bytes(format="JPEG", quality=85)
    assert isinstance(jpeg_bytes, bytes)
    assert len(jpeg_bytes) > 0
    assert jpeg_bytes.startswith(b"\xff\xd8")  # JPEG header

    # 2. to_base64
    b64_str = event.to_base64(format="JPEG", quality=85)
    assert isinstance(b64_str, str)
    decoded = base64.b64decode(b64_str)
    assert decoded == jpeg_bytes

    # 3. to_data_uri
    data_uri = event.to_data_uri(format="JPEG", quality=85)
    assert data_uri.startswith("data:image/jpeg;base64,")
    assert data_uri.endswith(b64_str)

    # 4. save to disk
    out_file = tmp_path / "nested" / "output.jpg"
    saved_path = event.save(out_file, quality=85)
    assert Path(saved_path).is_file()
    with Image.open(saved_path) as loaded_img:
        assert loaded_img.size == (320, 240)


def test_motion_gatekeeper_full_event_cycle() -> None:
    """Test full event cycle: static frames -> moving rectangle -> static frames -> event emitted."""
    config = SieveConfig(
        downscale_width=320,
        motion_threshold=0.015,
        pixel_diff_threshold=25,
        min_contour_area=16,
        cooldown_frames=12,
        min_event_frames=3,
    )
    gatekeeper = MotionGatekeeper(config=config)

    emitted_events: list[list[FramePacket]] = []
    frame_idx = 0

    # 1. Feed 15 static black frames (640x360)
    black_frame = np.zeros((360, 640, 3), dtype=np.uint8)
    for _ in range(15):
        res = gatekeeper.process_frame(black_frame, frame_idx, frame_idx / 30.0)
        if res is not None:
            emitted_events.append(res)
        frame_idx += 1

    assert len(emitted_events) == 0

    # 2. Feed 8 frames with a moving 100x80 white rectangle
    rect_positions: list[tuple[int, int]] = []
    for i in range(8):
        motion_frame = black_frame.copy()
        rx = 50 + i * 25
        ry = 100
        rect_positions.append((rx, ry))
        motion_frame[ry : ry + 80, rx : rx + 100] = 255

        res = gatekeeper.process_frame(motion_frame, frame_idx, frame_idx / 30.0)
        if res is not None:
            emitted_events.append(res)
        frame_idx += 1

    # 3. Feed 15 static black frames to let cooldown expire and finalize the event
    for _ in range(15):
        res = gatekeeper.process_frame(black_frame, frame_idx, frame_idx / 30.0)
        if res is not None:
            emitted_events.append(res)
        frame_idx += 1

    # Exactly 1 event should have been emitted
    assert len(emitted_events) == 1
    event = emitted_events[0]
    assert len(event) >= 8

    # Verify that motion_bbox accurately tracks the moving rectangle in 640x360 coordinates
    motion_packets = [p for p in event if p.motion_bbox is not None]
    assert len(motion_packets) >= 6

    for packet in motion_packets:
        bbox = packet.motion_bbox
        assert bbox is not None
        # Bounding box should be positioned around y ~ 100 and within bounds
        assert 80 <= bbox.y <= 120
        assert 60 <= bbox.height <= 110
        assert 70 <= bbox.width <= 160
        assert bbox.x2 <= 640
        assert bbox.y2 <= 360


def test_motion_gatekeeper_flicker_rejection() -> None:
    """Test camera flicker rejection (< min_event_frames=3) emits None."""
    config = SieveConfig(
        downscale_width=320,
        motion_threshold=0.015,
        cooldown_frames=12,
        min_event_frames=3,
    )
    gatekeeper = MotionGatekeeper(config=config)

    emitted_events: list[list[FramePacket]] = []
    frame_idx = 0
    black_frame = np.zeros((360, 640, 3), dtype=np.uint8)

    # 1. Feed 10 static frames
    for _ in range(10):
        res = gatekeeper.process_frame(black_frame, frame_idx, frame_idx / 30.0)
        if res is not None:
            emitted_events.append(res)
        frame_idx += 1

    # 2. Feed 1 single-frame flash
    flash_frame = black_frame.copy()
    flash_frame[100:180, 200:300] = 255
    res = gatekeeper.process_frame(flash_frame, frame_idx, frame_idx / 30.0)
    if res is not None:
        emitted_events.append(res)
    frame_idx += 1

    # 3. Feed 15 static frames
    for _ in range(15):
        res = gatekeeper.process_frame(black_frame, frame_idx, frame_idx / 30.0)
        if res is not None:
            emitted_events.append(res)
        frame_idx += 1

    # Flicker must be rejected, emitting no events
    assert len(emitted_events) == 0


def test_motion_gatekeeper_flush() -> None:
    """Test flush() returns buffered packets when stream terminates mid-event."""
    config = SieveConfig(min_event_frames=3)
    gatekeeper = MotionGatekeeper(config=config)

    black_frame = np.zeros((360, 640, 3), dtype=np.uint8)
    frame_idx = 0

    # Seed reference frame
    gatekeeper.process_frame(black_frame, frame_idx, 0.0)
    frame_idx += 1

    # Feed 5 active motion frames
    for i in range(5):
        frame = black_frame.copy()
        frame[50:150, 50 + i * 20 : 150 + i * 20] = 255
        res = gatekeeper.process_frame(frame, frame_idx, frame_idx / 30.0)
        assert res is None  # Event still open
        frame_idx += 1

    # Stream ends abruptly -> call flush()
    flushed = gatekeeper.flush()
    assert flushed is not None
    assert len(flushed) == 5

    # Subsequent flush returns None (clean state)
    second_flush = gatekeeper.flush()
    assert second_flush is None


def test_motion_gatekeeper_1080p_latency() -> None:
    """Verify frame gating latency on 1080p frames is well within production limits."""
    gatekeeper = MotionGatekeeper(SieveConfig(downscale_width=320))
    frame_1080p = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # Initialize reference
    gatekeeper.process_frame(frame_1080p, 0, 0.0)

    # Measure 20 static frames (the critical drop path evaluated on every frame)
    durations: list[float] = []
    for idx in range(1, 21):
        t0 = time.perf_counter()
        gatekeeper.process_frame(frame_1080p, idx, idx / 30.0)
        durations.append((time.perf_counter() - t0) * 1000.0)

    avg_ms = sum(durations) / len(durations)
    # Average latency must comfortably be sub-6ms on test environments
    assert avg_ms < 6.0, f"Average latency too high: {avg_ms:.2f}ms"
