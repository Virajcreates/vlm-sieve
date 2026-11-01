"""Comprehensive Unit & End-to-End Test Suite for vlm-sieve v0.2.1."""
from __future__ import annotations

import asyncio
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image
import pytest

from vlmsieve import (
    BoundingBox,
    FramePacket,
    MotionGatekeeper,
    SieveConfig,
    StoryboardPacker,
    TokenSavingsReport,
    VideoSieve,
    VisualEvent,
    __version__,
    compute_event_roi,
    is_camera_disturbance,
    merge_episodes_with_bridges,
    select_keyframes,
)
from vlmsieve.cli import main, parse_grid


def test_version_and_exports() -> None:
    assert __version__ == "0.2.1"


def test_bounding_box_geometry_and_padding() -> None:
    b1 = BoundingBox(x=10, y=20, width=40, height=50)
    b2 = BoundingBox(x=30, y=10, width=50, height=40)
    assert b1.x2 == 50 and b1.y2 == 70 and b1.area == 2000

    u = b1.union(b2)
    assert (u.x, u.y, u.width, u.height) == (10, 10, 70, 60)

    padded_kw = b1.pad(padding_ratio=0.15, max_w=200, max_h=200)
    padded_pos = b1.pad(0.5, 200, 200)
    assert padded_kw.width >= b1.width
    assert padded_pos.x == 0 and padded_pos.y == 0


def test_visual_event_serialization(tmp_path: Path) -> None:
    img = Image.new("RGB", (320, 240), color=(100, 150, 200))
    report = TokenSavingsReport(
        raw_frames_inspected=60,
        sampled_frames_decoded=20,
        frames_dropped=54,
        storyboards_emitted=1,
        estimated_naive_tokens=67200,
        estimated_sieve_tokens=1120,
        compression_ratio=60.0,
        tokens_saved_pct=98.33,
    )
    ev = VisualEvent(
        event_id=1,
        start_frame=10,
        end_frame=30,
        start_time_sec=0.33,
        end_time_sec=1.0,
        raw_frame_count=21,
        keyframes=[],
        roi_bbox=BoundingBox(x=0, y=0, width=320, height=240),
        storyboard_pil=img,
        savings=report,
    )
    raw_bytes = ev.to_bytes(format="JPEG", quality=85)
    assert isinstance(raw_bytes, bytes) and len(raw_bytes) > 100
    b64 = ev.to_base64(format="JPEG", quality=85)
    assert isinstance(b64, str) and len(b64) > 100
    uri = ev.to_data_uri(format="JPEG", quality=85)
    assert uri.startswith("data:image/jpeg;base64,")
    saved = ev.save(tmp_path / "test_ev.jpg")
    assert saved.is_file()


def test_pre_roll_ring_buffer_captures_subtle_onset() -> None:
    cfg = SieveConfig(
        motion_threshold=0.02,
        pre_roll_sec=2.0,
        pre_roll_min_score=0.002,
        cooldown_frames=4,
        min_event_frames=2,
    )
    gate = MotionGatekeeper(cfg, effective_fps=10.0)

    events = []
    for i in range(30):
        img = np.full((360, 640, 3), 20, dtype=np.uint8)
        if 5 <= i < 10:
            img[150:170, 100 + i * 4 : 120 + i * 4] = 180
        elif 10 <= i <= 16:
            img[100:240, 150 + i * 10 : 250 + i * 10] = 240
        ev = gate.process_frame(img, i, i * 0.1)
        if ev is not None:
            events.append(ev)

    assert len(events) == 1
    assert events[0][0].frame_index <= 7


def test_episode_merging_and_hybrid_keyframe_spacing() -> None:
    assert select_keyframes([], target_count=6) == []
    with pytest.raises(ValueError, match="target_count must be >= 1"):
        select_keyframes([], target_count=0)

    pkts_burst1 = [
        FramePacket(
            frame_index=i,
            timestamp_sec=float(i),
            rgb_array=np.full((120, 160, 3), (i * 15) % 255, dtype=np.uint8),
            motion_score=0.05 if i != 3 else 0.25,
        )
        for i in range(0, 6)
    ]
    pkts_burst2 = [
        FramePacket(
            frame_index=i,
            timestamp_sec=float(i),
            rgb_array=np.full((120, 160, 3), (i * 20) % 255, dtype=np.uint8),
            motion_score=0.06 if i != 12 else 0.30,
        )
        for i in range(10, 16)
    ]
    bridges = [
        FramePacket(
            frame_index=i,
            timestamp_sec=float(i),
            rgb_array=np.full((120, 160, 3), 50, dtype=np.uint8),
            motion_score=0.001,
        )
        for i in range(6, 10)
    ]

    merged = merge_episodes_with_bridges(
        [pkts_burst1, pkts_burst2], bridges, merge_gap_sec=6.0, max_frames=600
    )
    assert len(merged) == 1
    kfs = select_keyframes(merged[0], target_count=6)
    assert len(kfs) == 6
    times = [k.timestamp_sec for k in kfs]
    assert times[0] == 0.0 and times[-1] == 15.0
    gaps = [times[i + 1] - times[i] for i in range(len(times) - 1)]
    assert min(gaps) >= 0.75


def test_roi_computation_and_validation() -> None:
    with pytest.raises(ValueError, match="positive"):
        compute_event_roi([], frame_width=0, frame_height=360)

    full = compute_event_roi([], frame_width=640, frame_height=360)
    assert (full.x, full.y, full.width, full.height) == (0, 0, 640, 360)


def test_camera_disturbance_detection() -> None:
    shake_pkts = [
        FramePacket(
            frame_index=i,
            timestamp_sec=i * 0.1,
            rgb_array=np.zeros((360, 640, 3), dtype=np.uint8),
            motion_score=0.25,
            motion_bbox=BoundingBox(x=0, y=0, width=620, height=350),
        )
        for i in range(8)
    ]
    is_shake, ratio, loc_cnt, glob_cnt = is_camera_disturbance(shake_pkts, 640, 360)
    assert is_shake is True
    assert ratio > 0.70
    assert loc_cnt == 0 and glob_cnt == 8


def test_storyboard_packer_and_both_axis_clamping() -> None:
    packer = StoryboardPacker(grid_layout=(2, 3), tile_max_dim=400)
    pkts = [
        FramePacket(
            frame_index=i,
            timestamp_sec=float(i),
            rgb_array=np.full((1000, 300, 3), 120, dtype=np.uint8),
            motion_score=0.05,
        )
        for i in range(4)
    ]
    roi = BoundingBox(x=0, y=0, width=300, height=1000)
    board = packer.pack(pkts, roi)
    assert board.height == 2 * 400 + 3 * 4
    assert board.width < 3 * 400 + 4 * 4


def test_videosieve_sync_and_async_parity() -> None:
    frames = []
    for i in range(45):
        img = np.full((360, 640, 3), 25, dtype=np.uint8)
        if 10 <= i <= 25:
            img[100:220, i * 12 : i * 12 + 90] = 230
        frames.append(img)

    sieve_sync = VideoSieve(source=frames, min_event_frames=2, cooldown_frames=6)
    sync_events = list(sieve_sync.stream_events())
    assert len(sync_events) == 1
    assert sync_events[0].savings is not None
    assert sync_events[0].savings.tokens_saved_pct > 90.0

    async def _collect() -> list[VisualEvent]:
        sieve_async = VideoSieve(source=frames, min_event_frames=2, cooldown_frames=6)
        out = []
        async for ev in sieve_async.astream_events():
            out.append(ev)
        return out

    async_events = asyncio.run(_collect())
    assert len(async_events) == len(sync_events)
    assert async_events[0].start_frame == sync_events[0].start_frame


def test_cli_parse_grid_and_e2e_video(tmp_path: Path) -> None:
    assert parse_grid("2x3") == (2, 3)
    assert parse_grid(" 3 X 3 ") == (3, 3)

    video_path = tmp_path / "synthetic.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 15.0, (640, 360))
    for i in range(60):
        frame = np.full((360, 640, 3), 30, dtype=np.uint8)
        if 15 <= i <= 35:
            x = 50 + (i - 15) * 18
            frame[100:240, x : x + 100] = (40, 200, 240)
        writer.write(frame)
    writer.release()

    out_dir = tmp_path / "cli_out"
    rc = main([str(video_path), "-o", str(out_dir), "--grid", "2x3"])
    assert rc == 0
    assert (out_dir / "event_001.jpg").is_file()
    assert (out_dir / "summary.json").is_file()

    res = subprocess.run(
        [sys.executable, "-m", "vlmsieve", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0
    assert "vlmsieve" in res.stdout
