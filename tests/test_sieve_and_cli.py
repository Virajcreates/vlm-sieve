"""Unit tests for VideoSieve developer SDK and CLI application."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from vlmsieve import SieveConfig, VideoSieve, VisualEvent
from vlmsieve.cli import main, parse_grid


def _generate_synthetic_frames(
    static_before: int = 20,
    motion_frames: int = 12,
    static_after: int = 20,
    shape: tuple[int, int] = (360, 640),
) -> list[np.ndarray]:
    """Generate a sequence of synthetic BGR frames containing a moving white block."""
    h, w = shape
    frames: list[np.ndarray] = []
    base = np.zeros((h, w, 3), dtype=np.uint8)

    # Leading static frames
    for _ in range(static_before):
        frames.append(base.copy())

    # Active motion frames
    for i in range(motion_frames):
        f = base.copy()
        sx = 50 + i * 15
        sy = 100
        f[sy : sy + 80, sx : sx + 80] = 255
        frames.append(f)

    # Trailing static frames to close event
    for _ in range(static_after):
        frames.append(base.copy())

    return frames


def test_videosieve_iterable_stream() -> None:
    """Verify VideoSieve correctly processes an in-memory iterable of frames."""
    frames = _generate_synthetic_frames(20, 12, 20)
    cfg = SieveConfig(min_event_frames=3, cooldown_frames=10)
    sieve = VideoSieve(source=frames, config=cfg)

    events = list(sieve.stream_events())
    assert len(events) == 1

    event = events[0]
    assert event.event_id == 1
    assert event.raw_frame_count >= 12
    # Verify token savings > 90%
    assert event.savings.tokens_saved_pct > 90.0
    # Verify active ROI crop is tighter than full frame (640x360)
    assert event.roi_bbox.width < 640
    assert event.roi_bbox.height < 360


def test_videosieve_async_parity() -> None:
    """Verify astream_events matches synchronous stream_events output identically."""
    frames = _generate_synthetic_frames(15, 8, 15)
    cfg = SieveConfig(min_event_frames=3, cooldown_frames=8)

    sieve_sync = VideoSieve(source=frames, config=cfg)
    sync_events = list(sieve_sync.stream_events())

    async def _collect_async() -> list[VisualEvent]:
        sieve_async = VideoSieve(source=frames, config=cfg)
        collected = []
        async for ev in sieve_async.astream_events():
            collected.append(ev)
        return collected

    async_events = asyncio.run(_collect_async())

    assert len(sync_events) == len(async_events) == 1
    se, ae = sync_events[0], async_events[0]
    assert se.event_id == ae.event_id
    assert se.start_frame == ae.start_frame
    assert se.end_frame == ae.end_frame
    assert se.raw_frame_count == ae.raw_frame_count
    assert se.roi_bbox == ae.roi_bbox


def test_videosieve_process_video_to_dir_and_cli(tmp_path: Path) -> None:
    """Verify VideoSieve.process_video_to_dir and CLI execution on an encoded video file."""
    # Write a test video to disk
    video_path = tmp_path / "sample.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        30.0,
        (640, 360),
    )
    frames = _generate_synthetic_frames(20, 10, 20)
    for f in frames:
        writer.write(f)
    writer.release()

    out_dir = tmp_path / "cli_output"
    exit_code = main(
        [
            str(video_path),
            "-o",
            str(out_dir),
            "--grid",
            "2x3",
            "--threshold",
            "0.015",
        ]
    )

    assert exit_code == 0
    storyboard_file = out_dir / "event_001.jpg"
    summary_file = out_dir / "summary.json"

    assert storyboard_file.is_file()
    assert summary_file.is_file()

    with open(summary_file, encoding="utf-8") as f:
        data = json.load(f)

    assert "savings" in data
    assert "events" in data
    assert len(data["events"]) == 1
    assert data["events"][0]["image_file"] == "event_001.jpg"
    assert data["savings"]["storyboards_emitted"] == 1
    assert data["savings"]["tokens_saved_pct"] > 80.0


def test_cli_parse_grid() -> None:
    """Verify CLI grid parser validations and error handling."""
    assert parse_grid("2x3") == (2, 3)
    assert parse_grid("1x4") == (1, 4)
    assert parse_grid(" 3 X 3 ") == (3, 3)

    with pytest.raises(argparse.ArgumentTypeError, match="Invalid grid format"):
        parse_grid("2-3")

    with pytest.raises(argparse.ArgumentTypeError, match="must be integers"):
        parse_grid("axb")

    with pytest.raises(argparse.ArgumentTypeError, match="must be >= 1"):
        parse_grid("0x3")

    with pytest.raises(argparse.ArgumentTypeError, match="must be >= 1"):
        parse_grid("2x-1")
