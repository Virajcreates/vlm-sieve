"""End-to-End multi-event pipeline integration tests."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from vlmsieve import VideoSieve
from vlmsieve.cli import main


def _create_two_event_video(video_path: Path) -> Path:
    """Synthesize a 180-frame (6.0s at 30fps), 640x360 video containing two distinct events.

    - Frames 0-24: Static background with reference grid lines.
    - Frames 25-60: Event 1 (Cyan object moves horizontally in top-left quadrant).
    - Frames 61-104: 44 Static frames allowing Event 1 cooldown to expire cleanly.
    - Frames 105-145: Event 2 (Orange object moves diagonally in bottom-right quadrant).
    - Frames 146-179: 34 Static tail frames allowing Event 2 cooldown to expire cleanly.
    """
    video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30.0,
        (640, 360),
    )

    h, w = 360, 640
    base_bg = np.zeros((h, w, 3), dtype=np.uint8)
    base_bg[:, :] = (20, 25, 30)
    # Static structural grid lines
    cv2.line(base_bg, (50, 0), (50, 360), (35, 40, 48), 1)
    cv2.line(base_bg, (0, 180), (640, 180), (35, 40, 48), 1)

    for frame_idx in range(180):
        frame = base_bg.copy()

        if 25 <= frame_idx <= 60:
            # Event 1: moves from (60, 50) to (240, 90) in top-left quadrant
            t = (frame_idx - 25) / 35.0
            rx = round(60 + t * (240 - 60))
            ry = round(50 + t * (90 - 50))
            # Bright cyan/white BGR (255, 255, 100)
            cv2.rectangle(frame, (rx, ry), (rx + 60, ry + 50), (255, 255, 100), -1)

        elif 105 <= frame_idx <= 145:
            # Event 2: moves from (380, 200) to (560, 290) in bottom-right quadrant
            t = (frame_idx - 105) / 40.0
            rx = round(380 + t * (560 - 380))
            ry = round(200 + t * (290 - 200))
            # Bright orange/yellow BGR (0, 165, 255)
            cv2.rectangle(frame, (rx, ry), (rx + 60, ry + 50), (0, 165, 255), -1)

        writer.write(frame)

    writer.release()
    return video_path


def test_e2e_multi_event_video_pipeline(tmp_path: Path) -> None:
    """Verify that VideoSieve extracts both distinct events with localized ROIs and high savings."""
    video_path = tmp_path / "multi_event.mp4"
    _create_two_event_video(video_path)

    sieve = VideoSieve(
        video_path,
        grid_layout=(2, 3),
        auto_crop_roi=True,
        burn_timestamps=True,
        spatial_grid_overlay=True,
    )

    events = list(sieve.stream_events())
    assert len(events) == 2, f"Expected exactly 2 events, got {len(events)}"

    # 1. Inspect Event 1 (Top-Left trajectory)
    ev1 = events[0]
    assert ev1.event_id == 1
    assert len(ev1.keyframes) == 6

    # Verify at least 5 of the 6 keyframes have high motion score (no tail flood)
    motion_keyframes_1 = [
        k for k in ev1.keyframes if k.motion_score >= sieve.config.motion_threshold
    ]
    assert len(motion_keyframes_1) >= 5, (
        f"Expected >= 5 motion keyframes, got {len(motion_keyframes_1)}"
    )

    # Verify ROI is tighter than full frame
    assert ev1.roi_bbox.width < 640
    assert ev1.roi_bbox.height < 360
    assert ev1.storyboard_pil.size[0] > 0
    assert ev1.storyboard_pil.size[1] > 0

    # 2. Inspect Event 2 (Bottom-Right trajectory)
    ev2 = events[1]
    assert ev2.event_id == 2
    assert len(ev2.keyframes) == 6

    motion_keyframes_2 = [
        k for k in ev2.keyframes if k.motion_score >= sieve.config.motion_threshold
    ]
    assert len(motion_keyframes_2) >= 5, (
        f"Expected >= 5 motion keyframes, got {len(motion_keyframes_2)}"
    )

    # Verify spatial shift: Event 2 ROI is in bottom-right quadrant
    assert ev2.roi_bbox.x > ev1.roi_bbox.x, (
        f"Expected ev2.roi_bbox.x ({ev2.roi_bbox.x}) > ev1.roi_bbox.x ({ev1.roi_bbox.x})"
    )
    assert ev2.roi_bbox.y > ev1.roi_bbox.y, (
        f"Expected ev2.roi_bbox.y ({ev2.roi_bbox.y}) > ev1.roi_bbox.y ({ev1.roi_bbox.y})"
    )

    # 3. Verify cumulative savings report
    report = sieve.get_savings_report()
    assert report.raw_frames_inspected == 180
    assert report.storyboards_emitted == 2
    assert report.tokens_saved_pct >= 98.0


def test_e2e_cli_multi_event_output(tmp_path: Path) -> None:
    """Verify that CLI generates all storyboard JPEGs and summary.json for multiple events."""
    video_path = tmp_path / "cli_multi.mp4"
    _create_two_event_video(video_path)

    out_dir = tmp_path / "cli_e2e_out"
    exit_code = main(
        [str(video_path), "-o", str(out_dir), "--grid", "2x3", "--spatial-grid"]
    )

    assert exit_code == 0
    assert (out_dir / "event_001.jpg").is_file()
    assert (out_dir / "event_002.jpg").is_file()
    assert (out_dir / "summary.json").is_file()
