"""Generate a production demo storyboard showcase image for documentation and README."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from vlmsieve import SieveConfig, VideoSieve


def create_simulated_security_scene(total_frames: int = 150) -> list[np.ndarray]:
    """Generate 150 synthetic 1280x720 frames simulating a security camera feed.

    Includes a static background grid, a restricted boundary zone, telemetry overlays,
    and a moving autonomous vehicle crossing the scene from frame 30 to 105.

    Args:
        total_frames: Total number of frames (default 150 = 5.0s at 30fps).

    Returns:
        List of 1280x720 BGR numpy ndarrays.
    """
    h, w = 720, 1280
    base_frame = np.zeros((h, w, 3), dtype=np.uint8)
    base_frame[:, :] = (18, 20, 24)  # Dark charcoal slate

    # 1. Floor grid lines
    for y in range(80, h, 60):
        cv2.line(base_frame, (0, y), (w, y), (30, 34, 42), 1)
    for x in range(0, w, 80):
        cv2.line(base_frame, (x, 80), (x, h), (30, 34, 42), 1)

    # 2. Static restricted perimeter zone
    zone_x1, zone_y1 = 360, 180
    zone_x2, zone_y2 = 920, 560
    cv2.rectangle(
        base_frame,
        (zone_x1, zone_y1),
        (zone_x2, zone_y2),
        (0, 140, 255),
        2,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        base_frame,
        "[ RESTRICTED PERIMETER // SECTOR-04 ]",
        (zone_x1 + 16, zone_y1 + 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 140, 255),
        2,
        lineType=cv2.LINE_AA,
    )

    # 3. Top telemetry status HUD
    cv2.rectangle(base_frame, (0, 0), (w, 52), (12, 14, 18), -1)
    cv2.line(base_frame, (0, 52), (w, 52), (45, 52, 64), 1)
    cv2.putText(
        base_frame,
        "FEED: CAM-09 [SOUTH CORRIDOR] | 30.0 FPS | 720p HD | AI GATEWAY ACTIVE",
        (24, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (74, 222, 128),  # Crisp emerald green
        1,
        lineType=cv2.LINE_AA,
    )

    frames: list[np.ndarray] = []

    for idx in range(total_frames):
        frame = base_frame.copy()

        # Freeze clock during static periods to prevent artificial motion from ticking digits
        if idx < 30:
            clock_sec = 1.00
        elif idx <= 105:
            clock_sec = idx / 30.0
        else:
            clock_sec = 3.50

        hud_time = f"REC 00:00:{clock_sec:05.2f}"
        cv2.putText(
            frame,
            hud_time,
            (w - 230, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 230, 242),
            1,
            lineType=cv2.LINE_AA,
        )

        # Draw robot for all frames >= 30 (animated 30..105, then parked cleanly > 105)
        if idx >= 30:
            if idx <= 105:
                progress = (idx - 30) / 75.0
                # Spline-like trajectory from (80, 260) -> (580, 340) -> (960, 520)
                if progress <= 0.5:
                    sub_t = progress / 0.5
                    rx = round(80 + sub_t * (580 - 80))
                    ry = round(260 + sub_t * (340 - 260))
                else:
                    sub_t = (progress - 0.5) / 0.5
                    rx = round(580 + sub_t * (960 - 580))
                    ry = round(340 + sub_t * (520 - 340))
            else:
                # Parked cleanly at final destination
                rx, ry = 960, 520

            rw, rh = 120, 80

            # Outer bounding shadow & vehicle body
            cv2.rectangle(
                frame,
                (rx, ry),
                (rx + rw, ry + rh),
                (255, 200, 30),  # Bright electric cyan BGR
                -1,
            )
            cv2.rectangle(
                frame,
                (rx, ry),
                (rx + rw, ry + rh),
                (255, 255, 255),
                2,
                lineType=cv2.LINE_AA,
            )

            # High-intensity headlights
            cv2.circle(frame, (rx + rw - 6, ry + 16), 7, (255, 255, 255), -1)
            cv2.circle(frame, (rx + rw - 6, ry + rh - 16), 7, (255, 255, 255), -1)

            # Identification label
            cv2.putText(
                frame,
                "AGV-BOT #01",
                (rx + 8, ry - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (56, 189, 248),
                2,
                lineType=cv2.LINE_AA,
            )

        frames.append(frame)

    return frames


def main() -> None:
    """Run simulated video through VideoSieve and emit demo storyboard artifact."""
    print("Synthesizing 720p simulated robotics / security video feed (150 frames)...")
    frames = create_simulated_security_scene(total_frames=150)

    config = SieveConfig(
        grid_layout=(2, 3),
        motion_threshold=0.008,
        tile_max_width=640,
        auto_crop_roi=True,
        roi_padding_ratio=0.18,
        min_roi_size=280,
        burn_timestamps=True,
        spatial_grid_overlay=True,
    )

    sieve = VideoSieve(source=frames, config=config)
    print("Processing video frames through VideoSieve pipeline...")
    events = list(sieve.stream_events())

    if not events:
        print("Warning: No visual events were emitted.")
        return

    event = events[0]
    assets_dir = Path("assets")
    assets_dir.mkdir(parents=True, exist_ok=True)
    out_path = assets_dir / "demo_storyboard.jpg"

    saved_path = event.save(out_path, quality=92)
    savings = sieve.get_savings_report()

    print("\n" + "=" * 60)
    print("          vlm-sieve Showcase Demo Summary")
    print("=" * 60)
    print(f" Demo Storyboard Saved : {saved_path}")
    print(
        f" Storyboard Dimensions : {event.storyboard_pil.size[0]}x{event.storyboard_pil.size[1]}px"
    )
    print(f" Raw Frames Inspected  : {savings.raw_frames_inspected}")
    print(f" Frames Dropped        : {savings.frames_dropped}")
    print(f" Storyboards Emitted   : {savings.storyboards_emitted}")
    print(f" Naive Token Estimate  : {savings.estimated_naive_tokens:,}")
    print(f" Sieve Token Estimate  : {savings.estimated_sieve_tokens:,}")
    print(f" Compression Ratio     : {savings.compression_ratio:.1f}x")
    print(f" Token Savings         : {savings.tokens_saved_pct:.2f}%")
    print(
        f" Localized ROI Area    : {event.roi_bbox.width}x{event.roi_bbox.height}px at ({event.roi_bbox.x}, {event.roi_bbox.y})"
    )
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
