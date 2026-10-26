"""Command-line interface (CLI) for vlm-sieve."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vlmsieve.models import SieveConfig
from vlmsieve.sieve import VideoSieve


def parse_grid(grid_str: str) -> tuple[int, int]:
    """Parse a grid dimension string in RxC format into a tuple (rows, cols).

    Args:
        grid_str: Grid layout string (e.g., '2x3').

    Returns:
        Tuple of (rows, cols) as positive integers.

    Raises:
        argparse.ArgumentTypeError: If string is malformed or dimensions <= 0.
    """
    parts = grid_str.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"Invalid grid format '{grid_str}'. Expected format: RxC (e.g. 2x3)"
        )
    try:
        r = int(parts[0].strip())
        c = int(parts[1].strip())
    except ValueError as err:
        raise argparse.ArgumentTypeError(
            f"Grid dimensions must be integers in '{grid_str}'"
        ) from err

    if r <= 0 or c <= 0:
        raise argparse.ArgumentTypeError(f"Grid dimensions must be >= 1, got {r}x{c}")

    return (r, c)


def build_parser() -> argparse.ArgumentParser:
    """Construct and configure the argparse ArgumentParser for vlmsieve."""
    parser = argparse.ArgumentParser(
        prog="vlmsieve",
        description="vlm-sieve: Event-driven video frame gatekeeper and storyboard packer for VLMs.",
    )
    parser.add_argument(
        "source",
        help="Path to video file, RTSP/HTTP stream URL, or camera index (e.g. 0)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="./sieve_out",
        help="Output directory for generated storyboards and summary.json (default: ./sieve_out)",
    )
    parser.add_argument(
        "--grid",
        type=parse_grid,
        default=(2, 3),
        help="Storyboard grid layout RxC (default: 2x3)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.015,
        help="Motion score threshold for frame gating (default: 0.015)",
    )
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="Disable active ROI auto-cropping (use full frames)",
    )
    parser.add_argument(
        "--spatial-grid",
        action="store_true",
        help="Enable 4x4 spatial grid coordinate overlay on tiles",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum raw video frames to process",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for running vlm-sieve.

    Args:
        argv: Command-line arguments list. Defaults to sys.argv[1:].

    Returns:
        Exit code (0 on success).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    config = SieveConfig(
        grid_layout=args.grid,
        motion_threshold=args.threshold,
        auto_crop_roi=not args.no_crop,
        spatial_grid_overlay=args.spatial_grid,
    )

    sieve = VideoSieve(source=args.source, config=config)
    events = sieve.process_video_to_dir(
        output_dir=args.output_dir, max_frames=args.max_frames
    )
    savings = sieve.get_savings_report()

    # Formatted ASCII summary table
    out_dir_resolved = Path(args.output_dir).resolve()
    print("\n" + "=" * 60)
    print("                vlm-sieve Processing Summary")
    print("=" * 60)
    print(f" Raw Frames Inspected  : {savings.raw_frames_inspected}")
    print(f" Storyboards Emitted   : {savings.storyboards_emitted}")
    print(f" Frames Dropped        : {savings.frames_dropped}")
    print(f" Naive Token Estimate  : {savings.estimated_naive_tokens:,}")
    print(f" Sieve Token Estimate  : {savings.estimated_sieve_tokens:,}")
    print(f" Compression Ratio     : {savings.compression_ratio:.1f}x")
    print(f" Token Savings         : {savings.tokens_saved_pct:.2f}%")
    print("=" * 60)
    if events:
        print(f" Output Directory      : {out_dir_resolved}")
        print(f" Total Events Saved    : {len(events)}")
    print("=" * 60 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
