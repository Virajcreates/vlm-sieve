"""Command-line interface for vlm-sieve v0.2.0."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from vlmsieve import __version__
from vlmsieve.models import SieveConfig
from vlmsieve.sieve import VideoSieve


def parse_grid(value: str) -> tuple[int, int]:
    try:
        parts = value.strip().lower().split("x")
        if len(parts) != 2:
            raise ValueError
        rows, cols = int(parts[0]), int(parts[1])
        if rows < 1 or cols < 1:
            raise ValueError
        return rows, cols
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid grid format '{value}'. Expected ROWSxCOLS, e.g., 2x3."
        ) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vlmsieve",
        description=(
            "vlm-sieve: Local-first spatial-temporal video gatekeeper and "
            "storyboard packer for Vision-Language Models."
        ),
    )
    parser.add_argument(
        "source",
        help="Path to a video file or integer webcam index (e.g. 0)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="sieve_output",
        help="Directory to save storyboard JPEGs and summary.json (default: sieve_output)",
    )
    parser.add_argument(
        "--grid",
        type=parse_grid,
        default=(2, 3),
        help="Storyboard grid layout as ROWSxCOLS (default: 2x3)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.015,
        help="Motion fraction threshold in (0, 1] (default: 0.015)",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=10.0,
        help="Target inspection FPS using bit-stream frame striding (default: 10.0)",
    )
    parser.add_argument(
        "--merge-gap-sec",
        type=float,
        default=6.0,
        help="Merge motion bursts separated by <= N seconds (default: 6.0)",
    )
    parser.add_argument(
        "--pre-roll-sec",
        type=float,
        default=3.0,
        help="Pre-roll ring buffer duration in seconds (default: 3.0)",
    )
    parser.add_argument(
        "--tile-max-dim",
        type=int,
        default=640,
        help="Maximum pixel dimension (width or height) per tile (default: 640)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on raw frames inspected",
    )
    parser.add_argument(
        "--spatial-grid",
        action="store_true",
        default=True,
        help="Draw A1..D4 spatial grid overlay on tiles (enabled by default)",
    )
    parser.add_argument(
        "--no-spatial-grid",
        action="store_false",
        dest="spatial_grid",
        help="Disable A1..D4 spatial grid overlay on tiles",
    )
    parser.add_argument(
        "--keep-disturbances",
        action="store_true",
        help="Do not suppress global camera-shake / exposure disturbance storyboards",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"vlm-sieve {__version__}",
    )

    args = parser.parse_args(argv)
    source: str | int = int(args.source) if args.source.isdigit() else args.source

    cfg = SieveConfig(
        grid_layout=args.grid,
        motion_threshold=args.threshold,
        sample_fps=args.sample_fps,
        merge_gap_sec=args.merge_gap_sec,
        pre_roll_sec=args.pre_roll_sec,
        tile_max_dim=args.tile_max_dim,
        spatial_grid_overlay=args.spatial_grid,
        suppress_disturbances=not args.keep_disturbances,
    )

    sieve = VideoSieve(source=source, config=cfg)
    sieve.process_video_to_dir(args.output_dir, max_frames=args.max_frames)
    report = sieve.get_savings_report()

    print(f"vlm-sieve v{__version__} Complete!")
    print(f"  Raw frames scanned     : {report.raw_frames_inspected}")
    print(f"  Sampled frames decoded : {report.sampled_frames_decoded}")
    print(f"  Storyboards emitted    : {report.storyboards_emitted}")
    print(f"  Disturbances suppressed: {report.disturbances_suppressed}")
    print(f"  Frame token reduction  : {report.tokens_saved_pct:.2f}%")
    print(f"  Output directory       : {Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
