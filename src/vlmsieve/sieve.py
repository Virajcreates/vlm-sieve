"""Unified developer SDK and pipeline coordinator for vlm-sieve."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Iterable, Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vlmsieve.gatekeeper import MotionGatekeeper
from vlmsieve.models import FramePacket, SieveConfig, TokenSavingsReport, VisualEvent
from vlmsieve.roi import compute_event_roi
from vlmsieve.sampler import select_keyframes
from vlmsieve.storyboard import StoryboardPacker


class VideoSieve:
    """Unified video processing engine and frame gatekeeper for Vision-Language Models.

    Orchestrates motion-based frame gating, temporal-peak keyframe sampling,
    active ROI auto-cropping, and multi-frame structured storyboard packing.
    """

    def __init__(
        self,
        source: str | int | Path | Iterable[np.ndarray],
        config: SieveConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize VideoSieve with a video source and configuration.

        Args:
            source: Video file path, camera index, stream URL, or iterable of BGR frames.
            config: Optional pre-configured SieveConfig instance.
            **kwargs: Hyperparameter overrides passed to SieveConfig.
        """
        if config is not None and kwargs:
            self.config = config.model_copy(update=kwargs)
        elif config is not None:
            self.config = config
        else:
            self.config = SieveConfig(**kwargs)

        self.source = source
        self._packer = StoryboardPacker()

        # Cumulative pipeline metrics
        self.raw_frames_inspected: int = 0
        self.frames_dropped: int = 0
        self.storyboards_emitted: int = 0

    def _iter_raw_frames(
        self, max_frames: int | None = None
    ) -> Iterator[tuple[int, float, np.ndarray]]:
        """Yield sequential (frame_index, timestamp_sec, frame_bgr) tuples from source."""
        if isinstance(self.source, (str, int, Path)):
            is_live_camera = isinstance(self.source, int) or (
                isinstance(self.source, str) and self.source.isdigit()
            )
            src: str | int
            if is_live_camera:
                src = int(self.source)
            elif isinstance(self.source, Path):
                src = str(self.source)
            else:
                src = self.source

            cap = cv2.VideoCapture(src)
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video source: {self.source}")

            if is_live_camera and self.config.camera_warmup_frames > 0:
                for _ in range(self.config.camera_warmup_frames):
                    ret, _ = cap.read()
                    if not ret:
                        break

            fps = cap.get(cv2.CAP_PROP_FPS) if not is_live_camera else 0.0
            if fps <= 0 or np.isnan(fps):
                fps = 30.0

            start_wall_time = time.perf_counter()
            frame_idx = 0
            try:
                while max_frames is None or frame_idx < max_frames:
                    ret, frame_bgr = cap.read()
                    if not ret or frame_bgr is None:
                        break
                    if is_live_camera:
                        timestamp_sec = time.perf_counter() - start_wall_time
                    else:
                        timestamp_sec = frame_idx / fps
                    yield frame_idx, timestamp_sec, frame_bgr
                    frame_idx += 1
            finally:
                cap.release()

        elif isinstance(self.source, Iterable):
            fps = 30.0
            for frame_idx, frame_bgr in enumerate(self.source):
                if max_frames is not None and frame_idx >= max_frames:
                    break
                if not isinstance(frame_bgr, np.ndarray):
                    raise TypeError(
                        f"Expected video frame to be np.ndarray, got {type(frame_bgr)}"
                    )
                timestamp_sec = frame_idx / fps
                yield frame_idx, timestamp_sec, frame_bgr
        else:
            raise TypeError(f"Unsupported video source type: {type(self.source)}")

    def _build_visual_event(
        self,
        event_id: int,
        buffer: list[FramePacket],
        frame_w: int,
        frame_h: int,
    ) -> VisualEvent:
        """Construct a complete VisualEvent from an emitted motion frame buffer."""
        rows, cols = self.config.grid_layout
        target_keyframes = rows * cols

        keyframes = select_keyframes(buffer, target_count=target_keyframes)
        roi = compute_event_roi(buffer, frame_w, frame_h, self.config)
        storyboard = self._packer.pack(keyframes, roi, self.config)

        self.storyboards_emitted += 1
        slots_per_storyboard = rows * cols
        self.frames_dropped = max(
            0,
            self.raw_frames_inspected
            - (self.storyboards_emitted * slots_per_storyboard),
        )
        report = self.get_savings_report()

        return VisualEvent(
            event_id=event_id,
            start_frame=buffer[0].frame_index,
            end_frame=buffer[-1].frame_index,
            start_time_sec=buffer[0].timestamp_sec,
            end_time_sec=buffer[-1].timestamp_sec,
            raw_frame_count=len(buffer),
            keyframes=keyframes,
            roi_bbox=roi,
            storyboard_pil=storyboard,
            savings=report,
        )

    def get_savings_report(self) -> TokenSavingsReport:
        """Calculate cumulative VLM token savings and compression metrics."""
        naive_tokens = self.raw_frames_inspected * self.config.tokens_per_image
        sieve_tokens = self.storyboards_emitted * self.config.tokens_per_image

        if self.storyboards_emitted > 0:
            compression_ratio = float(self.raw_frames_inspected) / float(
                self.storyboards_emitted
            )
        else:
            compression_ratio = float(self.raw_frames_inspected)

        if naive_tokens > 0:
            saved_pct = (naive_tokens - sieve_tokens) / float(naive_tokens) * 100.0
        else:
            saved_pct = 0.0

        return TokenSavingsReport(
            raw_frames_inspected=self.raw_frames_inspected,
            frames_dropped=self.frames_dropped,
            storyboards_emitted=self.storyboards_emitted,
            estimated_naive_tokens=naive_tokens,
            estimated_sieve_tokens=sieve_tokens,
            compression_ratio=round(compression_ratio, 2),
            tokens_saved_pct=round(saved_pct, 2),
        )

    def stream_events(self, max_frames: int | None = None) -> Iterator[VisualEvent]:
        """Synchronously stream VisualEvent objects detected in the video source.

        Args:
            max_frames: Optional upper limit on raw video frames to inspect.

        Yields:
            VisualEvent instances as discrete motion episodes close.
        """
        self.raw_frames_inspected = 0
        self.frames_dropped = 0
        self.storyboards_emitted = 0

        gatekeeper = MotionGatekeeper(self.config)
        last_w, last_h = 0, 0
        event_id = 0

        for frame_idx, timestamp_sec, frame_bgr in self._iter_raw_frames(
            max_frames=max_frames
        ):
            self.raw_frames_inspected += 1
            last_h, last_w = frame_bgr.shape[:2]

            emitted_buffer = gatekeeper.process_frame(
                frame_bgr, frame_idx, timestamp_sec
            )
            if emitted_buffer is not None:
                event_id += 1
                yield self._build_visual_event(event_id, emitted_buffer, last_w, last_h)

        # End of stream flush
        tail_buffer = gatekeeper.flush()
        if tail_buffer is not None and last_w > 0:
            event_id += 1
            yield self._build_visual_event(event_id, tail_buffer, last_w, last_h)

        slots_per_storyboard = self.config.grid_layout[0] * self.config.grid_layout[1]
        self.frames_dropped = max(
            0,
            self.raw_frames_inspected
            - (self.storyboards_emitted * slots_per_storyboard),
        )

    async def astream_events(
        self, max_frames: int | None = None
    ) -> AsyncIterator[VisualEvent]:
        """Asynchronously stream VisualEvent objects without blocking the event loop.

        Args:
            max_frames: Optional upper limit on raw video frames to inspect.

        Yields:
            VisualEvent instances as discrete motion episodes close.
        """
        sync_iter = self.stream_events(max_frames=max_frames)
        sentinel = object()

        def _get_next() -> Any:
            return next(sync_iter, sentinel)

        while True:
            event = await asyncio.to_thread(_get_next)
            if event is sentinel:
                break
            yield event

    def process_video_to_dir(
        self, output_dir: str | Path, max_frames: int | None = None
    ) -> list[VisualEvent]:
        """Process video and write JPEG storyboards and summary.json to output_dir.

        Args:
            output_dir: Destination directory path.
            max_frames: Optional upper limit on raw video frames to inspect.

        Returns:
            List of emitted VisualEvent objects.
        """
        out_dir = Path(output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        events: list[VisualEvent] = []
        events_meta: list[dict[str, Any]] = []

        for event in self.stream_events(max_frames=max_frames):
            events.append(event)
            img_filename = f"event_{event.event_id:03d}.jpg"
            img_path = out_dir / img_filename
            event.save(img_path)

            events_meta.append(
                {
                    "event_id": event.event_id,
                    "start_frame": event.start_frame,
                    "end_frame": event.end_frame,
                    "start_time_sec": round(event.start_time_sec, 3),
                    "end_time_sec": round(event.end_time_sec, 3),
                    "raw_frame_count": event.raw_frame_count,
                    "roi_bbox": {
                        "x": event.roi_bbox.x,
                        "y": event.roi_bbox.y,
                        "width": event.roi_bbox.width,
                        "height": event.roi_bbox.height,
                    },
                    "image_file": img_filename,
                }
            )

        final_savings = self.get_savings_report()
        summary_payload = {
            "savings": final_savings.model_dump(),
            "events": events_meta,
        }

        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2)

        return events
