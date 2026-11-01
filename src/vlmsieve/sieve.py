"""High-level VideoSieve orchestrator for files, webcams, and frame iterables."""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from collections.abc import Iterable, Iterator
import json
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np

from vlmsieve.gatekeeper import (
    MotionGatekeeper,
    is_camera_disturbance,
    merge_episodes_with_bridges,
    select_keyframes,
)
from vlmsieve.models import (
    DisturbanceRecord,
    FramePacket,
    SieveConfig,
    TokenSavingsReport,
    VisualEvent,
)
from vlmsieve.roi import compute_event_roi
from vlmsieve.storyboard import StoryboardPacker


class VideoSieve:
    """High-level spatial-temporal gatekeeper and storyboard generator (v0.2.1)."""

    def __init__(
        self,
        source: str | Path | int | Iterable[np.ndarray],
        config: SieveConfig | None = None,
        **kwargs: Any,
    ) -> None:
        if config is not None and kwargs:
            merged_data = config.model_dump()
            merged_data.update(kwargs)
            self.config = SieveConfig(**merged_data)
        elif config is not None:
            self.config = config
        else:
            self.config = SieveConfig(**kwargs)

        self.source = source
        self.packer = StoryboardPacker(self.config)
        self._raw_frames_inspected: int = 0
        self._sampled_frames_decoded: int = 0
        self._storyboards_emitted: int = 0
        self._disturbances: list[DisturbanceRecord] = []
        self._last_processing_sec: float = 0.0

    def get_disturbances(self) -> list[DisturbanceRecord]:
        """Return camera-shake or exposure disturbances suppressed during the last run."""
        return list(self._disturbances)

    def _emit_visual_event(
        self,
        buf: list[FramePacket],
        event_id: int,
        episode_index: int,
        frame_w: int,
        frame_h: int,
    ) -> VisualEvent:
        max_slots = self.config.grid_layout[0] * self.config.grid_layout[1]
        keyframes = select_keyframes(buf, target_count=max_slots)
        roi = compute_event_roi(buf, frame_w, frame_h, self.config)
        storyboard_img = self.packer.pack(keyframes, roi)

        kf_times = [round(p.timestamp_sec, 2) for p in keyframes]
        kf_gaps = [
            round(kf_times[j + 1] - kf_times[j], 2)
            for j in range(len(kf_times) - 1)
        ]

        return VisualEvent(
            event_id=event_id,
            episode_index=episode_index,
            start_frame=buf[0].frame_index,
            end_frame=buf[-1].frame_index,
            start_time_sec=round(buf[0].timestamp_sec, 3),
            end_time_sec=round(buf[-1].timestamp_sec, 3),
            raw_frame_count=len(buf),
            keyframes=keyframes,
            roi_bbox=roi,
            storyboard_pil=storyboard_img,
            savings=self.get_savings_report(),
            keyframe_times_sec=kf_times,
            keyframe_gaps_sec=kf_gaps,
        )

    def stream_events(self, max_frames: int | None = None) -> Iterator[VisualEvent]:
        """Yield VisualEvent objects from a video file, webcam index, or frame iterable."""
        self._raw_frames_inspected = 0
        self._sampled_frames_decoded = 0
        self._storyboards_emitted = 0
        self._disturbances = []
        t0 = time.perf_counter()

        if not isinstance(self.source, (str, Path, int)):
            fps = 30.0
            gate = MotionGatekeeper(self.config, effective_fps=fps)
            raw_events: list[list[FramePacket]] = []
            frame_w, frame_h = 640, 360

            for idx, frame in enumerate(self.source):
                if max_frames is not None and idx >= max_frames:
                    break
                if not isinstance(frame, np.ndarray) or frame.ndim != 3:
                    raise ValueError("Each frame in source iterable must be a 3D numpy ndarray.")
                frame_h, frame_w = frame.shape[:2]
                self._raw_frames_inspected += 1
                self._sampled_frames_decoded += 1
                ev = gate.process_frame(frame, idx, idx / fps)
                if ev is not None:
                    raw_events.append(ev)

            tail = gate.flush()
            if tail is not None:
                raw_events.append(tail)

            merged = merge_episodes_with_bridges(
                raw_events,
                gate.bridge_pool,
                self.config.merge_gap_sec,
                self.config.max_event_frames,
            )
            for ep_idx, buf in enumerate(merged, 1):
                is_shake, mean_ratio, loc_cnt, glob_cnt = is_camera_disturbance(
                    buf, frame_w, frame_h, self.config.motion_threshold
                )
                if is_shake and self.config.suppress_disturbances:
                    scores = [p.motion_score for p in buf]
                    self._disturbances.append(
                        DisturbanceRecord(
                            episode_index=ep_idx,
                            start_frame=buf[0].frame_index,
                            end_frame=buf[-1].frame_index,
                            start_time_sec=round(buf[0].timestamp_sec, 3),
                            end_time_sec=round(buf[-1].timestamp_sec, 3),
                            duration_sec=round(buf[-1].timestamp_sec - buf[0].timestamp_sec, 3),
                            max_motion_score=round(max(scores), 5) if scores else None,
                            mean_bbox_area_ratio=round(mean_ratio, 3),
                            localized_frames=loc_cnt,
                            global_frames=glob_cnt,
                        )
                    )
                    continue
                self._storyboards_emitted += 1
                yield self._emit_visual_event(
                    buf, self._storyboards_emitted, ep_idx, frame_w, frame_h
                )
            self._last_processing_sec = time.perf_counter() - t0
            return

        is_live_camera = isinstance(self.source, int)
        cap_target = self.source if is_live_camera else str(Path(self.source).expanduser())
        cap = cv2.VideoCapture(cap_target)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0 or fps > 240:
                fps = 30.0
            frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
            frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 360

            if not is_live_camera and self.config.sample_fps and self.config.sample_fps > 0:
                stride = max(1, round(fps / self.config.sample_fps))
            else:
                stride = 1

            effective_fps = fps / float(stride)
            gate = MotionGatekeeper(self.config, effective_fps=effective_fps)
            raw_events = []
            idx = 0

            while True:
                if max_frames is not None and idx >= max_frames:
                    break

                if stride > 1 and (idx % stride != 0):
                    if not cap.grab():
                        break
                    idx += 1
                    self._raw_frames_inspected += 1
                    continue

                ok, frame = cap.read()
                if not ok or frame is None:
                    break

                frame_h, frame_w = frame.shape[:2]
                self._raw_frames_inspected += 1
                self._sampled_frames_decoded += 1
                ts = idx / fps
                ev = gate.process_frame(frame, idx, ts)
                idx += 1

                if ev is not None:
                    if is_live_camera and self.config.merge_gap_sec <= 0.0:
                        self._storyboards_emitted += 1
                        yield self._emit_visual_event(
                            ev,
                            self._storyboards_emitted,
                            self._storyboards_emitted,
                            frame_w,
                            frame_h,
                        )
                    else:
                        raw_events.append(ev)

            tail = gate.flush()
            if tail is not None:
                if is_live_camera and self.config.merge_gap_sec <= 0.0:
                    self._storyboards_emitted += 1
                    yield self._emit_visual_event(
                        tail,
                        self._storyboards_emitted,
                        self._storyboards_emitted,
                        frame_w,
                        frame_h,
                    )
                else:
                    raw_events.append(tail)

            if raw_events:
                merged = merge_episodes_with_bridges(
                    raw_events,
                    gate.bridge_pool,
                    self.config.merge_gap_sec,
                    self.config.max_event_frames,
                )
                for ep_idx, buf in enumerate(merged, 1):
                    is_shake, mean_ratio, loc_cnt, glob_cnt = is_camera_disturbance(
                        buf, frame_w, frame_h, self.config.motion_threshold
                    )
                    if is_shake and self.config.suppress_disturbances:
                        scores = [p.motion_score for p in buf]
                        self._disturbances.append(
                            DisturbanceRecord(
                                episode_index=ep_idx,
                                start_frame=buf[0].frame_index,
                                end_frame=buf[-1].frame_index,
                                start_time_sec=round(buf[0].timestamp_sec, 3),
                                end_time_sec=round(buf[-1].timestamp_sec, 3),
                                duration_sec=round(buf[-1].timestamp_sec - buf[0].timestamp_sec, 3),
                                max_motion_score=round(max(scores), 5) if scores else None,
                                mean_bbox_area_ratio=round(mean_ratio, 3),
                                localized_frames=loc_cnt,
                                global_frames=glob_cnt,
                            )
                        )
                        continue
                    self._storyboards_emitted += 1
                    yield self._emit_visual_event(
                        buf, self._storyboards_emitted, ep_idx, frame_w, frame_h
                    )
        finally:
            cap.release()
            self._last_processing_sec = time.perf_counter() - t0

    async def astream_events(self, max_frames: int | None = None) -> AsyncIterator[VisualEvent]:
        for event in self.stream_events(max_frames=max_frames):
            await asyncio.sleep(0)
            yield event

    def get_savings_report(self) -> TokenSavingsReport:
        """Compute token savings metrics for the most recent stream_events() run."""
        raw = self._raw_frames_inspected
        emitted = self._storyboards_emitted
        max_slots = self.config.grid_layout[0] * self.config.grid_layout[1]
        dropped = max(0, raw - emitted * max_slots)

        naive_tokens = raw * self.config.tokens_per_image
        sieve_tokens = emitted * self.config.tokens_per_image
        if naive_tokens > 0:
            saved_pct = round((naive_tokens - sieve_tokens) / naive_tokens * 100.0, 2)
        else:
            saved_pct = 0.0

        ratio = round(raw / float(emitted), 2) if emitted > 0 else float(raw)
        return TokenSavingsReport(
            raw_frames_inspected=raw,
            sampled_frames_decoded=self._sampled_frames_decoded,
            frames_dropped=dropped,
            storyboards_emitted=emitted,
            disturbances_suppressed=len(self._disturbances),
            estimated_naive_tokens=naive_tokens,
            estimated_sieve_tokens=sieve_tokens,
            compression_ratio=ratio,
            tokens_saved_pct=saved_pct,
        )

    def process_video_to_dir(
        self, output_dir: str | Path, max_frames: int | None = None
    ) -> list[VisualEvent]:
        """Process video source, save event_XXX.jpg storyboards + summary.json, and return events."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        for old_jpg in out_path.glob("event_*.jpg"):
            old_jpg.unlink()

        events: list[VisualEvent] = []
        event_summaries: list[dict[str, Any]] = []

        for event in self.stream_events(max_frames=max_frames):
            img_name = f"event_{event.event_id:03d}.jpg"
            event.save(out_path / img_name)
            events.append(event)
            event_summaries.append(event.to_summary_dict(img_name))

        report = self.get_savings_report()
        summary_payload = {
            "pipeline": {
                "version": "0.2.1",
                "sample_fps": self.config.sample_fps,
                "motion_threshold": self.config.motion_threshold,
                "merge_gap_sec": self.config.merge_gap_sec,
                "pre_roll_sec": self.config.pre_roll_sec,
                "tile_max_dim": self.config.tile_max_dim,
                "suppress_disturbances": self.config.suppress_disturbances,
                "processing_wall_seconds": round(self._last_processing_sec, 3),
            },
            "savings": report.model_dump(),
            "events": event_summaries,
            "disturbances": [d.model_dump() for d in self._disturbances],
        }
        (out_path / "summary.json").write_text(
            json.dumps(summary_payload, indent=2), encoding="utf-8"
        )
        return events
