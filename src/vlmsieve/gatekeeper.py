"""Stage 1 & 2: Motion Gatekeeper, Pre-Roll Buffer, Episode Merger, and Keyframe Sampler."""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from vlmsieve.models import BoundingBox, FramePacket, SieveConfig


class MotionGatekeeper:
    """Stateful motion detector with circular pre-roll buffering and bridge-frame retention."""

    def __init__(self, config: SieveConfig, effective_fps: float = 10.0) -> None:
        self.config = config
        self.effective_fps = max(1.0, float(effective_fps))
        self._prev_blurred: np.ndarray | None = None
        self._buffer: list[FramePacket] = []
        self._motion_count: int = 0
        self._cooldown_counter: int = 0
        self._in_event: bool = False
        self._frames_seen: int = 0

        pre_roll_len = max(1, round(config.pre_roll_sec * self.effective_fps))
        self._pre_roll: deque[FramePacket] = deque(maxlen=pre_roll_len)
        self.bridge_pool: list[FramePacket] = []

    def reset_event(self) -> None:
        self._buffer = []
        self._motion_count = 0
        self._cooldown_counter = 0
        self._in_event = False

    def _extract_pre_roll(self) -> list[FramePacket]:
        if not self._pre_roll or self.config.pre_roll_sec <= 0.0:
            return []
        items = list(self._pre_roll)
        self._pre_roll.clear()
        for idx, pkt in enumerate(items):
            if pkt.motion_score >= self.config.pre_roll_min_score:
                return items[idx:]
        return items[-1:]

    def process_frame(
        self, frame_bgr: np.ndarray, frame_index: int, timestamp_sec: float
    ) -> list[FramePacket] | None:
        self._frames_seen += 1
        h, w = frame_bgr.shape[:2]
        target_w = min(w, self.config.downscale_width)
        scale = target_w / float(w)
        target_h = max(1, round(h * scale))

        if target_w != w or target_h != h:
            resized = cv2.resize(
                frame_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA
            )
        else:
            resized = frame_bgr

        if resized.ndim == 3 and resized.shape[2] == 3:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = resized

        if gray.dtype != np.uint8:
            gray = np.clip(gray, 0, 255).astype(np.uint8)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        if self._frames_seen <= self.config.camera_warmup_frames:
            self._prev_blurred = blurred
            return None

        if self._prev_blurred is None or self._prev_blurred.shape != blurred.shape:
            self._prev_blurred = blurred
            return None

        diff = cv2.absdiff(self._prev_blurred, blurred)
        self._prev_blurred = blurred

        _, thresh = cv2.threshold(
            diff, self.config.pixel_diff_threshold, 255, cv2.THRESH_BINARY
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(thresh, kernel, iterations=2)

        motion_score = float(np.count_nonzero(dilated)) / float(dilated.size)
        is_motion = motion_score >= self.config.motion_threshold

        bbox: BoundingBox | None = None
        if motion_score >= min(self.config.motion_threshold, self.config.pre_roll_min_score):
            contours, _ = cv2.findContours(
                dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            valid = [
                c
                for c in contours
                if cv2.contourArea(c) >= self.config.min_contour_area
            ]
            if valid:
                pts = np.vstack(valid)
                bx, by, bw, bh = cv2.boundingRect(pts)
                sx = w / float(target_w)
                sy = h / float(target_h)
                ox = round(bx * sx)
                oy = round(by * sy)
                ow = round(bw * sx)
                oh = round(bh * sy)
                x1 = max(0, min(w, ox))
                y1 = max(0, min(h, oy))
                x2 = max(x1, min(w, ox + ow))
                y2 = max(y1, min(h, oy + oh))
                bbox = BoundingBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1)

        packet = FramePacket(
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            bgr_array=frame_bgr,
            motion_score=motion_score,
            motion_bbox=bbox,
        )

        bridge_step = max(1, round(self.effective_fps / 2.0))
        if self._frames_seen % bridge_step == 0:
            self.bridge_pool.append(packet)

        if is_motion:
            if not self._in_event:
                self._buffer.extend(self._extract_pre_roll())
            self._buffer.append(packet)
            self._in_event = True
            self._motion_count += 1
            self._cooldown_counter = 0

            if len(self._buffer) >= self.config.max_event_frames:
                if self._motion_count >= self.config.min_event_frames:
                    emitted = list(self._buffer)
                    self.reset_event()
                    return emitted
                self.reset_event()
                return None
        elif self._in_event:
            self._cooldown_counter += 1
            if self._cooldown_counter == 1:
                self._buffer.append(packet)
            if self._cooldown_counter >= self.config.cooldown_frames:
                if self._motion_count >= self.config.min_event_frames:
                    emitted = list(self._buffer)
                    self.reset_event()
                    return emitted
                self.reset_event()
                return None
        else:
            self._pre_roll.append(packet)

        return None

    def flush(self) -> list[FramePacket] | None:
        if self._in_event and self._motion_count >= self.config.min_event_frames:
            emitted = list(self._buffer)
            self.reset_event()
            return emitted
        self.reset_event()
        return None


def merge_episodes_with_bridges(
    raw_events: list[list[FramePacket]],
    bridge_pool: list[FramePacket] | None = None,
    merge_gap_sec: float = 6.0,
    max_frames: int = 600,
) -> list[list[FramePacket]]:
    """Merge adjacent motion bursts separated by <= merge_gap_sec and inject ~2 FPS bridge frames."""
    if not raw_events or merge_gap_sec <= 0.0:
        return raw_events

    bridges_all = bridge_pool or []
    merged: list[list[FramePacket]] = [list(raw_events[0])]

    for ev in raw_events[1:]:
        prev_ev = merged[-1]
        t_prev_end = prev_ev[-1].timestamp_sec
        t_curr_start = ev[0].timestamp_sec
        gap = t_curr_start - t_prev_end
        if gap <= merge_gap_sec and (len(prev_ev) + len(ev)) <= max_frames:
            bridges = [
                b
                for b in bridges_all
                if t_prev_end < b.timestamp_sec < t_curr_start
            ]
            prev_ev.extend(bridges)
            prev_ev.extend(ev)
        else:
            merged.append(list(ev))

    return merged


def is_camera_disturbance(
    packets: list[FramePacket],
    frame_w: int,
    frame_h: int,
    motion_threshold: float = 0.015,
) -> tuple[bool, float, int, int]:
    """Detect short global camera shake or auto-exposure flash with no localized object trajectory."""
    active = [
        p
        for p in packets
        if p.motion_score >= motion_threshold and p.motion_bbox is not None
    ]
    if not active:
        return True, 0.0, 0, 0

    frame_area = float(max(1, frame_w * frame_h))
    ratios = [p.motion_bbox.area / frame_area for p in active if p.motion_bbox is not None]
    mean_ratio = float(np.mean(ratios)) if ratios else 0.0
    localized_cnt = sum(1 for r in ratios if 0.005 < r < 0.65)
    global_cnt = sum(1 for r in ratios if r >= 0.65)

    is_disturbance = (
        len(active) <= 25 and mean_ratio >= 0.70 and localized_cnt < 6
    )
    return is_disturbance, mean_ratio, localized_cnt, global_cnt


def select_keyframes(
    packets: list[FramePacket], target_count: int = 6
) -> list[FramePacket]:
    """Peak-preserving keyframe sampler with anti-duplicate min_gap guard (v0.2.0)."""
    if target_count < 1:
        raise ValueError("target_count must be >= 1")
    seen: set[int] = set()
    p: list[FramePacket] = []
    for x in sorted(packets, key=lambda item: item.frame_index):
        if x.frame_index not in seen:
            seen.add(x.frame_index)
            p.append(x)

    if not p:
        return []
    if len(p) <= target_count:
        return list(p)
    if target_count == 1:
        return [max(p, key=lambda x: x.motion_score)]
    if target_count == 2:
        return [p[0], p[-1]]

    t_start = p[0].timestamp_sec
    t_end = p[-1].timestamp_sec
    duration = max(1e-6, t_end - t_start)
    ideal_step = duration / (target_count - 1)
    min_gap = min(1.5, 0.25 * ideal_step)

    selected = [p[0]]
    middle = p[1:-1]
    splits = np.array_split(np.arange(len(middle)), target_count - 2)

    for k, inds in enumerate(splits, 1):
        if len(inds) == 0:
            continue
        target_t = t_start + k * ideal_step
        bucket = [middle[i] for i in inds]
        prev_pkt = selected[-1]

        cands = [
            x
            for x in bucket
            if (x.timestamp_sec - prev_pkt.timestamp_sec) >= min_gap
            and (t_end - x.timestamp_sec) >= min_gap
        ]
        if not cands:
            cands = [
                x
                for x in bucket
                if (x.timestamp_sec - prev_pkt.timestamp_sec) >= min_gap
            ] or bucket

        max_m = max(x.motion_score for x in cands)

        def score(cand: FramePacket) -> float:
            norm_m = (cand.motion_score / max_m) if max_m > 1e-9 else 0.0
            diversity = float(np.mean(np.abs(cand.thumb64 - prev_pkt.thumb64))) / 255.0
            prox = max(
                0.0, 1.0 - abs(cand.timestamp_sec - target_t) / (1.5 * ideal_step)
            )
            return 0.55 * norm_m + 0.35 * diversity + 0.10 * prox

        best = max(cands, key=score)
        selected.append(best)

    selected.append(p[-1])
    return sorted(selected, key=lambda x: x.frame_index)
