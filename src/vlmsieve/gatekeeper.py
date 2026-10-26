"""Stage 1 CPU Motion Gatekeeper for temporal event segmentation."""

from __future__ import annotations

import cv2
import numpy as np

from vlmsieve.models import BoundingBox, FramePacket, SieveConfig


class MotionGatekeeper:
    """Ultra-low latency CPU motion gatekeeper and temporal event segmenter.

    Analyzes incoming video frames on downscaled grayscale representations to
    detect significant visual motion, track active motion bounding boxes,
    filter high-frequency camera flicker, and group contiguous motion into
    discrete event segments.
    """

    def __init__(self, config: SieveConfig | None = None) -> None:
        """Initialize the MotionGatekeeper with optional configuration.

        Args:
            config: SieveConfig hyperparameters. Uses default config if None.
        """
        self.config: SieveConfig = config or SieveConfig()
        self._prev_gray: np.ndarray | None = None
        self._active_buffer: list[FramePacket] = []
        self._active_motion_count: int = 0
        self._cooldown_counter: int = 0
        self._in_event: bool = False

    def reset(self) -> None:
        """Reset all internal state, history buffers, and event trackers."""
        self._prev_gray = None
        self._reset_event_state()

    def _reset_event_state(self) -> None:
        """Reset active event buffers and cooldown state."""
        self._active_buffer = []
        self._active_motion_count = 0
        self._cooldown_counter = 0
        self._in_event = False

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        frame_index: int,
        timestamp_sec: float,
    ) -> list[FramePacket] | None:
        """Process an incoming video frame through the motion gating pipeline.

        Args:
            frame_bgr: Input video frame (BGR or Grayscale numpy ndarray).
            frame_index: 0-indexed sequential frame number.
            timestamp_sec: Presentation timestamp in seconds.

        Returns:
            A list of FramePacket instances if an event closed on this frame;
            None otherwise.
        """
        if not isinstance(frame_bgr, np.ndarray):
            raise TypeError("frame_bgr must be a numpy ndarray")
        if frame_bgr.size == 0 or frame_bgr.ndim not in (2, 3):
            raise ValueError("frame_bgr must be a non-empty 2D or 3D numpy ndarray")
        if frame_bgr.ndim == 3 and frame_bgr.shape[2] not in (1, 3):
            raise ValueError("frame_bgr must have 1 or 3 color channels")

        orig_h, orig_w = frame_bgr.shape[:2]
        if orig_h <= 0 or orig_w <= 0:
            raise ValueError(f"Invalid frame dimensions: {orig_w}x{orig_h}")

        # Downscale preserving aspect ratio to sub-3ms evaluation resolution
        target_w = min(orig_w, self.config.downscale_width)
        scale_down = target_w / float(orig_w)
        target_h = max(1, round(orig_h * scale_down))

        if target_w != orig_w or target_h != orig_h:
            resized = cv2.resize(
                frame_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA
            )
        else:
            resized = frame_bgr

        # Convert to single-channel 8-bit grayscale
        if resized.ndim == 3 and resized.shape[2] == 3:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        elif resized.ndim == 3 and resized.shape[2] == 1:
            gray = resized[:, :, 0]
        else:
            gray = resized

        if gray.dtype != np.uint8:
            gray = np.clip(gray, 0, 255).astype(np.uint8)

        # Apply Gaussian blur to reduce high-frequency camera noise
        blurred_gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Initialize reference frame on first call or when dimension changes
        if self._prev_gray is None or self._prev_gray.shape != blurred_gray.shape:
            self._prev_gray = blurred_gray
            return None

        # Absolute frame difference
        diff = cv2.absdiff(self._prev_gray, blurred_gray)
        self._prev_gray = blurred_gray

        # Binary threshold and morphological dilation to bridge contours
        _, thresh = cv2.threshold(
            diff, self.config.pixel_diff_threshold, 255, cv2.THRESH_BINARY
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(thresh, kernel, iterations=2)

        # Compute normalized motion score
        motion_score = float(np.count_nonzero(dilated)) / float(dilated.size)
        is_motion = motion_score >= self.config.motion_threshold

        # Extract active motion bounding box if motion detected
        motion_bbox: BoundingBox | None = None
        if is_motion:
            contours, _ = cv2.findContours(
                dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            valid_contours = [
                c
                for c in contours
                if cv2.contourArea(c) >= self.config.min_contour_area
            ]
            if valid_contours:
                all_pts = np.vstack(valid_contours)
                bx, by, bw, bh = cv2.boundingRect(all_pts)

                scale_x = orig_w / float(target_w)
                scale_y = orig_h / float(target_h)

                orig_x = round(bx * scale_x)
                orig_y = round(by * scale_y)
                orig_box_w = round(bw * scale_x)
                orig_box_h = round(bh * scale_y)

                clamped_x = max(0, min(orig_w, orig_x))
                clamped_y = max(0, min(orig_h, orig_y))
                clamped_x2 = max(clamped_x, min(orig_w, orig_x + orig_box_w))
                clamped_y2 = max(clamped_y, min(orig_h, orig_y + orig_box_h))

                motion_bbox = BoundingBox(
                    x=clamped_x,
                    y=clamped_y,
                    width=clamped_x2 - clamped_x,
                    height=clamped_y2 - clamped_y,
                )

        # State machine processing
        if is_motion:
            rgb_array = self._to_rgb(frame_bgr)
            self._active_buffer.append(
                FramePacket(
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    rgb_array=rgb_array,
                    motion_score=motion_score,
                    motion_bbox=motion_bbox,
                )
            )
            self._in_event = True
            self._active_motion_count += 1
            self._cooldown_counter = 0

            # Guard against unbounded event lengths
            if len(self._active_buffer) >= self.config.max_event_frames:
                if self._active_motion_count >= self.config.min_event_frames:
                    emitted = list(self._active_buffer)
                    self._reset_event_state()
                    return emitted
                self._reset_event_state()
                return None

        elif self._in_event:
            self._cooldown_counter += 1
            if self._cooldown_counter == 1:
                rgb_array = self._to_rgb(frame_bgr)
                self._active_buffer.append(
                    FramePacket(
                        frame_index=frame_index,
                        timestamp_sec=timestamp_sec,
                        rgb_array=rgb_array,
                        motion_score=motion_score,
                        motion_bbox=None,
                    )
                )

            if self._cooldown_counter >= self.config.cooldown_frames:
                if self._active_motion_count >= self.config.min_event_frames:
                    emitted = list(self._active_buffer)
                    self._reset_event_state()
                    return emitted
                # Reject transient flicker
                self._reset_event_state()
                return None

        return None

    @staticmethod
    def _to_rgb(frame: np.ndarray) -> np.ndarray:
        """Convert BGR or grayscale numpy array to RGB."""
        if frame.ndim == 3 and frame.shape[2] == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame.ndim == 3 and frame.shape[2] == 1:
            return cv2.cvtColor(frame[:, :, 0], cv2.COLOR_GRAY2RGB)
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

    def flush(self) -> list[FramePacket] | None:
        """Flush any active event buffer at end-of-stream.

        Returns:
            Buffered frame packets if an active event meets minimum event length;
            otherwise None.
        """
        if self._in_event and self._active_motion_count >= self.config.min_event_frames:
            emitted = list(self._active_buffer)
            self._reset_event_state()
            return emitted
        self._reset_event_state()
        return None
