"""Core data models and configurations for vlm-sieve."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field


class BoundingBox(BaseModel):
    """Represents a 2D bounding box with integer pixel coordinates."""

    model_config = ConfigDict(frozen=True)

    x: int = Field(..., description="Top-left X coordinate")
    y: int = Field(..., description="Top-left Y coordinate")
    width: int = Field(..., description="Box width in pixels")
    height: int = Field(..., description="Box height in pixels")

    @property
    def x2(self) -> int:
        """Right boundary coordinate (x + width)."""
        return self.x + self.width

    @property
    def y2(self) -> int:
        """Bottom boundary coordinate (y + height)."""
        return self.y + self.height

    @property
    def area(self) -> int:
        """Total area enclosed by the bounding box in pixels."""
        return max(0, self.width) * max(0, self.height)

    def union(self, other: BoundingBox) -> BoundingBox:
        """Return the smallest BoundingBox enclosing both self and other."""
        min_x = min(self.x, other.x)
        min_y = min(self.y, other.y)
        max_x2 = max(self.x2, other.x2)
        max_y2 = max(self.y2, other.y2)
        return BoundingBox(
            x=min_x,
            y=min_y,
            width=max(0, max_x2 - min_x),
            height=max(0, max_y2 - min_y),
        )

    def pad(self, padding_ratio: float, max_w: int, max_h: int) -> BoundingBox:
        """Expand width and height symmetrically by padding_ratio with boundary clamping.

        Args:
            padding_ratio: Fractional padding added to each side (e.g., 0.15 adds 15%).
            max_w: Maximum allowable width boundary (e.g. frame width).
            max_h: Maximum allowable height boundary (e.g. frame height).

        Returns:
            A new BoundingBox clamped strictly within [0, max_w] and [0, max_h].
        """
        pad_w = round(self.width * padding_ratio)
        pad_h = round(self.height * padding_ratio)

        clamped_x = max(0, min(max_w, self.x - pad_w))
        clamped_y = max(0, min(max_h, self.y - pad_h))
        clamped_x2 = max(clamped_x, min(max_w, self.x2 + pad_w))
        clamped_y2 = max(clamped_y, min(max_h, self.y2 + pad_h))

        return BoundingBox(
            x=clamped_x,
            y=clamped_y,
            width=clamped_x2 - clamped_x,
            height=clamped_y2 - clamped_y,
        )


@dataclass
class FramePacket:
    """Encapsulates a processed video frame with associated motion metadata."""

    frame_index: int
    timestamp_sec: float
    rgb_array: np.ndarray
    motion_score: float
    motion_bbox: BoundingBox | None = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FramePacket):
            return False
        return (
            self.frame_index == other.frame_index
            and abs(self.timestamp_sec - other.timestamp_sec) < 1e-6
            and abs(self.motion_score - other.motion_score) < 1e-6
            and self.motion_bbox == other.motion_bbox
            and np.array_equal(self.rgb_array, other.rgb_array)
        )


class TokenSavingsReport(BaseModel):
    """Metrics quantifying VLM token and API savings achieved by sieving."""

    raw_frames_inspected: int
    frames_dropped: int
    storyboards_emitted: int
    estimated_naive_tokens: int
    estimated_sieve_tokens: int
    compression_ratio: float
    tokens_saved_pct: float


class SieveConfig(BaseModel):
    """Runtime configuration hyperparameters for the vlm-sieve pipeline."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    downscale_width: int = 320
    motion_threshold: float = 0.015
    pixel_diff_threshold: int = 25
    min_contour_area: int = 16
    cooldown_frames: int = 12
    min_event_frames: int = 3
    max_event_frames: int = 150
    grid_layout: tuple[int, int] = (2, 3)
    auto_crop_roi: bool = True
    roi_padding_ratio: float = 0.15
    min_roi_size: int = 224
    burn_timestamps: bool = True
    spatial_grid_overlay: bool = False
    tile_max_width: int = 640
    tokens_per_image: int = 1120
    camera_warmup_frames: int = 8


@dataclass
class VisualEvent:
    """Represents a discrete temporal event containing keyframes and a composite storyboard."""

    event_id: int
    start_frame: int
    end_frame: int
    start_time_sec: float
    end_time_sec: float
    raw_frame_count: int
    keyframes: list[FramePacket]
    roi_bbox: BoundingBox
    storyboard_pil: Image.Image
    savings: TokenSavingsReport

    def save(self, path: str | Path, quality: int = 90) -> str:
        """Save the storyboard PIL image to disk, creating parent directories if needed.

        Args:
            path: Destination file path.
            quality: Compression quality for lossy formats (JPEG/WebP).

        Returns:
            Resolved absolute path as a string.
        """
        out_path = Path(path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        save_kwargs: dict[str, Any] = {}
        ext = out_path.suffix.lower()
        img_to_save = self.storyboard_pil

        if ext in {".jpg", ".jpeg"}:
            if img_to_save.mode in ("RGBA", "LA", "P"):
                img_to_save = img_to_save.convert("RGB")
            save_kwargs["quality"] = quality
        elif ext == ".webp":
            save_kwargs["quality"] = quality

        img_to_save.save(out_path, **save_kwargs)
        return str(out_path)

    def to_bytes(self, format: str = "JPEG", quality: int = 88) -> bytes:
        """Encode the storyboard PIL image into an in-memory byte buffer.

        Args:
            format: Target image format (e.g. 'JPEG', 'PNG', 'WEBP').
            quality: Image quality parameter for lossy encoders.

        Returns:
            Raw encoded image bytes.
        """
        buf = io.BytesIO()
        save_kwargs: dict[str, Any] = {}
        fmt_upper = format.upper()
        img_to_save = self.storyboard_pil

        if fmt_upper in {"JPEG", "JPG"}:
            if img_to_save.mode in ("RGBA", "LA", "P"):
                img_to_save = img_to_save.convert("RGB")
            save_kwargs["quality"] = quality
        elif fmt_upper == "WEBP":
            save_kwargs["quality"] = quality

        img_to_save.save(buf, format=format, **save_kwargs)
        return buf.getvalue()

    def to_base64(self, format: str = "JPEG", quality: int = 88) -> str:
        """Return base64 encoded string of the encoded storyboard."""
        raw_bytes = self.to_bytes(format=format, quality=quality)
        return base64.b64encode(raw_bytes).decode("utf-8")

    def to_data_uri(self, format: str = "JPEG", quality: int = 88) -> str:
        """Return a data URI string representation of the storyboard."""
        b64 = self.to_base64(format=format, quality=quality)
        return f"data:image/{format.lower()};base64,{b64}"
