"""Data models and configuration for vlm-sieve v0.2.0."""
from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator


class BoundingBox(BaseModel):
    """Axis-aligned bounding box in pixel coordinates."""

    model_config = ConfigDict(frozen=True)

    x: int
    y: int
    width: int
    height: int

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    def union(self, other: BoundingBox) -> BoundingBox:
        min_x = min(self.x, other.x)
        min_y = min(self.y, other.y)
        max_x = max(self.x2, other.x2)
        max_y = max(self.y2, other.y2)
        return BoundingBox(
            x=min_x,
            y=min_y,
            width=max(0, max_x - min_x),
            height=max(0, max_y - min_y),
        )

    def pad(
        self,
        padding_ratio: float = 0.15,
        max_w: int = 1920,
        max_h: int = 1080,
        *,
        ratio: float | None = None,
    ) -> BoundingBox:
        r = ratio if ratio is not None else padding_ratio
        pad_w = round(self.width * r)
        pad_h = round(self.height * r)
        x1 = max(0, min(max_w, self.x - pad_w))
        y1 = max(0, min(max_h, self.y - pad_h))
        x2 = max(x1, min(max_w, self.x2 + pad_w))
        y2 = max(y1, min(max_h, self.y2 + pad_h))
        return BoundingBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1)


class SieveConfig(BaseModel):
    """Configuration parameters for the VideoSieve pipeline (v0.2.0)."""

    downscale_width: int = Field(default=320, ge=64)
    sample_fps: float | None = Field(default=10.0, gt=0.0)
    motion_threshold: float = Field(default=0.015, gt=0.0, le=1.0)
    pre_roll_sec: float = Field(default=3.0, ge=0.0)
    pre_roll_min_score: float = Field(default=0.0025, ge=0.0, le=1.0)
    pixel_diff_threshold: int = Field(default=25, ge=1, le=255)
    min_contour_area: int = Field(default=16, ge=1)
    cooldown_frames: int = Field(default=8, ge=1)
    min_event_frames: int = Field(default=2, ge=1)
    max_event_frames: int = Field(default=600, ge=2)
    merge_gap_sec: float = Field(default=6.0, ge=0.0)
    suppress_disturbances: bool = Field(default=True)
    grid_layout: tuple[int, int] = Field(default=(2, 3))
    auto_crop_roi: bool = Field(default=True)
    roi_padding_ratio: float = Field(default=0.35, ge=0.0, le=2.0)
    min_roi_size: int = Field(default=320, ge=32)
    burn_timestamps: bool = Field(default=True)
    spatial_grid_overlay: bool = Field(default=True)
    tile_max_dim: int = Field(default=640, ge=64)
    tile_max_width: int | None = Field(default=None, ge=64)
    tokens_per_image: int = Field(default=1120, ge=1)
    camera_warmup_frames: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _sync_tile_dimensions(self) -> SieveConfig:
        rows, cols = self.grid_layout
        if rows < 1 or cols < 1:
            raise ValueError("grid_layout rows and cols must be >= 1")
        if self.tile_max_width is not None and "tile_max_dim" not in self.model_fields_set:
            object.__setattr__(self, "tile_max_dim", self.tile_max_width)
        return self


class FramePacket:
    """Container for an inspected video frame with lazy RGB and 64x64 thumbnail caching."""

    __slots__ = (
        "frame_index",
        "timestamp_sec",
        "_bgr_array",
        "_rgb_array",
        "_thumb64_cache",
        "motion_score",
        "motion_bbox",
    )

    def __init__(
        self,
        frame_index: int,
        timestamp_sec: float,
        rgb_array: np.ndarray | None = None,
        motion_score: float = 0.0,
        motion_bbox: BoundingBox | None = None,
        *,
        bgr_array: np.ndarray | None = None,
    ) -> None:
        if rgb_array is None and bgr_array is None:
            raise ValueError("Either rgb_array or bgr_array must be provided.")
        self.frame_index = int(frame_index)
        self.timestamp_sec = float(timestamp_sec)
        self._rgb_array = rgb_array
        self._bgr_array = bgr_array
        self._thumb64_cache: np.ndarray | None = None
        self.motion_score = float(motion_score)
        self.motion_bbox = motion_bbox

    @property
    def rgb_array(self) -> np.ndarray:
        if self._rgb_array is None:
            assert self._bgr_array is not None
            self._rgb_array = cv2.cvtColor(self._bgr_array, cv2.COLOR_BGR2RGB)
        return self._rgb_array

    @property
    def bgr_array(self) -> np.ndarray:
        if self._bgr_array is None:
            assert self._rgb_array is not None
            self._bgr_array = cv2.cvtColor(self._rgb_array, cv2.COLOR_RGB2BGR)
        return self._bgr_array

    @property
    def thumb64(self) -> np.ndarray:
        if self._thumb64_cache is None:
            base = self._bgr_array if self._bgr_array is not None else self._rgb_array
            assert base is not None
            self._thumb64_cache = cv2.resize(
                base, (64, 64), interpolation=cv2.INTER_AREA
            ).astype(np.float32)
        return self._thumb64_cache


class DisturbanceRecord(BaseModel):
    """Metadata record for a suppressed global camera shake or exposure flash episode."""

    episode_index: int
    event_type: str = "camera_disturbance"
    start_frame: int
    end_frame: int
    start_time_sec: float
    end_time_sec: float
    duration_sec: float
    max_motion_score: float | None = None
    mean_bbox_area_ratio: float
    localized_frames: int
    global_frames: int
    note: str = (
        "Global camera shake or exposure disturbance detected with no "
        "localized foreground trajectory; storyboard image suppressed."
    )


class TokenSavingsReport(BaseModel):
    """Summary of frame reduction and estimated token savings."""

    raw_frames_inspected: int
    sampled_frames_decoded: int = 0
    frames_dropped: int
    storyboards_emitted: int
    disturbances_suppressed: int = 0
    estimated_naive_tokens: int
    estimated_sieve_tokens: int
    compression_ratio: float
    tokens_saved_pct: float


class VisualEvent(BaseModel):
    """Emitted visual episode containing the packed storyboard and temporal metadata."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    event_id: int
    episode_index: int = 1
    event_type: str = "visual_activity"
    start_frame: int
    end_frame: int
    start_time_sec: float
    end_time_sec: float
    raw_frame_count: int
    keyframes: list[FramePacket] = Field(default_factory=list)
    roi_bbox: BoundingBox
    storyboard_pil: Image.Image
    savings: TokenSavingsReport | None = None
    keyframe_times_sec: list[float] = Field(default_factory=list)
    keyframe_gaps_sec: list[float] = Field(default_factory=list)

    @property
    def duration_sec(self) -> float:
        return round(max(0.0, self.end_time_sec - self.start_time_sec), 3)

    def to_bytes(self, format: str = "JPEG", quality: int = 85) -> bytes:
        buf = io.BytesIO()
        fmt = format.upper()
        save_kwargs: dict[str, Any] = {"format": fmt}
        if fmt in {"JPEG", "JPG", "WEBP"}:
            save_kwargs["quality"] = quality
        img = self.storyboard_pil.convert("RGB") if fmt in {"JPEG", "JPG"} else self.storyboard_pil
        img.save(buf, **save_kwargs)
        return buf.getvalue()

    def to_base64(self, format: str = "JPEG", quality: int = 85) -> str:
        raw = self.to_bytes(format=format, quality=quality)
        return base64.b64encode(raw).decode("ascii")

    def to_data_uri(self, format: str = "JPEG", quality: int = 85) -> str:
        mime_sub = "jpeg" if format.upper() in {"JPEG", "JPG"} else format.lower()
        b64 = self.to_base64(format=format, quality=quality)
        return f"data:image/{mime_sub};base64,{b64}"

    def save(self, path: str | Path, format: str | None = None, quality: int = 90) -> Path:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs: dict[str, Any] = {}
        if format is not None:
            save_kwargs["format"] = format
        if format is None or format.upper() in {"JPEG", "JPG", "WEBP"}:
            save_kwargs["quality"] = quality
        self.storyboard_pil.save(out_path, **save_kwargs)
        return out_path

    def to_summary_dict(self, image_filename: str | None = None) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "episode_index": self.episode_index,
            "event_type": self.event_type,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_time_sec": round(self.start_time_sec, 3),
            "end_time_sec": round(self.end_time_sec, 3),
            "duration_sec": self.duration_sec,
            "raw_frame_count": self.raw_frame_count,
            "buffer_frame_count": self.raw_frame_count,
            "roi_bbox": self.roi_bbox.model_dump(),
            "image_file": image_filename or f"event_{self.event_id:03d}.jpg",
            "image_size": list(self.storyboard_pil.size),
            "keyframe_times_sec": self.keyframe_times_sec,
            "keyframe_gaps_sec": self.keyframe_gaps_sec,
        }
