"""Stage 4: Spatial-Temporal Storyboard Packer with Both-Axis Clamping & Grid Overlays."""
from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from vlmsieve.models import BoundingBox, FramePacket, SieveConfig
from vlmsieve.roi import crop_packet_to_roi


class StoryboardPacker:
    """Packs selected keyframes into a grid canvas with timestamp badges and A1..D4 grid cells."""

    def __init__(self, config: SieveConfig | None = None, **kwargs) -> None:
        self.config = config if config is not None else SieveConfig(**kwargs)

    def pack(self, keyframes: list[FramePacket], roi: BoundingBox) -> Image.Image:
        if not keyframes:
            raise ValueError("keyframes list cannot be empty.")

        rows, cols = self.config.grid_layout
        sample_crop = crop_packet_to_roi(keyframes[0], roi)
        crop_h, crop_w = sample_crop.shape[:2]

        tile_w, tile_h = max(1, crop_w), max(1, crop_h)
        max_side = max(tile_w, tile_h)
        if max_side > self.config.tile_max_dim:
            scale = self.config.tile_max_dim / float(max_side)
            tile_w = max(1, round(tile_w * scale))
            tile_h = max(1, round(tile_h * scale))

        if self.config.tile_max_width is not None and tile_w > self.config.tile_max_width:
            scale = self.config.tile_max_width / float(tile_w)
            tile_w = self.config.tile_max_width
            tile_h = max(1, round(tile_h * scale))

        max_slots = rows * cols
        tiles: list[Image.Image] = []

        badge_font = ImageFont.load_default(size=max(14, tile_w // 22))
        grid_font = ImageFont.load_default(size=max(11, tile_w // 30))
        watermark_font = ImageFont.load_default(size=max(14, tile_w // 20))

        base_ts = keyframes[0].timestamp_sec

        for idx, pkt in enumerate(keyframes[:max_slots]):
            cropped = crop_packet_to_roi(pkt, roi)
            im = Image.fromarray(cropped).convert("RGBA")
            if im.size != (tile_w, tile_h):
                im = im.resize((tile_w, tile_h), resample=Image.Resampling.LANCZOS)

            if self.config.spatial_grid_overlay:
                overlay = Image.new("RGBA", (tile_w, tile_h), (0, 0, 0, 0))
                draw = ImageDraw.Draw(overlay)
                for cc in range(1, 4):
                    lx = round(cc * (tile_w / 4.0))
                    draw.line([(lx, 0), (lx, tile_h)], fill=(56, 189, 248, 110), width=1)
                for rr in range(1, 4):
                    ly = round(rr * (tile_h / 4.0))
                    draw.line([(0, ly), (tile_w, ly)], fill=(56, 189, 248, 110), width=1)

                row_labels = ["A", "B", "C", "D"]
                col_labels = ["1", "2", "3", "4"]
                for ri in range(4):
                    y_end = round((ri + 1) * (tile_h / 4.0))
                    for ci in range(4):
                        x_end = round((ci + 1) * (tile_w / 4.0))
                        cell_label = f"{row_labels[ri]}{col_labels[ci]}"
                        bb = grid_font.getbbox(cell_label)
                        tw = bb[2] - bb[0]
                        th = bb[3] - bb[1]
                        pgx = max(2, tile_w // 150)
                        pgy = max(1, tile_h // 150)
                        pw = tw + 2 * pgx
                        ph = th + 2 * pgy
                        px2 = x_end - 2
                        py2 = y_end - 2
                        px1 = px2 - pw
                        py1 = py2 - ph
                        draw.rounded_rectangle(
                            [px1, py1, px2, py2], radius=2, fill=(18, 20, 23, 160)
                        )
                        draw.text(
                            (px1 + pgx - bb[0], py1 + pgy - bb[1]),
                            cell_label,
                            font=grid_font,
                            fill=(220, 230, 242, 230),
                        )
                im = Image.alpha_composite(im, overlay)

            if self.config.burn_timestamps:
                overlay = Image.new("RGBA", (tile_w, tile_h), (0, 0, 0, 0))
                draw = ImageDraw.Draw(overlay)
                delta = max(0.0, pkt.timestamp_sec - base_ts)
                badge_text = f"#{idx + 1} | +{delta:.2f}s"
                bb = badge_font.getbbox(badge_text)
                tw = bb[2] - bb[0]
                th = bb[3] - bb[1]
                px = max(6, tile_w // 50)
                py = max(4, tile_h // 50)
                x1, y1 = 6, 6
                x2 = x1 + tw + 2 * px
                y2 = y1 + th + 2 * py
                draw.rounded_rectangle(
                    [x1, y1, x2, y2],
                    radius=3,
                    fill=(18, 20, 23, 225),
                    outline=(56, 189, 248, 255),
                    width=1,
                )
                draw.text(
                    (x1 + px - bb[0], y1 + py - bb[1]),
                    badge_text,
                    font=badge_font,
                    fill=(255, 255, 255, 255),
                )
                im = Image.alpha_composite(im, overlay)

            tiles.append(im.convert("RGB"))

        gutter = 4
        canvas_w = cols * tile_w + (cols + 1) * gutter
        canvas_h = rows * tile_h + (rows + 1) * gutter
        canvas = Image.new("RGB", (canvas_w, canvas_h), (18, 20, 23))

        for slot in range(max_slots):
            r = slot // cols
            c = slot % cols
            x = gutter + c * (tile_w + gutter)
            y = gutter + r * (tile_h + gutter)
            if slot < len(tiles):
                canvas.paste(tiles[slot], (x, y))
            else:
                empty = Image.new("RGB", (tile_w, tile_h), (28, 31, 38))
                draw = ImageDraw.Draw(empty)
                label = "NO FRAME"
                bb = watermark_font.getbbox(label)
                tw = bb[2] - bb[0]
                th = bb[3] - bb[1]
                draw.text(
                    ((tile_w - tw) // 2 - bb[0], (tile_h - th) // 2 - bb[1]),
                    label,
                    font=watermark_font,
                    fill=(110, 118, 129),
                )
                canvas.paste(empty, (x, y))

        return canvas
# === VLM-SIEVE v0.3.0 UPGRADE ===
import inspect as _inspect
from typing import Any as _Any
import cv2 as _cv2
import numpy as _np
from PIL import Image as _Image, ImageDraw as _ImageDraw, ImageFont as _ImageFont
from vlmsieve.models import BoundingBox as _BoundingBox, SieveConfig as _SieveConfig


def choose_adaptive_grid(
    num_keyframes: int, max_grid: tuple[int, int] = (2, 3)
) -> tuple[int, int]:
    max_slots = max_grid[0] * max_grid[1]
    n = max(1, min(num_keyframes, max_slots))
    if max_grid == (1, 3):
        return (1, min(n, 3))
    if n == 1:
        return (1, 1)
    if n == 2:
        return (1, 2)
    if n == 3:
        return (1, 3)
    if n == 4:
        return (2, 2)
    return max_grid


def snap_to_multiple(val: int, multiple: int | None) -> int:
    if not multiple or multiple <= 1:
        return max(1, val)
    return max(multiple, round(val / float(multiple)) * multiple)


def _extract_array_from_packet(pkt: _Any) -> _np.ndarray:
    if isinstance(pkt, _np.ndarray):
        return pkt
    for attr in ("rgb_array", "rgb_frame", "frame", "image"):
        val = getattr(pkt, attr, None)
        if isinstance(val, _np.ndarray):
            return val
    for attr in ("bgr_array", "bgr_frame"):
        val = getattr(pkt, attr, None)
        if isinstance(val, _np.ndarray):
            return _cv2.cvtColor(val, _cv2.COLOR_BGR2RGB) if val.ndim == 3 else val
    for v in getattr(pkt, "__dict__", {}).values():
        if isinstance(v, _np.ndarray) and v.ndim >= 2:
            return v
    raise TypeError(f"Unsupported frame packet: {type(pkt)}")


def _extract_rgb_and_ts(pkt: _Any, idx: int, roi: _BoundingBox | None = None) -> tuple[_np.ndarray, float]:
    arr = _extract_array_from_packet(pkt)
    ts = float(getattr(pkt, "timestamp_sec", idx * 0.5)) if not isinstance(pkt, _np.ndarray) else float(idx) * 0.5
    if roi is not None:
        h, w = arr.shape[:2]
        x1 = max(0, min(w - 1, int(roi.x)))
        y1 = max(0, min(h - 1, int(roi.y)))
        x2 = max(x1 + 1, min(w, int(roi.x + roi.width)))
        y2 = max(y1 + 1, min(h, int(roi.y + roi.height)))
        arr = arr[y1:y2, x1:x2]
    return arr, ts


_OrigStoryboardPacker = StoryboardPacker


class StoryboardPacker(_OrigStoryboardPacker):  # type: ignore[no-redef]
    def __init__(self, config: _SieveConfig | tuple[int, int] | None = None, **kwargs: _Any) -> None:
        if isinstance(config, _SieveConfig):
            cfg = config.model_copy(update=kwargs) if kwargs else config
        elif isinstance(config, tuple):
            kwargs["grid_layout"] = config
            cfg = _SieveConfig(**kwargs)
        else:
            cfg = _SieveConfig(**kwargs)

        self.config = cfg
        self.grid_layout = cfg.grid_layout
        self.tile_max_dim = cfg.tile_max_dim
        self.gutter_px = cfg.gutter_px
        self.burn_timestamps = cfg.burn_timestamps
        self.spatial_grid_overlay = cfg.spatial_grid_overlay
        self.adaptive_grid = cfg.adaptive_grid
        self.patch_align = cfg.patch_align
        self.spatial_grid_style = cfg.spatial_grid_style

        sig = _inspect.signature(_OrigStoryboardPacker.__init__)
        super_kwargs = {}
        for pname in list(sig.parameters.keys())[1:]:
            if pname == "config":
                super_kwargs["config"] = cfg
            elif hasattr(cfg, pname):
                super_kwargs[pname] = getattr(cfg, pname)
        try:
            super().__init__(**super_kwargs)
        except Exception:
            pass

    def pack_with_tiles(
        self, keyframes: list[_Any], roi: _BoundingBox | None = None
    ) -> tuple[_Image.Image, list[_Image.Image]]:
        if not keyframes:
            raise ValueError("keyframes list cannot be empty")

        cfg = self.config
        grid_layout = getattr(self, "grid_layout", cfg.grid_layout)
        tile_max_dim = getattr(self, "tile_max_dim", cfg.tile_max_dim)
        gutter_px = getattr(self, "gutter_px", cfg.gutter_px)
        burn_timestamps = getattr(self, "burn_timestamps", cfg.burn_timestamps)
        spatial_overlay = getattr(self, "spatial_grid_overlay", cfg.spatial_grid_overlay)
        adaptive = getattr(self, "adaptive_grid", False)
        patch_align = getattr(self, "patch_align", None)
        grid_style = getattr(self, "spatial_grid_style", "cells")
        if not spatial_overlay:
            grid_style = "none"

        rows, cols = (
            choose_adaptive_grid(len(keyframes), grid_layout)
            if adaptive
            else grid_layout
        )
        max_slots = rows * cols
        used_kfs = keyframes[:max_slots]

        sample_crop, base_ts = _extract_rgb_and_ts(used_kfs[0], 0, roi)
        crop_h, crop_w = sample_crop.shape[:2]

        tile_w, tile_h = max(1, int(crop_w)), max(1, int(crop_h))
        max_side = max(tile_w, tile_h)
        if max_side > tile_max_dim:
            scale = tile_max_dim / float(max_side)
            tile_w = max(1, round(tile_w * scale))
            tile_h = max(1, round(tile_h * scale))

        if patch_align:
            tile_w = snap_to_multiple(tile_w, patch_align)
            tile_h = snap_to_multiple(tile_h, patch_align)

        badge_font = _ImageFont.load_default(size=max(13, tile_w // 22))
        grid_font = _ImageFont.load_default(size=max(11, tile_w // 30))
        watermark_font = _ImageFont.load_default(size=max(14, tile_w // 20))

        rendered_tiles: list[_Image.Image] = []
        discrete_clean_crops: list[_Image.Image] = []

        row_labels = ["A", "B", "C", "D"]
        col_labels = ["1", "2", "3", "4"]

        for idx, pkt in enumerate(used_kfs):
            cropped, pkt_ts = _extract_rgb_and_ts(pkt, idx, roi)
            clean_pil = _Image.fromarray(cropped).convert("RGB")
            if clean_pil.size != (tile_w, tile_h):
                clean_pil = clean_pil.resize(
                    (tile_w, tile_h), resample=_Image.Resampling.LANCZOS
                )
            discrete_clean_crops.append(clean_pil)

            im = clean_pil.convert("RGBA")

            if grid_style in {"cells", "axes"}:
                overlay = _Image.new("RGBA", (tile_w, tile_h), (0, 0, 0, 0))
                draw = _ImageDraw.Draw(overlay)
                for cc in range(1, 4):
                    lx = round(cc * (tile_w / 4.0))
                    draw.line([(lx, 0), (lx, tile_h)], fill=(56, 189, 248, 95), width=1)
                for rr in range(1, 4):
                    ly = round(rr * (tile_h / 4.0))
                    draw.line([(0, ly), (tile_w, ly)], fill=(56, 189, 248, 95), width=1)

                if grid_style == "cells":
                    for ri in range(4):
                        y_end = round((ri + 1) * (tile_h / 4.0))
                        for ci in range(4):
                            x_end = round((ci + 1) * (tile_w / 4.0))
                            cell_label = f"{row_labels[ri]}{col_labels[ci]}"
                            bb = grid_font.getbbox(cell_label)
                            tw, th = bb[2] - bb[0], bb[3] - bb[1]
                            pgx, pgy = max(2, tile_w // 150), max(1, tile_h // 150)
                            pw, ph = tw + 2 * pgx, th + 2 * pgy
                            px2, py2 = x_end - 2, y_end - 2
                            px1, py1 = px2 - pw, py2 - ph
                            draw.rounded_rectangle(
                                [px1, py1, px2, py2], radius=2, fill=(18, 20, 23, 160)
                            )
                            draw.text(
                                (px1 + pgx - bb[0], py1 + pgy - bb[1]),
                                cell_label,
                                font=grid_font,
                                fill=(220, 230, 242, 230),
                            )
                elif grid_style == "axes":
                    for ci in range(4):
                        cx = round((ci + 0.5) * (tile_w / 4.0))
                        lbl = col_labels[ci]
                        bb = grid_font.getbbox(lbl)
                        tw, th = bb[2] - bb[0], bb[3] - bb[1]
                        px1, py1 = cx - tw // 2 - 3, tile_h - th - 6
                        draw.rounded_rectangle(
                            [px1, py1, px1 + tw + 6, tile_h - 2],
                            radius=2,
                            fill=(18, 20, 23, 185),
                        )
                        draw.text(
                            (px1 + 3 - bb[0], py1 + 2 - bb[1]),
                            lbl,
                            font=grid_font,
                            fill=(56, 189, 248, 245),
                        )
                    for ri in range(4):
                        cy = round((ri + 0.5) * (tile_h / 4.0))
                        lbl = row_labels[ri]
                        bb = grid_font.getbbox(lbl)
                        tw, th = bb[2] - bb[0], bb[3] - bb[1]
                        px1, py1 = tile_w - tw - 8, cy - th // 2 - 2
                        draw.rounded_rectangle(
                            [px1, py1, tile_w - 2, py1 + th + 4],
                            radius=2,
                            fill=(18, 20, 23, 185),
                        )
                        draw.text(
                            (px1 + 3 - bb[0], py1 + 2 - bb[1]),
                            lbl,
                            font=grid_font,
                            fill=(56, 189, 248, 245),
                        )

                im = _Image.alpha_composite(im, overlay)

            if burn_timestamps:
                overlay = _Image.new("RGBA", (tile_w, tile_h), (0, 0, 0, 0))
                draw = _ImageDraw.Draw(overlay)
                delta = max(0.0, pkt_ts - base_ts)
                badge_text = f"#{idx + 1} | +{delta:.2f}s"
                bb = badge_font.getbbox(badge_text)
                tw, th = bb[2] - bb[0], bb[3] - bb[1]
                px, py = max(5, tile_w // 55), max(3, tile_h // 55)
                x1, y1 = 6, 6
                x2, y2 = x1 + tw + 2 * px, y1 + th + 2 * py
                draw.rounded_rectangle(
                    [x1, y1, x2, y2],
                    radius=3,
                    fill=(18, 20, 23, 225),
                    outline=(56, 189, 248, 255),
                    width=1,
                )
                draw.text(
                    (x1 + px - bb[0], y1 + py - bb[1]),
                    badge_text,
                    font=badge_font,
                    fill=(255, 255, 255, 255),
                )
                im = _Image.alpha_composite(im, overlay)

            rendered_tiles.append(im.convert("RGB"))

        canvas_w = (
            cols * tile_w + (cols + 1) * gutter_px if gutter_px > 0 else cols * tile_w
        )
        canvas_h = (
            rows * tile_h + (rows + 1) * gutter_px if gutter_px > 0 else rows * tile_h
        )
        canvas = _Image.new("RGB", (canvas_w, canvas_h), (18, 20, 23))

        for slot in range(max_slots):
            r = slot // cols
            c = slot % cols
            x = (gutter_px + c * (tile_w + gutter_px)) if gutter_px > 0 else (c * tile_w)
            y = (gutter_px + r * (tile_h + gutter_px)) if gutter_px > 0 else (r * tile_h)
            if slot < len(rendered_tiles):
                canvas.paste(rendered_tiles[slot], (x, y))
            else:
                empty = _Image.new("RGB", (tile_w, tile_h), (28, 31, 38))
                draw = _ImageDraw.Draw(empty)
                label = "NO FRAME"
                bb = watermark_font.getbbox(label)
                tw, th = bb[2] - bb[0], bb[3] - bb[1]
                draw.text(
                    ((tile_w - tw) // 2 - bb[0], (tile_h - th) // 2 - bb[1]),
                    label,
                    font=watermark_font,
                    fill=(110, 118, 129),
                )
                canvas.paste(empty, (x, y))

        if patch_align and (
            canvas.width % patch_align != 0 or canvas.height % patch_align != 0
        ):
            aw = snap_to_multiple(canvas.width, patch_align)
            ah = snap_to_multiple(canvas.height, patch_align)
            canvas = canvas.resize((aw, ah), resample=_Image.Resampling.LANCZOS)

        canvas._vlmsieve_grid_used = (rows, cols)  # type: ignore[attr-defined]
        canvas._vlmsieve_cropped_tiles = discrete_clean_crops  # type: ignore[attr-defined]
        return canvas, discrete_clean_crops

    def pack(self, keyframes: list[_Any], roi: _BoundingBox | None = None, **kwargs: _Any) -> _Image.Image:
        board, _ = self.pack_with_tiles(keyframes, roi)
        return board


def pack_storyboard(
    keyframes: list[_Any],
    roi: _BoundingBox | None = None,
    config: _SieveConfig | None = None,
) -> _Image.Image:
    return StoryboardPacker(config).pack(keyframes, roi=roi)
