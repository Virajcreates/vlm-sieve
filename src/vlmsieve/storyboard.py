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
