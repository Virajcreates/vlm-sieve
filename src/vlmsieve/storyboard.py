"""Stage 4 Structured Storyboard Packer for multi-frame grid composites."""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from vlmsieve.models import BoundingBox, FramePacket, SieveConfig
from vlmsieve.roi import crop_packet_to_roi


class StoryboardPacker:
    """Consolidates keyframes into structured, temporal storyboard image grids."""

    GUTTER_PX: int = 4
    CANVAS_BG_RGB: tuple[int, int, int] = (18, 20, 23)
    EMPTY_TILE_RGB: tuple[int, int, int] = (28, 31, 38)

    def pack(
        self,
        keyframes: list[FramePacket],
        roi: BoundingBox,
        config: SieveConfig | None = None,
    ) -> Image.Image:
        """Pack keyframe ROI crops into a structured multi-panel grid canvas.

        Args:
            keyframes: Chronological list of FramePacket instances to pack.
            roi: Global event BoundingBox defining the spatial crop region.
            config: SieveConfig runtime hyperparameters.

        Returns:
            A PIL.Image in RGB mode containing the assembled storyboard grid.
        """
        if not keyframes:
            raise ValueError("keyframes cannot be empty")

        cfg = config or SieveConfig()
        rows, cols = cfg.grid_layout
        if rows <= 0 or cols <= 0:
            raise ValueError("grid_layout dimensions must be >= 1")

        # Determine target tile dimensions from the base ROI crop
        sample_crop = crop_packet_to_roi(keyframes[0], roi)
        crop_h, crop_w = sample_crop.shape[:2]

        if crop_w > cfg.tile_max_width:
            scale = cfg.tile_max_width / float(crop_w)
            tile_w = cfg.tile_max_width
            tile_h = max(1, round(crop_h * scale))
        else:
            tile_w = max(1, crop_w)
            tile_h = max(1, crop_h)

        # Process each keyframe into a styled tile
        max_slots = rows * cols
        processed_tiles: list[Image.Image] = []
        badge_font = ImageFont.load_default(size=max(14, tile_w // 22))
        grid_font = ImageFont.load_default(size=max(11, tile_w // 30))
        watermark_font = ImageFont.load_default(size=max(14, tile_w // 20))

        for idx, packet in enumerate(keyframes[:max_slots]):
            cropped_rgb = crop_packet_to_roi(packet, roi)
            tile_img = Image.fromarray(cropped_rgb).convert("RGBA")

            if tile_img.size != (tile_w, tile_h):
                tile_img = tile_img.resize(
                    (tile_w, tile_h), resample=Image.Resampling.LANCZOS
                )

            # Optional 4x4 spatial grid overlay
            if cfg.spatial_grid_overlay:
                grid_overlay = Image.new("RGBA", (tile_w, tile_h), (0, 0, 0, 0))
                draw_grid = ImageDraw.Draw(grid_overlay)

                # Draw 3 vertical and 3 horizontal dividing lines
                for c in range(1, 4):
                    lx = round(c * (tile_w / 4.0))
                    draw_grid.line(
                        [(lx, 0), (lx, tile_h)], fill=(56, 189, 248, 110), width=1
                    )

                for r in range(1, 4):
                    ly = round(r * (tile_h / 4.0))
                    draw_grid.line(
                        [(0, ly), (tile_w, ly)], fill=(56, 189, 248, 110), width=1
                    )

                # Draw coordinate labels in bottom-right corner of each of the 16 cells
                row_names = ["A", "B", "C", "D"]
                col_names = ["1", "2", "3", "4"]

                for r_idx in range(4):
                    cell_y_end = round((r_idx + 1) * (tile_h / 4.0))
                    for c_idx in range(4):
                        cell_x_end = round((c_idx + 1) * (tile_w / 4.0))
                        cell_label = f"{row_names[r_idx]}{col_names[c_idx]}"

                        bbox = grid_font.getbbox(cell_label)
                        tw = bbox[2] - bbox[0]
                        th = bbox[3] - bbox[1]

                        pad_gx = max(2, tile_w // 150)
                        pad_gy = max(1, tile_h // 150)
                        pill_w = tw + 2 * pad_gx
                        pill_h = th + 2 * pad_gy
                        px2 = cell_x_end - 2
                        py2 = cell_y_end - 2
                        px1 = px2 - pill_w
                        py1 = py2 - pill_h

                        draw_grid.rounded_rectangle(
                            [px1, py1, px2, py2],
                            radius=2,
                            fill=(18, 20, 23, 160),
                        )
                        draw_grid.text(
                            (px1 + pad_gx - bbox[0], py1 + pad_gy - bbox[1]),
                            cell_label,
                            font=grid_font,
                            fill=(220, 230, 242, 230),
                        )

                tile_img = Image.alpha_composite(tile_img, grid_overlay)

            # Chronological timestamp badge overlay
            if cfg.burn_timestamps:
                badge_overlay = Image.new("RGBA", (tile_w, tile_h), (0, 0, 0, 0))
                draw_badge = ImageDraw.Draw(badge_overlay)

                delta_sec = max(0.0, packet.timestamp_sec - keyframes[0].timestamp_sec)
                label = f"#{idx + 1} | +{delta_sec:.2f}s"

                bbox = badge_font.getbbox(label)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]

                pad_x = max(6, tile_w // 50)
                pad_y = max(4, tile_h // 50)
                pill_x1 = 6
                pill_y1 = 6
                pill_x2 = pill_x1 + tw + 2 * pad_x
                pill_y2 = pill_y1 + th + 2 * pad_y

                draw_badge.rounded_rectangle(
                    [pill_x1, pill_y1, pill_x2, pill_y2],
                    radius=3,
                    fill=(18, 20, 23, 225),
                    outline=(56, 189, 248, 255),
                    width=1,
                )
                draw_badge.text(
                    (pill_x1 + pad_x - bbox[0], pill_y1 + pad_y - bbox[1]),
                    label,
                    font=badge_font,
                    fill=(255, 255, 255, 255),
                )

                tile_img = Image.alpha_composite(tile_img, badge_overlay)

            processed_tiles.append(tile_img.convert("RGB"))

        # Compute canvas dimensions including outer borders and inter-tile gutters
        canvas_w = cols * tile_w + (cols + 1) * self.GUTTER_PX
        canvas_h = rows * tile_h + (rows + 1) * self.GUTTER_PX
        canvas = Image.new("RGB", (canvas_w, canvas_h), self.CANVAS_BG_RGB)

        # Assemble grid slots
        for slot_idx in range(max_slots):
            r = slot_idx // cols
            c = slot_idx % cols
            x = self.GUTTER_PX + c * (tile_w + self.GUTTER_PX)
            y = self.GUTTER_PX + r * (tile_h + self.GUTTER_PX)

            if slot_idx < len(processed_tiles):
                canvas.paste(processed_tiles[slot_idx], (x, y))
            else:
                # Placeholder slot for events with fewer keyframes than grid cells
                empty_tile = Image.new("RGB", (tile_w, tile_h), self.EMPTY_TILE_RGB)
                draw_empty = ImageDraw.Draw(empty_tile)
                watermark = "NO FRAME"
                bbox = watermark_font.getbbox(watermark)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
                tx = (tile_w - tw) // 2 - bbox[0]
                ty = (tile_h - th) // 2 - bbox[1]
                draw_empty.text(
                    (tx, ty), watermark, font=watermark_font, fill=(110, 118, 129)
                )
                canvas.paste(empty_tile, (x, y))

        return canvas
