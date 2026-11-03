from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
from vlmsieve import SieveConfig, VideoSieve
from vlmsieve.models import BoundingBox, FramePacket
from vlmsieve.packer import StoryboardPacker, choose_adaptive_grid, snap_to_multiple


def _make_packets(count: int, w: int = 640, h: int = 480) -> list[FramePacket]:
    fields = getattr(FramePacket, "model_fields", {})
    img_key = next(
        (k for k in ("rgb_array", "bgr_array", "frame", "bgr_frame", "rgb_frame", "image") if k in fields),
        "rgb_array",
    )
    packets: list[FramePacket] = []
    for i in range(count):
        img = np.full((h, w, 3), 40 + i * 20, dtype=np.uint8)
        packets.append(
            FramePacket(
                frame_index=i * 3,
                timestamp_sec=float(i) * 0.5,
                motion_score=0.05,
                motion_bbox=BoundingBox(x=100, y=80, width=220, height=260),
                **{img_key: img},
            )
        )
    return packets


def test_target_vlm_profiles() -> None:
    c_qwen = SieveConfig(target_vlm="qwen-vl")
    assert c_qwen.tile_max_dim == 448 and c_qwen.patch_align == 28 and c_qwen.gutter_px == 0
    c_loc = SieveConfig(target_vlm="local-7b")
    assert c_loc.tile_max_dim == 336 and c_loc.patch_align == 28 and c_loc.spatial_grid_style == "axes"


def test_adaptive_grid_and_28px_patch_alignment() -> None:
    assert choose_adaptive_grid(2, (2, 3)) == (1, 2)
    assert choose_adaptive_grid(4, (2, 3)) == (2, 2)
    assert snap_to_multiple(190, 28) == 196
    packer = StoryboardPacker(SieveConfig(target_vlm="qwen-vl"))
    roi = BoundingBox(x=50, y=40, width=310, height=270)
    board, crops = packer.pack_with_tiles(_make_packets(3), roi)
    assert board.width % 28 == 0 and board.height % 28 == 0 and len(crops) == 3


def test_end_to_end_sieve_v030(tmp_path: Path) -> None:
    vid_path = tmp_path / "synthetic_motion.mp4"
    writer = cv2.VideoWriter(str(vid_path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (640, 480))
    for i in range(75):
        frame = np.full((480, 640, 3), 25, dtype=np.uint8)
        if 15 <= i <= 50:
            frame[120:360, 80 + (i - 15) * 8 : 200 + (i - 15) * 8] = (40, 220, 240)
        writer.write(frame)
    writer.release()

    sieve = VideoSieve(vid_path, config=SieveConfig(target_vlm="local-7b"))
    events = sieve.process_video()
    assert len(events) >= 1
    assert events[0].storyboard.width % 28 == 0
    assert events[0].storyboard.height % 28 == 0
    assert len(events[0].get_cropped_tiles()) >= 1
    rep = sieve.get_savings_report()
    assert rep.target_vlm == "local-7b" and rep.qwen_vl_patch_tokens > 0
