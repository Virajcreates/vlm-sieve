"""Stage 2 Temporal-Peak Keyframe Sampler for representative frame selection."""

from __future__ import annotations

import cv2
import numpy as np

from vlmsieve.models import FramePacket


def _visual_diversity_score(
    candidate_rgb: np.ndarray, reference_rgb: np.ndarray
) -> float:
    """Compute normalized visual diversity between two RGB frames.

    Downscales both frames to 64x64 and calculates the normalized mean
    absolute pixel difference.

    Args:
        candidate_rgb: Candidate frame RGB array.
        reference_rgb: Reference frame RGB array to compare against.

    Returns:
        Diversity score in [0.0, 1.0].
    """
    if candidate_rgb.shape[:2] != (64, 64):
        cand = cv2.resize(candidate_rgb, (64, 64), interpolation=cv2.INTER_AREA)
    else:
        cand = candidate_rgb

    if reference_rgb.shape[:2] != (64, 64):
        ref = cv2.resize(reference_rgb, (64, 64), interpolation=cv2.INTER_AREA)
    else:
        ref = reference_rgb

    cand_f = cand.astype(np.float32)
    ref_f = ref.astype(np.float32)
    diff = np.abs(cand_f - ref_f)
    return float(np.mean(diff)) / 255.0


def select_keyframes(
    packets: list[FramePacket], target_count: int = 6
) -> list[FramePacket]:
    """Select the most informative keyframes across an event segment.

    Preserves the onset (first frame) and conclusion (last frame) of the action,
    and samples representative frames across contiguous temporal buckets using a
    weighted combination of motion intensity (60%) and visual diversity (40%).

    Args:
        packets: Chronological list of FramePacket instances in an event.
        target_count: Desired number of keyframes to select. Must be >= 1.

    Returns:
        List of selected FramePacket instances strictly ordered by frame_index.
    """
    if target_count <= 0:
        raise ValueError("target_count must be >= 1")

    if not packets:
        return []

    sorted_packets = sorted(packets, key=lambda p: p.frame_index)

    if len(sorted_packets) <= target_count:
        return list(sorted_packets)

    if target_count == 1:
        return [max(sorted_packets, key=lambda p: p.motion_score)]

    if target_count == 2:
        return [sorted_packets[0], sorted_packets[-1]]

    # target_count >= 3: Pin onset and termination frames
    selected: list[FramePacket] = [sorted_packets[0]]
    prev_selected: FramePacket = sorted_packets[0]

    middle = sorted_packets[1:-1]
    num_buckets = target_count - 2
    splits = np.array_split(np.arange(len(middle)), num_buckets)

    for bucket_indices in splits:
        if len(bucket_indices) == 0:
            continue

        bucket = [middle[i] for i in bucket_indices]
        max_m = max(p.motion_score for p in bucket)

        def _candidate_score(
            candidate: FramePacket,
            m_max: float = max_m,
            prev_p: FramePacket = prev_selected,
        ) -> float:
            norm_m = (candidate.motion_score / m_max) if m_max > 1e-9 else 0.0
            div_score = _visual_diversity_score(candidate.rgb_array, prev_p.rgb_array)
            return 0.60 * norm_m + 0.40 * div_score

        best_p = max(bucket, key=_candidate_score)
        selected.append(best_p)
        prev_selected = best_p

    selected.append(sorted_packets[-1])
    return selected
