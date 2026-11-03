# vlm-sieve: Python Library for Vision-Language Model (VLM) Video Preprocessing (v0.2.1)

> **PyPI Package:** `pip install vlm-sieve` | **Python Import:** `from vlmsieve import VideoSieve`

**Local-First, Zero-GPU Spatial-Temporal Video Gatekeeper & Storyboard Packer for Vision-Language Models (VLMs)**

`vlm-sieve` sits in front of Vision-Language Models (Gemini, GPT-4o, Claude, Qwen-VL, Llama-Vision) to filter out dead air, merge fragmented motion bursts into coherent episodes, crop active Regions-of-Interest (ROIs) with generous context padding, and pack peak-preserving keyframes into composite spatial-temporal storyboards.

---

## What's New in v0.2.1

1. **Bit-Stream Frame Striding (`sample_fps=10.0`) + Lazy RGB Caching:** Uses `cap.grab()` at the demuxer level to skip intermediate frames without decoding pixel buffers, cutting local CPU preprocessing time by **60%+**.
2. **Episode Merging (`merge_gap_sec=6.0`) + Bridge Frames:** Combines adjacent motion bursts separated by brief pauses and retains ~2 FPS bridge frames across merged gaps so multi-second interactions emit a single unified storyboard instead of fragmenting.
3. **Micro-Motion Circular Pre-Roll Buffer (`pre_roll_sec=3.0`, `pre_roll_min_score=0.0025`):** Captures subtle early approaches (e.g., distant sidewalk footsteps) before the main foreground threshold is crossed.
4. **Hybrid Peak-Preserving Keyframe Sampler:** Prevents `<1.5s` duplicate frames while preserving transient action peaks (such as a 1-second doorbell button press).
5. **Global Camera-Disturbance Classifier (`suppress_disturbances=True`):** Automatically identifies full-frame camera shake or auto-exposure flashes, logs the timestamp in `summary.json`, and suppresses redundant empty-room storyboard images.
6. **Both-Axis Tile Clamping (`tile_max_dim=640`) & Wider Context Padding (`roi_padding_ratio=0.35`):** Clamps both `max(width, height)` per tile and preserves surrounding scene context around objects.

---

## Real-World Provider-Counted Gemini Benchmarks

Measured using provider-reported `usage_metadata.prompt_token_count` on Gemini 3.8 Flash across both visual-only and full audiovisual (`Storyboards + 32kbps MP3 Audio`) pipelines:

### 1. 5m 22s Ring Home Security Footage (`1280x720` @ 30 FPS, 9,658 frames)

| Pipeline Mode | Emitted Payload | Provider Input Tokens | Input Token Savings | Est. API Cost | Local CPU Time |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Direct Gemini Video Upload** | Full `.webm` (`322s`) | `29,403` | Baseline | `$0.02895` | N/A (`18.3s` upload/wait) |
| **`vlm-sieve v0.2.1` (Visual-Only)** | **4 Storyboards + 1 Shake Log** | **`4,823`** | **83.60% saved** | **`$0.01197` (58.7% saved)** | **`17.08s`** |
| **`vlm-sieve v0.2.1` (Visual + Audio)** | **4 Storyboards + 1.2 MB `.mp3`** | **`12,925`** | **56.04% saved** | **`$0.01835` (36.6% saved)** | **`18.34s`** |

### 2. 40s Vivint Doorbell Footage (`720x720` @ 15 FPS, 600 frames)

| Pipeline Mode | Emitted Payload | Provider Input Tokens | Input Token Savings | Est. API Cost | End-to-End Latency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Direct Gemini Video Upload** | Full `.mp4` (`40s`) | `3,659` | Baseline | `$0.00641` | `19.20s` |
| **`vlm-sieve v0.2.1` (Visual-Only)** | **1 Storyboard (`1408x1292`)** | **`1,292`** | **64.69% saved** | **`$0.00431` (32.8% saved)** | **`11.62s` (39.5% faster)** |
| **`vlm-sieve v0.2.1` (Visual + Audio)** | **1 Storyboard + 157 KB `.mp3`** | **`2,346`** | **35.88% saved** | **`$0.00536` (16.4% saved)** | **`16.04s` (16.5% faster)** |

---

## Installation

```bash
pip install -U vlm-sieve
```

---

## Quickstart (Python API)

```python
from vlmsieve import VideoSieve

sieve = VideoSieve(
    source="security_footage.mp4",
    sample_fps=10.0,
    pre_roll_sec=3.0,
    merge_gap_sec=6.0,
    tile_max_dim=640,
    suppress_disturbances=True,
)

events = sieve.process_video_to_dir("./sieve_output")
report = sieve.get_savings_report()

for ev in events:
    print(
        f"Event {ev.event_id}: {ev.start_time_sec:.2f}s - {ev.end_time_sec:.2f}s | "
        f"Keyframes: {ev.keyframe_times_sec} | Size: {ev.storyboard_pil.size}"
    )
```

---

## Command-Line Interface (CLI)

```bash
vlmsieve ./input_video.mp4 -o ./sieve_output --pre-roll-sec 3.5 --merge-gap-sec 6.0
python -m vlmsieve ./input_video.mp4 -o ./sieve_output
```

---

## License

MIT License

---

## New in v0.3.0: Local VLM (`Qwen2.5-VL`) 28px Patch Alignment & Target Profiles

`v0.3.0` adds one-line target profiles (`target_vlm="gemini" | "gpt-4o" | "qwen-vl" | "local-7b"`), **28px NaViT patch alignment**, **adaptive grids** (`1x1`..`2x3` to eliminate empty `"NO FRAME"` patches), **outer-border `A..D` / `1..4` axis ticks**, and **FP16 KV-cache VRAM telemetry**:

```python
from vlmsieve import SieveConfig, VideoSieve

# Optimize for Qwen2.5-VL (448px tiles, 28px aligned, 0px gutter, clean outer-border axis ticks)
sieve = VideoSieve("security_footage.mp4", config=SieveConfig(target_vlm="qwen-vl"))
events = sieve.process_video()

report = sieve.get_savings_report()
print(f"Qwen-VL Patch Tokens: {report.qwen_vl_patch_tokens:,}")
print(f"Qwen2.5-VL-7B KV Cache: {report.qwen7b_kv_cache_mb} MB (Saved {report.qwen7b_kv_cache_saved_mb:,.1f} MB)")
```

| Video Dataset | Profile | Canvas (`WxH`) | `28px` Aligned | Gemini Tokens | Qwen2.5-VL Patch Tokens | Reduction vs `v0.2.1` | Qwen2.5-VL-7B KV Cache |
|---|---|---|---|---|---|---|---|
| **5m 22s Ring Video** | `v0.2.1` Baseline | `1936x732` | No | `4,823` | `9,776` | Baseline | `534.62 MB` |
| **5m 22s Ring Video** | `v0.3.0 qwen-vl` | `1344x504` | **Yes** | `4,485` | **`4,608`** | **`-52.9%`** | **`252.00 MB`** |
| **5m 22s Ring Video** | `v0.3.0 local-7b` | `1008x392` | **Yes** | **`4,426`** | **`2,592`** | **`-73.5%`** | **`141.75 MB`** |
| **40s Vivint Video** | `v0.2.1` Baseline | `1408x1292` | No | `1,292` | `2,397` | Baseline | `131.09 MB` |
| **40s Vivint Video** | `v0.3.0 qwen-vl` | `1008x896` | **Yes** | **`1,205`** | **`1,152`** | **`-51.9%`** | **`63.00 MB`** |
| **40s Vivint Video** | `v0.3.0 local-7b` | `756x672` | **Yes** | **`1,205`** | **`648`** | **`-73.0%`** | **`35.44 MB`** |

