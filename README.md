# vlm-sieve

[![PyPI version](https://img.shields.io/pypi/v/vlm-sieve.svg)](https://pypi.org/project/vlm-sieve/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Zero-GPU CPU Engine](https://img.shields.io/badge/CPU%20Engine-%3C2ms%2Fframe-purple.svg)](#performance--benchmarks)

> **Event-Driven Video Frame Gatekeeper, Active ROI Cropper & Temporal Storyboard Packer for Vision-Language Models.**

---

## Overview

Sending raw 30fps or 60fps video directly to frontier Vision-Language Models (such as **Google Gemini**, **OpenAI GPT-4o**, or **Anthropic Claude 3.5**) quickly becomes cost-prohibitive, introduces severe latency, and floods context windows with static background tokens.

`vlm-sieve` (`vlmsieve`) is an ultra-fast, CPU-only intelligent pre-filter that monitors live webcams, RTSP security feeds, or recorded video files. It discards redundant or idle frames, tracks active motion regions, and packs salient temporal moments into structured, timestamped, and spatially grounded **storyboard image grids**—reducing VLM token consumption by **>95%** while boosting temporal and spatial comprehension.

---

## Visual Showcase

![vlm-sieve Structured Storyboard Demo](https://raw.githubusercontent.com/Virajcreates/vlm-sieve/main/assets/demo_storyboard.jpg)

In the example above, `vlm-sieve` monitored a 150-frame ($5.0\text{s}$) simulated surveillance feed. Instead of transmitting 150 separate images ($168,000$ naive tokens), `vlm-sieve` emitted a **single $2 \times 3$ storyboard** ($1,120$ tokens)—achieving **99.33% token savings (150x compression)** while providing:
- **Chronological Indexing:** Each tile contains a high-contrast badge (`#1 | +0.00s` to `#6 | +2.53s`) specifying exact relative offsets.
- **$4 \times 4$ Spatial Coordinate Grounding:** Tiles feature cell coordinate labels (`A1`–`D4`) allowing VLMs to reason about exact spatial trajectories (e.g., *"vehicle entered at B1, traversed C2-C3, and parked at D4"*).
- **Active ROI Zooming:** Automatically crops out uninformative backgrounds without losing context.

---

## The 4-Stage Architecture Funnel

```
Raw Video Stream (30/60 FPS)
 │
 ├──► [Stage 1: Downscaled CPU Motion Gatekeeper]
 │     • Sub-2ms frame evaluation on CPU via 320p grayscale differencing
 │     • Rejects camera sensor noise, exposure flickers, and static intervals
 │     • Automated webcam sensor warmup & 1-frame settling tail capture
 │
 ├──► [Stage 2: Temporal-Peak Keyframe Sampler]
 │     • Pins event onset (first motion) and termination (settled action)
 │     • Stratified bucket sampling combining motion energy (60%) + visual diversity (40%)
 │
 ├──► [Stage 3: Active ROI Auto-Cropper]
 │     • Computes union bounding box over entire motion trajectory
 │     • Applies contextual padding with edge-aware minimum ROI size clamping
 │
 └──► [Stage 4: Structured Storyboard Packer]
       • Assembles RxC composite image grid with uniform gutters
       • Burns chronological relative timestamps and optional A1-D4 spatial overlays
       • Outputs ready-to-prompt PIL.Image, raw bytes, or base64 data URIs
```

---

## Installation

```bash
pip install vlm-sieve
```

### Core Dependency Guardrails
`vlm-sieve` is engineered for ultra-lean edge and cloud deployment. Core dependencies are strictly limited to `numpy`, `opencv-python-headless`, `Pillow`, and `pydantic`. **Zero heavy deep-learning frameworks (PyTorch, TensorFlow, Transformers) are installed or required.**

---

## Quickstart Code Examples

### 1. Basic 5-Line Python SDK

```python
from vlmsieve import VideoSieve

# Works with local video files, RTSP/HTTP streams, or webcam index (0)
sieve = VideoSieve("warehouse_cam.mp4", grid_layout=(2, 3), spatial_grid_overlay=True)

for event in sieve.stream_events():
    print(f"Captured Event #{event.event_id} ({event.raw_frame_count} frames)")
    # Save storyboard directly to disk or send to VLM
    event.save(f"event_{event.event_id}.jpg")

report = sieve.get_savings_report()
print(f"Tokens saved: {report.tokens_saved_pct:.1f}% ({report.compression_ratio:.1f}x compression)")
```

---

### 2. Google Gemini (`google-genai`) Integration

Pass storyboard PIL images directly to Google Gemini 2.0 / 1.5 Flash:

```python
from google import genai
from vlmsieve import VideoSieve

client = genai.Client()
sieve = VideoSieve("security_feed.mp4", spatial_grid_overlay=True)

for event in sieve.stream_events():
    prompt = (
        "Analyze this chronological 2x3 video storyboard. Each tile has a relative timestamp badge "
        "and a 4x4 spatial grid (rows A-D, columns 1-4). "
        "Describe what occurred, noting exact grid coordinates and timestamps."
    )
    
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[event.storyboard_pil, prompt]
    )
    print(f"Gemini Analysis for Event #{event.event_id}:\n{response.text}\n")
```

---

### 3. OpenAI (`gpt-4o` / `gpt-4o-mini`) Integration

Encode the storyboard to a base64 Data URI for OpenAI's Chat Completions API:

```python
from openai import OpenAI
from vlmsieve import VideoSieve

client = OpenAI()
sieve = VideoSieve("traffic_cam.mp4", spatial_grid_overlay=True)

for event in sieve.stream_events():
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Identify any safety incidents in this 6-panel storyboard.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": event.to_data_uri(format="JPEG", quality=88)},
                    },
                ],
            }
        ],
    )
    print(response.choices[0].message.content)
```

---

### 4. Non-Blocking Async Streaming (FastAPI / Agent Loops)

```python
import asyncio
from vlmsieve import VideoSieve

async def live_monitoring_agent():
    sieve = VideoSieve("rtsp://admin:pass@192.168.1.50/stream")
    
    # astream_events runs CPU frame gating on worker threads without blocking asyncio
    async for event in sieve.astream_events():
        print(f"New event #{event.event_id} captured at {event.start_time_sec:.1f}s")
        # Trigger background webhook, alerts, or asynchronous LLM calls
        await dispatch_vlm_task(event.to_bytes())

asyncio.run(live_monitoring_agent())
```

---

## Command-Line Interface (CLI)

`vlm-sieve` ships with a command-line tool `vlmsieve`:

```bash
# Process video into 2x3 storyboards with spatial coordinate grid
vlmsieve warehouse.mp4 --output-dir ./events --grid 2x3 --spatial-grid

# Inspect live webcam feed (index 0) with higher sensitivity threshold
vlmsieve 0 --threshold 0.01 --max-frames 300 -o ./webcam_events

# Process RTSP feed with full frames (disable active ROI cropping)
vlmsieve "rtsp://camera.local/live" --no-crop -o ./rtsp_events
```

### Sample CLI Output
```
============================================================
                vlm-sieve Processing Summary
============================================================
 Raw Frames Inspected  : 1,800
 Storyboards Emitted   : 4
 Frames Dropped        : 1,776
 Naive Token Estimate  : 2,016,000
 Sieve Token Estimate  : 4,480
 Compression Ratio     : 450.0x
 Token Savings         : 99.78%
============================================================
 Output Directory      : C:\Users\user\Documents\events
 Total Events Saved    : 4
============================================================
```

---

## `SieveConfig` Reference Guide

Every hyperparameter can be fine-tuned via `SieveConfig`:

| Parameter | Type | Default | Description |
|:---|:---:|:---:|:---|
| `downscale_width` | `int` | `320` | Downscaled frame width for motion evaluation. Guarantees sub-2ms CPU gating latency. |
| `motion_threshold` | `float` | `0.015` | Minimum fraction of moving pixels to trigger an active motion state. |
| `pixel_diff_threshold` | `int` | `25` | Pixel intensity delta threshold in $[0, 255]$ for frame differencing. |
| `min_contour_area` | `int` | `16` | Minimum contour area in downscaled pixels required to eliminate sensor noise. |
| `cooldown_frames` | `int` | `12` | Static frames required after motion ceases before closing an event. |
| `min_event_frames` | `int` | `3` | Minimum motion frames required to emit an event (rejects single-frame flicker). |
| `max_event_frames` | `int` | `150` | Maximum frames in a single event buffer before flushing to prevent unbounded latency. |
| `grid_layout` | `tuple[int, int]` | `(2, 3)` | Storyboard grid layout specified as `(rows, cols)` (e.g., `(2, 3)` or `(3, 3)`). |
| `auto_crop_roi` | `bool` | `True` | Automatically detect and crop the active motion region across the action trajectory. |
| `roi_padding_ratio` | `float` | `0.15` | Symmetrical padding ratio added around the motion bounding box. |
| `min_roi_size` | `int` | `224` | Minimum cropped ROI width and height, preserving framing for small objects. |
| `burn_timestamps` | `bool` | `True` | Burn relative elapsed timestamp badges (`#1 \| +0.00s`) onto each tile. |
| `spatial_grid_overlay` | `bool` | `False` | Render a $4 \times 4$ spatial grid (`A1`–`D4`) across each tile for coordinate grounding. |
| `tile_max_width` | `int` | `640` | Maximum pixel width per tile before proportional Lanczos downscaling. |
| `tokens_per_image` | `int` | `1120` | Standard VLM image tile token cost used to compute savings reports. |
| `camera_warmup_frames` | `int` | `8` | Number of initial frames discarded when opening live webcams for exposure stabilization. |

---

## Performance & Benchmarks

- **CPU Frame Gating Latency:** `< 2.5ms` per 1080p frame and `< 1.2ms` per 720p frame on single-core standard x86_64 CPUs.
- **Zero GPU Memory:** All downscaling, differencing, and morphological operations execute within CPU L2/L3 cache without requiring CUDA or GPU buffers.
- **Token Compression:** Typically compresses continuous video by **95% to 99%**, translating directly to a 20x to 100x reduction in API costs.

---

## License

This project is licensed under the terms of the [MIT License](LICENSE).
