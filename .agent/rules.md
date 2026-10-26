# vlm-sieve: Workspace Architectural Rules & Engineering Guidelines

## 1. Project Identity & Purpose
- **Distribution Name:** `vlm-sieve`
- **Import Package Name:** `vlmsieve`
- **Core Purpose:** High-throughput, event-driven video frame gatekeeper, active Region-of-Interest (ROI) cropper, and temporal storyboard packer. It selectively filters redundant video frames and intelligently packs salient visual information to reduce Vision-Language Model (VLM) image token consumption by >95% while operating purely on CPU.

---

## 2. Strict Architectural Constraints

### 2.1 Core Dependency Guardrails
- **Heavy Deep-Learning Prohibition:** Heavy deep-learning frameworks (`torch`, `tensorflow`, `transformers`, `jax`, `onnxruntime`, etc.) are **STRICTLY FORBIDDEN** in core dependencies.
- **Allowed Core Dependencies Only:**
  - `numpy>=1.24.0`
  - `opencv-python-headless>=4.8.0`
  - `Pillow>=10.0.0`
  - `pydantic>=2.5.0`
- Any optional neural/model integrations must live in separated optional extras (e.g., `vlm-sieve[torch]`) and must never be imported in core modules without graceful fallbacks.

### 2.2 Performance & Latency Targets
- **CPU Frame Gating Latency:** `< 3ms` per frame at 1080p input resolution.
- **Downscaled Grayscale Processing:** Gating decisions must operate on downscaled grayscale representations (e.g., 360p or lower, pyramid representations) to guarantee sub-3ms evaluation on commodity single-core CPUs.
- **Memory Footprint:** Zero-copy operations and buffer reuse wherever possible when piping NumPy arrays between OpenCV and Pillow. Avoid redundant array allocations.

### 2.3 Implementation & Code Integrity
- **Zero Placeholder Stubs:** Every class, function, and method must be completely implemented, mathematically sound, and functional. No empty `pass` bodies, no `...` ellipses in concrete logic, and no `TODO` or unhandled edge cases.
- **Strict Typing:** All modules must enforce Python 3.9+ type annotations on all function signatures, return values, and class attributes. Utilize `typing` / `pydantic` types and adhere to strict static typing checks.

---

## 3. Core Functional Pillars

### 3.1 Temporal Frame Gating (`vlmsieve.gating`)
- Detect meaningful visual delta between consecutive and sliding-window frames.
- Employ lightweight algorithms:
  - Structural Similarity Metric (SSIM) approximations on downscaled tiles.
  - Pixel-difference statistics with adaptive luminosity normalization.
  - Motion energy accumulation (frame differencing / background subtraction).
- Fast drop of static or sub-threshold frames before any expensive image processing.

### 3.2 Active Region-of-Interest Cropper (`vlmsieve.cropper`)
- Identify dynamic bounding boxes containing the active motion or high-frequency visual change.
- Morphological dilation and contour clustering to avoid fragmented crops.
- Dynamic aspect-ratio padding to preserve contextual cues without transmitting vast static backgrounds to VLMs.

### 3.3 Temporal Storyboard Packer (`vlmsieve.packer`)
- Composite multiple sequential keyframes or ROI patches into high-density temporal grid storyboards.
- Timestamp and sequence metadata overlay/badges for multi-frame contextual alignment within a single VLM input image.
- Drastic reduction of API calls and token count by consolidating $N$ frames into single composite tensors/images.

---

## 4. Coding & Quality Standards

- **Python Version Support:** `>=3.9`
- **Linting & Formatting:** `ruff` rules must pass cleanly.
- **Data Validation:** All user-facing configuration, thresholds, and outputs must be validated using `pydantic.BaseModel`.
- **Testing:** Unit tests under `tests/` verifying mathematical correctness, threshold edge cases, memory cleanup, and throughput benchmarking.
