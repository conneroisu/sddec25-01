# Final Design Decisions Summary

**Change ID:** `add-live-webcam-demo`
**Status:** Ready for implementation
**Validation:** ✓ Passed `spectr validate --strict`

## All 27 Finalized Decisions

### Core Configuration (1-5)
1. **Eye Selection:** Left eye only - simpler, faster, half the inference cost
2. **Webcam Resolution:** 1920x1080 (Full HD) native capture - high quality for presentations
3. **Display Resolution:** Full 1080p window - impressive presentation quality
4. **Model File:** User provides via required `--model` flag - keeps repo lightweight, explicit
5. **Fallback Mode:** Clear overlay and show "No Face Detected" message - honest and transparent

### Eye Detection & Cropping (6-10)
6. **Landmark Points:** Full 12-point eye contour (MediaPipe indices: 362, 385, 387, 263, 373, 380, 374, 381, 382, 384, 398, 466)
7. **Crop Padding:** 20% padding around eye landmarks - balanced, captures eye plus surrounding context
8. **Aspect Ratio Handling:** Smart crop - expand bounding box to 1.6:1 (640:400) before extraction
9. **Minimum Size Check:** Eye region must be > 100px - show "Move Closer" warning if smaller
10. **Landmark Smoothing:** Use MediaPipe built-in tracking mode smoothing - reduces jitter

### Preprocessing Pipeline (11-13)
11. **Preprocessing Order:** Gamma → CLAHE on original crop → Resize to 640x400 → Normalize
12. **Gamma Correction:** Fixed γ=0.8 matching training - most accurate, no runtime complexity
13. **CLAHE Timing:** Apply to original crop before resize - better quality, matches training

### Visualization (14-19)
14. **Overlay Color:** Green (50% alpha) - tech aesthetic, complements skin tones
15. **Overlay Resolution:** Match model output (640x400 upscaled) - shows direct model prediction
16. **Bounding Box:** 3-pixel green line - bold and clear for presentations
17. **Font Size:** Medium (1.0 scale) - readable at desk distance
18. **Message Position:** Top center banner - clear without blocking main view
19. **Debug View:** No separate debug panels - clean single-window interface

### Performance & Error Handling (20-24)
20. **Frame Processing:** Process every frame, accept actual FPS - honest performance reporting
21. **GPU Strategy:** Prefer GPU (CUDA), fallback to CPU with warning - optimized for 1080p
22. **Slow Inference:** Accept and show actual FPS/timing - simple, transparent
23. **Model Validation:** Strict startup validation of input/output shapes - fail fast with clear errors
24. **Logging:** Minimal by default, `--verbose` logs everything (timing, shapes, values, coordinates)

### User Experience (25-27)
25. **Capture Feature:** No screenshot/recording - keep it simple
26. **Model Path:** Always require explicit `--model` flag - clear and unambiguous
27. **Controls:** ESC (exit), Space (pause/resume) - simple and standard

## Documentation Files

### Core Proposal Files
- ✅ `proposal.md` - Why, what changes, impact, technical approach, rationale
- ✅ `tasks.md` - 8 phases, 64 implementation tasks with 1080p details
- ✅ `design.md` - Full architecture, 27 decisions, 7 component breakdowns, data flow, risks
- ✅ `specs/webcam-demo/spec.md` - 9 requirements, 32 scenarios, all decisions encoded

### Reference Files
- ✅ `LANDMARKS.md` - MediaPipe landmark indices, code examples, algorithm details
- ✅ `DECISIONS.md` - This file, comprehensive decision summary

## Key Technical Specifications

### Pipeline
```
1080p Webcam (1920x1080)
    ↓
MediaPipe Face Mesh (tracking mode, smoothing enabled)
    ↓
12-point Left Eye Contour Extraction (indices documented in LANDMARKS.md)
    ↓
Size Check (>100px, else "Move Closer")
    ↓
20% Padding + Smart Crop to 1.6:1 Aspect Ratio
    ↓
Preprocessing: Grayscale → Gamma(0.8) → CLAHE(1.5, 8x8) → Resize(640x400) → Normalize(0.5, 0.5)
    ↓
ONNX Inference (CUDAExecutionProvider preferred, CPUExecutionProvider fallback with warning)
    ↓
Strict Model Validation: Input (1,1,400,640), Output (1,2,400,640)
    ↓
Post-processing: argmax → Binary Mask (0=bg, 1=pupil)
    ↓
Coordinate Transform: 640x400 → Eye Crop Region → 1920x1080 Frame
    ↓
Green Overlay (50% alpha) + 3px Bounding Box + Status Banner
    ↓
Full 1080p Display with Performance Metrics (FPS, Inference Time, GPU/CPU)
```

### Command-Line Interface
```bash
# Required
python demo/demo.py --model path/to/model.onnx

# Optional
python demo/demo.py --model path/to/model.onnx --camera 1 --verbose
```

### Keyboard Controls
- **ESC**: Exit application
- **Space**: Pause/resume video feed

### Display Layout
```
┌─────────────────────────────────────────────────────────────┐
│ FPS: 28.3                    [FACE DETECTED]                │ ← Top
│ Inference: 12.4ms                                           │
│ GPU: CUDA                                                   │
│                                                             │
│                                                             │
│                    [1080p Webcam Feed]                      │
│                                                             │
│                  ┌──────────────┐                           │ ← Eye
│                  │  Left Eye    │ ← Green 3px box           │   Region
│                  │  + Green     │                           │
│                  │    Overlay   │                           │
│                  └──────────────┘                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Dependencies
- Python 3.10+
- OpenCV 4.5+
- ONNX Runtime 1.17+ (with CUDA support recommended)
- MediaPipe 0.10+
- NumPy 1.21+

## Hardware Requirements
- **Webcam:** 1080p (tested with laptop webcam)
- **GPU:** CUDA-capable GPU strongly recommended for real-time 1080p processing
- **CPU:** Supports CPU-only mode but may achieve <30 FPS

## Validation Status
- ✅ All 27 decisions documented in design.md
- ✅ proposal.md updated with 1080p specifics
- ✅ tasks.md updated with 64 detailed tasks
- ✅ spec.md updated with 9 requirements, 32 scenarios
- ✅ LANDMARKS.md created with MediaPipe details
- ✅ `spectr validate add-live-webcam-demo --strict` passes
- ✅ All files cross-referenced for consistency
- ✅ Green overlay (not red) confirmed throughout
- ✅ 1080p resolution confirmed throughout
- ✅ No screenshot feature confirmed (removed from tasks)
- ✅ No frame skipping confirmed (removed from tasks)
- ✅ Verbose logging details confirmed throughout

## Ready for Implementation
All design decisions are finalized and documented. The proposal is ready for the `coder` agent to implement.
