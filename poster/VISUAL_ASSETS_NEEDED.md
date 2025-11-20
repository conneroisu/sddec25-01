# Visual Assets Needed for Poster

This document lists the visual assets that should be created to replace text-heavy sections and improve poster readability.

## Priority 1: Essential Visuals

### 1. System Architecture Diagram
**Location:** `sections/architecture.tex`
**Purpose:** Show end-to-end processing pipeline
**Content:**
```
Camera Input → Preprocessing → U-Net Segmentation (DPU) → Feature Extraction → Eye Tracking Output
```
**Annotations:**
- 8.3ms per frame
- 60 FPS total throughput
- Pipelined DPU scheduling visualization
**Format:** Block diagram with arrows, timing annotations
**Size:** 90% of column width

### 2. Eye Segmentation Example
**Location:** `sections/results.tex` (to be added)
**Purpose:** Show actual system output - proof it works
**Content:**
- Side-by-side: Raw eye image | Segmentation mask
- Label: "Pupil detection with 98.8% IoU accuracy"
**Format:** Two images side-by-side
**Size:** 80% of column width

### 3. Performance Comparison Chart
**Location:** `sections/results.tex` (to be added)
**Purpose:** Visualize 5× throughput improvement
**Content:**
- Bar chart: "Baseline (12 FPS)" vs "Pipelined (60 FPS)"
- Y-axis: Frames per second
- Annotation: "5× speedup from intelligent scheduling"
**Format:** Horizontal bar chart
**Size:** 70% of column width

## Priority 2: Recommended Visuals

### 4. Before/After Scheduling Diagram
**Purpose:** Explain how pipelined scheduling achieves 5× improvement
**Content:**
- Sequential scheduling: [DPU task 1] → [idle] → [DPU task 2] → [idle]
- Pipelined scheduling: [DPU task 1] [DPU task 2] [DPU task 3] [overlapped]
**Format:** Timeline diagram
**Size:** 80% of column width

### 5. Clinical Use Case Storyboard
**Purpose:** Make audience emotionally connect to application
**Content:**
- Wheelchair user with eye tracking camera
- System detecting anomaly
- Wheelchair repositioning to safe position
**Format:** 3-panel illustration
**Size:** Full column width

### 6. Lighting Robustness Examples
**Purpose:** Demonstrate >98% accuracy across conditions
**Content:**
- 3 images: Bright light | Normal | Low light
- All with successful segmentation overlay
**Format:** Three images in row
**Size:** 90% of column width

## Asset Creation Tools

Recommended tools for creating these assets:

1. **Block diagrams:** draw.io, Lucidchart, or TikZ (LaTeX)
2. **Charts:** Python (matplotlib), R (ggplot2), or Excel
3. **Eye segmentation:** Export from actual system output
4. **Illustrations:** Adobe Illustrator, Inkscape, or Figma

## File Format Guidelines

- **Format:** PNG or PDF (vector preferred for diagrams)
- **Resolution:** 300 DPI minimum for print
- **Color scheme:** Use ISU colors (Cardinal Red #C8102E, Gold #F1BE48)
- **Consistency:** Match ISU branding in poster-config.tex

## Integration Notes

Once created, replace placeholder boxes in:
- `sections/architecture.tex` - System diagram
- `sections/results.tex` - Performance charts and segmentation examples

Update `\includegraphics{}` commands with actual file paths in `assets/` directory.
