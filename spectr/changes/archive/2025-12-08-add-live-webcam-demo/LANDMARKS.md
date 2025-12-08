# MediaPipe Face Mesh Landmark Reference

## Left Eye Contour (12 points)

These landmark indices define the complete left eye contour for accurate bounding box computation:

```
Landmark Indices: 362, 385, 387, 263, 373, 380, 374, 381, 382, 384, 398, 466
```

### Landmark Positions:
- **362**: Outer corner (temporal side)
- **385**: Upper outer lid
- **387**: Upper lid (mid-outer)
- **263**: Inner corner (nasal side)
- **373**: Upper lid (mid-inner)
- **380**: Upper lid center
- **374**: Lower lid (mid-inner)
- **381**: Lower lid center
- **382**: Lower lid (mid-outer)
- **384**: Lower outer lid
- **398**: Lower outer corner
- **466**: Additional outer contour point

### Algorithm:
1. Extract all 12 landmark coordinates from MediaPipe Face Mesh results
2. Compute min/max x and y coordinates to form bounding box
3. Check if bounding box width or height > 100px
4. Add 20% padding on all sides
5. Expand box to 1.6:1 (width:height) aspect ratio for model input
6. Clip to frame boundaries if needed

### Code Example:
```python
# MediaPipe landmark indices for left eye
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380, 374, 381, 382, 384, 398, 466]

# Extract coordinates
eye_points = []
for idx in LEFT_EYE_INDICES:
    landmark = face_landmarks.landmark[idx]
    x = int(landmark.x * frame_width)
    y = int(landmark.y * frame_height)
    eye_points.append((x, y))

# Compute bounding box
x_coords = [p[0] for p in eye_points]
y_coords = [p[1] for p in eye_points]
x_min, x_max = min(x_coords), max(x_coords)
y_min, y_max = min(y_coords), max(y_coords)

# Check minimum size
width = x_max - x_min
height = y_max - y_min
if width < 100 or height < 100:
    # Show "Move Closer" warning
    return None

# Add 20% padding
padding_x = int(width * 0.2)
padding_y = int(height * 0.2)
x_min = max(0, x_min - padding_x)
x_max = min(frame_width, x_max + padding_x)
y_min = max(0, y_min - padding_y)
y_max = min(frame_height, y_max + padding_y)

# Expand to 1.6:1 aspect ratio (smart crop)
current_width = x_max - x_min
current_height = y_max - y_min
target_ratio = 640 / 400  # 1.6

if current_width / current_height < target_ratio:
    # Too tall, expand width
    target_width = int(current_height * target_ratio)
    diff = target_width - current_width
    x_min = max(0, x_min - diff // 2)
    x_max = min(frame_width, x_max + diff // 2)
else:
    # Too wide, expand height
    target_height = int(current_width / target_ratio)
    diff = target_height - current_height
    y_min = max(0, y_min - diff // 2)
    y_max = min(frame_height, y_max + diff // 2)

# Extract crop
eye_crop = frame[y_min:y_max, x_min:x_max]
```

## Alternative Eye Landmark Sets (Not Used)

### Iris Landmarks (5 points per eye) - NOT USED
- Left iris: 468, 469, 470, 471, 472
- Right iris: 473, 474, 475, 476, 477
- **Why not used:** Too small, doesn't capture full eye context

### Simple Eye Box (6 points) - NOT USED
- Left: 133 (outer), 33 (inner), 159 (top-outer), 145 (top-inner), 386 (bottom-outer), 263 (bottom-inner)
- **Why not used:** Less accurate than full 12-point contour

## Reference
- MediaPipe Face Mesh has 468 facial landmarks
- Left eye region: indices 33-133 (approximate range)
- Right eye region: indices 362-263 (approximate range)
- Full specification: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
