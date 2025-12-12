#!/usr/bin/env python3
"""
Dataset Viewer for OpenEDS Pupil Segmentation Dataset.
Tkinter GUI application to view and verify the HuggingFace dataset
used in train.py for pupil segmentation training.
"""
import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import Image
from datasets import load_dataset
import io

# Constants (matching train.py)
IMAGE_HEIGHT = 400
IMAGE_WIDTH = 640
HF_DATASET_REPO = "Conner/sddec25-01"

# Colormap for heatmaps (viridis-like)
def viridis_colormap(value):
    """Simple viridis-like colormap for 0-1 normalized values."""
    # Approximate viridis colors at key points
    colors = [
        (0.267, 0.004, 0.329),  # dark purple
        (0.282, 0.140, 0.458),  # purple
        (0.254, 0.265, 0.530),  # blue-purple
        (0.207, 0.372, 0.553),  # blue
        (0.163, 0.471, 0.558),  # teal
        (0.127, 0.566, 0.551),  # green-teal
        (0.135, 0.659, 0.518),  # green
        (0.267, 0.749, 0.441),  # yellow-green
        (0.478, 0.821, 0.318),  # lime
        (0.741, 0.873, 0.150),  # yellow
        (0.993, 0.906, 0.144),  # bright yellow
    ]
    idx = min(int(value * (len(colors) - 1)), len(colors) - 2)
    t = value * (len(colors) - 1) - idx
    r = colors[idx][0] * (1 - t) + colors[idx + 1][0] * t
    g = colors[idx][1] * (1 - t) + colors[idx + 1][1] * t
    b = colors[idx][2] * (1 - t) + colors[idx + 1][2] * t
    return int(r * 255), int(g * 255), int(b * 255)


def apply_colormap(data):
    """Apply viridis colormap to 2D array, returns RGB image."""
    # Normalize to 0-1
    data = np.asarray(data, dtype=np.float32)
    if data.max() > data.min():
        data = (data - data.min()) / (data.max() - data.min())
    else:
        data = np.zeros_like(data)

    h, w = data.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            rgb[i, j] = viridis_colormap(data[i, j])
    return rgb


def apply_colormap_fast(data):
    """Fast vectorized colormap application."""
    data = np.asarray(data, dtype=np.float32)
    if data.max() > data.min():
        data = (data - data.min()) / (data.max() - data.min())
    else:
        data = np.zeros_like(data)

    # Precomputed viridis LUT (256 colors)
    lut = np.array([viridis_colormap(i / 255.0) for i in range(256)], dtype=np.uint8)
    indices = (data * 255).astype(np.uint8)
    return lut[indices]


class DatasetViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Dataset Viewer - OpenEDS Pupil Segmentation")
        self.root.geometry("1400x900")

        # Dataset state
        self.dataset = None
        self.current_split = "train"
        self.current_idx = 0
        self.photo_refs = []  # Keep references to prevent garbage collection

        # Create UI
        self.create_ui()

        # Load dataset
        self.load_dataset()

        # Display first sample
        self.update_display()

        # Bind resize event
        self.root.bind("<Configure>", self.on_resize)
        self.last_size = (0, 0)

    def create_ui(self):
        """Create the UI components."""
        # Main container
        self.main_frame = ttk.Frame(self.root, padding="5")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Navigation frame
        nav_frame = ttk.Frame(self.main_frame)
        nav_frame.pack(fill=tk.X, pady=(0, 10))

        # Split selector
        ttk.Label(nav_frame, text="Split:").pack(side=tk.LEFT, padx=(0, 5))
        self.split_var = tk.StringVar(value="train")
        split_combo = ttk.Combobox(
            nav_frame,
            textvariable=self.split_var,
            values=["train", "validation"],
            state="readonly",
            width=12
        )
        split_combo.pack(side=tk.LEFT, padx=(0, 15))
        split_combo.bind("<<ComboboxSelected>>", self.on_split_change)

        # Sample index
        ttk.Label(nav_frame, text="Sample:").pack(side=tk.LEFT, padx=(0, 5))
        self.idx_var = tk.StringVar(value="0")
        self.idx_entry = ttk.Entry(nav_frame, textvariable=self.idx_var, width=8)
        self.idx_entry.pack(side=tk.LEFT, padx=(0, 5))
        self.idx_entry.bind("<Return>", self.on_idx_entry)

        self.total_label = ttk.Label(nav_frame, text="/ 0")
        self.total_label.pack(side=tk.LEFT, padx=(0, 15))

        # Navigation buttons
        ttk.Button(nav_frame, text="< Prev", command=self.prev_sample).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="Next >", command=self.next_sample).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="Go", command=self.go_to_sample).pack(side=tk.LEFT, padx=(10, 0))

        # Keyboard bindings
        self.root.bind("<Left>", lambda e: self.prev_sample())
        self.root.bind("<Right>", lambda e: self.next_sample())

        # Image grid frame (2 rows x 4 cols)
        self.grid_frame = ttk.Frame(self.main_frame)
        self.grid_frame.pack(fill=tk.BOTH, expand=True)

        # Configure grid weights for equal sizing
        for i in range(2):
            self.grid_frame.rowconfigure(i, weight=1)
        for j in range(4):
            self.grid_frame.columnconfigure(j, weight=1)

        # Create image panels with labels
        self.panels = {}
        panel_configs = [
            ("image", "Image", 0, 0),
            ("label", "Label (Pupil Mask)", 0, 1),
            ("spatial_weights", "Spatial Weights", 0, 2),
            ("dist_map_0", "Dist Map (Ch 0)", 0, 3),
            ("dist_map_1", "Dist Map (Ch 1)", 1, 0),
            ("eye_mask", "Eye Mask", 1, 1),
            ("eye_weight", "Eye Weight", 1, 2),
        ]

        for name, title, row, col in panel_configs:
            frame = ttk.LabelFrame(self.grid_frame, text=title, padding="2")
            frame.grid(row=row, column=col, sticky="nsew", padx=2, pady=2)

            canvas = tk.Canvas(frame, bg="black", highlightthickness=0)
            canvas.pack(fill=tk.BOTH, expand=True)

            self.panels[name] = {
                "frame": frame,
                "canvas": canvas,
                "photo": None
            }

        # Statistics frame
        self.stats_frame = ttk.Frame(self.main_frame)
        self.stats_frame.pack(fill=tk.X, pady=(10, 0))

        self.stats_label = ttk.Label(
            self.stats_frame,
            text="Loading dataset...",
            font=("TkDefaultFont", 10)
        )
        self.stats_label.pack(side=tk.LEFT)

    def load_dataset(self):
        """Load the HuggingFace dataset."""
        self.stats_label.config(text="Loading dataset from HuggingFace cache...")
        self.root.update()

        try:
            self.hf_dataset = load_dataset(
                HF_DATASET_REPO,
                cache_dir="./hf_cache",
            )
            # Set format for efficient loading
            for split in self.hf_dataset:
                self.hf_dataset[split].set_format("numpy")

            self.dataset = self.hf_dataset[self.current_split]
            self.total_label.config(text=f"/ {len(self.dataset) - 1}")
            self.stats_label.config(text="Dataset loaded successfully")
        except Exception as e:
            self.stats_label.config(text=f"Error loading dataset: {e}")

    def on_split_change(self, event=None):
        """Handle split selection change."""
        new_split = self.split_var.get()
        if new_split != self.current_split:
            self.current_split = new_split
            self.dataset = self.hf_dataset[self.current_split]
            self.current_idx = 0
            self.idx_var.set("0")
            self.total_label.config(text=f"/ {len(self.dataset) - 1}")
            self.update_display()

    def prev_sample(self):
        """Navigate to previous sample."""
        if self.current_idx > 0:
            self.current_idx -= 1
            self.idx_var.set(str(self.current_idx))
            self.update_display()

    def next_sample(self):
        """Navigate to next sample."""
        if self.dataset and self.current_idx < len(self.dataset) - 1:
            self.current_idx += 1
            self.idx_var.set(str(self.current_idx))
            self.update_display()

    def go_to_sample(self):
        """Go to specific sample index."""
        try:
            idx = int(self.idx_var.get())
            if self.dataset and 0 <= idx < len(self.dataset):
                self.current_idx = idx
                self.update_display()
            else:
                self.idx_var.set(str(self.current_idx))
        except ValueError:
            self.idx_var.set(str(self.current_idx))

    def on_idx_entry(self, event=None):
        """Handle Enter key in index entry."""
        self.go_to_sample()

    def on_resize(self, event=None):
        """Handle window resize."""
        new_size = (self.root.winfo_width(), self.root.winfo_height())
        if new_size != self.last_size and self.dataset:
            self.last_size = new_size
            # Debounce: only update if size actually changed significantly
            self.root.after(100, self.update_display)

    def get_panel_size(self, panel_name):
        """Get the current size of a panel canvas."""
        canvas = self.panels[panel_name]["canvas"]
        canvas.update_idletasks()
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        # Minimum size to avoid issues
        return max(w, 100), max(h, 75)

    def tensor_to_grayscale_image(self, data):
        """Convert image tensor to displayable grayscale PIL Image."""
        # Data shape: (H, W) after reshape, stored as raw uint8 (0-255)
        data = np.asarray(data, dtype=np.float32)
        data = data.reshape(IMAGE_HEIGHT, IMAGE_WIDTH)
        # Data is already 0-255 uint8 in the dataset, just clip and convert
        data = np.clip(data, 0, 255).astype(np.uint8)
        return Image.fromarray(data, mode='L')

    def mask_to_overlay(self, image_data, mask_data, color=(255, 0, 0), alpha=0.5):
        """Create colored overlay of mask on grayscale image."""
        # Convert grayscale to RGB
        img = self.tensor_to_grayscale_image(image_data)
        img_rgb = img.convert('RGB')
        img_array = np.array(img_rgb)

        # Get mask
        mask = np.asarray(mask_data, dtype=np.int32).reshape(IMAGE_HEIGHT, IMAGE_WIDTH)

        # Apply overlay where mask == 1
        overlay = img_array.copy()
        mask_bool = mask == 1
        for c in range(3):
            overlay[:, :, c] = np.where(
                mask_bool,
                (1 - alpha) * img_array[:, :, c] + alpha * color[c],
                img_array[:, :, c]
            )

        return Image.fromarray(overlay.astype(np.uint8))

    def heatmap_to_image(self, data):
        """Convert weight/distance map to colormap image."""
        data = np.asarray(data, dtype=np.float32)
        if len(data.shape) == 1:
            data = data.reshape(IMAGE_HEIGHT, IMAGE_WIDTH)
        rgb = apply_colormap_fast(data)
        return Image.fromarray(rgb)

    def pil_to_tkphoto(self, pil_image):
        """Convert PIL image to Tkinter PhotoImage using PPM format."""
        # Convert to RGB if not already
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        # Save to PPM format in memory (Tkinter can read this natively)
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PPM')
        ppm_data = buffer.getvalue()
        return tk.PhotoImage(data=ppm_data)

    def display_image_on_canvas(self, panel_name, pil_image):
        """Display PIL image on canvas, auto-scaling to fit."""
        canvas = self.panels[panel_name]["canvas"]
        canvas_w, canvas_h = self.get_panel_size(panel_name)

        # Calculate scale to fit
        img_w, img_h = pil_image.size
        scale = min(canvas_w / img_w, canvas_h / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)

        if new_w > 0 and new_h > 0:
            resized = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            photo = self.pil_to_tkphoto(resized)

            # Clear canvas and display centered
            canvas.delete("all")
            x = (canvas_w - new_w) // 2
            y = (canvas_h - new_h) // 2
            canvas.create_image(x, y, anchor=tk.NW, image=photo)

            # Keep reference
            self.panels[panel_name]["photo"] = photo

    def update_display(self):
        """Update all panels with current sample."""
        if not self.dataset:
            return

        # Clear old photo references
        self.photo_refs = []

        # Get sample
        sample = self.dataset[self.current_idx]

        # Extract data
        image = sample["image"]
        label = sample["label"]
        spatial_weights = sample["spatial_weights"]
        dist_map = sample["dist_map"]
        eye_mask = sample["eye_mask"]
        eye_weight = sample["eye_weight"]

        # Image (grayscale)
        img_pil = self.tensor_to_grayscale_image(image)
        self.display_image_on_canvas("image", img_pil.convert("RGB"))

        # Label overlay (pupil in red)
        label_pil = self.mask_to_overlay(image, label, color=(255, 50, 50), alpha=0.6)
        self.display_image_on_canvas("label", label_pil)

        # Spatial weights heatmap
        sw_pil = self.heatmap_to_image(spatial_weights.reshape(IMAGE_HEIGHT, IMAGE_WIDTH))
        self.display_image_on_canvas("spatial_weights", sw_pil)

        # Distance map channel 0
        dm = dist_map.reshape(2, IMAGE_HEIGHT, IMAGE_WIDTH)
        dm0_pil = self.heatmap_to_image(dm[0])
        self.display_image_on_canvas("dist_map_0", dm0_pil)

        # Distance map channel 1
        dm1_pil = self.heatmap_to_image(dm[1])
        self.display_image_on_canvas("dist_map_1", dm1_pil)

        # Eye mask overlay (green)
        eye_mask_pil = self.mask_to_overlay(image, eye_mask, color=(50, 255, 50), alpha=0.5)
        self.display_image_on_canvas("eye_mask", eye_mask_pil)

        # Eye weight heatmap
        ew_pil = self.heatmap_to_image(eye_weight.reshape(IMAGE_HEIGHT, IMAGE_WIDTH))
        self.display_image_on_canvas("eye_weight", ew_pil)

        # Update statistics
        label_flat = np.asarray(label).flatten()
        pupil_pixels = np.sum(label_flat == 1)
        total_pixels = len(label_flat)
        pupil_pct = 100 * pupil_pixels / total_pixels if total_pixels > 0 else 0

        img_array = np.asarray(image, dtype=np.float32)
        img_min = img_array.min()
        img_max = img_array.max()
        img_mean = img_array.mean()

        sw_array = np.asarray(spatial_weights, dtype=np.float32)
        sw_min = sw_array.min()
        sw_max = sw_array.max()

        stats_text = (
            f"Pupil: {pupil_pixels:,} px ({pupil_pct:.2f}%) | "
            f"Image: [{img_min:.3f}, {img_max:.3f}] mean={img_mean:.3f} | "
            f"Spatial weights: [{sw_min:.3f}, {sw_max:.3f}]"
        )
        self.stats_label.config(text=stats_text)


def main():
    root = tk.Tk()
    app = DatasetViewer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
