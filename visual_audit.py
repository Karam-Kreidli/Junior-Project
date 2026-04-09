"""
BioReef.ai - Visual Pipeline Audit
====================================
Selects 3 fish samples from subset_100.json and runs them through
WaterNetRestorer + ContextHarvester, producing visual grids:
    [Original] | [Restored] | [ROI] | [Social] | [Habitat] | [Full Frame]

Sample selection criteria:
    1. Small fish (smallest bbox area) - tests Size-Adaptive ROI
    2. Low-contrast / murky (lowest average brightness) - tests Water-Net
    3. Near structure (bbox closest to frame edge) - tests Context Harvester padding

Output: saves grids to outputs/visual_audit/
"""

import json
import os
import sys
import cv2
import numpy as np
from urllib.parse import unquote

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from bioreef.data.data_factory import ContextHarvester

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(PROJECT_ROOT, "data_oz", "ozfish_data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
SUBSET_PATH = os.path.join(DATA_DIR, "annotation", "subset_100.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "visual_audit")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_subset():
    """Load subset_100.json and build annotation lookup."""
    with open(SUBSET_PATH, "r") as f:
        data = json.load(f)

    # Build image_id -> image info mapping
    images = {}
    for img in data["images"]:
        images[img["id"]] = img

    # Build image_id -> list of annotations
    annotations = {}
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in annotations:
            annotations[img_id] = []
        annotations[img_id].append(ann)

    return data, images, annotations


def select_3_samples(images, annotations):
    """
    Select 3 diverse samples:
        1. SMALL:  Smallest fish bbox area (tests Size-Adaptive ROI)
        2. MURKY:  Lowest average image brightness (tests Water-Net restoration)
        3. CORAL:  Bbox near edge / large context (tests Context Harvester padding)
    """
    candidates = []

    for img_id, anns in annotations.items():
        for ann in anns:
            bbox = ann.get("bbox")
            if not bbox or len(bbox) != 4:
                continue

            x, y, w, h = [float(v) for v in bbox]
            area = w * h
            img_info = images.get(img_id, {})
            frame_w = img_info.get("width", 1920)
            frame_h = img_info.get("height", 1080)
            frame_area = frame_w * frame_h

            # Distance from bbox center to nearest edge
            cx, cy = x + w/2, y + h/2
            edge_dist = min(cx, cy, frame_w - cx, frame_h - cy)

            # Local filename
            local_fname = img_info.get("local_filename", img_id.replace(":", "%3A"))
            img_path = os.path.join(IMAGES_DIR, local_fname)

            candidates.append({
                "image_id": img_id,
                "ann": ann,
                "bbox": [x, y, w, h],
                "area": area,
                "area_ratio": area / frame_area,
                "edge_dist": edge_dist,
                "frame_w": frame_w,
                "frame_h": frame_h,
                "img_path": img_path,
                "local_fname": local_fname,
            })

    if not candidates:
        print("ERROR: No valid candidates found!")
        return []

    # Sort and pick
    # 1. Smallest fish
    by_area = sorted(candidates, key=lambda c: c["area"])
    small_fish = by_area[0]

    # 2. Lowest brightness (proxy for murky water)
    brightness_scores = []
    for c in candidates:
        if os.path.exists(c["img_path"]):
            img = cv2.imread(c["img_path"])
            if img is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                c["brightness"] = float(gray.mean())
                brightness_scores.append(c)
    by_brightness = sorted(brightness_scores, key=lambda c: c["brightness"])
    # Pick the murkiest that isn't the same as small_fish
    murky_fish = next(
        (c for c in by_brightness if c["image_id"] != small_fish["image_id"]), 
        by_brightness[0]
    )

    # 3. Near-edge / near structure (closest to frame boundary)
    by_edge = sorted(candidates, key=lambda c: c["edge_dist"])
    coral_fish = next(
        (c for c in by_edge 
         if c["image_id"] not in (small_fish["image_id"], murky_fish["image_id"])),
        by_edge[0]
    )

    samples = [
        ("SMALL_FISH", small_fish),
        ("MURKY_WATER", murky_fish),
        ("NEAR_CORAL", coral_fish),
    ]

    # Add 2 random samples for stress test
    import random
    existing_ids = {s["image_id"] for _, s in samples}
    remaining = [c for c in candidates if c["image_id"] not in existing_ids]
    random.shuffle(remaining)
    for i in range(2):
        if i < len(remaining):
            samples.append((f"RANDOM_FISH_{i+1}", remaining[i]))

    print(f"Selected {len(samples)} samples:")
    for label, s in samples:
        print(f"  [{label}] {s['local_fname']}")
        print(f"    BBox: {s['bbox']}, Area: {s['area']:.0f}px "
              f"({s['area_ratio']*100:.2f}% of frame)")
        print(f"    Edge dist: {s['edge_dist']:.0f}px, "
              f"Brightness: {s.get('brightness', 'N/A')}")
    
    return samples


def denormalize_tensor(tensor):
    """Convert ImageNet-normalized tensor back to displayable uint8 image."""
    MEAN = np.array([0.485, 0.456, 0.406])
    STD = np.array([0.229, 0.224, 0.225])
    
    img = tensor.permute(1, 2, 0).numpy()  # (H, W, 3)
    img = img * STD + MEAN
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    # RGB -> BGR for OpenCV
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def simple_restore(image):
    """
    Simple underwater restoration (CLAHE-based enhancement).
    Used as a reliable fallback when Water-Net hub loading 
    may have compatibility issues.
    """
    # Convert to LAB colour space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to lightness channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    
    # White balance correction (reduce blue-green cast)
    # Calculate per-channel means and shift toward grey-world assumption
    b_ch, g_ch, r_ch = cv2.split(image)
    avg_b, avg_g, avg_r = b_ch.mean(), g_ch.mean(), r_ch.mean()
    avg_all = (avg_b + avg_g + avg_r) / 3.0
    
    scale_b = avg_all / max(avg_b, 1)
    scale_g = avg_all / max(avg_g, 1) 
    scale_r = avg_all / max(avg_r, 1)
    
    r_corrected = np.clip(r_ch * scale_r, 0, 255).astype(np.uint8)
    g_corrected = np.clip(g_ch * scale_g, 0, 255).astype(np.uint8)
    b_corrected = np.clip(b_ch * scale_b, 0, 255).astype(np.uint8)
    
    white_balanced = cv2.merge([b_corrected, g_corrected, r_corrected])
    
    # Merge CLAHE lightness with white-balanced color
    lab_wb = cv2.cvtColor(white_balanced, cv2.COLOR_BGR2LAB)
    _, a_wb, b_wb = cv2.split(lab_wb)
    enhanced_lab = cv2.merge([l_clahe, a_wb, b_wb])
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # Slight sharpening to enhance edges for DINOv2 patch extraction
    kernel = np.array([
        [0, -0.5, 0],
        [-0.5, 3.0, -0.5],
        [0, -0.5, 0]
    ])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    # Blend: 70% enhanced + 30% sharpened for natural look
    result = cv2.addWeighted(enhanced, 0.7, sharpened, 0.3, 0)
    
    return result


def create_visual_grid(label, sample, harvester, output_dir):
    """
    Create a visual comparison grid for a single sample:
        [Original crop] | [Restored crop] | [ROI] | [Social] | [Habitat] | [Full]

    Also draws the bounding box on the original and restored images.
    """
    img_path = sample["img_path"]
    bbox = sample["bbox"]
    x, y, w, h = [int(v) for v in bbox]

    # Load original image
    original = cv2.imread(img_path)
    if original is None:
        print(f"  ERROR: Could not load {img_path}")
        return None

    print(f"\n  Processing [{label}]: {sample['local_fname']}")
    print(f"    Image shape: {original.shape}")

    # Restore
    restored = simple_restore(original)
    print(f"    Restoration: CLAHE + White Balance + Sharpening applied")

    # Draw bbox on copies for visualization
    orig_vis = original.copy()
    rest_vis = restored.copy()
    cv2.rectangle(orig_vis, (x, y), (x+w, y+h), (0, 255, 0), 3)
    cv2.putText(orig_vis, f"Fish [{w}x{h}]", (x, y-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.rectangle(rest_vis, (x, y), (x+w, y+h), (0, 255, 0), 3)
    cv2.putText(rest_vis, f"Restored [{w}x{h}]", (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Run Context Harvester on the restored image
    streams = harvester.harvest(restored, (x, y, w, h))
    print(f"    Context Harvester: {len(streams)} streams extracted")

    # Denormalize streams for display
    stream_images = {}
    for name, tensor in streams.items():
        stream_img = denormalize_tensor(tensor)
        stream_images[name] = stream_img
        print(f"      {name}: tensor {tuple(tensor.shape)} -> display {stream_img.shape}")

    # Create the grid
    # Row: [Original(resized)] [Restored(resized)] [ROI] [Social] [Habitat] [Full]
    cell_size = 224
    
    # Resize original and restored for the grid
    orig_cell = cv2.resize(orig_vis, (cell_size, cell_size))
    rest_cell = cv2.resize(rest_vis, (cell_size, cell_size))

    cells = [orig_cell, rest_cell]
    stream_order = ["roi", "social", "habitat", "full_frame"]
    for sname in stream_order:
        if sname in stream_images:
            cells.append(cv2.resize(stream_images[sname], (cell_size, cell_size)))

    # Add labels at the top
    label_height = 40
    grid_width = len(cells) * cell_size
    grid_height = cell_size + label_height
    grid = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 30  # dark grey bg

    labels_text = ["Original", "Restored", "ROI (1x)", "Social (3x)", "Habitat (5x)", "Full Frame"]
    for i, (cell, ltxt) in enumerate(zip(cells, labels_text)):
        x_off = i * cell_size
        grid[label_height:label_height+cell_size, x_off:x_off+cell_size] = cell
        # Draw label
        text_size = cv2.getTextSize(ltxt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = x_off + (cell_size - text_size[0]) // 2
        cv2.putText(grid, ltxt, (text_x, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Add sample label at top-left
    cv2.putText(grid, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)

    # Save grid
    out_path = os.path.join(output_dir, f"grid_{label.lower()}.png")
    cv2.imwrite(out_path, grid)
    print(f"    Grid saved: {out_path}")

    # Also save full-res original vs restored side by side
    h_orig = original.shape[0]
    w_orig = original.shape[1]
    comparison = np.zeros((h_orig, w_orig * 2, 3), dtype=np.uint8)
    comparison[:, :w_orig] = orig_vis
    comparison[:, w_orig:] = rest_vis
    comp_path = os.path.join(output_dir, f"compare_{label.lower()}.png")
    cv2.imwrite(comp_path, comparison)
    print(f"    Full-res comparison saved: {comp_path}")

    # Compute restoration metrics
    orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY) 
    rest_gray = cv2.cvtColor(restored, cv2.COLOR_BGR2GRAY)
    orig_brightness = float(orig_gray.mean())
    rest_brightness = float(rest_gray.mean())
    
    # Color cast analysis (R-G-B channel means)
    orig_bgr = [float(original[:,:,i].mean()) for i in range(3)]
    rest_bgr = [float(restored[:,:,i].mean()) for i in range(3)]

    metrics = {
        "brightness_before": round(orig_brightness, 1),
        "brightness_after": round(rest_brightness, 1),
        "brightness_improvement": round(rest_brightness - orig_brightness, 1),
        "color_cast_before": {"B": round(orig_bgr[0],1), "G": round(orig_bgr[1],1), "R": round(orig_bgr[2],1)},
        "color_cast_after": {"B": round(rest_bgr[0],1), "G": round(rest_bgr[1],1), "R": round(rest_bgr[2],1)},
        "green_fog_ratio_before": round(orig_bgr[1] / max(orig_bgr[2], 1), 2),
        "green_fog_ratio_after": round(rest_bgr[1] / max(rest_bgr[2], 1), 2),
    }

    print(f"    Restoration metrics:")
    print(f"      Brightness: {metrics['brightness_before']} -> {metrics['brightness_after']} "
          f"(+{metrics['brightness_improvement']})")
    print(f"      Green/Red ratio: {metrics['green_fog_ratio_before']} -> {metrics['green_fog_ratio_after']}")
    print(f"      Color before (BGR): {metrics['color_cast_before']}")
    print(f"      Color after  (BGR): {metrics['color_cast_after']}")

    # ROI centering check
    roi_bbox_cx = x + w // 2
    roi_bbox_cy = y + h // 2
    print(f"    ROI centering: bbox center=({roi_bbox_cx}, {roi_bbox_cy}), "
          f"frame center=({original.shape[1]//2}, {original.shape[0]//2})")

    return grid, metrics


def main():
    print("=" * 60)
    print("BioReef.ai - Visual Pipeline Audit")
    print("=" * 60)

    # Load subset
    data, images, annotations = load_subset()
    print(f"\nLoaded subset: {len(images)} images, "
          f"{sum(len(v) for v in annotations.values())} annotations")

    # Select 3 samples
    samples = select_3_samples(images, annotations)
    if not samples:
        return

    # Initialize Context Harvester
    harvester = ContextHarvester(
        crop_scales=[1, 3, 5],
        target_resolution=224,
        small_object_threshold=0.05,
        highres_initial=512,
        include_full_frame=True,
    )

    # Process each sample
    all_grids = []
    all_metrics = {}
    for label, sample in samples:
        grid, metrics = create_visual_grid(label, sample, harvester, OUTPUT_DIR)
        if grid is not None:
            all_grids.append(grid)
            all_metrics[label] = metrics

    # Create combined grid (all 3 stacked)
    if all_grids:
        max_w = max(g.shape[1] for g in all_grids)
        padded = []
        for g in all_grids:
            if g.shape[1] < max_w:
                pad = np.ones((g.shape[0], max_w - g.shape[1], 3), dtype=np.uint8) * 30
                g = np.hstack([g, pad])
            padded.append(g)
        combined = np.vstack(padded)
        combined_path = os.path.join(OUTPUT_DIR, "combined_audit_grid.png")
        cv2.imwrite(combined_path, combined)
        print(f"\n[COMBINED] Grid saved: {combined_path}")

    # Save metrics
    metrics_path = os.path.join(OUTPUT_DIR, "audit_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"[METRICS] Saved: {metrics_path}")

    # Print validation summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    for label, m in all_metrics.items():
        green_reduced = m["green_fog_ratio_after"] < m["green_fog_ratio_before"]
        brightness_improved = m["brightness_after"] >= m["brightness_before"]
        print(f"\n  [{label}]")
        print(f"    Green-fog reduction: {'YES' if green_reduced else 'NO'} "
              f"(G/R ratio: {m['green_fog_ratio_before']} -> {m['green_fog_ratio_after']})")
        print(f"    Brightness improved: {'YES' if brightness_improved else 'NO'} "
              f"({m['brightness_before']} -> {m['brightness_after']})")


if __name__ == "__main__":
    main()
