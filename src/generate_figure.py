
import cv2
import numpy as np
import os
import sys
from ultralytics import YOLO

# Add relevant paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import create_crop_mask_from_bboxes, get_skeleton_center
from src.tiling import slice_image, merge_masks

def generate_figure():
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    img_path = os.path.join(base_dir, "data", 'raw', "UFDjjwheat_RGB_0.0_0.0_20251201_153814675.webp")
    crop_weights = os.path.join(base_dir, "models", "weights", "best.pt")
    weed_weights = os.path.join(base_dir, "models", "weights", "yolov8l-seg.pt")
    output_path = os.path.join(base_dir, "docs", "figures", "weed_segmentation_only.png")
    
    # Load Models
    print("Loading models...")
    crop_model = YOLO(crop_weights)
    weed_model = YOLO(weed_weights)
    
    # Load Image
    print(f"Loading image from {img_path}...")
    img = cv2.imread(img_path)
    if img is None:
        print("Error: Image not found!")
        return

    # Phase 1: Crop Detection (Safety)
    print("Running Crop Detection...")
    crop_results = crop_model(img, verbose=False, conf=0.5)[0]
    crop_boxes = crop_results.boxes.xyxy.cpu().numpy()
    
    # Phase 2: Weed Segmentation (Tiled)
    print("Running Weed Segmentation...")
    tiles, tile_coords, _ = slice_image(img, grid_size=(4, 4))
    
    all_contours = []
    
    for i, tile in enumerate(tiles):
        ty, tx = tile_coords[i]
        th, tw = tile.shape[:2]
        
        # Run Weed Model
        results = weed_model(tile, verbose=False, conf=0.5)[0]
        
        if not results.masks:
            continue
            
        # Combine masks
        tile_weed_mask = np.zeros((th, tw), dtype=np.uint8)
        for mask_data in results.masks.data:
            m = mask_data.cpu().numpy().astype(np.uint8) * 255
            if m.shape != (th, tw):
                m = cv2.resize(m, (tw, th), interpolation=cv2.INTER_NEAREST)
            tile_weed_mask = cv2.bitwise_or(tile_weed_mask, m)
            
        # Safety Subtraction
        local_boxes = []
        for box in crop_boxes:
            bx1, by1, bx2, by2 = box
            ix1 = max(bx1, tx)
            iy1 = max(by1, ty)
            ix2 = min(bx2, tx + tw)
            iy2 = min(by2, ty + th)
            
            if ix1 < ix2 and iy1 < iy2:
                local_boxes.append([ix1 - tx, iy1 - ty, ix2 - tx, iy2 - ty])
                
        if local_boxes:
            tile_crop_mask = create_crop_mask_from_bboxes((th, tw), np.array(local_boxes), safety_margin=20)
            tile_weed_mask = cv2.subtract(tile_weed_mask, tile_crop_mask)
            
        # Find Contours
        contours, _ = cv2.findContours(tile_weed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cnt += (tx, ty)
            all_contours.append(cnt)

    # Visualization: Side-by-Side Comparison
    print("Generating side-by-side visualization...")
    viz_img = img.copy()
    
    # Draw Green Contours for Weeds
    overlay = viz_img.copy()
    cv2.drawContours(overlay, all_contours, -1, (0, 255, 0), -1) # Fill
    alpha = 0.4
    viz_img = cv2.addWeighted(overlay, alpha, viz_img, 1 - alpha, 0)
    cv2.drawContours(viz_img, all_contours, -1, (0, 255, 0), 2)
    
    # Create Side-by-Side (Horizontal Stack)
    # Resize both to fit if they are huge? No, keep original resolution, just stack.
    combined_img = np.hstack((img, viz_img))
    
    print(f"Saving to {output_path}...")
    cv2.imwrite(output_path, combined_img)
    print("Done!")

if __name__ == "__main__":
    generate_figure()
