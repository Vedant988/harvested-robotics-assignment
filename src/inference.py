import cv2
import numpy as np
import os
import argparse
from ultralytics import YOLO
import sys

# Add the project root directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import create_crop_mask_from_bboxes, get_skeleton_center
from src.tiling import slice_image, merge_masks

def separate_instances(mask):
    """
    Separate connected components and find the SKELETON target point.
    """
    # 1. Morphological Opening
    kernel = np.ones((3,3), np.uint8)
    opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # 2. Connected Components
    num_labels, labels, stats, centroids_data = cv2.connectedComponentsWithStats(opened_mask, connectivity=8)
    
    instances = []
    target_points = [] 
    
    # Label 0 is background
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Filter noise
        if area < 50: 
            continue
            
        # --- Create ROI for this instance ---
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        
        # Extract binary mask ROI
        instance_roi = (labels[y:y+h, x:x+w] == i).astype(np.uint8) * 255
        
        # Reconstruct full mask
        full_instance_mask = np.zeros_like(mask)
        full_instance_mask[y:y+h, x:x+w] = instance_roi
        instances.append(full_instance_mask)
        
        # --- Calculate SKELETON Target ---
        # Use simple centroid of the skeleton
        local_point = get_skeleton_center(instance_roi)
        
        if local_point:
            # Convert local ROI to Global
            global_cx = local_point[0] + x
            global_cy = local_point[1] + y
            target_points.append((global_cx, global_cy))
        else:
            # Fallback
            cx, cy = centroids_data[i]
            target_points.append((int(cx), int(cy)))
        
    return instances, target_points

def run_inference(input_dir, output_dir, crop_weights, weed_weights, conf_threshold):
    # 1. Load Crop Model (Global Detection)
    print(f"Loading Crop Model from {crop_weights}...")
    crop_model = YOLO(crop_weights)
    
    # 2. Load Weed Model (Tiled Segmentation) - REQUIRED
    if not os.path.exists(weed_weights):
        raise FileNotFoundError(f"Weed Segmentation Model not found at {weed_weights}. Please provide the model file.")
        
    print(f"Loading Weed Segmentation Model from {weed_weights}...")
    weed_model = YOLO(weed_weights)

    if os.path.exists(output_dir):
        import shutil
        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
            try:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            except Exception as e:
                print(f"Failed to delete {item_path}. Reason: {e}")
    else:
        os.makedirs(output_dir)
        
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
    print(f"Found {len(image_files)} images.")
    
    for filename in image_files:
        try:
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            print(f"Processing {filename}...")
            # Resize strictly for visualization/display if needed, but we keep full res for processing

            # --- Phase A: Crop Detection (Global) ---
            # We predict on full image (YOLO handles resizing internally)
            crop_results = crop_model(img, verbose=False, conf=conf_threshold)[0]
            crop_boxes = crop_results.boxes.xyxy.cpu().numpy() # Global coords
            
            # --- Phase B: Tile-Based Processing (Memory Optimized) ---
            # Instead of creating a full-size mask, we process everything per-tile
            tiles, tile_coords, _ = slice_image(img, grid_size=(4, 4))
            
            all_contours = []
            all_centroids = []
            
            for i, tile in enumerate(tiles):
                ty, tx = tile_coords[i]
                th, tw = tile.shape[:2]
                
                # 1. Weed Segmentation on Tile
                results = weed_model(tile, verbose=False, conf=conf_threshold)[0]
                
                if not results.masks:
                    continue
                    
                # Merge masks for this tile
                tile_weed_mask = np.zeros((th, tw), dtype=np.uint8)
                for mask_data in results.masks.data:
                    m = mask_data.cpu().numpy().astype(np.uint8) * 255
                    if m.shape != (th, tw):
                        m = cv2.resize(m, (tw, th), interpolation=cv2.INTER_NEAREST)
                    tile_weed_mask = cv2.bitwise_or(tile_weed_mask, m)
                
                # 2. Crop Safety Mask (Local for this tile)
                # Adjust global boxes to local tile coordinates
                local_boxes = []
                for box in crop_boxes:
                    bx1, by1, bx2, by2 = box
                    # Intersection logic
                    ix1 = max(bx1, tx)
                    iy1 = max(by1, ty)
                    ix2 = min(bx2, tx + tw)
                    iy2 = min(by2, ty + th)
                    
                    if ix1 < ix2 and iy1 < iy2:
                        # Box overlaps with tile. Convert to local coords.
                        local_boxes.append([ix1 - tx, iy1 - ty, ix2 - tx, iy2 - ty])
                
                if local_boxes:
                    tile_crop_mask = create_crop_mask_from_bboxes((th, tw), np.array(local_boxes), safety_margin=20)
                    tile_weed_mask = cv2.subtract(tile_weed_mask, tile_crop_mask)
                
                # 3. Instance Separation & Targeting (Local)
                # separate_instances returns coordinates relative to the MASK passed in (local tile).
                # We need to add (tx, ty) to them later.
                
                local_instances, local_points = separate_instances(tile_weed_mask)
                
                # 4. Convert to Global
                for inst_mask, point in zip(local_instances, local_points):
                    # Convert contour
                    contours, _ = cv2.findContours(inst_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in contours:
                        cnt += (tx, ty) # Offset coordinates
                        all_contours.append(cnt)
                    
                    # Convert Point
                    if point:
                        gx = point[0] + tx
                        gy = point[1] + ty
                        all_centroids.append((gx, gy))

                # Cleanup per tile
                del tile_weed_mask
                del results
                import gc
                gc.collect()

            # --- Visualization (On Global Image) ---
            viz_img = img.copy()
            
            # Draw Crops (Commented out for speed/clarity)
            # for box in crop_boxes:
            #     x1, y1, x2, y2 = map(int, box)
            #     cv2.rectangle(viz_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
            #     cx = (x1 + x2) // 2
            #     cy = (y1 + y2) // 2
            #     w = x2 - x1
            #     h = y2 - y1
            #     diagonal = np.sqrt(w**2 + h**2)
            #     radius = int((diagonal / 2) + 20)
            #     cv2.circle(viz_img, (cx, cy), radius, (255, 255, 0), 2)

            # # Draw Weeds (Global Contours) (Commented out)
            # cv2.drawContours(viz_img, all_contours, -1, (0, 0, 255), 2)
            
            # Draw Targets (Active)
            for (cx, cy) in all_centroids:
                cv2.circle(viz_img, (cx, cy), 5, (0, 0, 255), -1) # Made larger (5) and Red for visibility

            save_path = os.path.join(output_dir, f"result_{filename}")
            cv2.imwrite(save_path, viz_img)
            
            # Cleanup Image
            del img
            del viz_img
            gc.collect()

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()

    print("Inference completed.")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default=os.path.join(BASE_DIR, "data", "raw"), help="Input directory")
    parser.add_argument("--output_dir", default=os.path.join(BASE_DIR, "inference_results"), help="Output directory")
    parser.add_argument("--weights", default=os.path.join(BASE_DIR, "models", "weights", "best.pt"), help="Path to best.pt (Crops)")
    parser.add_argument("--weed_weights", default=os.path.join(BASE_DIR, "models", "weights", "yolov8l-seg.pt"), help="Path to yolov8l-seg.pt (Weeds)")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold (0.0 - 1.0)")
    args = parser.parse_args()
    
    run_inference(args.input_dir, args.output_dir, args.weights, args.weed_weights, args.conf)
