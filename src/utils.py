import cv2
import numpy as np
from skimage.morphology import skeletonize

def get_skeleton_center(mask_roi):
    """
    Finds the 'Hub' of the weed (Stem/Center) by combining Skeletonization
    and Distance Transform.
    Target = The point on the skeleton that is THICKEST (center of mass)
             and LOWEST (closest to ground).
    """
    # Ensure mask is binary and uint8
    mask_u8 = mask_roi.astype(np.uint8)
    if mask_u8.max() > 1:
        mask_u8 = (mask_u8 > 0).astype(np.uint8)
        
    binary = mask_u8 > 0
    
    # 1. Skeletonize
    if not np.any(binary):
        return None
        
    skeleton = skeletonize(binary)
    y_idxs, x_idxs = np.where(skeleton)
    
    if len(y_idxs) == 0:
        return None

    # 2. Distance Transform (Thickness at each point)
    # We use the original mask (not skeleton) to get thickness
    dist_map = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    
    # 3. Get Distance values ONLY at Skeleton points
    # We want to find which part of the skeleton is the "thickest" (the hub)
    skel_dists = dist_map[y_idxs, x_idxs]
    
    if len(skel_dists) == 0:
        return None
        
    # 4. Find candidates with Maximum Thickness
    # (Use a small threshold, e.g., 90% of max, to handle noise)
    max_dist = np.max(skel_dists)
    # Filter: keep points that are "very thick" (e.g. within 10% of max thickness)
    thick_threshold = max_dist * 0.90
    
    candidate_indices = np.where(skel_dists >= thick_threshold)[0]
    
    # Get coordinates of these thick candidates
    cand_y = y_idxs[candidate_indices]
    cand_x = x_idxs[candidate_indices]
    
    # 5. Break Ties: Choose the LOWEST (Max Y) point among the thickest
    # This biases towards the base of the stem if the stem is uniformly thick
    best_idx = np.argmax(cand_y)
    
    target_x = cand_x[best_idx]
    target_y = cand_y[best_idx]
    
    return (int(target_x), int(target_y))


def create_crop_mask_from_bboxes(image_shape, boxes, safety_margin=20):
    """
    Create a binary mask from YOLO bounding boxes using CIRCULAR protection zones.
    Radius = (Diagonal of Box / 2) + Safety Margin.
    
    Args:
        image_shape (tuple): (H, W) or (H, W, C).
        boxes (list): List of [x1, y1, x2, y2] coordinates.
        safety_margin (int): Extra pixels to expand the radius for safety.
        
    Returns:
        numpy.ndarray: Binary mask (255 for crop, 0 for background).
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        
        # Calculate Center
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        
        # Calculate Radius (Half Diagonal)
        w = x2 - x1
        h = y2 - y1
        diagonal = np.sqrt(w**2 + h**2)
        radius = int((diagonal / 2) + safety_margin)
        
        # Draw Circle
        cv2.circle(mask, (cx, cy), radius, 255, -1)
        
    return mask
