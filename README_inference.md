# Weed Targeting Strategy Checkpoint

This document describes the current "Laser Targeting Logic" used in the Harvested Robotics pipeline. This strategy was finalized after verifying that it successfully targets the **stem base** of the weeds rather than leaf tips or geometric centers.

**Date**: 2026-02-08
**Core Objective**: Identify the optimal "kill point" for a laser weeder, prioritizing the structural center (hub) and the lowest part of the stem (closest to the ground).

---

## 1. Targeting Algorithm: "Thickest & Lowest Hub"

The logic is implemented in `src/utils.py` -> `get_skeleton_center`.

### Step-by-Step Process:

1.  **Input**: A binary mask of a single weed instance (from YOLOv8-seg).
2.  **Distance Transform**: 
    *   Compute `cv2.distanceTransform(mask, cv2.DIST_L2, 5)`.
    *   This generates a map where each pixel's value represents its distance to the nearest background pixel (effectively the failing "thickness" of the plant at that point).
3.  **Skeletonization**: 
    *   Compute `skimage.morphology.skeletonize(mask)`.
    *   This reduces the plant shape to its 1-pixel wide structural core (skeleton).
4.  **Feature Extraction**: 
    *   Extract the thickness values *only at the skeleton pixel locations*.
    *   This gives us the thickness of the "branches" vs the "trunk".
5.  **Candidate Selection (The "Hub")**: 
    *   Identify the maximum thickness found on the skeleton (the `max_dist`).
    *   Filter for skeleton pixels that are at least **90% of this maximum thickness**.
    *   This isolates the central "hub" or main stem, ignoring thin leaves or branches.
6.  **Target Refinement (The "Base")**: 
    *   From the candidates identified in step 5, select the one with the **MAXIMUM Y-COORDINATE** (the lowest point in the image).
    *   This biases the final target point towards the base of the stem where it enters the ground.

### Why this works:
*   **vs. Centroid**: The geometric centroid often lands in the middle of leaf clusters or empty space for "C-shaped" weeds. This logic stays on the plant structure.
*   **vs. Skeleton Mean**: The mean of the skeleton can be pulled up by long leaves. Our logic prioritizes the *thickest* part.
*   **vs. Lowest Pixel**: The absolute lowest pixel might be a drooping leaf tip. Our logic only considers the *thick structural core*.

---

## 2. Segmentation Strategy: Memory-Optimized Tiling

To handle high-resolution field images without crashing (OOM), we use a strictly **Tiled Processing** approach in `src/inference.py`.

*   **No Full-Res Masks**: We *never* allocate a full-resolution (e.g., 8K) mask array for the entire image.
*   **Tile-Local Logic**: 
    1.  Crop the image into 4x4 tiles.
    2.  For each tile:
        *   Run YOLOv8-seg inference.
        *   Generate a local "Crop Safety Mask" for that tile (protecting crops).
        *   Subtract safety mask from weed mask locally.
        *   Find weed instances and calculate target points locally.
        *   Convert results to **Global Coordinates** (contours and points) and store them in a lightweight list.
        *   **IMMEDIATELY DELETE** the heavy tile masks and run garbage collection.
*   **Visualization**: Reconstructs the visual output by drawing the stored global contours and points onto the original image.

This ensures the pipeline uses minimal RAM regardless of the input image size.

---

## 3. Visualization

*   **Red Dots (Radius 5)**: Represent the calculated laser target points.
*   **Weed Contours**: Red outlines showing the segmented weed area.
*   **Crop Protection**: Blue boxes and Cyan circles representing the protected crop zones.
