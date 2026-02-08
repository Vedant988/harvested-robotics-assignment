
import cv2
import numpy as np
import os

def slice_image(image, grid_size=(4, 4)):
    """
    Slices an image into grid_size (rows, cols) tiles.
    Returns:
        tiles (list of numpy arrays): The image tiles.
        tile_coords (list of tuples): (y, x) top-left coordinates for each tile.
    """
    h, w = image.shape[:2]
    grid_rows, grid_cols = grid_size
    
    tile_h = h // grid_rows
    tile_w = w // grid_cols
    
    tiles = []
    tile_coords = []
    
    for r in range(grid_rows):
        for c in range(grid_cols):
            y_start = r * tile_h
            x_start = c * tile_w
            
            # Handle edge cases for last row/col to include remainder pixels
            y_end = (r + 1) * tile_h if r < grid_rows - 1 else h
            x_end = (c + 1) * tile_w if c < grid_cols - 1 else w
            
            tile = image[y_start:y_end, x_start:x_end]
            tiles.append(tile)
            tile_coords.append((y_start, x_start))
            
    return tiles, tile_coords, (tile_h, tile_w)

def merge_masks(tile_masks, tile_coords, original_shape):
    """
    Merges local tile masks back into a global mask.
    Args:
        tile_masks (list): List of binary masks for each tile.
        tile_coords (list): List of (y, x) top-left coordinates.
        original_shape (tuple): (H, W) of the original image.
    Returns:
        full_mask (numpy.ndarray): Reconstructed global mask.
    """
    full_mask = np.zeros(original_shape[:2], dtype=np.uint8)
    
    for mask, (y, x) in zip(tile_masks, tile_coords):
        if mask is None:
            continue
            
        h, w = mask.shape[:2]
        full_mask[y:y+h, x:x+w] = mask
        
    return full_mask
