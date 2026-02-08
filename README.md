# Instance-Level Weed & Crop Segmentation

## Submission Overview
This repository contains the complete solution for the **Harvested Robotics** technical assessment. The pipeline performs robust instance-level segmentation of crops and weeds, designed specifically for laser weeding applications.

## Key Features
*   **Pure Deep Learning**: Uses **YOLOv8-seg** for robust, high-resolution weed segmentation.
*   **Precision Targeting**: Implements advanced skeleton-based targeting logic to hit the weed stem base (see [Detailed Strategy](README_inference.md)).
*   **Safety Critical**: Implements **Circular Safety Exclusion Zones** around detected crops to ensure 0% crop damage from laser actuation.
*   **Dockerized & Portable**: Fully containerized environment ensuring reproducibility across any machine.
*   **Memory Optimized**: Uses **Dynamic Tiling** to process high-resolution field images without OOM errors.

---

##  How to Run (One-Click)
The submission includes an interactive script `run_pipeline.bat` that handles environment setup for you.

### Prerequisite
*   **Docker Desktop** (Recommended) - OR - **Python 3.8+**

### Steps
1.  Place your test images in `data/raw/` (50 samples provided).
2.  Type on terminal **`.\run_pipeline.bat`**.
3.  Select your preferred mode:
    *  ### Option 1: **Docker Mode** (Recommended for Guaranteed Reproducibility)
- **Concept**: Runs inside an isolated container with all dependencies pre-installed by the base image.
- **Pros**:
    - **Guaranteed to Work**: Eliminates "it works on my machine" issues.
    - **Zero Configuration**: Handles complex dependencies like CUDA, PyTorch, and OpenCV automatically.
    - **Cross-Platform**: Consistent behavior on Windows, Linux, and Mac.
- **Cons**:
    - **Initial Download**: Docker must pull the `ultralytics` base image (~14GB) on the first run.
    - **Virtualization**: Requires Docker Desktop to be running.

### Option 2: **Local Python / VENV Mode** (Recommended - Fast & Easy)
- **Concept**: Automatically creates a local virtual environment and runs the pipeline.
- **Pros**:
    - **Fastest Setup**: No 14GB Docker image downloads. Ready in seconds.
    - **Easy**: The script handles `venv` creation and dependency installation automatically.
    - **Native Performance**: Runs directly on your hardware.
- **Cons**:
    - **Prerequisites**: Requires Python 3.8+ installed.
    - **Dependency Risks**: Minimal (venv isolates dependencies), but relies on system drivers.

##  Output Format
Processed images are saved to the `results/` directory with the following visualization:
*   **Blue Bounding Box**: YOLOv8 Crop Detection.
*   **Cyan Circle**: Computed Safety Exclusion Zone (Radius = Half-Diagonal + Safety Margin).
*   **Red Contour**: Weed Instances targeted for laser actuation (outside the safety zone).

## Technical Constraints Handling
*   **Unseen Data**: The pipeline is agnostic to image dimensions and lighting conditions (using dynamic thresholding).
*   **Inference Speed**: Optimized for batch processing with CUDA support (if available).
