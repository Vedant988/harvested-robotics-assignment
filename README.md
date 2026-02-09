# Instance-Level Weed & Crop Segmentation

>  **Technical Report & Outcomes**
>
> For a detailed breakdown of the methodology, architectural decisions, and visual performance metrics, please refer to the **`docs/`** directory in the repository:
> [**View Technical Report & Outcomes**](https://github.com/Vedant988/harvested-robotics-assignment/tree/main/docs)

##  Submission Overview

This repository contains the complete solution for the **Harvested Robotics** technical assessment R2. The pipeline performs robust instance-level segmentation of crops and weeds, designed specifically for laser weeding applications.

## ✨ Key Features

*  **Pure Deep Learning**
    Utilizes **YOLOv8-seg** for robust, high-resolution weed and crop segmentation.
*  **Precision Targeting**
    Implements advanced skeleton-based targeting logic to identify the specific stem base of the weed (see [`README_inference.md`](https://github.com/Vedant988/harvested-robotics-assignment/blob/main/README_inference.md) for strategy details).
*  **Safety Critical**
    Implements **Circular Safety Exclusion Zones** around detected crops to ensure **0% crop damage** during laser actuation.
*  **Dockerized & Portable**
    Fully containerized environment ensuring guaranteed reproducibility across any machine.
*  **Memory Optimized**
    Uses **Dynamic Tiling** to process high-resolution field images without encountering Out-Of-Memory (OOM) errors.

---

## ⚡ How to Run (One-Click)

The submission includes an interactive script `run_pipeline.bat` that handles environment setup and execution for you.

### Prerequisites
* **Git** (Required for cloning)
* **Docker Desktop** (Option 1)
    **OR**
* **Python 3.8+** (Recommended Option 2)

### Steps
1.  **Clone the repository and navigate to the project folder:**
    ```powershell
    git clone https://github.com/Vedant988/harvested-robotics-assignment.git
    cd harvested-robotics-assignment
    ```

2.  *(Optional)* Place your test images in the `data/raw/` folder (11 samples are provided by default).

3.  **Run the pipeline script:**
    ```powershell
    .\run_pipeline.bat
    ```

4.  Select your preferred mode when prompted:

#### Option 1: Docker Mode (Best for Reproducibility)
Executes the pipeline within an isolated container to ensure an identical runtime environment to development.

> **Note:** This method guarantees cross-platform consistency and automatically handles complex dependencies (CUDA, PyTorch). However, it requires **Docker Desktop** to be running and necessitates an initial download of the base image (~14GB).

#### Option 2: Local Python / VENV Mode (Fastest & Recommended)
Automatically provisions a local virtual environment (`venv`) to run the pipeline directly on your hardware.

> **Note:** This is the quickest deployment method, bypassing virtualization overhead to start in seconds. It requires an existing **Python 3.8+** installation and relies on your local system drivers for GPU acceleration.

---

##  Output Format

Processed images are saved to the `results/` directory with the following visualization logic:

| Indicator | Visual Element | Description |
| :--- | :--- | :--- |
| **Crop** | **Blue Box** | YOLOv8 Crop Detection. |
| **Safety Zone** | **Cyan Circle** | Computed Exclusion Zone (Radius = Half-Diagonal + Safety Margin). |
| **Target** | **Red Contour** | Weed instances targeted for laser actuation (strictly outside the safety zone). |

##  Technical Constraints Handling

* **Unseen Data:** The pipeline is agnostic to image dimensions and lighting conditions, utilizing dynamic thresholding to adapt to various field environments.
* **Inference Speed:** The architecture is optimized for batch processing with CUDA support (automatically detected if available).
