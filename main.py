
from src.inference import run_inference
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Harvested Robotics Instance Segmentation Pipeline")
    parser.add_argument("--input_dir", type=str, default="data/raw", help="Path to input images directory")
    parser.add_argument("--output_dir", type=str, default="results", help="Path to output directory")
    parser.add_argument("--weights", type=str, default="models/weights/best.pt", help="Path to model weights")
    
    args = parser.parse_args()
    
    run_inference(args.input_dir, args.output_dir, args.weights)
