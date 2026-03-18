# --------------------------------------------------------------------------- #
# Simple demo script for ProxiVideoFriends multitask model from a short video
# --------------------------------------------------------------------------- #
# Example:
#
# cd scripts
# python3 run_demo.py \
#   --videoPath /path/to/video.mp4 \
#   --model_dir best_model_multitask/ \
#   --outputDir ../../demo_output
# --------------------------------------------------------------------------- #

import os
import sys
import json
import argparse
import warnings
import torch

warnings.filterwarnings("ignore")
from utils.utils_demo import get_video_frames_24fps, detect_persons_and_save_clippings, generate_model_input, decode_predictions

sys.path.append("..")

from evaluation.evaluator import load_saved_config, load_best_checkpoint, PROXEMICS_CLASS_NAMES, RELATIONSHIP_CLASS_NAMES
from train.model import MultiTaskCLStoken


def parse_args():
    parser = argparse.ArgumentParser(description="Demo script for a trained multitask model")
    parser.add_argument("--videoPath", type=str, required=True, help="Path to input short video")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to trained model directory")
    parser.add_argument("--checkpoint_name", type=str, choices=["model_best.pt", "model_last.pt"], default="model_best.pt")
    parser.add_argument("--outputDir", type=str, required=True, help="Directory where demo outputs will be saved")
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
args = parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

videoPath = args.videoPath
model_dir = args.model_dir
ckpt_path = os.path.join(model_dir, args.checkpoint_name)
outputDir = args.outputDir

framesDir = os.path.join(outputDir, "frames_24fps")
clippingsDir = os.path.join(outputDir, "clippings")
jsonOutput = os.path.join(outputDir, "demo_prediction.json")

os.makedirs(outputDir, exist_ok=True)

print("\n" + "=" * 70)
print("Demo configuration")
print("=" * 70)
print(f"Video path     : {videoPath}")
print(f"Checkpoint     : {ckpt_path}")
print(f"Output dir     : {outputDir}")


print("\n" + "=" * 70)
print("Frames extraction and clippings generation")
print("=" * 70)
# 1. Save frames at 24 fps
print("[INFO] Extracting frames...")
get_video_frames_24fps(videoPath, framesDir)
# 2. Detect 2 persons and generate clippings
print("[INFO] Detecting people and generating clippings...")
frame_info = detect_persons_and_save_clippings(framesDir=framesDir,clippingsDir=clippingsDir)


# 3. Load config
print("\n" + "=" * 70)
print("Model loading")
print("=" * 70)
print("[INFO] Loading model config...")
cfg = load_saved_config(model_dir)

if cfg["task"] != "multitask":
    raise ValueError("This demo script expects a multitask model.")

# 4. Build model
print("[INFO] Building model...")
model = MultiTaskCLStoken(typeImg=cfg["typeImg"],onlyPairRGB=cfg["onlyPairRGB"],onlyPairPose=False,nlayersFreeze=cfg["nlayersFreeze"],audio=cfg["audio"])
model.to(device)
# 5. Load checkpoint
print(f"    [INFO] Loading checkpoint: {ckpt_path}")
load_best_checkpoint(model, ckpt_path, device)
model.eval()


# 6. Generate model input
print("\n" + "=" * 70)
print("Input generation")
print("=" * 70)
print("[INFO] Building model input...")
X = generate_model_input(clippingsDir=clippingsDir, frames_per_sequence=cfg["frames_per_sequence"],backbone=cfg["backbone"],onlyPairRGB=cfg["onlyPairRGB"])
X = [x.to(device) for x in X]


# 7. Prediction
print("\n" + "=" * 70)
print("Inference")
print("=" * 70)
print("[INFO] Running inference...")
with torch.no_grad():
    output = model(X)

results = decode_predictions(output)

# 8. Save results
final_output = {
    "videoPath": videoPath,
    "model_dir": model_dir,
    "checkpoint_name": args.checkpoint_name,
    "config": {
        "task": cfg["task"],
        "backbone": cfg["backbone"],
        "fusion": cfg["fusion"],
        "frames_per_sequence": cfg["frames_per_sequence"],
        "onlyPairRGB": cfg["onlyPairRGB"],
        "audio": cfg["audio"]
    },
    "n_valid_frames": len(frame_info),
    "prediction": results
}

with open(jsonOutput, "w") as f:
    json.dump(final_output, f, indent=4)

print("\n" + "=" * 70)
print("Demo prediction")
print("=" * 70)
print(json.dumps(results, indent=4))
print(f"\n[INFO] Saved output JSON to: {jsonOutput}")
print(f"[INFO] Saved clippings to : {clippingsDir}")