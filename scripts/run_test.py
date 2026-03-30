# ---------------------------------------------------------------------------
# PyTorch test / evaluation script for ProxemicsNet++ model trained with video sequences and optional audio branch.
# ---------------------------------------------------------------------------

import os
import sys
import argparse

import torch
from torch.utils.data import DataLoader
import wandb

sys.path.append("..")
from evaluation.evaluator import *
from train.model import TemporalBranch, MultiTaskCLStoken, MultiTaskCrossAttention, MViTv2SmallBackbone
from dataset_utils.load_sequences_from_episodes import load_sequences_from_episodes, filter_set
from dataset_utils.dataGeneratorSequences import PyTorchSequenceDataset, BalancedSampler

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Test a trained model from its saved directory")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to trained model directory")
    parser.add_argument("--checkpoint_name", type=str, choices=["model_best.pt", "model_last.pt"], default="model_best.pt")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
args = parse_args()

# -----------------------------
# Basic paths and device
# -----------------------------
model_dir = args.model_dir
ckpt_path = os.path.join(model_dir, args.checkpoint_name)
test_output_dir = os.path.join(model_dir, "test_results")

device = torch.device(args.device if torch.cuda.is_available() else "cpu")

# -----------------------------
# Basic paths and device
# -----------------------------
print(f"[INFO] Loading config from: {model_dir}")

cfg = load_saved_config(model_dir)
shots_dir = os.path.join(cfg["datasetDir"], 'shots/season3')
labels_dir = os.path.join(cfg["datasetDir"], 'labels/')

base_dataset_path = os.path.join(cfg["datasetDir"], 'preprocessed_dataset')

base_audio_path = os.path.join(cfg["datasetDir"], 'labeled_wavs') 
audio_emmbeddings_dir = os.path.join(cfg["datasetDir"], 'audio_embeddings')
multitask=True
if cfg["task"] == "proxemics":
    multitask=False

# -----------------------------
# Print evaluation summary
# -----------------------------
print("\n" + "=" * 70)
print("Evaluation configuration")
print("=" * 70)
print(f"Device               : {device}")
print(f"Model dir            : {model_dir}")
print(f"Checkpoint           : {ckpt_path}")
print(f"Task                 : {cfg['task']}")
print(f"Backbone             : {cfg['backbone']}")
print(f"Fusion               : {cfg.get('fusion', 'N/A')}")
print(f"Batch size           : {cfg['batch_size']}")
print(f"Frames per sequence  : {cfg['frames_per_sequence']}")
print(f"Stride               : {cfg['stride']}")
print(f"Audio branch         : {cfg['audio']}")
print(f"Only Pair RGB        : {cfg['onlyPairRGB']}")
print(f"Dataset dir          : {cfg['datasetDir']}")
print(f"Test output dir      : {test_output_dir}")
print("=" * 70 + "\n")


# -----------------------------------------------------------------------
# 1. Load test split
# -----------------------------------------------------------------------
print("\n" + "=" * 70)
print("Dataset")
print("=" * 70)

print("[INFO] Loading test sequences...")
train_set, val_set, test_set = load_sequences_from_episodes(shots_dir, labels_dir, cfg["set"], frames_per_sequence=cfg["frames_per_sequence"], stride=cfg["stride"])
test_set=filter_set(test_set)
print(f"[INFO] Test sequences: {len(test_set)}")


# -----------------------------------------------------------------------
# 1.1. Audio embeddings (optional)
# -----------------------------------------------------------------------
audio_embeddings_path=""
if cfg["audio"]:
    #Check if audio embeddings already exist for the current configuration
    audio_embeddings_path = os.path.join(audio_emmbeddings_dir, f'window_{cfg["frames_per_sequence"]}_stride_{cfg["stride"]}')
    print("\n[INFO] Generating / loading audio embeddings...")
    print(f"[INFO] Audio embeddings path: {audio_embeddings_path}")
    from dataset_utils.generate_audio_embeddings import generate_audio_embeddings
    generate_audio_embeddings(test_set, base_audio_path, audio_embeddings_path, fps=24, model_name="base")

# -----------------------------------------------------------------------
# 2. Build datasets and dataloaders
# -----------------------------------------------------------------------
print("\n" + "=" * 70)
print("DataLoaders")
print("=" * 70)
print("[INFO] Creating datasets and dataloaders...")
#Datagenerator Parameters
params = dict(base_dataset_path=base_dataset_path,typeImg=cfg["typeImg"],onlyPairRGB=cfg["onlyPairRGB"],onlyPairPose=False,frames_per_sequence=cfg["frames_per_sequence"], use_backbone=cfg["backbone"], audio=cfg["audio"], audio_embeddings_path=audio_embeddings_path, multitask=multitask)
#generetors
test_ds = PyTorchSequenceDataset(list_sequences=test_set,augmentation=False, **params)
test_loader   = DataLoader(test_ds,   batch_size=cfg["batch_size"], shuffle=False, num_workers=4, pin_memory=True)


# -----------------------------------------------------------------------
# 3. Build model
# -----------------------------------------------------------------------
print("\n" + "=" * 70)
print("Model")
print("=" * 70)
print("[INFO] Building model...")

if cfg["task"] != "multitask":
    if cfg["backbone"] == "ResNet18":
        model = TemporalBranch(typeImg=cfg["typeImg"], onlyPairRGB=cfg["onlyPairRGB"], onlyPairPose=False,nlayersFreeze=cfg["nlayersFreeze"], audio=cfg["audio"])  #model returns raw logits (no sigmoid)
    else:
        model = TemporalBranch(typeImg=cfg["typeImg"], onlyPairRGB=cfg["onlyPairRGB"], onlyPairPose=False,nlayersFreeze=cfg["nlayersFreeze"], backbone_class=MViTv2SmallBackbone, dim=768, audio=cfg["audio"])  #model returns raw logits (no sigmoid)
else:
    if cfg["fusion"] == "crossAttention":
        if cfg["backbone"] == "ResNet18":
            model = MultiTaskCrossAttention(typeImg=cfg["typeImg"], onlyPairRGB=cfg["onlyPairRGB"], onlyPairPose=False,nlayersFreeze=cfg["nlayersFreeze"], audio=cfg["audio"])  #model returns raw logits (no sigmoid)
        else:
            model = MultiTaskCrossAttention(typeImg=cfg["typeImg"], onlyPairRGB=cfg["onlyPairRGB"], onlyPairPose=False,nlayersFreeze=cfg["nlayersFreeze"], backbone_class=MViTv2SmallBackbone, dim=768, audio=cfg["audio"])  #model returns raw logits (no sigmoid)
    else:
        if cfg["backbone"] == "ResNet18":
            model = MultiTaskCLStoken(typeImg=cfg["typeImg"], onlyPairRGB=cfg["onlyPairRGB"], onlyPairPose=False,nlayersFreeze=cfg["nlayersFreeze"], audio=cfg["audio"])  #model returns raw logits (no sigmoid) from both tasks
        else:
            model = MultiTaskCLStoken(typeImg=cfg["typeImg"], onlyPairRGB=cfg["onlyPairRGB"], onlyPairPose=False,nlayersFreeze=cfg["nlayersFreeze"], backbone_class=MViTv2SmallBackbone, dim=768, audio=cfg["audio"])  #model returns raw logits (no sigmoid)
model.to(device)

# -----------------------------------------------------------------------
# 5. Load checkpoint
# -----------------------------------------------------------------------
print(f"\n[INFO] Loading checkpoint: {ckpt_path}")
load_best_checkpoint(model, ckpt_path, device)

# -----------------------------------------------------------------------
# 6. Reopen same W&B run
# -----------------------------------------------------------------------
print("\n" + "=" * 70)
print("Weights & Biases")
print("=" * 70)
print("[INFO] Reopening same W&B run...")
wandb_run = wandb.init(
    project=cfg["wandb"]["project"],
    group=cfg["wandb"]["group"],
    name=cfg["wandb"]["name"],
    id=cfg["wandb"]["id"],
    resume="allow",
)

# -----------------------------------------------------------------------
# 7. Evaluate on test set
# -----------------------------------------------------------------------
print("\n" + "=" * 70)
print("Evaluation")
print("=" * 70)
print("[INFO] Running test evaluation...")
results = evaluate_on_test(
    task=cfg["task"],
    model=model,
    test_loader=test_loader,
    device=device,
)

# -----------------------------------------------------------------------
# 8. Print, save and log results
# -----------------------------------------------------------------------
print("\n" + "=" * 70)
print("Test Results")
print("=" * 70)
print("[INFO] Test results:")
print_test_results(cfg["task"], results)
print(f"[INFO] Saving test results to: {test_output_dir}")
save_test_results_json(results, test_output_dir)
print("[INFO] Logging test results to W&B...")
log_test_results_to_wandb_same_run(
    task=cfg["task"],
    results=results,
    wandb_run=wandb_run,
    output_dir=test_output_dir,
)
print("[INFO] Test evaluation finished.")
wandb.finish()


