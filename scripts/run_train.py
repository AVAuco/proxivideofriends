# ---------------------------------------------------------------------------
# PyTorch training script for ProxemicsNet++ with video sequences and optional audio branch.
# ---------------------------------------------------------------------------
 
import os
import sys
import argparse
import warnings

import torch
from torch.utils.data import DataLoader
import wandb


import wandb

warnings.filterwarnings("ignore")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

sys.path.append("..")
from train.model import TemporalBranch, MultiTaskCLStoken, MultiTaskCrossAttention, MViTv2SmallBackbone
from dataset_utils.load_sequences_from_episodes import load_sequences_from_episodes, filter_set
from dataset_utils.dataGeneratorSequences import PyTorchSequenceDataset, BalancedSampler
from train.trainer import fit


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Training and testing script.')
    # Training hyperparameters
    parser.add_argument('--batch',type=int,  help='Size of each batch', required=False, default=16)
    parser.add_argument('--epoch', type=int,  help='Number of epochs', required=False, default=25)
    parser.add_argument('--opt',type=str,  help='optimizer', required=False, default="AdamW")
    parser.add_argument('--lr', type=float,  help='lrate', required=False, default=0.0001)
    parser.add_argument('--set', type=int,  help='Set (1 or 2)', required=False, default=1)
    # Temporal configuration
    parser.add_argument('--window',type=int,  help='Frames sequence size', required=False, default=3)
    parser.add_argument('--stride',type=int,  help='Stride for the sliding window', required=False, default=3)
    # Input branches
    parser.add_argument('--onlyPairRGB',action='store_true',help='Only context brach of RGB model',default=False)
    parser.add_argument('--audio',action='store_true',help='Use audio branch',default=False)
    # Backbone and model configuration
    parser.add_argument('--backbone',type=str,  help='Temporal Backbone', required=False, choices=["ResNet18", "mViTv2"], default="ResNet18")
    parser.add_argument('--nlayersFreeze', type=int,  help='n layers frozen', choices=[0,1,2,3,4,5], required=False, default=0)
    parser.add_argument("--fusion", help="Fusion type - crossAttention/CLS", choices=['crossAttention', 'CLS'], required=False, default='crossAttention')
    parser.add_argument("--task", help="Task type - Proxemics/Relationship/Multitask", choices=['proxemics', 'relationship', 'multitask'], required=False, default='multitask')
    # Paths
    parser.add_argument('--datasetDIR',type=str,  help='Main Dir where dataset is located', required=True, default="../../dataset/")
    parser.add_argument('--outModelsDIR',type=str,  help='Dir where model will be saved', required=True, default="../../")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
args = parse_args()

# -----------------------------
# Basic setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batchsize = args.batch  # Size of each batch
nepochs = args.epoch
optimizer = args.opt  # or 'AdamW'
lr = args.lr
augmentation=True
use_set=args.set  # Set (1 or 2)

frames_per_sequence = args.window  # Number of frames in each sequence
stride = args.stride # Stride for the sliding window

typeImg="RGB"
onlyPairRGB=args.onlyPairRGB  # Only context branch of RGB model
audio = args.audio  # Use audio branch

use_backbone = args.backbone  # Backbone to use (ResNet18 or mViTv2)
backbone_size = "base"  # Size of the pre-trained backbone (small or base)
nlayersFreeze=args.nlayersFreeze  # Number of layers to freeze in the backbone (0-5)

task= args.task  # Task type (Proxemics, Relationship, or Multitask)
multitask=True
if task == "proxemics":
    multitask=False
fusion = args.fusion  # Fusion type (crossAttention or CLS)

datasetDir=args.datasetDIR
outputDir=args.outModelsDIR

# -----------------------------
# Backbone-specific and task-specific checks
# -----------------------------
if use_backbone == "mViTv2":
    backbone_size = "small"
    if frames_per_sequence != 16:
        print(f"[INFO] Using backbone: {use_backbone} ({backbone_size})")
        print("[WARNING] MViTv2 expects 16-frame clips.")
        print(f"[WARNING] Overriding --window from {frames_per_sequence} to 16.")
    frames_per_sequence = 16
if task != "multitask":
    print(f"[WARNING]  The --fusion argument is not applicable for single-task models. Ignoring fusion choice and using crossAttention by default.")
    fusion = "crossAttention"  # CLS Fusion is not applicable for single-task models

# -----------------------------
# Paths
# ----------------------------- 
shots_dir = os.path.join(datasetDir, 'shots/season3')
labels_dir = os.path.join(datasetDir, 'labels/')
base_dataset_path = os.path.join(datasetDir, 'preprocessed_dataset')
output_dir_models = os.path.join(outputDir, 'output_models')

base_audio_path = os.path.join(datasetDir, 'wavs') 
audio_emmbeddings_dir = os.path.join(datasetDir, 'audio_embeddings')

# -----------------------------
#  W&B naming
# -----------------------------
projectname = f'proxiVideosFriends_{task}_{use_backbone}_{fusion}_temporal'

if onlyPairRGB:
    groupname=typeImg+'_onlypair_'+backbone_size
else:
    groupname=typeImg+'_p0p1pair_'+backbone_size


# Modelname format
modelname = (
        f"Model_aug{int(augmentation)}_bs{batchsize}_set{use_set}"
        f"_lr{lr:1.5f}_o{optimizer}_seq{frames_per_sequence}_str{stride}_fr{nlayersFreeze}"
        f"_audio{int(audio)}"
    )

id=f"id_{groupname}_{modelname}"

model_filepath = os.path.join(output_dir_models,task,use_backbone,fusion,typeImg, groupname, modelname)
os.makedirs(model_filepath, exist_ok=True)
#print(model_filepath)

# -----------------------------
# Print experiment summary
# -----------------------------
print("\n" + "=" * 70)
print("Experiment configuration")
print("=" * 70)
print(f"Device               : {device}")
print(f"Task                 : {task}")
print(f"Backbone             : {use_backbone}")
print(f"Fusion               : {fusion}")
print(f"Batch size           : {batchsize}")
print(f"Epochs               : {nepochs}")
print(f"Learning rate        : {lr}")
print(f"Optimizer            : {optimizer}")
print(f"Frames per sequence  : {frames_per_sequence}")
print(f"Stride               : {stride}")
print(f"Freeze level         : {nlayersFreeze}")
print(f"Audio branch         : {audio}")
print(f"Only Pair RGB        : {onlyPairRGB}")
print(f"Dataset dir          : {datasetDir}")
print(f"Output dir           : {model_filepath}")
print("=" * 70 + "\n")


# -----------------------------------------------------------------------
# 1. Load dataset splits
# -----------------------------------------------------------------------
print("\n" + "=" * 70)
print("Dataset")
print("=" * 70)
print("[INFO] Loading sequences...")
train_set, val_set, test_set = load_sequences_from_episodes(
    shots_dir, labels_dir, use_set, frames_per_sequence=frames_per_sequence,stride=stride
)

# Optional filtering for class balancing on val/test
val_set=filter_set(val_set)  
test_set=filter_set(test_set)
print(f"[INFO] Train sequences      : {len(train_set)}")
print(f"[INFO] Validation sequences : {len(val_set)}")
print(f"[INFO] Test sequences       : {len(test_set)}")

# -----------------------------------------------------------------------
# 1.1. Audio embeddings (optional)
# -----------------------------------------------------------------------
audio_embeddings_path=""
if audio:
    #Check if audio embeddings already exist for the current configuration
    audio_embeddings_path = os.path.join(audio_emmbeddings_dir, f'window_{frames_per_sequence}_stride_{stride}')
    print("\n[INFO] Generating / loading audio embeddings...")
    print(f"[INFO] Audio embeddings path: {audio_embeddings_path}")
    from dataset_utils.generate_audio_embeddings import generate_audio_embeddings
    generate_audio_embeddings(train_set, base_audio_path, audio_embeddings_path, fps=24, model_name="base")
    generate_audio_embeddings(val_set, base_audio_path, audio_embeddings_path, fps=24, model_name="base")
    generate_audio_embeddings(test_set, base_audio_path, audio_embeddings_path, fps=24, model_name="base")


# -----------------------------------------------------------------------
# 2. Build datasets and dataloaders
# -----------------------------------------------------------------------
print("\n" + "=" * 70)
print("DataLoaders")
print("=" * 70)
print("[INFO] Creating datasets and dataloaders...")
#Datagenerator Parameters
params = dict(base_dataset_path=base_dataset_path,typeImg=typeImg,onlyPairRGB=onlyPairRGB,onlyPairPose=False,frames_per_sequence=frames_per_sequence, use_backbone=use_backbone, audio=audio, audio_embeddings_path=audio_embeddings_path, multitask=multitask)

#generetaors
train_ds = PyTorchSequenceDataset(list_sequences=train_set,augmentation=True, **params)
val_ds= PyTorchSequenceDataset(list_sequences=val_set,augmentation=False, **params)
test_ds = PyTorchSequenceDataset(list_sequences=test_set,augmentation=False, **params)

#samplers
train_sampler = BalancedSampler(train_set, num_classes=7, seed=0)

#loaders
train_loader = DataLoader(train_ds,batch_size=batchsize,sampler=train_sampler,num_workers=4,pin_memory=True,persistent_workers=True)
val_loader   = DataLoader(val_ds,   batch_size=batchsize, shuffle=False, num_workers=4, pin_memory=True)
test_loader   = DataLoader(test_ds,   batch_size=batchsize, shuffle=False, num_workers=4, pin_memory=True)


# -----------------------------------------------------------------------
# 3. Build model
# -----------------------------------------------------------------------
print("\n" + "=" * 70)
print("Model")
print("=" * 70)
print("[INFO] Building model...")

if task != "multitask":
    if use_backbone == "ResNet18":
        model = TemporalBranch(typeImg=typeImg, onlyPairRGB=onlyPairRGB, onlyPairPose=False,nlayersFreeze=nlayersFreeze, audio=audio)  #model returns raw logits (no sigmoid)
    else:
        model = TemporalBranch(typeImg=typeImg, onlyPairRGB=onlyPairRGB, onlyPairPose=False,nlayersFreeze=nlayersFreeze, backbone_class=MViTv2SmallBackbone, dim=768, audio=audio)  #model returns raw logits (no sigmoid)
else:
    if fusion == "crossAttention":
        if use_backbone == "ResNet18":
            model = MultiTaskCrossAttention(typeImg=typeImg, onlyPairRGB=onlyPairRGB, onlyPairPose=False,nlayersFreeze=nlayersFreeze, audio=audio)  #model returns raw logits (no sigmoid)
        else:
            model = MultiTaskCrossAttention(typeImg=typeImg, onlyPairRGB=onlyPairRGB, onlyPairPose=False,nlayersFreeze=nlayersFreeze, backbone_class=MViTv2SmallBackbone, dim=768, audio=audio)  #model returns raw logits (no sigmoid)
    else:
        if use_backbone == "ResNet18":
            model = MultiTaskCLStoken(typeImg=typeImg, onlyPairRGB=onlyPairRGB, onlyPairPose=False,nlayersFreeze=nlayersFreeze, audio=audio)  #model returns raw logits (no sigmoid) from both tasks
        else:
            model = MultiTaskCLStoken(typeImg=typeImg, onlyPairRGB=onlyPairRGB, onlyPairPose=False,nlayersFreeze=nlayersFreeze, backbone_class=MViTv2SmallBackbone, dim=768, audio=audio)  #model returns raw logits (no sigmoid)
model.to(device)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"[INFO] Total parameters     : {total_params:,}")
print(f"[INFO] Trainable parameters : {trainable_params:,}")

# -----------------------------------------------------------------------
# 4. Initialize Weights & Biases
# -----------------------------------------------------------------------
print("\n" + "=" * 70)
print("Weights & Biases")
print("=" * 70)
print("[INFO] Initializing W&B...")

wandb_run = wandb.init(
    project=projectname,
    group=groupname,
    name=modelname,
    id=id,
    resume="allow",
    config={
        "task": task,
        "epochs": nepochs,
        "batch_size": batchsize,
        "learning_rate": lr,
        "optimizer": optimizer,
        "set": use_set,
        "weight_decay": 0.01,
        "alpha": 1.0,
        "output_dir": model_filepath,
    }
)
wandb.watch(model)

# -----------------------------------------------------------------------
# 5. Save config for reproducibility and testing
# -----------------------------------------------------------------------
config_to_save = {
    "task": task,
    "epochs": nepochs,
    "batch_size": batchsize,
    "learning_rate": lr,
    "optimizer": optimizer,
    "set": use_set,
    "weight_decay": 0.01,
    "alpha": 1.0,
    "frames_per_sequence": frames_per_sequence,
    "stride": stride,
    "typeImg": typeImg,
    "onlyPairRGB": onlyPairRGB,
    "backbone": use_backbone,
    "backbone_size": backbone_size,
    "nlayersFreeze": nlayersFreeze,
    "fusion": fusion,
    "audio": audio,
    "datasetDir": datasetDir,

    "output_dir": model_filepath,
    "wandb": {
        "project": projectname,
        "group": groupname,
        "name": modelname,
        "id": id,
    }
}

# -----------------------------------------------------------------------
# 6. Train
# -----------------------------------------------------------------------
print("\n" + "=" * 70)
print("Starting training...")
print("=" * 70)

fit(
    task=task,
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    train_sampler=train_sampler,
    train_set=train_set,
    device=device,
    output_dir=model_filepath,
    epochs=nepochs,
    batch_size=batchsize,
    learning_rate=lr,
    weight_decay=0.01,
    alpha=1.0,
    early_stop_patience=8,
    wandb_run=wandb_run,
    config_to_save=config_to_save,
)

print("[INFO] Training finished.")
wandb.finish()
