"""
datageneratorSequences.py

PyTorch dataset and sampling utilities for multimodal proxemics learning.

This file includes:
- PyTorchSequenceDataset: loads temporal RGB, pose, and optional audio inputs.
- Sequence-wise preprocessing and data augmentation.
- BalancedSampler: mitigates class imbalance by undersampling the all-zero
  class and oversampling underrepresented interaction classes.
"""
import os, random, copy
from zipfile import ZipFile
from collections import defaultdict,  Counter
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda

# ---------- DATASET (loading, normalization, and augmentation) ----------
class PyTorchSequenceDataset(Dataset):
    """
    Returns:
        X -> tuple of tensors with shape (T, C, H, W)
        y -> label tensor with shape (6,)
    """
    def __init__(self,list_sequences,base_dataset_path, typeImg='RGB', onlyPairRGB=False, onlyPairPose=False,  frames_per_sequence=3, augmentation=True, use_backbone="ResNet18", audio=False, audio_embeddings_path=None, multitask=False):

        self.list_sequences   = copy.deepcopy(list_sequences)
        self.base_dir         = base_dataset_path
        self.typeImg          = typeImg
        self.onlyPairRGB      = onlyPairRGB
        self.onlyPairPose     = onlyPairPose
        self.T                = frames_per_sequence
        self.augmentation     = augmentation
        self.audio = audio
        self.audio_embeddings_path = audio_embeddings_path
        self.multitask = multitask

        # Caches to avoid repeatedly reopening ZIP files and reloading images
        self.img_cache        = {}
        self.zip_rgb_opened   = {}
        self.zip_pose_opened  = {}

        ############ r2plus1d_18 / video backbones preprocessing ############
        # Different backbones expect different input resolutions and normalization values.
        # The "ResNet18" branch here actually uses a 112x112 preprocessing pipeline,
        # which is commonly used for video models such as R(2+1)D.
        if use_backbone =="ResNet18":
            self.rgb_normalizer = Compose([
                Lambda(lambda img: F.to_pil_image(img) if isinstance(img, np.ndarray) else img),
                transforms.Resize((128, 171)),
                transforms.CenterCrop((112, 112)),
                ToTensor(),
                Normalize(mean=[0.43216, 0.394666, 0.37645],
                            std=[0.22803, 0.22145, 0.216989])
                ])
        else: # Default preprocessing for larger image backbones (e.g. 224x224 inputs)
            self.rgb_normalizer = Compose([
                Lambda(lambda img: F.to_pil_image(img) if isinstance(img, np.ndarray) else img),
                transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
                transforms.CenterCrop(224),
                ToTensor(),
                Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
            ])

    # ------- required Dataset methods -------
    def __len__(self):
        return len(self.list_sequences)

    def __getitem__(self, idx):
        seq = self.list_sequences[idx]
        return self._load_sequence(seq)

    # ------- helper methods -------
    def _open_zip(self, episode, rgb=True):
        """
        Open the ZIP file for a given episode only once and reuse it afterwards.
        """
        d = self.zip_rgb_opened if rgb else self.zip_pose_opened
        if episode in d:
            return d[episode]
        sub = 'recortes.zip' if rgb else 'poseImg_I_recortes_thr0_pair.zip'
        path = os.path.join(self.base_dir, f'episode{episode}', sub)
        d[episode] = ZipFile(path, mode='r')
        return d[episode]

    def _load_img_from_zip(self, zip_ref, key):
        """
        Read an image from a ZIP archive and cache it globally.
        This avoids decoding the same image multiple times.
        """
        if key in self.img_cache:
            return self.img_cache[key]
        try:
            buf  = zip_ref.read(key)
            img  = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f'[WARN] Error reading {key}: {e}')
            # Fallback to a black image if loading fails
            img  = np.zeros((224,224,3), np.uint8)
        self.img_cache[key] = img
        return img

    def _get_random_transform(self):
        """
        Generate a random augmentation configuration -ImageDataGenerator.get_random_transform((224,224))
        This roughly mimics Keras ImageDataGenerator-style transforms.
        """
        return {
            "shear":           random.uniform(-5, 5),      # shear_range=5
            "scale":           random.uniform(0.9, 1.1),   # zoom_range=0.1
            "brightness":      random.uniform(0.8, 1.2),   # brightness_range=[0.8,1.2]
            "flip_horizontal": random.randint(0, 1)        # 0 ó 1
        }

    def _apply_transform(self, img, t):
        """
        Apply geometric and photometric augmentation to an RGB image.
        Operations are done in PIL format for torchvision compatibility.
        """
        img = F.to_pil_image(img)  if isinstance(img, np.ndarray) else img
        # Apply zoom and shear without rotation or translation
        img = F.affine(img,
                    angle=0.0,
                    translate=(0, 0),
                    scale=t["scale"],
                    shear=t["shear"],
                    interpolation=InterpolationMode.BILINEAR)
        # Apply brightness change
        img = F.adjust_brightness(img, t["brightness"])

        # Apply horizontal flip if selected
        if t["flip_horizontal"] == 1:
            img = F.hflip(img)

        return img

    # ---------- full sequence loading + augmentation ----------
    def _load_sequence(self, seq):
        ep, frames = seq['episode'], seq['frames']
        p0, p1     = seq['p0'], seq['p1']
        label      = torch.tensor(seq['label'], dtype=torch.float32)

        imgs_p0, imgs_p1, imgs_pair  = [], [], []

        # Store individual and pair crops for both RGB and pose modalities
        for fr in frames:
            if 'RGB' in self.typeImg:
                zrgb  = self._open_zip(ep, rgb=True)
                imgs_p0 .append(self._load_img_from_zip(zrgb, f'recortes/{fr}_{p0}.jpg'))
                imgs_p1 .append(self._load_img_from_zip(zrgb, f'recortes/{fr}_{p1}.jpg'))
                imgs_pair.append(self._load_img_from_zip(zrgb, f'recortes/{fr}_pair_{p0}-{p1}.jpg'))

            if 'Pose' in self.typeImg:
                zpose = self._open_zip(ep, rgb=False)
                poses_p0.append(self._load_img_from_zip(
                        zpose, f'poseImg_I_recortes_thr0_pair/pose_{fr}_{p0}.jpg'))
                poses_p1.append(self._load_img_from_zip(
                        zpose, f'poseImg_I_recortes_thr0_pair/pose_{fr}_{p1}.jpg'))
                poses_pr.append(self._load_img_from_zip(
                        zpose, f'poseImg_I_recortes_thr0_pair/pose_{fr}_pair_{p0}-{p1}.jpg'))

        # ---- 2. Data augmentation (training only) ----
        if self.augmentation:
            t = self._get_random_transform()   
            # Apply the same random transform to all RGB frames in the sequence
            # to preserve temporal consistency
            # ----- RGB -----
            if 'RGB' in self.typeImg:
                imgs_p0   = [self._apply_transform(im, t) for im in imgs_p0]
                imgs_p1   = [self._apply_transform(im, t) for im in imgs_p1]
                imgs_pair = [self._apply_transform(im, t) for im in imgs_pair]

            # For pose images, only horizontal flip is applied.
            # Channel swapping is used to preserve left/right body-part semantics.
            if 'Pose' in self.typeImg and t["flip_horizontal"] == 1:
                poses_p0 = [np.fliplr(im)[..., [0, 2, 1]] for im in poses_p0]
                poses_p1 = [np.fliplr(im)[..., [0, 2, 1]] for im in poses_p1]
                poses_pr = [np.fliplr(im)[..., [0, 2, 1]] for im in poses_pr]

        # ---- 3. Normalize RGB frames ---- [-1, 1]
        if 'RGB' in self.typeImg:
            imgs_p0 = [self.rgb_normalizer(img) for img in imgs_p0]
            imgs_p1 = [self.rgb_normalizer(img) for img in imgs_p1]
            imgs_pair = [self.rgb_normalizer(img) for img in imgs_pair]
        
        # ---- 4. Convert lists of frames into tensors ----
        X = []
        # Stack frames along the temporal dimension: (T, C, H, W)
        stack = lambda l: torch.stack(l)                   
        # Pose images are transformed at stacking time
        stack_pose = lambda l: torch.stack([self.pose_transform(im) for im in l])
        if 'RGB' in self.typeImg:
            if not self.onlyPairRGB:
                X.append(stack(imgs_p0))
                X.append(stack(imgs_p1))
            X.append(stack(imgs_pair))

        if 'Pose' in self.typeImg:
            if not self.onlyPairPose:
                X.append(stack_pose(poses_p0))
                X.append(stack_pose(poses_p1))
            X.append(stack_pose(poses_pr))
        
        # ---- 5. Load optional audio embedding ----
        if self.audio:
            # Expected format:
            # audio_embeddings/.../{episode}_{first_frame}_{last_frame}.npy
            emb_path = os.path.join(self.audio_embeddings_path, f"{ep}_{frames[0]}_{frames[-1]}.npy")
            try:
                audio_emb = np.load(emb_path)
                audio_emb = torch.tensor(audio_emb, dtype=torch.float32)
            except FileNotFoundError:
                print(f"[WARN] Audio embedding not found: {emb_path}")
                audio_emb = torch.zeros(80, dtype=torch.float32) 
            X.append(audio_emb)
        # ---- 6. Return data and labels ----
        if self.multitask:
            # Additional relationship label for multitask training
            relationship = torch.tensor(seq['relationship'], dtype=torch.long)  # scalar (0–5)
            return tuple(X), (label, relationship)  
        else:
            return tuple(X), label




class BalancedSampler(Sampler):
    def __init__(self, list_sequences, num_classes=7, seed=None):
        self.list_sequences = list_sequences
        self.num_classes = num_classes
        self.labels = [np.array(seq['label']) for seq in list_sequences]
        self.dataset_size = len(list_sequences)
        # Base seed is used together with the epoch index for deterministic reshuffling
        self.base_seed = seed if seed is not None else 0
        self.epoch = 0  
        self.last_epoch_indices = []
        # Compute actual class frequency
        # Class 6 represents the "all-zero" case (no active label among classes 0-5)
        self.class_freq = np.zeros(num_classes, dtype=int)
        for lbl in self.labels:
            if lbl.sum() == 0:
                self.class_freq[6] += 1
            else:
                self.class_freq[:6] += lbl[:6]
        # Relationship labels used to better balance the all-zero class internally
        self.rel_labels = np.array([seq['relationship'] for seq in list_sequences], dtype=np.int64)
        self.num_rel_classes = 6

        # Target frequency: match all classes to the most frequent class among 0-5
        self.target_count = int(self.class_freq[:6].max())  # solo cuenta clases 0 a 5
        self.class_deficit = np.clip(self.target_count - self.class_freq, 0, None)
    
    def set_epoch(self, epoch):
        """
        Update the internal epoch value so sampling remains deterministic across epochs.
        """
        self.epoch = epoch

    def __iter__(self):
        seed = self.base_seed + self.epoch
        
        random.seed(seed)
        np.random.seed(seed)

        # 1. Start with full dataset coverage
        epoch_indices = list(range(self.dataset_size))

         # 2. Explicitly limit class 6 (all-zero samples) to target_count
        actual_6 = sum(1 for idx in epoch_indices if self.labels[idx].sum() == 0)
        excess = actual_6 - self.target_count
        if actual_6 > self.target_count:
            # Separate all-zero samples
            all_zero_idx = [idx for idx in epoch_indices if self.labels[idx].sum() == 0]
            # Relationship distribution within all-zero samples
            az_rel = self.rel_labels[np.array(all_zero_idx, dtype=np.int64)]
            counts = np.bincount(az_rel, minlength=self.num_rel_classes).astype(np.int64)

            # Decide how many all-zero samples to keep per relationship class
            # while distributing them as evenly as possible
            keep_total = self.target_count
            keep_per_c = np.zeros(self.num_rel_classes, dtype=np.int64)
            base = keep_total // self.num_rel_classes
            # First pass: assign an equal base amount per relationship class
            for c in range(self.num_rel_classes):
                keep_per_c[c] = min(base, counts[c])

            # Distribute the remaining quota to the most available classes
            rem = int(keep_total - keep_per_c.sum())
            if rem > 0:
                order = np.argsort(-counts)  # clases con más disponibles primero
                for c in order:
                    if rem == 0:
                        break
                    add = int(min(rem, counts[c] - keep_per_c[c]))
                    if add > 0:
                        keep_per_c[c] += add
                        rem -= add

            # Removal budget per relationship class
            remove_budget = counts - keep_per_c  

            # Remove excess all-zero samples, prioritizing majority relationship classes
            removed_total = 0
            new_epoch = []
            for idx in epoch_indices:
                if removed_total < excess and self.labels[idx].sum() == 0:
                    c = int(self.rel_labels[idx])
                    if remove_budget[c] > 0:
                        remove_budget[c] -= 1
                        removed_total += 1
                        continue  # eliminamos este all-zero (de clase mayoritaria)
                new_epoch.append(idx)
            epoch_indices = new_epoch


        # 3. Prepare oversampling for classes 0-5 with remaining deficit
        class_counter = self.class_freq.copy()
        deficit = self.class_deficit.copy()

        # Valid candidates are samples whose active labels belong only to classes
        # that are still underrepresented (deficit > 0). This ensures that oversampling focuses on the classes that need it most.
        candidates = []
        for idx, lbl in enumerate(self.labels):
            if lbl.sum() == 0:
                continue  # ya cubierto arriba
            active = [i for i in range(6) if lbl[i] == 1]
            if all(deficit[i] > 0 for i in active):
                candidates.append((idx, active))

        # 4. Oversample until all classes 0-5 reach target_count
        while np.any(deficit[:6] > 0) and candidates:
            idx, acts = random.choice(candidates)

            # Remove candidates that are no longer useful
            if any(class_counter[a] >= self.target_count for a in acts):
                candidates.remove((idx, acts))
                continue

            epoch_indices.append(idx)
            for a in acts:
                class_counter[a] += 1
                if class_counter[a] >= self.target_count:
                    deficit[a] = 0
        
        random.shuffle(epoch_indices)
        
    
        # Store sampling statistics for debugging/inspection
        from collections import Counter
        self.last_epoch_indices = epoch_indices.copy()
        self.index_counter = Counter(epoch_indices)
       
        self.last_epoch_indices= epoch_indices.copy()
        return iter(epoch_indices)

    def __len__(self):
        """
        Approximate epoch length after oversampling.
        """
        return len(self.list_sequences) + sum(self.class_deficit)
