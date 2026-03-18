# =========================================================================== #
# Useful functions for the demo script
# =========================================================================== #

import os
import cv2
import numpy as np 
import torch
from ultralytics import YOLO
import sys
sys.path.append("..")
from evaluation.evaluator import PROXEMICS_CLASS_NAMES, RELATIONSHIP_CLASS_NAMES

# --------------------------------------------------------------------------- #
# Extract video frames at a fixed rate of 24 FPS
# This function reads a video file and saves frames uniformly at 24 FPS.
# --------------------------------------------------------------------------- #
def get_video_frames_24fps(videoPath, outputFramesDir):
    os.makedirs(outputFramesDir, exist_ok=True)
    # Open the input video
    cap = cv2.VideoCapture(videoPath)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {videoPath}")
    # Read the original FPS of the video
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0:
        original_fps = 24.0 # Fallback to 24 FPS if the original FPS cannot be read

    frame_idx = 0
    saved_idx = 0
    next_t = 0.0
    # Read the video frame by frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Compute the current timestamp of the frame
        current_t = frame_idx / original_fps
        # Save frames at a fixed rate of 24 FPS
        if current_t >= next_t - 1e-8:
            out_name = os.path.join(outputFramesDir, f"{saved_idx:06d}.jpg")
            cv2.imwrite(out_name, frame)
            saved_idx += 1
            next_t += 1.0 / 24.0

        frame_idx += 1

    cap.release()
    print(f"\t[INFO] Saved {saved_idx} frames at 24 FPS into: {outputFramesDir}")


# --------------------------------------------------------------------------- #
# Detection functions (BBox extraction and clippings generation)
# --------------------------------------------------------------------------- #

# Resize an image to a square format and then to 224x224
# This function pads the shortest side with black borders and resizes the image.
def resizeImage(img):
    h, w, c = img.shape  # height, width, channel
    padding1 = padding2 = int(abs(h - w) / 2)  # Round to the integer
    if abs(h - w) % 2 != 0:
        padding1 += 1
    if h < w:  # Apply padding to height
        img = cv2.copyMakeBorder(img, padding1, padding2, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    else:  # Apply padding to width
        img = cv2.copyMakeBorder(img, 0, 0, padding1, padding2, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
    return img

# Select the two main detected persons in a frame
# This function keeps the two highest-confidence person detections and orders
# them from left to right.
def get_main_pair_boxes(results):
    if len(results) == 0 or results[0].boxes is None:
        return None

    boxes = results[0].boxes.xyxy.detach().cpu().numpy()
    confs = results[0].boxes.conf.detach().cpu().numpy()

    # At least two persons are needed
    if len(boxes) < 2:
        return None

    # Sort detections by confidence and keep the top 2
    order = np.argsort(-confs)
    boxes = boxes[order[:2]]

    # Sort the two boxes from left to right according to their x-center
    centers_x = [(b[0] + b[2]) / 2.0 for b in boxes]
    if centers_x[0] <= centers_x[1]:
        return boxes[0], boxes[1]
    else:
        return boxes[1], boxes[0]
    
# Detect two persons in each frame and save the corresponding cropped images - use YOLOv8 segmentation model to get the bounding boxes and masks. 
# This function generates crops for person 0, person 1, and the joint pair area.
def detect_persons_and_save_clippings(framesDir, clippingsDir):
    # Create the output directory for the cropped images
    os.makedirs(clippingsDir, exist_ok=True)
    # Load the YOLO person detector
    detector = YOLO("yolov8x-seg.pt")
    frames = sorted(os.listdir(framesDir))

    frame_info = []

    for frame_name in frames:
        frame_path = os.path.join(framesDir, frame_name)
        img = cv2.imread(frame_path)
        if img is None:
            continue
        # Run person detection (class 0 = person)
        results = detector(img, classes=[0], conf=0.3, iou=0.3, verbose=False)
        pair_boxes = get_main_pair_boxes(results)
        if pair_boxes is None:
            continue

        p0_box, p1_box = pair_boxes

        # Ensure boxes are within image boundaries
        # Get BBox coordinates and clamp them to image dimensions
        h, w, _ = img.shape

        x1_0, y1_0, x2_0, y2_0 = [int(v) for v in p0_box]
        x1_1, y1_1, x2_1, y2_1 = [int(v) for v in p1_box]
        # Clamp the coordinates for person 0
        x1_0 = max(0, min(x1_0, w - 1))
        x2_0 = max(1, min(x2_0, w))
        y1_0 = max(0, min(y1_0, h - 1))
        y2_0 = max(1, min(y2_0, h))
        # Clamp the coordinates for person 1
        x1_1 = max(0, min(x1_1, w - 1))
        x2_1 = max(1, min(x2_1, w))
        y1_1 = max(0, min(y1_1, h - 1))
        y2_1 = max(1, min(y2_1, h))
        # Compute the union box containing both persons
        pair_x1 = min(x1_0, x1_1)
        pair_y1 = min(y1_0, y1_1)
        pair_x2 = max(x2_0, x2_1)
        pair_y2 = max(y2_0, y2_1)

        # Crop person 0, person 1, and the pair region
        p0 = img[y1_0:y2_0, x1_0:x2_0]
        p1 = img[y1_1:y2_1, x1_1:x2_1]
        pair = img[pair_y1:pair_y2, pair_x1:pair_x2]

        if p0.size == 0 or p1.size == 0 or pair.size == 0:
            continue
        # Resize crops to the expected input size
        p0 = resizeImage(p0)
        p1 = resizeImage(p1)
        pair = resizeImage(pair)
        # Save the cropped images with a consistent naming scheme
        frame_base = frame_name[:-4]
        cv2.imwrite(os.path.join(clippingsDir, frame_base + "_p0.jpg"), p0)
        cv2.imwrite(os.path.join(clippingsDir, frame_base + "_p1.jpg"), p1)
        cv2.imwrite(os.path.join(clippingsDir, frame_base + "_pair_p0-p1.jpg"), pair)
        
        # Store metadata for the processed frame
        frame_info.append({
            "frame": frame_name,
            "p0": [x1_0, y1_0, x2_0, y2_0],
            "p1": [x1_1, y1_1, x2_1, y2_1],
            "pair": [pair_x1, pair_y1, pair_x2, pair_y2],
        })

    if len(frame_info) == 0:
        raise RuntimeError("No valid frames with 2 detected persons were found.")

    print(f"\t[INFO] Saved clippings for {len(frame_info)} valid frames")
    return frame_info


# --------------------------------------------------------------------------- #
# Transform functions
# --------------------------------------------------------------------------- #

# Build the RGB preprocessing pipeline for the selected backbone
# This function returns the image transformations required before inference.
def get_rgb_transform(backbone="ResNet18"):
    from torchvision.transforms import Compose, Lambda, Normalize, ToTensor, InterpolationMode
    from torchvision.transforms import v2 as transforms
    from torchvision.transforms import functional as F
    if backbone == "ResNet18":
        transform = Compose([
            Lambda(lambda img: F.to_pil_image(img) if isinstance(img, np.ndarray) else img),
            transforms.Resize((128, 171)),
            transforms.CenterCrop((112, 112)),
            ToTensor(),
            Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
        ])
    else:
        transform = Compose([
            Lambda(lambda img: F.to_pil_image(img) if isinstance(img, np.ndarray) else img),
            transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
        ])
    return transform




# --------------------------------------------------------------------------- #
# Input generation
# --------------------------------------------------------------------------- #
# Generate the model input tensors from the saved clippings
# This function builds temporal sequences for person 0, person 1, and pair crops.
def generate_model_input(clippingsDir, frames_per_sequence=16, backbone="ResNet18", onlyPairRGB=False):
    # Build the image transform for the selected backbone
    transform = get_rgb_transform(backbone)

    # Collect and sort all saved crops for person 0, person 1, and pairs
    p0_imgs = sorted([f for f in os.listdir(clippingsDir) if f.endswith("_p0.jpg")])
    p1_imgs = sorted([f for f in os.listdir(clippingsDir) if f.endswith("_p1.jpg")])
    pair_imgs = sorted([f for f in os.listdir(clippingsDir) if f.endswith("_pair_p0-p1.jpg")])
    # Use the minimum count to avoid index mismatches
    nframes = min(len(p0_imgs), len(p1_imgs), len(pair_imgs))
    
    # Select frame indices:
    # - center clip if enough frames are available
    # - otherwise sample frames uniformly with repetition if necessary
    if nframes >= frames_per_sequence:
        start = (nframes - frames_per_sequence) // 2
        idxs = np.arange(start, start + frames_per_sequence)
    else:
        idxs = np.linspace(0, nframes - 1, frames_per_sequence).round().astype(int)

    p0_sequence = []
    p1_sequence = []
    pair_sequence = []
    # Load and transform each selected frame
    for idx in idxs:
        img_p0 = cv2.imread(os.path.join(clippingsDir, p0_imgs[idx]))
        img_p1 = cv2.imread(os.path.join(clippingsDir, p1_imgs[idx]))
        img_pair = cv2.imread(os.path.join(clippingsDir, pair_imgs[idx]))

        p0_sequence.append(transform(img_p0))
        p1_sequence.append(transform(img_p1))
        pair_sequence.append(transform(img_pair))

    X = []
    # Add individual person streams unless only the pair stream is requested
    if not onlyPairRGB:
        X.append(torch.stack(p0_sequence, dim=0).unsqueeze(0))
        X.append(torch.stack(p1_sequence, dim=0).unsqueeze(0))
    # Always add the pair stream
    X.append(torch.stack(pair_sequence, dim=0).unsqueeze(0))

    return X


# --------------------------------------------------------------------------- #
# Prediction functions
# --------------------------------------------------------------------------- #

# Convert raw model outputs into readable predictions
# This function decodes proxemics and relationship probabilities into labels.
def decode_predictions(output, proxThreshold=0.5):
    # Extract raw logits from the model output
    prox_logits = output["proxemics"]
    rel_logits = output["relationship"]

    # Convert logits to probabilities
    prox_probs = torch.sigmoid(prox_logits[0]).detach().cpu().numpy()
    rel_probs = torch.softmax(rel_logits[0], dim=0).detach().cpu().numpy()
    # Multi-label prediction for proxemics using a threshold
    prox_pred = []
    for i, p in enumerate(prox_probs):
        if p >= proxThreshold:
            prox_pred.append(PROXEMICS_CLASS_NAMES[i])
    # If no proxemics label passes the threshold, keep the most probable one
    if len(prox_pred) == 0:
        prox_pred.append(PROXEMICS_CLASS_NAMES[int(np.argmax(prox_probs))])

    # Predict the relationship label using the maximum probability
    # The last class is excluded from the argmax since it represents "no relationship"
    rel_idx = int(np.argmax(rel_probs[:-1]))
    rel_pred = RELATIONSHIP_CLASS_NAMES[rel_idx]
    # Build a readable output dictionary
    results = {
        "proxemics": {
            "predicted_labels": prox_pred,
            "probabilities": {PROXEMICS_CLASS_NAMES[i]: float(prox_probs[i]) for i in range(len(PROXEMICS_CLASS_NAMES))}
        },
        "relationship": {
            "predicted_label": rel_pred,
            "probabilities": {RELATIONSHIP_CLASS_NAMES[i]: float(rel_probs[i]) for i in range(len(RELATIONSHIP_CLASS_NAMES)-1)}
        }
    }
    return results

