"""
load_sequences_from_episodes.py

Utilities for loading labeled temporal sequences from episode annotations,
building train/validation/test splits, and analyzing class distributions.
"""
import os
import json
import numpy as np
import random
import torch
from collections import defaultdict, Counter

def get_max_class_occurrence(val_set):
    """
    Count occurrences per class, ignoring the null label [0, 0, 0, 0, 0, 0],
    and return the maximum count among non-null classes.
    """
    counter = Counter()
    for item in val_set:
        label = tuple(item["label"])
        # Ignore the null class and count only active labels
        if any(label): 
            for i, v in enumerate(label):
                if v == 1:
                    counter[i] += 1
    return max(counter.values()) if counter else 0

def filter_set(val_set):
    """
    Reduce the number of samples with null label [0, 0, 0, 0, 0, 0]
    so that it does not exceed the maximum occurrence of any non-null class.
    """
    null_entries = []
    non_null_entries = []

    for item in val_set:
        label = np.array(item["label"])
        if np.all(label == 0):
            null_entries.append(item)
        else:
            non_null_entries.append(item)

    max_null_samples = get_max_class_occurrence(non_null_entries)
    # Shuffle with a fixed seed for reproducibility
    random.seed(42)
    random.shuffle(null_entries)
    # Keep only as many null samples as the largest non-null class
    null_entries = null_entries[:max_null_samples]
    
    balanced_val_set = non_null_entries + null_entries
    random.shuffle(balanced_val_set)

    return balanced_val_set


# ----------------------------- Class distribution analysis ------------------------------ #
def analyze_val_distribution(val_ds):
    """
    Print the class distribution of a validation dataset, including the number
    of null-label samples.
    """
    counts_per_class = defaultdict(int)
    null_label_count = 0
    total = len(val_ds)

    for i in range(total):
        _, label = val_ds[i]
        # Convert PyTorch tensors to NumPy for easier processing
        if isinstance(label, torch.Tensor):
            label = label.numpy()
        if np.all(label == 0):
            null_label_count += 1
        else:
            for j, val in enumerate(label):
                if val == 1:
                    counts_per_class[j] += 1

    print(f"Total samples: {total}")
    print(f"Samples with label [0, 0, 0, 0, 0, 0]: {null_label_count}")
    for cls in range(6):
        print(f"Class {cls}: {counts_per_class[cls]} samples")
# -------------------------------------------------------------------------------------------- #

def load_shots(shots_file):
    """
    Load shot boundaries from a text file.
    Each line is expected to contain: start_frame end_frame
    """
    shots = []
    with open(shots_file, 'r') as f:
        for line in f:
            start, end = map(int, line.strip().split())
            shots.append((start, end))
    return shots

def frame_in_shot(frame_number, shots):
    """
    Return True if the frame number falls inside any shot interval.
    """
    for start, end in shots:
        if start <= frame_number <= end:
            return True
    return False

def load_labels(label_file):
    """
    Load a JSON label file.
    """
    with open(label_file, 'r') as f:
        labels = json.load(f)
    return labels

def get_frame_number(imgname):
    """
    Convert a frame name into an integer frame index.
    Assumes imgname does not include the '.jpg' suffix.
    """
    return int(imgname)  # imgname without '.jpg'


def get_real_relation_from_pair(pair, frame_name, labels_data, labels_ids_data):
    """
    Given a pair such as 'p0-p1' and a frame name, return the real identity-based
    relationship string if it is available in both JSON files.

    The returned relationship is normalized by:
    - converting names to lowercase
    - trimming spaces
    - sorting both names alphabetically

    Returns:
        normalized_relation (str) if successful,
        None otherwise.
    """
    try:
        # Retrieve pair keys in both files for the given frame
        keys_indices = list(labels_data[frame_name]["labels"].keys())
        keys_names = list(labels_ids_data[frame_name]["labels"].keys())

        # Match the pair by index position
        rel_index = keys_indices.index(pair)
        real_relation = keys_names[rel_index]

        # Normalize and sort, e.g. 'Rachel-Ross' -> 'rachel-ross'
        normalized_relation = '-'.join(sorted([name.strip().lower() for name in real_relation.split('-')]))

        return normalized_relation

    except Exception as e:
        print(f"[ERROR] Exception in get_real_relation_from_pair for frame {frame_name}, pair {pair}: {e}")
        return None


def load_sequences_from_episodes(shots_dir, labels_dir, use_set, frames_per_sequence=3, stride=1,multitask=True):
    """
    Build temporal sequences from episode annotations and split them into
    train / validation / test sets.

    Sequence generation is centered on a valid middle frame and padded when
    needed at shot boundaries. If a pair is missing in surrounding frames,
    the closest valid frame for that pair is duplicated.

    Args:
        shots_dir: Directory containing shot boundary files.
        labels_dir: Directory containing label annotations.
        use_set: Controls which predefined episode split is used.
        frames_per_sequence: Number of frames per sequence.
        stride: Step size when sliding through frames.
        multitask: Whether to also load relationship identity labels.

    Returns:
        train_set, val_set, test_set
    """
    '''
    if frames_per_sequence % 2 != 1:
        print("frames_per_sequence must be odd to have a clear middle frame.")
        exit()
    '''
    # Predefined episode splits
    if use_set==1:
        train_episodes = ['01', '07', '14', '15', '16', '17', '18', '19', '20', '24', '25']
        val_episodes = ['02', '06']
        test_episodes = ['03', '04', '05', '08', '09', '10', '11', '12', '13', '21', '22', '23']
    else:
        train_episodes = ['03', '04', '08', '09', '10', '11', '12', '21', '22', '23']
        val_episodes = ['05', '13']
        test_episodes = ['01','02', '06', '07', '14', '15', '16', '17', '18', '19', '20', '24', '25']
    


    all_sequences = []

    # Debug / bookkeeping counters
    duplication_due_to_missing_pair = 0
    duplication_due_to_padding = 0
    discarded_due_to_minus_one = 0
    cont=0
    inconsistent_identity_per_class = [0] * 6
    empty_label_in_inconsistent = 0
    
    if multitask:
        print("Filtering with multitask")
        relationship_file = os.path.join(labels_dir, 'relationship.json')
        if not os.path.exists(relationship_file):
            print(f"Relationship file {relationship_file} not found. Exiting.")
            exit()
        relationship_data = load_labels(relationship_file)

    episodes = sorted([name for name in os.listdir(shots_dir) if name.startswith('episode')])

    for ep_filename in episodes:
        ep_number = ep_filename.replace('episode', '').replace('_shots.txt', '').zfill(2)

        #print(f"Processing episode {ep_number}...")

        shots_file = os.path.join(shots_dir, f'episode{ep_number}_shots.txt')
        labels_file = os.path.join(labels_dir, f'labels/ep{ep_number}_labels_6classes_pair_BBs.json')

        if not os.path.exists(shots_file) or not os.path.exists(labels_file):
            print(f"Skipping episode {ep_number}: missing files.")
            continue

        labels_data = load_labels(labels_file)
        shots = load_shots(shots_file)

        if multitask:
            labels_ids_file = os.path.join(labels_dir, f'labels_with_ids/ep{ep_number}_labels_6classes_pair_BBs.json')
            if not os.path.exists(labels_ids_file):
                print(f"Skipping episode {ep_number}: missing files.")
                continue
            labels_ids_data = load_labels(labels_ids_file)
            
        # Build a direct mapping from frame number to pair labels
        # Example:
        # frame_to_labels = {
        #   1: {'p0-p1': [0,0,0,0,0,0], 'p1-p2': [0,1,0,0,0,0]}
        # }
        frames_in_episode = sorted(labels_data.keys())
        frame_numbers = [get_frame_number(f.replace('.jpg', '')) for f in frames_in_episode]
        frame_to_labels = {get_frame_number(f.replace('.jpg', '')): labels_data[f]['labels'] for f in frames_in_episode}

        #Process shot by shot
        for start, end in shots:
            shot_frames = [f for f in frame_numbers if start <= f <= end]
            shot_frames = sorted(shot_frames)

            if len(shot_frames) == 0:
                continue
                
            # Collect all pairs appearing at any point in this shot
            pairs_in_shot = set()
            for f in shot_frames:
                pairs_in_shot.update(frame_to_labels[f].keys())
            pairs_in_shot = sorted(pairs_in_shot)
            
            # Process each pair independently within the shot
            for pair in pairs_in_shot:
                idx = 0
                while idx < len(shot_frames):
                    mid_frame = shot_frames[idx]
                    # The middle frame must contain the pair to be valid, otherwise we skip it (and advance only 1 frame to try to find the next valid center)
                    if pair not in frame_to_labels[mid_frame]:
                        idx += 1
                        continue

                    mid_label = frame_to_labels[mid_frame][pair]

                    # Skip sequences whose central frame has an invalid label
                    if isinstance(mid_label, list) and (-1 in mid_label or len(mid_label) == 0):
                        discarded_due_to_minus_one += 1
                        idx += 1  
                        continue

                    # Compute how many frames are taken before and after the center
                    if frames_per_sequence % 2 == 1: #odd
                        pre = post = frames_per_sequence // 2
                    else: #even
                        pre = frames_per_sequence // 2
                        post = frames_per_sequence - pre - 1
                    seq_frames = []

                    # Previous frames
                    for h in range(pre, 0, -1):
                        frame_idx = idx - h
                        if frame_idx < 0:
                            # Pad with the first frame if we go out of bounds
                            seq_frames.append(shot_frames[0])  
                            duplication_due_to_padding += 1
                        else:
                            seq_frames.append(shot_frames[frame_idx])

                    # Central frame
                    seq_frames.append(mid_frame)

                    # Next frames
                    for h in range(1, post + 1):
                        frame_idx = idx + h
                        if frame_idx >= len(shot_frames):
                            # Pad with the last frame if we go out of bounds
                            seq_frames.append(shot_frames[-1])  
                            duplication_due_to_padding += 1
                        else:
                            seq_frames.append(shot_frames[frame_idx])

                    # Ensure the selected pair exists in all frames of the sequence.
                    # If not, replace missing frames with the nearest valid frame for that pair.
                    fixed_seq_frames = []
                    first_valid_frame = None

                    # Find the first frame in the sequence where the pair exists
                    for f in seq_frames:
                        if pair in frame_to_labels.get(f, {}):
                            first_valid_frame = f
                            break

                    # Replace invalid positions before/after valid occurrences
                    for f in seq_frames:
                        if pair in frame_to_labels.get(f, {}):
                            last_valid_frame = f
                            fixed_seq_frames.append(f)
                        else:
                            # Before the first valid occurrence, duplicate first_valid_frame
                            if f < first_valid_frame:
                                fixed_seq_frames.append(first_valid_frame)
                                duplication_due_to_missing_pair += 1
                            else:
                                # After that, duplicate the most recent valid frame
                                fixed_seq_frames.append(last_valid_frame)
                                duplication_due_to_missing_pair += 1
                    
                    # The label used for the full sequence is the label of the middle frame
                    used_label = mid_label

                    # Save the sequence
                    p0, p1 = pair.split('-')
                    if not multitask:
                        sequence = {
                            'episode': ep_number,
                            'frames': [f"{f:06d}" for f in fixed_seq_frames],
                            'p0': p0,
                            'p1': p1,
                            'label': used_label
                        }
                    else:
                        frame_name = f"{mid_frame:06d}.jpg"

                        normalized_relation = get_real_relation_from_pair(pair, frame_name, labels_data, labels_ids_data)

                        if normalized_relation is None:
                            print(f"[WARNING] No valid relation for pair {pair} in frame {frame_name}")
                            idx += 1
                            continue

                        # Identity consistency check:
                        # the real relationship associated with the pair must remain
                        # constant across all frames in the sequence
                        consistent = True
                        for f in fixed_seq_frames:
                            fname = f"{f:06d}.jpg"
                            norm_rel = get_real_relation_from_pair(pair, fname, labels_data, labels_ids_data)
                            if norm_rel is None or norm_rel != normalized_relation:
                                cont += 1
                                if all(v == 0 for v in used_label):
                                    empty_label_in_inconsistent += 1
                                else:
                                    inconsistent_identity_per_class = [
                                        a + b for a, b in zip(inconsistent_identity_per_class, used_label)
                                    ]
                                consistent = False
                                break

                        if not consistent:
                            idx += 1
                            continue
                        
                        if normalized_relation not in relationship_data:
                            print(f"[WARNING] Relationship '{normalized_relation}' not found in relationship_data.")
                            continue
                        else:
                            class_id = int(relationship_data[normalized_relation])
                            if 1 <= class_id <= 6:
                                rel_class = class_id - 1  # integer class in [0, 5]
                            sequence = {
                                'episode': ep_number,
                                'frames': [f"{f:06d}" for f in fixed_seq_frames],
                                'p0': p0,
                                'p1': p1,
                                'label': used_label,
                                'relationship': rel_class 
                            }
                        
                    all_sequences.append(sequence)
                    # Advance only after successfully generating a sequence
                    idx += stride  

    # Split into train / val / test according to episode membership
    train_set = [seq for seq in all_sequences if seq['episode'] in train_episodes]
    val_set = [seq for seq in all_sequences if seq['episode'] in val_episodes]
    test_set = [seq for seq in all_sequences if seq['episode'] in test_episodes]

    return train_set, val_set, test_set


