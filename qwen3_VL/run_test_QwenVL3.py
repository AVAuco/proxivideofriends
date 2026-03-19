"""
Evaluate Qwen3-VL on the ProxiVideoFriends proxemics benchmark.

Before running:
    conda activate qwenVL3

This script loads the test split, generates missing video sequences,
runs inference with Qwen3-VL, caches responses, computes AP/mAP metrics,
and saves the final results to JSON files.

Example:
    python run_test_QwenVL3.py --datasetDIR ../../dataset/ --resultDIR ../../ --set 1 --t 16 --s 16
"""

import sys
import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor


from utils.qwen_utils import generate_prompt, parse_response_to_vector, compute_map
from utils.generate_video_sequences import generate_video_sequences
sys.path.append("..")
from dataset_utils.load_sequences_from_episodes import *



# -----------------------------
# Argument parsing
# -----------------------------
parser = argparse.ArgumentParser(description="Test QwenVL3 on ProxivideoFriends dataset.")
parser.add_argument('--t',type=int,  help='Frames sequence size', required=False, default=16)
parser.add_argument('--s',type=int,  help='Stride for the sliding window', required=False, default=16)
parser.add_argument('--set', type=int,  help='Set (1 or 2)', required=False, default=1)

parser.add_argument('--datasetDIR',type=str,  help='Main Dir where dataset is located', required=True, default="../../dataset/")
parser.add_argument('--resultDIR',type=str,  help='Dir where model will be saved', required=True, default="../../")
    
    
args = parser.parse_args()

# -----------------------------
# Basic setup
# -----------------------------
use_set= args.set
frames_per_sequence= args.t
stride= args.s

datasetDir=args.datasetDIR
resultDir=args.resultDIR

# -----------------------------
# Paths
# ----------------------------- 
shots_dir = os.path.join(datasetDir, 'shots/season3')
labels_dir = os.path.join(datasetDir, 'labels/')
base_dataset_path = os.path.join(datasetDir, 'output_circles_release')

questions_json_path="generated_questions_options_proxemics_circles.json"
# List of questions to be asked to the model, with their corresponding options and the expected output format, stored in a json file
with open(questions_json_path, "r") as file:
    questions_json=json.load(file)

suffix=f"set{use_set}_window{frames_per_sequence}_stride{stride}_proxemics" 

output_dir_models = os.path.join(resultDir, 'resultsQwenVL3')
output_responses_json_file = f"{output_dir_models}/results_using_pretrain_model/circles/output_responses_QwenVL3_{suffix}.json"
output_test_json_file =f"{output_dir_models}/results_using_pretrain_model/circles/results_QwenVL3_{suffix}.json"
os.makedirs(os.path.dirname(output_responses_json_file), exist_ok=True)
os.makedirs(os.path.dirname(output_test_json_file), exist_ok=True)


# -----------------------------
# 1. Model initialization
# -----------------------------
print("\n" + "=" * 70)
print("Model Initialization")
print("=" * 70)
print("[INFO] Loading model... Qwen3-VL-30B-A3B-Instruct")
model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-30B-A3B-Instruct",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
    device_map="auto",
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-30B-A3B-Instruct")


# -----------------------------------------------------------------------
# 2. Load dataset splits
# -----------------------------------------------------------------------
print("\n" + "=" * 70)
print("Dataset")
print("=" * 70)
print("[INFO] Loading sequences...")
_, _, test_set = load_sequences_from_episodes(shots_dir, labels_dir, use_set, frames_per_sequence=frames_per_sequence,stride=stride)
test_set=filter_set(test_set)
# Order the test set by episode number to ensure consistent ordering across runs
test_set = sorted(test_set, key=lambda x: x['episode'])
print(f"[INFO] Test sequences       : {len(test_set)}")


# -----------------------------
# 3. Generate videos if needed
# -----------------------------
print("\n" + "=" * 70)
print("Video Generation")
print("=" * 70)
print("[INFO] Generating video sequences...")
videos_path=f"{datasetDir}video_frames_sequences_circles/window{frames_per_sequence}_stride{stride}"
if os.path.exists(videos_path):
    print(f"\t[!] Videos already exist: {videos_path}")
else:
    generate_video_sequences(test_set, base_dataset_path, videos_path, fps=24)


# -----------------------------
# 4. Load previous responses (resume) or initialize new responses dictionary
# -----------------------------
print("\n" + "=" * 70)
print("Resume or Initialize Responses")
print("=" * 70)
responses = {}
if os.path.exists(output_responses_json_file):
    print(f"[INFO] Loading previous responses from {output_responses_json_file}")
    with open(output_responses_json_file, "r") as f:
        responses = json.load(f)
else:
    print(f"[INFO] No previous responses found from {output_responses_json_file}. Starting fresh.")


# -----------------------------
# Initialize storage for predictions and real labels
# -----------------------------
num_questions=1
all_predictions = [[] for _ in range(num_questions)]
real_labels = []

# -----------------------------
# 5. Main inference loop
# -----------------------------
print("\n" + "=" * 70)
print("Inference Loop")
print("=" * 70)
for video in tqdm(test_set, desc="Processing videos"):
    episode = video['episode']
    frames = video['frames']
    p0 = video['p0']
    p1 = video['p1']

    videoname=f"{episode}_{frames[0]}_{frames[-1]}_pair_{p0}-{p1}.mp4"
    video_file = os.path.join(videos_path, videoname)
    print('Processing: ',video_file )
      
    label=video['label']
    real_labels.append(label)
    
    video_responses={}

    for idx_question in range (num_questions):
        # -----------------------------
        # Check cached response
        # -----------------------------
        if videoname in responses and str(idx_question) in responses[videoname].keys():
            #print(f"Cached response found for {videoname}")
            # we can directly use the cached response
            response = responses[videoname][str(idx_question)]
        else:
            # -----------------------------
            # Generate prompt
            # -----------------------------
            question=generate_prompt(idx_question, questions_json)
            
            try:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": video_file,
                            },
                            {"type": "text", "text": question},
                        ],
                    }
                ]
                

                # Preparation for inference
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt"
                )  
                inputs = inputs.to(model.device)
                generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )

                response=output_text[0]

            except Exception:
                print("Inference error:")
                response = ""
        
        video_responses[str(idx_question)] = response
        # -----------------------------
        # Parse prediction
        # -----------------------------
        pred_vector = parse_response_to_vector(response)
        all_predictions[idx_question].append(pred_vector)
        
    # -----------------------------
    # Save responses progressively
    # -----------------------------
    responses[videoname] = video_responses

    with open(output_responses_json_file, 'w') as f:
        json.dump(responses, f, indent=4)

 
# -----------------------------
# Compute metrics (mAP)
# -----------------------------
print("\n" + "=" * 70)
print("Computing Metrics")
print("=" * 70)
print("[INFO] Computing mAP...")
result_json = compute_map(real_labels, all_predictions)


# -----------------------------
# Save results
# -----------------------------
print("\n" + "=" * 70)
print("Saving Results")
print("=" * 70)
print("[INFO] Saving results")
with open(output_test_json_file, "w") as f:
    json.dump(result_json, f, indent=4)

print("\t[INFO] Results saved to:", output_test_json_file)

