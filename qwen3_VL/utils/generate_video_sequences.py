"""
Utility functions for generating video clips from frame sequences.
"""
import os
import cv2
from pathlib import Path

# -----------------------------
# Video sequence generation
# -----------------------------
def generate_video_sequences(data, input_base, output_base, fps=24):
    """
    Generate video files from frame sequences.

    Each item in `data` is expected to contain:
        - episode: episode identifier
        - frames: list of frame indices
        - p0: first person id
        - p1: second person id

    Args:
        data (list): List of dictionaries describing the video sequences.
        input_base (str): Root directory containing the input frame folders.
        output_base (str): Directory where generated videos will be saved.
        fps (int, optional): Output video frame rate. Defaults to 24.
    """

    for entry in data:
        episode = entry['episode']
        frames = entry['frames']
        p0 = entry['p0']
        p1 = entry['p1']
        print(
            f"[i] Generating video for episode {episode}, "
            f"frames {frames[0]} to {frames[-1]}, pair {p0}-{p1}..."
        )
        # Build the input directory for the current episode
        episode_dir = os.path.join(input_base, f"episode{episode}", "release")

        # Build the output video filename and full path
        output_filename = f"{episode}_{frames[0]}_{frames[-1]}_pair_{p0}-{p1}.mp4"
        output_path = os.path.join(output_base, output_filename)
        Path(output_base).mkdir(parents=True, exist_ok=True)

        # Read the first frame to get the video dimensions
        first_frame_path = os.path.join(episode_dir, f"{frames[0]}_pair_{p0}-{p1}.jpg")
        if not os.path.exists(first_frame_path):
            print(f"[!] First frame not found: {first_frame_path}")
            continue

        frame = cv2.imread(first_frame_path)
        height, width, _ = frame.shape

        # Create the video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Write all frames into the output video
        for f in frames:
            img_path = os.path.join(episode_dir, f"{f}_pair_{p0}-{p1}.jpg")
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                out.write(img)
            else:
                print(f"[!] Frame not found: {img_path}")

        out.release()
        print(f"\t[✓] Video generated: {output_path}")
  