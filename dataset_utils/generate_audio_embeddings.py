"""
audio_embeddings.py

Utilities for extracting and saving audio embeddings from sequence-aligned
audio segments using Whisper.
"""

import os
import torchaudio
import whisper
import torch
import numpy as np

def generate_audio_embeddings(sequences, audio_dir, embedding_path, fps=24, model_name="base"):
    """
    Extract and save Whisper-based audio embeddings for each sequence
    if they do not already exist.

    Args:
        sequences: List of sequence dictionaries. Each sequence must contain
            at least "episode" and "frames".
        audio_dir: Directory containing episode-level audio files.
        embedding_path: Output directory where embeddings will be saved.
        fps: Video frame rate used to align frame indices with audio time.
        model_name: Whisper model variant to load.
    """
    # Load Whisper model on GPU if available, otherwise use CPU
    model = whisper.load_model(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Create output directory if it does not exist
    os.makedirs(embedding_path, exist_ok=True)

    # Cache files per episode to optimize search
    files_by_episode = {}

    for seq in sequences:
        episode = seq["episode"]
        start_frame = int(seq["frames"][0])
        end_frame = int(seq["frames"][-1])
        
        # Save one embedding per temporal sequence
        out_path = os.path.join(embedding_path, f"{episode}_{seq['frames'][0]}_{seq['frames'][-1]}.npy")
        # Skip if the embedding was already generated
        if os.path.exists(out_path):
            continue  # Already generated

        # Get list of audio files for this episode
        if episode not in files_by_episode:
            # Files expected as episode{ep}_{start}_{end}.wav
            files_by_episode[episode] = [f for f in os.listdir(audio_dir) if f.startswith(f"episode{episode}_") and f.endswith(".wav")]

        # Find the file that contains the sequence
        audio_path = None
        t_start = 0
        for f in files_by_episode[episode]:
            try:
                # Parse boundaries from filename: episodeXX_START_END.wav
                parts = f.replace(".wav", "").split("_")
                ts = int(parts[1])
                te = int(parts[2])
                if ts <= start_frame and end_frame <= te:
                    audio_path = os.path.join(audio_dir, f)
                    t_start = ts
                    break
            except (IndexError, ValueError):
                continue

        if not audio_path:
            # print(f"[WARNING] No tramo found for episode {episode} frames {start_frame}-{end_frame}")
            continue
        
        print(f"[INFO] Generating embedding for {out_path} using {os.path.basename(audio_path)}")

        # Load tramo audio
        audio, sr = torchaudio.load(audio_path)
        
        # Relative offset in the tramo
        rel_start = start_frame - t_start
        rel_end = end_frame - t_start
        
        start_sample = int(rel_start / fps * sr)
        end_sample = int(rel_end / fps * sr)
        
        # Crop the audio segment
        segment = audio[:, start_sample:end_sample]

        # Resample to 16 kHz if needed, since Whisper expects 16 kHz audio
        if sr != 16000:
            segment = torchaudio.functional.resample(segment, sr, 16000)
        # Convert to mono by averaging channels when the audio is stereotéreo)
        if segment.shape[0] > 1:
            segment = segment.mean(dim=0, keepdim=True)

        # Extract features with Whisper
        if segment.shape[1] >= 30 * 16000:  # 30s
            # For long segments, use Whisper encoder outputs directly
            # pad_or_trim ensures the input matches Whisper's expected duration
            segment = whisper.pad_or_trim(segment.squeeze(0))
            mel = whisper.log_mel_spectrogram(segment).to(model.device)
            with torch.no_grad():
                emb = model.encoder(mel.unsqueeze(0)) # shape: (1, T', D)
                # Temporal average pooling to obtain a fixed-length vector
                emb_mean = emb.mean(dim=1).squeeze(0).cpu().numpy() # (D,)
        else:
            # For short segments, fall back to averaging the mel spectrogram
            # over time to obtain a fixed-length representation
            # squeeze removes the channel dimension -> Whisper expects (samples,)
            mel = whisper.log_mel_spectrogram(segment.squeeze(0)).to(model.device)  # (80, T)
            # Average over time to get fixed-length representation
            emb_mean = mel.mean(dim=1).cpu().numpy()  # shape (80,)

        # Save embedding as a NumPy array
        np.save(out_path, emb_mean)

       