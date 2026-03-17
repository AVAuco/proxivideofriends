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

    for seq in sequences:
        episode = seq["episode"]
        start_frame = seq["frames"][0]
        end_frame = seq["frames"][-1]
        # Expected audio file for the full episode
        audio_path = os.path.join(audio_dir, f"episode{episode}audio.wav")
        if not os.path.exists(audio_path):
            print(f"[WARNING] Audio not found: {audio_path}")
            continue
        # Save one embedding per temporal sequence
        out_path = os.path.join(embedding_path, f"{episode}_{start_frame}_{end_frame}.npy")
        # Skip if the embedding was already generated
        if os.path.exists(out_path):
            continue  # Already generated
        
        print(f"[INFO] Generating embedding for {out_path}")

        # Load full episode audio
        # audio shape: (channels, num_samples)
        audio, sr = torchaudio.load(audio_path)  # audio.shape: (channels, samples rate)
        # Convert frame indices into audio sample positions
        start_sample = int(int(start_frame) / fps * sr) # *sr para pasar a Hz del vídeo
        end_sample = int(int(end_frame) / fps * sr)
        
        # Crop the audio segment aligned with the visual sequence
        segment = audio[:, start_sample:end_sample]  # (channels, segment_samples)

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

       