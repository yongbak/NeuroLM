"""
by Wei-Bang Jiang
https://github.com/935963004/NeuroLM
"""

#from pyhealth.metrics import binary_metrics_fn, multiclass_metrics_fn
import math
import numpy as np
import os
from downstream_dataset import TUABLoader, TUEVLoader, TUSLLoader, HMCLoader, WorkloadLoader
from metrics import binary_metrics_fn, multiclass_metrics_fn

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule

def get_metrics(output, target, metrics, is_binary):
    if is_binary:
        if 'roc_auc' not in metrics or sum(target) * (len(target) - sum(target)) != 0:  # to prevent all 0 or all 1 and raise the AUROC error
            results = binary_metrics_fn(
                target,
                output,
                metrics=metrics
            )
        else:
            results = {
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "pr_auc": 0.0,
                "roc_auc": 0.0,
            }
    else:
        results = multiclass_metrics_fn(
            target, output, metrics=metrics
        )
    return results

import os
import argparse
import numpy as np
import torch
import pickle
from pathlib import Path

from model.model_vq import VQ_Align
from model.model_neural_transformer import NTConfig

def load_vq_model(checkpoint_path, device='cuda', offline=True):
    """Load trained VQ model from checkpoint"""
    print(f"Loading VQ model from {checkpoint_path}")
    
    # weights_only=False to load full checkpoint, or True
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get model configuration from checkpoint
    encoder_args = checkpoint['encoder_args']
    decoder_args = checkpoint['decoder_args']
    
    print(f"Encoder config: {encoder_args}")
    print(f"Decoder config: {decoder_args}")
    
    # Create model
    encoder_conf = NTConfig(**encoder_args)
    decoder_conf = NTConfig(**decoder_args)
    model = VQ_Align(encoder_conf, decoder_conf, offline=offline)
    
    # Load state dict
    state_dict = checkpoint['model']
    
    # Fix keys if needed (remove unwanted prefix)
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f"✅ VQ model loaded successfully!")
    print(f"📊 Codebook size: {model.VQ.quantize.num_tokens}")
    
    return model

# output_pkl_path 지정 시 저장, None인 경우 저장하지 않음
def txt_to_full_pickle(txt_file_path, output_pkl_path=None, sampling_rate=2000.0, notch=60.0):
    """
    Convert a single TXT/CSV file to one full pickle file (no windowing).
    For inference: entire signal → single pkl file
    
    Args:
        txt_file_path: path to input txt/csv file
        output_pkl_path: path to output pickle file
        sampling_rate: target sampling rate (default 2000 Hz)
        notch: notch filter frequency (default 60 Hz)
    """
    import mne
    import numpy as np
    import pickle
    
    # Read signal
    with open(txt_file_path, 'r', encoding='utf-8') as f:
        signal = f.read().split("\n")[:-1]
        signal = [float(s) for s in signal]
    
    signal = np.asarray(signal, dtype=np.float32)
    
    # Create MNE RawArray
    ch_names = ['DEVICE']
    ch_types = ['misc']
    info = mne.create_info(ch_names=ch_names, sfreq=sampling_rate, ch_types=ch_types)
    raw = mne.io.RawArray(signal[np.newaxis, :], info)
    
    # Preprocessing
    nchan = raw.info.get('nchan', raw.get_data().shape[0])
    picks = np.arange(nchan)
    
    raw.filter(l_freq=0.5, h_freq=None, picks=picks, 
               filter_length='auto', l_trans_bandwidth='auto')
    raw.notch_filter(notch, picks=picks)
    raw.resample(sampling_rate, n_jobs=5)
    
    # Get data
    signals = raw.get_data()  # shape: (1, n_samples)
    
    # Save as pickle - entire signal as one sample
    sample = {
        "X": signals,  # shape: (1, n_samples)
        "ch_names": ['DEVICE'],
        "y": 1,  # dummy label
    }
    
    if output_pkl_path is not None:
        with open(output_pkl_path, 'wb') as f:
            pickle.dump(sample, f)
        
        print(f"✅ Saved full signal pickle: {output_pkl_path}")
        print(f"   Shape: {signals.shape}, Duration: {signals.shape[1]/sampling_rate:.2f}s")
        
    return sample

def extract_tokens_from_single_file(model, file_path, chunk_size=40, sequence_unit=200, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Extract tokens from a txt signal file by chunking into 40-token segments"""
    print(f"🔄 Processing: {file_path}")
    
    # Load data
    sample = txt_to_full_pickle(file_path, output_pkl_path=None)
    
    data = sample["X"]  # Shape: (channels, time_samples)
    ch_names = sample["ch_names"]
    
    # print(f"📊 Original data shape: {data.shape}, channels: {len(ch_names)}")
    # print(f"📊 Channel names: {ch_names}")
    
    # Preprocess like in dataset.py
    data = torch.FloatTensor(data / 100)
    time_segments = data.size(1) // sequence_unit
    
    # Rearrange to (time_segments * channels, sequence_unit)
    from einops import rearrange
    data = rearrange(data, 'N (A T) -> (A N) T', T=sequence_unit)
    # print(f"📊 After rearrange: {data.shape}, time segments: {time_segments}")
    
    n_channels = len(ch_names)
    total_tokens = data.size(0)
    
    # Calculate number of chunks (40 tokens each)
    num_chunks = (total_tokens + chunk_size - 1) // chunk_size
    # print(f"📦 Splitting into {num_chunks} chunks of {chunk_size} tokens each")
    
    # Create proper input_chans like in dataset.py
    from dataset import standard_1020
    
    def get_chans(ch_names_list):
        chans = []
        for ch_name in ch_names_list:
            try:
                chans.append(standard_1020.index(ch_name))
            except ValueError:
                # If channel not found, use 'pad' token
                chans.append(standard_1020.index('pad'))
        return chans
    
    # Process each chunk
    all_tokens = []
    
    for chunk_idx in range(num_chunks):
        print("[*] TRIAL %d" % chunk_idx)
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, total_tokens)
        chunk_data = data[start_idx:end_idx]
        actual_chunk_size = chunk_data.size(0)
        
        # print(f"\n📦 Chunk {chunk_idx + 1}/{num_chunks}: tokens [{start_idx}:{end_idx}] ({actual_chunk_size} tokens)")
        
        # Calculate time_segments for this chunk
        chunk_time_segments = (actual_chunk_size + n_channels - 1) // n_channels
        
        # Create input_chans: repeat channel names for each time segment
        input_chans_list = []
        for t_idx in range(chunk_time_segments):
            for ch_idx in range(n_channels):
                token_idx = t_idx * n_channels + ch_idx
                if token_idx < actual_chunk_size:
                    input_chans_list.append(ch_names[ch_idx])
        
        # Pad to chunk_size
        if len(input_chans_list) < chunk_size:
            input_chans_list.extend(['pad'] * (chunk_size - len(input_chans_list)))
        else:
            input_chans_list = input_chans_list[:chunk_size]
        
        # Convert to indices
        # IMPORTANT: pos_embed has size 1, so all channel indices must be 0
        input_chans_indices = [0] * len(input_chans_list)
        
        # Pad data to chunk_size
        X = torch.zeros((chunk_size, sequence_unit))
        X[:actual_chunk_size] = chunk_data
        X = X.unsqueeze(0)  # Add batch dimension: (1, chunk_size, sequence_unit)

        input_chans = torch.IntTensor(input_chans_indices).unsqueeze(0)  # (1, chunk_size)
        
        # Create input_time (time segment indices within this chunk)
        input_time_list = []
        for t_idx in range(chunk_time_segments):
            for ch_idx in range(n_channels):
                token_idx = t_idx * n_channels + ch_idx
                if token_idx < actual_chunk_size:
                    input_time_list.append(t_idx)
        
        # Pad to chunk_size
        if len(input_time_list) < chunk_size:
            input_time_list.extend([0] * (chunk_size - len(input_time_list)))
        else:
            input_time_list = input_time_list[:chunk_size]
        
        input_time = torch.IntTensor(input_time_list).unsqueeze(0)  # (1, chunk_size)
        
        # Create mask (True for actual data, False for padding)
        input_mask = torch.zeros(chunk_size).bool()
        input_mask[:actual_chunk_size] = True
        input_mask = input_mask.unsqueeze(0)  # (1, chunk_size)
        
        # Move to device
        X = X.to(device)
        input_chans = input_chans.to(device)
        input_time = input_time.to(device)
        input_mask = input_mask.to(device)
        
        # print(f"  Input time range: {input_time[input_mask].min().item()}-{input_time[input_mask].max().item()}")
        # print(f"  Active tokens: {input_mask.sum().item()}")
        
        # Extract discrete tokens
        with torch.no_grad():
            discrete_tokens = model.VQ.get_codebook_indices(
                X, input_chans, input_time, input_mask
            )
        
        # Only keep the actual tokens (not padding)
        chunk_tokens = discrete_tokens[0, :actual_chunk_size].cpu().numpy()
        all_tokens.append(chunk_tokens)
        
        # print(f"  ✅ Extracted {len(chunk_tokens)} tokens, range: {chunk_tokens.min()}-{chunk_tokens.max()}")
        # print(f"  Sample tokens: {chunk_tokens[:10]}")
    
    # Concatenate all chunks
    all_tokens = np.concatenate(all_tokens, axis=0)
    
    # print(f"\n🎯 Total tokens extracted: {len(all_tokens)}")
    # print(f"🎯 Token range: {all_tokens.min()} - {all_tokens.max()}")
    # print(f"🎯 First 20 tokens: {all_tokens[:20]}")
    
    # Create list[int] version for easy use
    token_sequence = all_tokens.tolist()  # Convert numpy array to list[int]
    # print(token_sequence)
    # print(len(token_sequence))
    
    return all_tokens.tolist(), {
        'ch_names': ch_names,
        'total_tokens': len(all_tokens),
        'num_chunks': num_chunks,
        'chunk_size': chunk_size,
        'file_path': file_path
    }


def main():
    parser = argparse.ArgumentParser('Simple token extraction')
    parser.add_argument('--checkpoint_path', required=True, 
                        help='Path to trained VQ model checkpoint')
    parser.add_argument('--input_file', required=True,
                        help='Path to input pickle file')
    parser.add_argument('--output_dir', default='./simple_tokens',
                        help='Output directory')
    parser.add_argument('--chunk_size', type=int, default=64,
                        help='Number of tokens per chunk (default: 64)')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                        help='Device to use')
    
    args = parser.parse_args()
    
    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    print(f"🔧 Using device: {args.device}")
    print(f"📦 Chunk size: {args.chunk_size} tokens")
    
    # Load VQ model
    model = load_vq_model(args.checkpoint_path, args.device)
    
    # Extract tokens from single file
    tokens, metadata = extract_tokens_from_single_file(
        model, args.input_file, args.device, args.chunk_size
    )
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save tokens
    output_file = os.path.join(args.output_dir, 'tokens.npy')
    np.save(output_file, tokens)
    print(f"\n💾 Saved {len(tokens)} tokens to: {output_file}")
    
    # Save metadata
    metadata_file = os.path.join(args.output_dir, 'metadata.npz')
    np.savez(metadata_file, **metadata)
    print(f"💾 Saved metadata to: {metadata_file}")
    
    print(f"\n✅ Token extraction completed!")
    print(f"📊 Summary:")
    print(f"  - Total tokens: {len(tokens)}")
    print(f"  - Number of chunks: {metadata['num_chunks']}")
    print(f"  - Chunk size: {metadata['chunk_size']}")
    print(f"  - Token range: {tokens.min()} - {tokens.max()}")


if __name__ == "__main__":
    # Example usage
    txt_signal_file = "/home/yongbak/research/NeuroLM/datasets/PMD_samples/s0_b_2024_07.csv"
    output_pkl = "/home/yongbak/research/NeuroLM/datasets/processed/tmp/s0_b_2024_07.pkl"
    #txt_to_full_pickle(txt_file, output_pkl)

    model = load_vq_model("./vq_output/checkpoints/VQ/ckpt_best.pt", device='cuda' if torch.cuda.is_available() else 'cpu')
    
    #print(model.VQ.get_tokens(txt_signal_file))

    from constants import NUM_OF_SAMPLES_PER_TOKEN, NUM_OF_TOTAL_TOKENS

    token_sequence, _ = extract_tokens_from_single_file(model, txt_signal_file, sequence_unit=NUM_OF_SAMPLES_PER_TOKEN, chunk_size=NUM_OF_TOTAL_TOKENS, device='cuda' if torch.cuda.is_available() else 'cpu')
    print(type(token_sequence))
    print(len(token_sequence))
    print(token_sequence)

    # To-Do
    # train.py early stopping
    # skipping pkl file creation
    # data augmentation - dataset_maker/prepare_from_txt_signals.py