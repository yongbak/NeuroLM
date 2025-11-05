"""
Simple script to extract discrete tokens from neural signals
"""

import os
import argparse
import numpy as np
import torch
import pickle
from pathlib import Path

from model.model_vq import VQ_Align
from model.model_neural_transformer import NTConfig


def load_vq_model(checkpoint_path, device='cuda'):
    """Load trained VQ model from checkpoint"""
    print(f"Loading VQ model from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model configuration from checkpoint
    encoder_args = checkpoint['encoder_args']
    decoder_args = checkpoint['decoder_args']
    
    print(f"Encoder config: {encoder_args}")
    print(f"Decoder config: {decoder_args}")
    
    # Create model
    encoder_conf = NTConfig(**encoder_args)
    decoder_conf = NTConfig(**decoder_args)
    model = VQ_Align(encoder_conf, decoder_conf)
    
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


def extract_tokens_from_single_file(model, file_path, device='cuda', max_length=1024):
    """Extract tokens from a single pickle file"""
    print(f"🔄 Processing: {file_path}")
    
    # Load data
    with open(file_path, 'rb') as f:
        sample = pickle.load(f)
    
    data = sample["X"]  # Shape: (channels, time_samples)
    ch_names = sample["ch_names"]
    
    print(f"📊 Original data shape: {data.shape}, channels: {len(ch_names)}")
    print(f"📊 Channel names: {ch_names}")
    
    # Preprocess like in dataset.py
    data = torch.FloatTensor(data / 100)
    time_segments = data.size(1) // 200
    
    # Rearrange to (time_segments * channels, 200)
    from einops import rearrange
    data = rearrange(data, 'N (A T) -> (A N) T', T=200)
    print(f"📊 After rearrange: {data.shape}, time segments: {time_segments}")
    
    # Truncate if too long
    if data.size(0) > max_length:
        data = data[:max_length]
        # Also truncate channel list accordingly
        n_channels = len(ch_names)
        max_time_segments = max_length // n_channels
        time_segments = min(time_segments, max_time_segments)
        print(f"⚠️  Truncated to {max_length} tokens ({max_time_segments} time segments)")
    
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
    
    # Create input_chans: repeat channel names for each time segment
    input_chans_list = list(ch_names) * time_segments
    # Pad to max_length
    if len(input_chans_list) < max_length:
        input_chans_list.extend(['pad'] * (max_length - len(input_chans_list)))
    else:
        input_chans_list = input_chans_list[:max_length]
    
    # Convert to indices
    input_chans_indices = get_chans(input_chans_list)
    
    # Create input tensors
    batch_size = 1
    n_tokens = data.size(0)
    
    # Pad data to max_length
    X = torch.zeros((max_length, 200))
    X[:n_tokens] = data
    X = X.unsqueeze(0)  # Add batch dimension: (1, max_length, 200)
    
    input_chans = torch.IntTensor(input_chans_indices).unsqueeze(0)  # (1, max_length)
    
    # Create input_time (time segment indices)
    input_time_list = []
    for i in range(time_segments):
        input_time_list.extend([i] * len(ch_names))
    # Pad to max_length
    if len(input_time_list) < max_length:
        input_time_list.extend([0] * (max_length - len(input_time_list)))
    else:
        input_time_list = input_time_list[:max_length]
    
    input_time = torch.IntTensor(input_time_list).unsqueeze(0)  # (1, max_length)
    
    # Create mask (True for actual data, False for padding)
    input_mask = torch.zeros(max_length).bool()
    input_mask[:n_tokens] = True
    input_mask = input_mask.unsqueeze(0)  # (1, max_length)
    
    # Move to device
    X = X.to(device)
    input_chans = input_chans.to(device)
    input_time = input_time.to(device)
    input_mask = input_mask.to(device)
    
    print(f"📊 Input tensor shapes:")
    print(f"  X: {X.shape}")
    print(f"  input_chans: {input_chans.shape}, range: {input_chans.min().item()}-{input_chans.max().item()}")
    print(f"  input_time: {input_time.shape}, range: {input_time.min().item()}-{input_time.max().item()}")
    print(f"  input_mask: {input_mask.shape}, active tokens: {input_mask.sum().item()}")
    
    # Extract discrete tokens
    with torch.no_grad():
        discrete_tokens = model.VQ.get_codebook_indices(
            X, input_chans, input_time, input_mask
        )
    
    print(f"🎯 Output tokens shape: {discrete_tokens.shape}")
    print(f"🎯 Token range: {discrete_tokens.min().item()} - {discrete_tokens.max().item()}")
    print(f"🎯 Sample tokens: {discrete_tokens[0, :20].cpu().numpy()}")
    
    return discrete_tokens.cpu().numpy(), {
        'input_chans': input_chans.cpu().numpy(),
        'input_time': input_time.cpu().numpy(),
        'input_mask': input_mask.cpu().numpy(),
        'original_shape': data.shape,
        'ch_names': ch_names,
        'time_segments': time_segments,
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
    parser.add_argument('--max_length', type=int, default=1024,
                        help='Maximum sequence length')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                        help='Device to use')
    
    args = parser.parse_args()
    
    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    print(f"🔧 Using device: {args.device}")
    
    # Load VQ model
    model = load_vq_model(args.checkpoint_path, args.device)
    
    # Extract tokens from single file
    tokens, metadata = extract_tokens_from_single_file(
        model, args.input_file, args.device, args.max_length
    )
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save tokens
    output_file = os.path.join(args.output_dir, 'tokens.npy')
    np.save(output_file, tokens)
    print(f"💾 Saved tokens to: {output_file}")
    
    # Save metadata
    metadata_file = os.path.join(args.output_dir, 'metadata.npz')
    np.savez(metadata_file, **metadata)
    print(f"💾 Saved metadata to: {metadata_file}")
    
    print(f"✅ Token extraction completed!")


if __name__ == '__main__':
    main()