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


def extract_tokens_from_single_file(model, file_path, device='cuda', chunk_size=64):
    """Extract tokens from a single pickle file by chunking into 64-token segments"""
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
    
    n_channels = len(ch_names)
    total_tokens = data.size(0)
    
    # Calculate number of chunks (64 tokens each)
    num_chunks = (total_tokens + chunk_size - 1) // chunk_size
    print(f"📦 Splitting into {num_chunks} chunks of {chunk_size} tokens each")
    
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
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, total_tokens)
        chunk_data = data[start_idx:end_idx]
        actual_chunk_size = chunk_data.size(0)
        
        print(f"\n📦 Chunk {chunk_idx + 1}/{num_chunks}: tokens [{start_idx}:{end_idx}] ({actual_chunk_size} tokens)")
        
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
        input_chans_indices = get_chans(input_chans_list)
        
        # Pad data to chunk_size
        X = torch.zeros((chunk_size, 200))
        X[:actual_chunk_size] = chunk_data
        X = X.unsqueeze(0)  # Add batch dimension: (1, chunk_size, 200)
        
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
        
        print(f"  Input time range: {input_time[input_mask].min().item()}-{input_time[input_mask].max().item()}")
        print(f"  Active tokens: {input_mask.sum().item()}")
        
        # Extract discrete tokens
        with torch.no_grad():
            discrete_tokens = model.VQ.get_codebook_indices(
                X, input_chans, input_time, input_mask
            )
        
        # Only keep the actual tokens (not padding)
        chunk_tokens = discrete_tokens[0, :actual_chunk_size].cpu().numpy()
        all_tokens.append(chunk_tokens)
        
        print(f"  ✅ Extracted {len(chunk_tokens)} tokens, range: {chunk_tokens.min()}-{chunk_tokens.max()}")
        print(f"  Sample tokens: {chunk_tokens[:10]}")
    
    # Concatenate all chunks
    all_tokens = np.concatenate(all_tokens, axis=0)
    
    print(f"\n🎯 Total tokens extracted: {len(all_tokens)}")
    print(f"🎯 Token range: {all_tokens.min()} - {all_tokens.max()}")
    print(f"🎯 First 20 tokens: {all_tokens[:20]}")
    
    return all_tokens, {
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


if __name__ == '__main__':
    main()