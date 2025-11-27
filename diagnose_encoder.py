"""
Diagnose encoder output similarity and codebook collapse

python diagnose_encoder.py `
    --checkpoint "./vq_output/checkpoints/VQ/ckpt.pt" `
    --samples "path/to/signal1.pkl" "path/to/signal2.pkl" "path/to/signal3.pkl" "path/to/signal4.pkl" `
    --device cuda `
    --output encoder_diagnosis.png
"""

import torch
import numpy as np
import pickle
from pathlib import Path
from model.model_vq import VQ_Align
from model.model_neural_transformer import NTConfig
import matplotlib.pyplot as plt
import seaborn as sns


def load_vq_model(checkpoint_path, device='cuda'):
    """Load trained VQ model from checkpoint"""
    print(f"Loading VQ model from {checkpoint_path}")
    
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


def load_pickle_sample(pkl_path):
    """Load a single pickle file sample"""
    with open(pkl_path, 'rb') as f:
        sample = pickle.load(f)
    
    X = sample['X']  # Shape: (1, 8000) or similar
    ch_names = sample['ch_names']
    
    print(f"Loaded {pkl_path}")
    print(f"  Shape: {X.shape}, Channels: {ch_names}")
    
    return X, ch_names


def prepare_input_for_encoder(X, ch_names, block_size=40, sequence_unit=200):
    """
    Prepare pickle data for encoder input
    
    Args:
        X: numpy array, shape (n_channels, n_samples)
        ch_names: list of channel names
        block_size: number of tokens (default 40)
        sequence_unit: samples per token (default 200)
    
    Returns:
        X_tensor, input_chans, input_time, input_mask
    """
    from dataset import standard_1020
    from einops import rearrange
    
    # Normalize
    X_norm = X / 100.0
    
    # Convert to tensor
    X_tensor = torch.FloatTensor(X_norm)  # (n_channels, n_samples)
    
    # Rearrange to tokens
    # (n_channels, n_samples) → (n_tokens, sequence_unit)
    X_rearranged = rearrange(X_tensor, 'N (A T) -> (A N) T', T=sequence_unit)
    
    n_tokens = X_rearranged.shape[0]
    n_channels = len(ch_names)
    
    # Pad or truncate to block_size
    if n_tokens < block_size:
        # Pad
        X_padded = torch.zeros((block_size, sequence_unit))
        X_padded[:n_tokens] = X_rearranged
        actual_tokens = n_tokens
    else:
        # Truncate
        X_padded = X_rearranged[:block_size]
        actual_tokens = block_size
    
    # Create input_chans (channel indices)
    def get_chans(ch_names_list):
        chans = []
        for ch_name in ch_names_list:
            try:
                chans.append(standard_1020.index(ch_name))
            except ValueError:
                chans.append(standard_1020.index('pad'))
        return chans
    
    # Repeat channel names for each time segment
    time_segments = (actual_tokens + n_channels - 1) // n_channels
    input_chans_list = []
    for t_idx in range(time_segments):
        for ch_idx in range(n_channels):
            token_idx = t_idx * n_channels + ch_idx
            if token_idx < actual_tokens:
                input_chans_list.append(ch_names[ch_idx])
    
    # Pad to block_size
    while len(input_chans_list) < block_size:
        input_chans_list.append('pad')
    input_chans_list = input_chans_list[:block_size]
    
    input_chans_indices = get_chans(input_chans_list)
    input_chans = torch.IntTensor(input_chans_indices)
    
    # Create input_time
    input_time_list = []
    for t_idx in range(time_segments):
        for ch_idx in range(n_channels):
            token_idx = t_idx * n_channels + ch_idx
            if token_idx < actual_tokens:
                input_time_list.append(t_idx)
    
    # Pad to block_size
    while len(input_time_list) < block_size:
        input_time_list.append(0)
    input_time_list = input_time_list[:block_size]
    
    input_time = torch.IntTensor(input_time_list)
    
    # Create mask
    input_mask = torch.zeros(block_size).bool()
    input_mask[:actual_tokens] = True
    
    # Add batch dimension
    X_batch = X_padded.unsqueeze(0)  # (1, block_size, sequence_unit)
    input_chans_batch = input_chans.unsqueeze(0)  # (1, block_size)
    input_time_batch = input_time.unsqueeze(0)  # (1, block_size)
    input_mask_batch = input_mask.unsqueeze(0)  # (1, block_size)
    
    return X_batch, input_chans_batch, input_time_batch, input_mask_batch


def get_encoder_output(model, X, input_chans, input_time, input_mask, device='cuda'):
    """
    Get encoder output (before quantization)
    
    Returns:
        encoder_output: (batch, n_tokens, embed_dim)
        quantized_indices: (batch, n_tokens)
    """
    X = X.to(device)
    input_chans = input_chans.to(device)
    input_time = input_time.to(device)
    input_mask = input_mask.to(device)
    
    with torch.no_grad():
        # Get encoder features
        encoder_features = model.VQ.encoder(X, input_chans, input_time, input_mask, 
                                           return_all_tokens=True)
        
        # Project to quantizer space
        to_quantizer_features = model.VQ.encode_task_layer(encoder_features)
        
        # Get quantized indices
        quantize, loss, embed_ind = model.VQ.quantize(to_quantizer_features)
    
    return to_quantizer_features.cpu(), embed_ind.cpu()


def compute_similarity_metrics(embeddings_list):
    """
    Compute various similarity metrics between encoder outputs
    
    Args:
        embeddings_list: list of (n_tokens, embed_dim) tensors
    
    Returns:
        dict with similarity metrics
    """
    n_samples = len(embeddings_list)
    
    # Flatten embeddings: (n_tokens * embed_dim,)
    flat_embeddings = [emb.flatten() for emb in embeddings_list]
    
    # Compute pairwise cosine similarities
    cosine_sim_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            emb_i = flat_embeddings[i].numpy()
            emb_j = flat_embeddings[j].numpy()
            
            # Cosine similarity
            cos_sim = np.dot(emb_i, emb_j) / (np.linalg.norm(emb_i) * np.linalg.norm(emb_j))
            cosine_sim_matrix[i, j] = cos_sim
    
    # Compute pairwise L2 distances
    l2_distance_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            emb_i = flat_embeddings[i].numpy()
            emb_j = flat_embeddings[j].numpy()
            
            # L2 distance
            l2_dist = np.linalg.norm(emb_i - emb_j)
            l2_distance_matrix[i, j] = l2_dist
    
    # Compute mean and std of embeddings (per sample)
    embedding_stats = []
    for i, emb in enumerate(embeddings_list):
        emb_np = emb.numpy()
        stats = {
            'mean': np.mean(emb_np),
            'std': np.std(emb_np),
            'min': np.min(emb_np),
            'max': np.max(emb_np),
            'norm': np.linalg.norm(emb_np)
        }
        embedding_stats.append(stats)
    
    return {
        'cosine_similarity': cosine_sim_matrix,
        'l2_distance': l2_distance_matrix,
        'embedding_stats': embedding_stats
    }


def analyze_codebook_usage(indices_list, codebook_size):
    """
    Analyze codebook usage patterns
    
    Args:
        indices_list: list of (n_tokens,) tensors with codebook indices
        codebook_size: total number of codes in codebook
    
    Returns:
        dict with codebook usage statistics
    """
    # Flatten all indices
    all_indices = torch.cat([idx.flatten() for idx in indices_list])
    
    # Count unique codes used
    unique_codes = torch.unique(all_indices)
    n_unique = len(unique_codes)
    
    # Compute usage histogram
    usage_counts = torch.zeros(codebook_size)
    for idx in all_indices:
        usage_counts[idx] += 1
    
    # Find most and least used codes
    top_k = 10
    top_codes = torch.topk(usage_counts, k=min(top_k, codebook_size))
    
    # Calculate entropy (measure of diversity)
    probs = usage_counts / usage_counts.sum()
    probs = probs[probs > 0]  # Remove zero probabilities
    entropy = -torch.sum(probs * torch.log(probs + 1e-10))
    max_entropy = np.log(codebook_size)
    normalized_entropy = entropy / max_entropy
    
    return {
        'total_codes': codebook_size,
        'unique_codes_used': n_unique,
        'usage_percentage': (n_unique / codebook_size) * 100,
        'top_codes': top_codes.indices.tolist(),
        'top_counts': top_codes.values.tolist(),
        'entropy': entropy.item(),
        'normalized_entropy': normalized_entropy.item(),
        'usage_counts': usage_counts
    }


def visualize_results(similarity_metrics, codebook_stats, sample_names):
    """Create visualization plots"""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Cosine similarity heatmap
    ax = axes[0, 0]
    sns.heatmap(similarity_metrics['cosine_similarity'], 
                annot=True, fmt='.3f', cmap='coolwarm',
                xticklabels=sample_names, yticklabels=sample_names,
                vmin=-1, vmax=1, ax=ax)
    ax.set_title('Cosine Similarity between Encoder Outputs')
    
    # 2. L2 distance heatmap
    ax = axes[0, 1]
    sns.heatmap(similarity_metrics['l2_distance'],
                annot=True, fmt='.2f', cmap='viridis',
                xticklabels=sample_names, yticklabels=sample_names,
                ax=ax)
    ax.set_title('L2 Distance between Encoder Outputs')
    
    # 3. Codebook usage histogram
    ax = axes[1, 0]
    usage_counts = codebook_stats['usage_counts'].numpy()
    non_zero_counts = usage_counts[usage_counts > 0]
    ax.hist(non_zero_counts, bins=50, edgecolor='black')
    ax.set_xlabel('Usage Count')
    ax.set_ylabel('Number of Codes')
    ax.set_title(f'Codebook Usage Distribution\n'
                 f'({codebook_stats["unique_codes_used"]}/{codebook_stats["total_codes"]} codes used, '
                 f'{codebook_stats["usage_percentage"]:.1f}%)')
    ax.set_yscale('log')
    
    # 4. Embedding statistics
    ax = axes[1, 1]
    stats = similarity_metrics['embedding_stats']
    means = [s['mean'] for s in stats]
    stds = [s['std'] for s in stats]
    norms = [s['norm'] for s in stats]
    
    x = np.arange(len(sample_names))
    width = 0.25
    
    ax.bar(x - width, means, width, label='Mean', alpha=0.8)
    ax.bar(x, stds, width, label='Std', alpha=0.8)
    ax.bar(x + width, [n/1000 for n in norms], width, label='Norm/1000', alpha=0.8)
    
    ax.set_xlabel('Sample')
    ax.set_ylabel('Value')
    ax.set_title('Encoder Output Statistics')
    ax.set_xticks(x)
    ax.set_xticklabels(sample_names, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig


def main():
    import argparse
    
    parser = argparse.ArgumentParser('Diagnose Encoder Output Similarity')
    parser.add_argument('--checkpoint', required=True, help='Path to VQ model checkpoint')
    parser.add_argument('--samples', nargs='+', required=True, 
                       help='Paths to pickle files (at least 2)')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--output', default='encoder_diagnosis.png', 
                       help='Output plot filename')
    
    args = parser.parse_args()
    
    if len(args.samples) < 2:
        print("Error: Need at least 2 samples to compare")
        return
    
    print(f"{'='*60}")
    print(f"Encoder Output Similarity Diagnosis")
    print(f"{'='*60}\n")
    
    # Load model
    model = load_vq_model(args.checkpoint, args.device)
    codebook_size = model.VQ.quantize.num_tokens
    
    # Load samples and get encoder outputs
    embeddings_list = []
    indices_list = []
    sample_names = []
    
    for i, pkl_path in enumerate(args.samples):
        print(f"\n[Sample {i+1}] {pkl_path}")
        
        # Load sample
        X, ch_names = load_pickle_sample(pkl_path)
        
        # Prepare input
        X_batch, input_chans, input_time, input_mask = prepare_input_for_encoder(X, ch_names)
        
        # Get encoder output
        encoder_output, quantized_indices = get_encoder_output(
            model, X_batch, input_chans, input_time, input_mask, args.device
        )
        
        embeddings_list.append(encoder_output[0])  # Remove batch dim
        indices_list.append(quantized_indices[0])  # Remove batch dim
        sample_names.append(f"Sample_{i+1}")
        
        print(f"  Encoder output shape: {encoder_output.shape}")
        print(f"  Quantized indices shape: {quantized_indices.shape}")
        print(f"  Unique codes used: {len(torch.unique(quantized_indices))}")
    
    # Compute similarity metrics
    print(f"\n{'='*60}")
    print("Computing Similarity Metrics...")
    print(f"{'='*60}\n")
    
    similarity_metrics = compute_similarity_metrics(embeddings_list)
    
    print("Cosine Similarity Matrix:")
    print(similarity_metrics['cosine_similarity'])
    print()
    
    print("L2 Distance Matrix:")
    print(similarity_metrics['l2_distance'])
    print()
    
    print("Embedding Statistics:")
    for i, stats in enumerate(similarity_metrics['embedding_stats']):
        print(f"  {sample_names[i]}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
              f"norm={stats['norm']:.2f}")
    print()
    
    # Analyze codebook usage
    print(f"{'='*60}")
    print("Analyzing Codebook Usage...")
    print(f"{'='*60}\n")
    
    codebook_stats = analyze_codebook_usage(indices_list, codebook_size)
    
    print(f"Total codebook size: {codebook_stats['total_codes']}")
    print(f"Unique codes used: {codebook_stats['unique_codes_used']}")
    print(f"Usage percentage: {codebook_stats['usage_percentage']:.2f}%")
    print(f"Entropy (normalized): {codebook_stats['normalized_entropy']:.4f}")
    print(f"Top 10 most used codes: {codebook_stats['top_codes']}")
    print(f"Top 10 usage counts: {codebook_stats['top_counts']}")
    print()
    
    # Diagnosis
    print(f"{'='*60}")
    print("DIAGNOSIS")
    print(f"{'='*60}\n")
    
    # Check encoder similarity
    avg_cosine_sim = np.mean(similarity_metrics['cosine_similarity'][np.triu_indices(len(sample_names), k=1)])
    print(f"Average cosine similarity between different samples: {avg_cosine_sim:.4f}")
    
    if avg_cosine_sim > 0.95:
        print("⚠️  WARNING: Encoder outputs are very similar (>0.95)")
        print("   → Problem: Encoder is producing nearly identical outputs for different inputs")
        print("   → This suggests encoder collapse")
    elif avg_cosine_sim > 0.8:
        print("⚠️  CAUTION: Encoder outputs are quite similar (>0.8)")
        print("   → Encoder may not be learning diverse representations")
    else:
        print("✅ Encoder outputs show reasonable diversity")
    print()
    
    # Check codebook usage
    if codebook_stats['usage_percentage'] < 10:
        print(f"⚠️  CRITICAL: Only {codebook_stats['usage_percentage']:.1f}% of codebook is used")
        print("   → Problem: Severe codebook collapse")
    elif codebook_stats['usage_percentage'] < 50:
        print(f"⚠️  WARNING: Only {codebook_stats['usage_percentage']:.1f}% of codebook is used")
        print("   → Problem: Moderate codebook collapse")
    else:
        print(f"✅ Good codebook usage: {codebook_stats['usage_percentage']:.1f}%")
    print()
    
    if codebook_stats['normalized_entropy'] < 0.3:
        print(f"⚠️  WARNING: Low entropy ({codebook_stats['normalized_entropy']:.3f})")
        print("   → Codebook usage is very unbalanced")
    else:
        print(f"✅ Reasonable entropy: {codebook_stats['normalized_entropy']:.3f}")
    print()
    
    # Visualize
    print(f"{'='*60}")
    print("Creating Visualization...")
    print(f"{'='*60}\n")
    
    fig = visualize_results(similarity_metrics, codebook_stats, sample_names)
    fig.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"✅ Plot saved to: {args.output}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
