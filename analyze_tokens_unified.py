"""
Unified VQ-VAE Token Analysis for PMD Dataset
Analyzes token distribution for different checkpoints
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

from utils import load_vq_model, extract_tokens_from_single_file, txt_to_full_pickle
from constants import NUM_OF_SAMPLES_PER_TOKEN, NUM_OF_TOTAL_TOKENS
import pickle
import torch.nn.functional as F

# ============================================================================
# 🔧 설정: 이 값을 바꿔서 다른 체크포인트를 분석하세요
# 옵션: 9, 19, 29, 39, 49, "best"
CHECKPOINT_VERSION = 29  # ← 이 값을 바꾸세요!

# 옵션: train, test, val
TYPE = "test"

ONLY_ORIGINAL = False
# 모든 결과는 ./token_analysis/{type} 디렉토리에 저장됩니다
# ============================================================================

def get_checkpoint_path(version):
    """Get checkpoint path from version"""
    if version == "best":
        return "./vq_output/checkpoints/VQ/ckpt_best.pt"
    else:
        return f"./vq_output/checkpoints/VQ/ckpt-{version}.pt"

def get_output_dir(memo):
    """Get output directory from version - all in ./token_analysis"""
    return "./logs/token_analysis_"+memo

def get_label_from_filename(filename):
    """Extract raw label character from filename (b/cc/m/s)"""
    parts = filename.split('-')
    if len(parts) > 1:      # is_augmented == True
        name = parts[1]
    else:
        name = parts[0]
    return name.split('_')[1]


def compute_reconstruction_loss(model, file_path, device='cuda'):
    """Compute reconstruction loss using VQ model's forward pass (same as training)"""
    try:
        # Load signal
        if file_path.endswith('.pkl'):
            with open(file_path, 'rb') as f:
                sample = pickle.load(f)
            signal = sample['X']  # (n_channels, n_samples)
            ch_names = sample.get('ch_names', ['DEVICE'])
        else:
            sample = txt_to_full_pickle(file_path, output_pkl_path=None)
            signal = sample['X']
            ch_names = sample.get('ch_names', ['DEVICE'])
        
        # Convert to tensor
        X = torch.FloatTensor(signal).unsqueeze(0).to(device)  # (1, n_channels, n_samples)
        n_channels, n_samples = signal.shape
        
        # Prepare targets (y_freq and y_raw are same as X for evaluation)
        Y_freq = X.clone()
        Y_raw = X.clone()
        
        # Prepare input_chans and input_time
        chunk_size = NUM_OF_TOTAL_TOKENS
        sequence_unit = NUM_OF_SAMPLES_PER_TOKEN
        chunk_time_segments = chunk_size // n_channels
        
        input_chans_list = []
        input_time_list = []
        for t_idx in range(chunk_time_segments):
            for ch_idx in range(n_channels):
                input_chans_list.append(ch_idx)
                input_time_list.append(t_idx)
        
        input_chans = torch.IntTensor(input_chans_list).unsqueeze(0).to(device)
        input_time = torch.IntTensor(input_time_list).unsqueeze(0).to(device)
        input_mask = torch.ones(1, chunk_size).bool().to(device)
        
        # Forward pass through VQ model (same as training)
        # Ensure model is in eval mode
        model.eval()
        model.VQ.eval()
        
        with torch.no_grad():
            loss, encoder_features, log = model.VQ(X, Y_freq, Y_raw, input_chans, input_time, input_mask)
        
        # Extract losses from log (same format as training)
        raw_loss = log['val/rec_raw_loss']
        freq_loss = log['val/rec_freq_loss']
        vq_loss = log['val/quant_loss']
        total_loss = log['val/total_loss']
        
        return raw_loss, freq_loss, vq_loss, total_loss
    
    except Exception as e:
        print(f"Error computing loss: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

def process_all_files(model, data_dir, device='cuda'):
    """Process all files and extract tokens with labels"""
    csv_files = list(Path(data_dir).glob('*.csv'))
    pkl_files = list(Path(data_dir).glob('*.pkl'))
    files = csv_files + pkl_files

    print(f"🔍 Found {len(csv_files)} CSV files")
    print(f"🔍 Found {len(pkl_files)} PKL files")

    tokens_by_label = defaultdict(list)
    file_info = []
    losses_by_label = defaultdict(list)
    
    for i, file in enumerate(files):
        filename = file.name
        raw_label = get_label_from_filename(filename)
        # Convert raw label to normal/abnormal
        label = 'normal' if raw_label == 'b' else 'abnormal'

        print(f"[{i+1}/{len(files)}] {filename} ({raw_label} -> {label})", end=' ')

        try:
            token_sequence, metadata = extract_tokens_from_single_file(
                model, 
                str(file), 
                use_pkl=False if file.suffix == '.csv' else True, 
                sequence_unit=NUM_OF_SAMPLES_PER_TOKEN, 
                chunk_size=NUM_OF_TOTAL_TOKENS,
                device=device
            )
            
            # Compute reconstruction losses
            raw_loss, freq_loss, vq_loss, total_loss = compute_reconstruction_loss(
                model, str(file), device=device
            )
            
            tokens_by_label[label].append(token_sequence)
            file_info.append({
                'filename': filename,
                'label': label,
                'num_tokens': len(token_sequence),
                'unique_tokens': len(set(token_sequence)),
                'token_sequence': token_sequence,
                'raw_loss': raw_loss,
                'freq_loss': freq_loss,
                'vq_loss': vq_loss,
                'total_loss': total_loss
            })
            
            if raw_loss is not None:
                losses_by_label[label].append({
                    'raw': raw_loss,
                    'freq': freq_loss,
                    'vq': vq_loss,
                    'total': total_loss
                })
            
            print(f"✅ ({len(token_sequence)} tokens, {len(set(token_sequence))} unique, raw={raw_loss:.4f}, freq={freq_loss:.4f}, total={total_loss:.4f})")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    return tokens_by_label, file_info, losses_by_label

def analyze_token_distribution(tokens_by_label):
    """Analyze token distribution for each label"""
    print("\n" + "="*80)
    print("📊 TOKEN DISTRIBUTION ANALYSIS")
    print("="*80)
    
    analysis_results = {}
    
    for label in ['normal', 'abnormal']:
        if label not in tokens_by_label or len(tokens_by_label[label]) == 0:
            print(f"\n⚠️  No files for label: {label}")
            continue
        
        print(f"\n{'='*80}")
        print(f"Label: {label.upper()}")
        print(f"{'='*80}")
        
        all_tokens = []
        for token_seq in tokens_by_label[label]:
            all_tokens.extend(token_seq)
        
        all_tokens = np.array(all_tokens)
        
        print(f"\n1️⃣  BASIC STATISTICS:")
        print(f"   - Total files: {len(tokens_by_label[label])}")
        print(f"   - Total tokens: {len(all_tokens)}")
        print(f"   - Unique tokens: {len(set(all_tokens))}")
        print(f"   - Token range: {all_tokens.min()} - {all_tokens.max()}")
        
        token_counts = Counter(all_tokens)
        sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n2️⃣  TOKEN FREQUENCY (Top 20):")
        for rank, (token, count) in enumerate(sorted_tokens[:20], 1):
            percentage = (count / len(all_tokens)) * 100
            print(f"   {rank:2d}. Token {token:4d}: {count:6d} times ({percentage:5.2f}%)")
        
        top_10_count = sum(count for _, count in sorted_tokens[:10])
        top_10_percentage = (top_10_count / len(all_tokens)) * 100
        print(f"\n   Top 10 tokens concentration: {top_10_percentage:.2f}%")
        
        print(f"\n3️⃣  DISTRIBUTION STATISTICS:")
        print(f"   - Mean frequency: {np.mean(list(token_counts.values())):.2f}")
        print(f"   - Median frequency: {np.median(list(token_counts.values())):.2f}")
        print(f"   - Std dev: {np.std(list(token_counts.values())):.2f}")
        print(f"   - Min frequency: {min(token_counts.values())}")
        print(f"   - Max frequency: {max(token_counts.values())}")
        
        total_codebook = 1024
        usage_percentage = (len(set(all_tokens)) / total_codebook) * 100
        print(f"\n4️⃣  CODEBOOK USAGE:")
        print(f"   - Used tokens: {len(set(all_tokens))}/{total_codebook} ({usage_percentage:.2f}%)")
        print(f"   - Unused tokens: {total_codebook - len(set(all_tokens))}")
        
        analysis_results[label] = {
            'all_tokens': all_tokens,
            'token_counts': token_counts,
            'sorted_tokens': sorted_tokens,
            'unique_count': len(set(all_tokens)),
            'total_tokens': len(all_tokens)
        }
    
    return analysis_results

def analyze_reconstruction_losses(losses_by_label):
    """Analyze reconstruction losses for each label"""
    print("\n" + "="*80)
    print("📉 RECONSTRUCTION LOSS ANALYSIS")
    print("="*80)
    
    for label in ['normal', 'abnormal']:
        if label not in losses_by_label or len(losses_by_label[label]) == 0:
            print(f"\n⚠️  No loss data for label: {label}")
            continue
        
        losses = losses_by_label[label]
        
        raw_losses = [l['raw'] for l in losses if l['raw'] is not None]
        freq_losses = [l['freq'] for l in losses if l['freq'] is not None]
        vq_losses = [l['vq'] for l in losses if l['vq'] is not None]
        total_losses = [l['total'] for l in losses if l['total'] is not None]
        
        print(f"\n{'='*80}")
        print(f"Label: {label.upper()}")
        print(f"{'='*80}")
        
        print(f"\n📊 RAW MSE LOSS (Time Domain):")
        print(f"   - Mean: {np.mean(raw_losses):.6f}")
        print(f"   - Std:  {np.std(raw_losses):.6f}")
        print(f"   - Min:  {np.min(raw_losses):.6f}")
        print(f"   - Max:  {np.max(raw_losses):.6f}")
        
        print(f"\n📊 FREQUENCY MSE LOSS (Frequency Domain):")
        print(f"   - Mean: {np.mean(freq_losses):.6f}")
        print(f"   - Std:  {np.std(freq_losses):.6f}")
        print(f"   - Min:  {np.min(freq_losses):.6f}")
        print(f"   - Max:  {np.max(freq_losses):.6f}")
        
        print(f"\n📊 VQ LOSS (Commitment Loss):")
        print(f"   - Mean: {np.mean(vq_losses):.6f}")
        print(f"   - Std:  {np.std(vq_losses):.6f}")
        print(f"   - Min:  {np.min(vq_losses):.6f}")
        print(f"   - Max:  {np.max(vq_losses):.6f}")
        
        print(f"\n📊 TOTAL LOSS (Raw + Freq + VQ):")
        print(f"   - Mean: {np.mean(total_losses):.6f}")
        print(f"   - Std:  {np.std(total_losses):.6f}")
        print(f"   - Min:  {np.min(total_losses):.6f}")
        print(f"   - Max:  {np.max(total_losses):.6f}")
        
        print(f"\n💡 Loss 해석:")
        print(f"   - Raw loss: 시간 도메인 복원 오차 (낮을수록 좋음)")
        print(f"   - Freq loss: 주파수 도메인 복원 오차 (낮을수록 좋음)")
        print(f"   - VQ loss: 코드북 벡터와 인코더 출력 간 거리 (낮을수록 좋음)")
        print(f"   - Total loss: 전체 학습 loss (train_vq.py와 동일한 계산 방식)")

def compare_labels(analysis_results):
    """Compare token usage between normal and abnormal signals"""
    print("\n" + "="*80)
    print("🔍 LABEL COMPARISON")
    print("="*80)
    
    if 'normal' not in analysis_results or 'abnormal' not in analysis_results:
        print("⚠️  Cannot compare: both labels required")
        return
    
    normal_tokens = set(analysis_results['normal']['all_tokens'])
    abnormal_tokens = set(analysis_results['abnormal']['all_tokens'])
    
    overlap = normal_tokens & abnormal_tokens
    normal_only = normal_tokens - abnormal_tokens
    abnormal_only = abnormal_tokens - normal_tokens
    
    print(f"\n1️⃣  TOKEN SET OVERLAP:")
    print(f"   - Normal only tokens: {len(normal_only)} tokens")
    print(f"   - Abnormal only tokens: {len(abnormal_only)} tokens")
    print(f"   - Shared tokens: {len(overlap)} tokens")
    print(f"   - Total unique tokens: {len(normal_tokens | abnormal_tokens)} tokens")
    
    overlap_ratio = (len(overlap) / len(normal_tokens | abnormal_tokens)) * 100
    print(f"   - Overlap ratio: {overlap_ratio:.2f}%")
    
    print(f"\n2️⃣  TOP TOKENS COMPARISON:")
    print(f"\n   Normal (Top 10):")
    for rank, (token, count) in enumerate(analysis_results['normal']['sorted_tokens'][:10], 1):
        percentage = (count / analysis_results['normal']['total_tokens']) * 100
        print(f"      {rank:2d}. Token {token:4d}: {percentage:5.2f}%")
    
    print(f"\n   Abnormal (Top 10):")
    for rank, (token, count) in enumerate(analysis_results['abnormal']['sorted_tokens'][:10], 1):
        percentage = (count / analysis_results['abnormal']['total_tokens']) * 100
        print(f"      {rank:2d}. Token {token:4d}: {percentage:5.2f}%")
    
    normal_top_10 = set(token for token, _ in analysis_results['normal']['sorted_tokens'][:10])
    abnormal_top_10 = set(token for token, _ in analysis_results['abnormal']['sorted_tokens'][:10])
    
    overlap_top = normal_top_10 & abnormal_top_10
    print(f"\n3️⃣  TOP 10 TOKENS OVERLAP:")
    print(f"   - Shared in top 10: {len(overlap_top)} tokens")
    print(f"   - Different in top 10: {10 - len(overlap_top)} tokens per class")
    
    if len(overlap_top) < 10:
        print(f"\n✅ Good discriminability: Top 10 tokens are different!")
    else:
        print(f"\n⚠️  Low discriminability: Top 10 tokens are mostly the same")

def create_visualizations(analysis_results, version, output_dir):
    """Create comprehensive visualization plot (6-panel)"""
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint_label = f"ckpt-{version}" if version != "best" else "ckpt_best"
    
    print(f"\n🎨 Creating visualizations in {output_dir}")
    
    # Single comprehensive 6-panel plot
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Panel 1: Normal token distribution
    ax1 = fig.add_subplot(gs[0, 0])
    all_tokens_normal = analysis_results['normal']['all_tokens']
    ax1.hist(all_tokens_normal, bins=100, edgecolor='black', alpha=0.7, color='#2ecc71')
    ax1.set_xlabel('Token Index', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title(f'Normal Signal Token Distribution ({checkpoint_label})', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Panel 2: Abnormal token distribution
    ax2 = fig.add_subplot(gs[0, 1])
    all_tokens_abnormal = analysis_results['abnormal']['all_tokens']
    ax2.hist(all_tokens_abnormal, bins=100, edgecolor='black', alpha=0.7, color='#e74c3c')
    ax2.set_xlabel('Token Index', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title(f'Abnormal Signal Token Distribution ({checkpoint_label})', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Panel 3: Codebook usage comparison
    ax3 = fig.add_subplot(gs[1, 0])
    labels_list = ['Normal', 'Abnormal']
    unique_counts = [analysis_results['normal']['unique_count'], analysis_results['abnormal']['unique_count']]
    colors = ['#2ecc71', '#e74c3c']
    
    bars = ax3.bar(labels_list, unique_counts, color=colors, edgecolor='black', linewidth=2, width=0.6)
    ax3.axhline(y=1024, color='gray', linestyle='--', linewidth=2, label='Codebook Size (1024)')
    ax3.set_ylabel('Number of Unique Tokens', fontsize=11)
    ax3.set_title(f'Codebook Usage ({checkpoint_label})', fontsize=12, fontweight='bold')
    ax3.set_ylim(0, 1100)
    
    for bar, count in zip(bars, unique_counts):
        height = bar.get_height()
        percentage = (count / 1024) * 100
        ax3.text(bar.get_x() + bar.get_width()/2., height + 20,
                f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax3.legend(fontsize=10)
    ax3.grid(axis='y', alpha=0.3)
    
    # Panel 4: Top 10 tokens comparison
    ax4 = fig.add_subplot(gs[1, 1])
    normal_top_10 = analysis_results['normal']['sorted_tokens'][:10]
    abnormal_top_10 = analysis_results['abnormal']['sorted_tokens'][:10]
    
    x_pos = np.arange(10)
    normal_percentages = [(count / analysis_results['normal']['total_tokens']) * 100 for _, count in normal_top_10]
    abnormal_percentages = [(count / analysis_results['abnormal']['total_tokens']) * 100 for _, count in abnormal_top_10]
    
    width = 0.35
    ax4.bar(x_pos - width/2, normal_percentages, width, label='Normal', color='#2ecc71', edgecolor='black')
    ax4.bar(x_pos + width/2, abnormal_percentages, width, label='Abnormal', color='#e74c3c', edgecolor='black')
    
    ax4.set_xlabel('Rank', fontsize=11)
    ax4.set_ylabel('Percentage (%)', fontsize=11)
    ax4.set_title(f'Top 10 Tokens Distribution ({checkpoint_label})', fontsize=12, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f'{i+1}' for i in range(10)])
    ax4.legend(fontsize=10)
    ax4.grid(axis='y', alpha=0.3)
    
    # Panel 5: Top 20 tokens for Normal
    ax5 = fig.add_subplot(gs[2, 0])
    sorted_tokens_normal = analysis_results['normal']['sorted_tokens'][:20]
    tokens_n, counts_n = zip(*sorted_tokens_normal)
    percentages_n = [(count / analysis_results['normal']['total_tokens']) * 100 for count in counts_n]
    
    ax5.barh(range(len(tokens_n)), percentages_n, color='#2ecc71', edgecolor='black')
    ax5.set_yticks(range(len(tokens_n)))
    ax5.set_yticklabels([f'Token {t}' for t in tokens_n], fontsize=9)
    ax5.set_xlabel('Percentage (%)', fontsize=11)
    ax5.set_title(f'Top 20 Tokens - Normal ({checkpoint_label})', fontsize=12, fontweight='bold')
    ax5.grid(axis='x', alpha=0.3)
    ax5.invert_yaxis()
    
    # Panel 6: Top 20 tokens for Abnormal
    ax6 = fig.add_subplot(gs[2, 1])
    sorted_tokens_abnormal = analysis_results['abnormal']['sorted_tokens'][:20]
    tokens_a, counts_a = zip(*sorted_tokens_abnormal)
    percentages_a = [(count / analysis_results['abnormal']['total_tokens']) * 100 for count in counts_a]
    
    ax6.barh(range(len(tokens_a)), percentages_a, color='#e74c3c', edgecolor='black')
    ax6.set_yticks(range(len(tokens_a)))
    ax6.set_yticklabels([f'Token {t}' for t in tokens_a], fontsize=9)
    ax6.set_xlabel('Percentage (%)', fontsize=11)
    ax6.set_title(f'Top 20 Tokens - Abnormal ({checkpoint_label})', fontsize=12, fontweight='bold')
    ax6.grid(axis='x', alpha=0.3)
    ax6.invert_yaxis()
    
    plt.savefig(f'{output_dir}/token_analysis_{checkpoint_label}.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: token_analysis_{checkpoint_label}.png")
    plt.close()

def save_summary(analysis_results, version, output_dir):
    """Save detailed summary statistics"""
    checkpoint_label = f"ckpt-{version}" if version != "best" else "ckpt_best"
    
    summary_file = f'{output_dir}/summary_{checkpoint_label}.txt'
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"VQ-VAE Token Analysis Summary ({checkpoint_label})\n")
        f.write("="*80 + "\n\n")
        
        for label in ['normal', 'abnormal']:
            if label not in analysis_results:
                continue
            
            f.write(f"\n{'='*80}\n")
            f.write(f"Label: {label.upper()}\n")
            f.write(f"{'='*80}\n")
            f.write(f"Total tokens: {analysis_results[label]['total_tokens']}\n")
            f.write(f"Unique tokens: {analysis_results[label]['unique_count']}\n")
            f.write(f"Codebook usage: {(analysis_results[label]['unique_count']/1024)*100:.2f}%\n\n")
            
            f.write("Top 20 tokens:\n")
            for rank, (token, count) in enumerate(analysis_results[label]['sorted_tokens'][:20], 1):
                percentage = (count / analysis_results[label]['total_tokens']) * 100
                f.write(f"  {rank:2d}. Token {token:4d}: {count:6d} ({percentage:5.2f}%)\n")
    
    print(f"   ✅ Saved: {summary_file}")

def main():
    print("="*80)
    print(f"🎯 VQ-VAE Token Analysis - Checkpoint: {CHECKPOINT_VERSION}")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if ONLY_ORIGINAL:
        data_dir = f'./datasets/PMD_samples'
    else:
        data_dir = f'./datasets/processed/PMD_samples/{TYPE}'
    
    checkpoint_path = get_checkpoint_path(CHECKPOINT_VERSION)
    output_dir = get_output_dir(f"{CHECKPOINT_VERSION}_{TYPE}")
    
    print(f"\n📌 Configuration:")
    print(f"   - Device: {device}")
    print(f"   - Checkpoint: {checkpoint_path}")
    print(f"   - Output directory: {output_dir}")
    
    if not os.path.exists(data_dir):
        print(f"\n❌ Data directory not found: {data_dir}")
        return
    
    if not os.path.exists(checkpoint_path):
        print(f"\n❌ Checkpoint not found: {checkpoint_path}")
        return
    
    print(f"\n🔧 Loading VQ model...")
    try:
        model = load_vq_model(checkpoint_path, device=device, offline=False)
        
        # Verify model is in eval mode
        print(f"   - Model training mode: {model.training}")
        print(f"   - VQ training mode: {model.VQ.training}")
        
        if model.training or model.VQ.training:
            print("   ⚠️  WARNING: Model is in training mode! Setting to eval...")
            model.eval()
            model.VQ.eval()
        else:
            print("   ✅ Model is in eval mode")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    print(f"\n📂 Processing all 'PKL' and 'CSV' files (single-sample inference)...")
    print(f"   💡 Note: Training uses batch processing, which may result in different codebook usage")
    tokens_by_label, file_info, losses_by_label = process_all_files(model, data_dir, device=device)
    
    print(f"\n📊 Analyzing token distribution...")
    analysis_results = analyze_token_distribution(tokens_by_label)
    
    print(f"\n📉 Analyzing reconstruction losses...")
    analyze_reconstruction_losses(losses_by_label)
    
    print(f"\n🔍 Comparing labels...")
    compare_labels(analysis_results)
    
    print(f"\n🎨 Creating visualizations...")
    create_visualizations(analysis_results, CHECKPOINT_VERSION, output_dir)
    
    print(f"\n💾 Saving summary...")
    os.makedirs(output_dir, exist_ok=True)
    save_summary(analysis_results, CHECKPOINT_VERSION, output_dir)
    
    print("\n" + "="*80)
    print("✅ Analysis completed!")
    print("="*80)

if __name__ == "__main__":
    main()
