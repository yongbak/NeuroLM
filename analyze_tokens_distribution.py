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

from utils import load_vq_model, extract_tokens_from_single_file
from constants import NUM_OF_SAMPLES_PER_TOKEN, NUM_OF_TOTAL_TOKENS, CODEBOOK_SIZE

# ============================================================================
# 🔧 설정: 이 값을 바꿔서 다른 체크포인트를 분석하세요
ID = "token_1000"

# 옵션: 9, 19, 29, 39, 49, "best"
CHECKPOINT_VERSION = 29  # ← 이 값을 바꾸세요!

# 옵션: train, test, val
TYPE = "test"

# 비교할 상위 K개 토큰 (0이면 전체 토큰 비교)
TOP_K = 30  # ← 상위 몇 개 토큰을 볼지 결정 (0 = 전체)

ONLY_ORIGINAL = False
# 모든 결과는 ./token_analysis/{type} 디렉토리에 저장됩니다
# ============================================================================

def get_checkpoint_path(version):
    """Get checkpoint path from version"""
    if version == "best":
        return f"./vq_output_{ID}/checkpoints/VQ/ckpt_best.pt"
    else:
        return f"./vq_output_{ID}/checkpoints/VQ/ckpt-{version}.pt"

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


def process_all_files(model, data_dir, device='cuda'):
    """Process all files and extract tokens with labels"""
    csv_files = list(Path(data_dir).glob('*.csv'))
    pkl_files = list(Path(data_dir).glob('*.pkl'))
    files = csv_files + pkl_files

    print(f"🔍 Found {len(csv_files)} CSV files")
    print(f"🔍 Found {len(pkl_files)} PKL files")

    tokens_by_label = defaultdict(list)
    file_info = []
    
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
            
            tokens_by_label[label].append(token_sequence)
            file_info.append({
                'filename': filename,
                'label': label,
                'num_tokens': len(token_sequence),
                'unique_tokens': len(set(token_sequence)),
                'token_sequence': token_sequence
            })
            
            print(f"✅ ({len(token_sequence)} tokens, {len(set(token_sequence))} unique)")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    return tokens_by_label, file_info

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

        top_k_display = len(sorted_tokens) if TOP_K == 0 else min(TOP_K, len(sorted_tokens))
        print(f"\n2️⃣  TOKEN FREQUENCY (Top {top_k_display}):")
        for rank, (token, count) in enumerate(sorted_tokens[:top_k_display], 1):
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
        
        total_codebook = CODEBOOK_SIZE
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
    ax3.axhline(y=CODEBOOK_SIZE, color='gray', linestyle='--', linewidth=2, label=f'Codebook Size ({CODEBOOK_SIZE})')
    ax3.set_ylabel('Number of Unique Tokens', fontsize=11)
    ax3.set_title(f'Codebook Usage ({checkpoint_label})', fontsize=12, fontweight='bold')
    ax3.set_ylim(0, CODEBOOK_SIZE*1.1)
    
    for bar, count in zip(bars, unique_counts):
        height = bar.get_height()
        percentage = (count / CODEBOOK_SIZE) * 100
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
    
    # Panel 5: Top K tokens for Normal
    ax5 = fig.add_subplot(gs[2, 0])
    top_k_display = len(analysis_results['normal']['sorted_tokens']) if TOP_K == 0 else min(TOP_K, len(analysis_results['normal']['sorted_tokens']))
    sorted_tokens_normal = analysis_results['normal']['sorted_tokens'][:top_k_display]
    tokens_n, counts_n = zip(*sorted_tokens_normal)
    percentages_n = [(count / analysis_results['normal']['total_tokens']) * 100 for count in counts_n]
    
    ax5.barh(range(len(tokens_n)), percentages_n, color='#2ecc71', edgecolor='black')
    ax5.set_yticks(range(len(tokens_n)))
    ax5.set_yticklabels([f'Token {t}' for t in tokens_n], fontsize=9)
    ax5.set_xlabel('Percentage (%)', fontsize=11)
    ax5.set_title(f'Top {top_k_display} Tokens - Normal ({checkpoint_label})', fontsize=12, fontweight='bold')
    ax5.grid(axis='x', alpha=0.3)
    ax5.invert_yaxis()
    
    # Panel 6: Top K tokens for Abnormal
    ax6 = fig.add_subplot(gs[2, 1])
    sorted_tokens_abnormal = analysis_results['abnormal']['sorted_tokens'][:top_k_display]
    tokens_a, counts_a = zip(*sorted_tokens_abnormal)
    percentages_a = [(count / analysis_results['abnormal']['total_tokens']) * 100 for count in counts_a]
    
    ax6.barh(range(len(tokens_a)), percentages_a, color='#e74c3c', edgecolor='black')
    ax6.set_yticks(range(len(tokens_a)))
    ax6.set_yticklabels([f'Token {t}' for t in tokens_a], fontsize=9)
    ax6.set_xlabel('Percentage (%)', fontsize=11)
    ax6.set_title(f'Top {top_k_display} Tokens - Abnormal ({checkpoint_label})', fontsize=12, fontweight='bold')
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
            f.write(f"Codebook usage: {(analysis_results[label]['unique_count']/CODEBOOK_SIZE)*100:.2f}%\n\n")

            top_k_display = len(analysis_results[label]['sorted_tokens']) if TOP_K == 0 else min(TOP_K, len(analysis_results[label]['sorted_tokens']))
            f.write(f"Top {top_k_display} tokens:\n")
            for rank, (token, count) in enumerate(analysis_results[label]['sorted_tokens'][:top_k_display], 1):
                percentage = (count / analysis_results[label]['total_tokens']) * 100
                f.write(f"  {rank:2d}. Token {token:4d}: {count:6d} ({percentage:5.2f}%)\n")
            
            # 전체 토큰 분포 출력
            f.write(f"\n{'-'*80}\n")
            f.write(f"ALL TOKENS ({analysis_results[label]['unique_count']} unique tokens):\n")
            f.write(f"{'-'*80}\n")
            f.write(f"{'Rank':<6} {'Token':<8} {'Count':<8} {'Percentage':<12} {'Cumulative %':<15}\n")
            f.write(f"{'-'*80}\n")
            
            cumulative_percentage = 0.0
            for rank, (token, count) in enumerate(analysis_results[label]['sorted_tokens'], 1):
                percentage = (count / analysis_results[label]['total_tokens']) * 100
                cumulative_percentage += percentage
                f.write(f"{rank:<6d} {token:<8d} {count:<8d} {percentage:>10.2f}% {cumulative_percentage:>13.2f}%\n")
        
        # 겹치는 토큰 분석 추가
        f.write(f"\n{'='*80}\n")
        f.write(f"OVERLAPPING TOKENS ANALYSIS\n")
        f.write(f"{'='*80}\n")
        
        if 'normal' in analysis_results and 'abnormal' in analysis_results:
            # 각 레이블의 토큰 세트 추출
            normal_tokens = set(token for token, _ in analysis_results['normal']['sorted_tokens'])
            abnormal_tokens = set(token for token, _ in analysis_results['abnormal']['sorted_tokens'])
            
            # 겹치는 토큰
            overlapping_tokens = normal_tokens & abnormal_tokens
            
            # 정상에만 있는 토큰
            normal_only = normal_tokens - abnormal_tokens
            
            # 비정상에만 있는 토큰
            abnormal_only = abnormal_tokens - normal_tokens
            
            f.write(f"\n📊 TOKEN SET COMPARISON:\n")
            f.write(f"   - NORMAL unique tokens: {len(normal_tokens)}\n")
            f.write(f"   - ABNORMAL unique tokens: {len(abnormal_tokens)}\n")
            f.write(f"   - Overlapping tokens: {len(overlapping_tokens)}\n")
            f.write(f"   - NORMAL only: {len(normal_only)}\n")
            f.write(f"   - ABNORMAL only: {len(abnormal_only)}\n\n")
            
            # 겹치는 토큰들의 점유율 계산
            normal_token_dict = dict(analysis_results['normal']['sorted_tokens'])
            abnormal_token_dict = dict(analysis_results['abnormal']['sorted_tokens'])
            
            overlap_normal_count = sum(normal_token_dict.get(token, 0) for token in overlapping_tokens)
            overlap_abnormal_count = sum(abnormal_token_dict.get(token, 0) for token in overlapping_tokens)
            
            overlap_normal_ratio = (overlap_normal_count / analysis_results['normal']['total_tokens']) * 100
            overlap_abnormal_ratio = (overlap_abnormal_count / analysis_results['abnormal']['total_tokens']) * 100
            
            f.write(f"📈 OVERLAPPING TOKENS CONTRIBUTION:\n")
            f.write(f"   NORMAL:\n")
            f.write(f"      - Total count: {overlap_normal_count}\n")
            f.write(f"      - Percentage: {overlap_normal_ratio:.2f}%\n\n")
            f.write(f"   ABNORMAL:\n")
            f.write(f"      - Total count: {overlap_abnormal_count}\n")
            f.write(f"      - Percentage: {overlap_abnormal_ratio:.2f}%\n\n")
            
            # 상세 겹치는 토큰 리스트
            f.write(f"{'Rank':<6} {'Token':<8} {'Normal Count':<14} {'Normal %':<12} {'Abnormal Count':<16} {'Abnormal %':<12}\n")
            f.write(f"{'-'*80}\n")
            
            # 겹치는 토큰을 정상에서의 빈도로 정렬
            sorted_overlapping = sorted(overlapping_tokens, 
                                       key=lambda t: normal_token_dict.get(t, 0), 
                                       reverse=True)
            
            for rank, token in enumerate(sorted_overlapping, 1):
                normal_count = normal_token_dict.get(token, 0)
                abnormal_count = abnormal_token_dict.get(token, 0)
                normal_pct = (normal_count / analysis_results['normal']['total_tokens']) * 100
                abnormal_pct = (abnormal_count / analysis_results['abnormal']['total_tokens']) * 100
                
                f.write(f"{rank:<6d} {token:<8d} {normal_count:<14d} {normal_pct:>10.2f}% {abnormal_count:<16d} {abnormal_pct:>10.2f}%\n")
    
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
    output_dir = get_output_dir(f"{ID}_{CHECKPOINT_VERSION}_{TYPE}")
    
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
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    print(f"\n📂 Processing all 'PKL' and 'CSV' files...")
    tokens_by_label, file_info = process_all_files(model, data_dir, device=device)
    
    print(f"\n📊 Analyzing token distribution...")
    analysis_results = analyze_token_distribution(tokens_by_label)
    
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
