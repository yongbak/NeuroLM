"""
테스트용 가짜 EEG 데이터 생성 스크립트
"""

import numpy as np
import pickle
import os
from pathlib import Path

# 표준 10-20 채널 (실제 EEG 채널명)
standard_channels = [
    'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ'
]

def create_dummy_eeg_data(n_channels=19, duration_sec=40, sampling_rate=200):
    """
    가짜 EEG 데이터 생성
    
    Args:
        n_channels: 채널 수 (기본 19개)
        duration_sec: 길이 (초)
        sampling_rate: 샘플링 레이트 (Hz)
    
    Returns:
        eeg_data: (n_channels, n_samples) 형태의 EEG 데이터
        ch_names: 채널명 리스트
    """
    n_samples = duration_sec * sampling_rate
    
    # 실제 EEG와 비슷한 특성을 가진 신호 생성
    eeg_data = np.zeros((n_channels, n_samples))
    
    for ch in range(n_channels):
        # 1. 기본 주파수 성분들 (알파파, 베타파 등)
        time = np.linspace(0, duration_sec, n_samples)
        
        # 알파파 (8-12Hz) - 주로 후두부
        alpha = 10 * np.sin(2 * np.pi * 10 * time) if ch >= 8 else 5 * np.sin(2 * np.pi * 10 * time)
        
        # 베타파 (13-30Hz) - 주로 전두부
        beta = 5 * np.sin(2 * np.pi * 20 * time) if ch < 8 else 2 * np.sin(2 * np.pi * 20 * time)
        
        # 세타파 (4-8Hz)
        theta = 3 * np.sin(2 * np.pi * 6 * time)
        
        # 델타파 (0.5-4Hz)
        delta = 2 * np.sin(2 * np.pi * 2 * time)
        
        # 2. 백색 잡음
        noise = np.random.normal(0, 5, n_samples)
        
        # 3. 1/f 잡음 (더 현실적인 EEG)
        freqs = np.fft.fftfreq(n_samples, 1/sampling_rate)
        freqs[0] = 1  # 0으로 나누기 방지
        psd = 1 / np.abs(freqs)  # 1/f 특성
        phase = np.random.uniform(-np.pi, np.pi, len(freqs))
        complex_noise = np.sqrt(psd) * np.exp(1j * phase)
        pink_noise = np.real(np.fft.ifft(complex_noise)) * 10
        
        # 모든 성분 합성
        signal = alpha + beta + theta + delta + noise + pink_noise
        
        # μV 단위로 스케일링 (일반적인 EEG 범위: -100 ~ +100 μV)
        signal = signal * 5  # 약 -50 ~ +50 μV 범위
        
        eeg_data[ch] = signal
    
    # 채널명 생성
    ch_names = standard_channels[:n_channels]
    
    return eeg_data, ch_names

def create_dummy_dataset(output_dir, n_files=10, train_ratio=0.8):
    """
    여러 개의 가짜 EEG 파일 생성
    
    Args:
        output_dir: 저장할 디렉토리
        n_files: 생성할 파일 수
        train_ratio: 훈련/검증 비율
    """
    
    # 디렉토리 생성
    train_dir = Path(output_dir) / 'train'
    val_dir = Path(output_dir) / 'val'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    n_train = int(n_files * train_ratio)
    n_val = n_files - n_train
    
    print(f"Creating {n_train} training files and {n_val} validation files...")
    
    # 훈련 데이터 생성
    for i in range(n_train):
        # 랜덤한 특성의 EEG 데이터 생성
        n_channels = np.random.randint(16, 22)  # 16-21 채널
        duration = np.random.randint(30, 60)    # 30-60초
        
        eeg_data, ch_names = create_dummy_eeg_data(
            n_channels=n_channels, 
            duration_sec=duration
        )
        
        # NeuroLM 형식으로 저장
        data_dict = {
            "X": eeg_data,
            "ch_names": ch_names
        }
        
        file_path = train_dir / f"dummy_eeg_{i:04d}.pkl"
        with open(file_path, 'wb') as f:
            pickle.dump(data_dict, f)
        
        if i % 10 == 0:
            print(f"Created training file {i+1}/{n_train}")
    
    # 검증 데이터 생성
    for i in range(n_val):
        n_channels = np.random.randint(16, 22)
        duration = np.random.randint(30, 60)
        
        eeg_data, ch_names = create_dummy_eeg_data(
            n_channels=n_channels, 
            duration_sec=duration
        )
        
        data_dict = {
            "X": eeg_data,
            "ch_names": ch_names
        }
        
        file_path = val_dir / f"dummy_eeg_val_{i:04d}.pkl"
        with open(file_path, 'wb') as f:
            pickle.dump(data_dict, f)
    
    print(f"✅ Dataset created successfully!")
    print(f"📁 Training files: {train_dir}")
    print(f"📁 Validation files: {val_dir}")
    print(f"📊 Total files: {n_files}")

def create_text_data(output_dir):
    """
    텍스트 데이터도 생성 (train_pretrain.py용)
    """
    text_dir = Path(output_dir) / 'text'
    text_dir.mkdir(parents=True, exist_ok=True)
    
    # 간단한 더미 텍스트 데이터
    vocab_size = 50257  # GPT-2 vocab size
    sequence_length = 100000
    
    # 훈련용
    train_data = np.random.randint(0, vocab_size, sequence_length, dtype=np.uint16)
    train_path = text_dir / 'train.bin'
    train_data.tofile(train_path)
    
    # 검증용
    val_data = np.random.randint(0, vocab_size, sequence_length // 10, dtype=np.uint16)
    val_path = text_dir / 'val.bin'
    val_data.tofile(val_path)
    
    print(f"📝 Text data created: {text_dir}")

if __name__ == "__main__":
    # 출력 디렉토리 설정
    output_dir = "../dummy_dataset"
    
    # 데이터셋 생성
    create_dummy_dataset(output_dir, n_files=50)  # 50개 파일 생성
    create_text_data(output_dir)  # 텍스트 데이터도 생성
    
    print("\n🎉 Dummy dataset creation completed!")
    print(f"\n사용법:")
    print(f"python train_vq.py --dataset_dir {output_dir}")
