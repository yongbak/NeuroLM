"""
by Yongbak
Copy from prepare_workload.py and modify to read .txt signal files
"""

import mne
import numpy as np
import os
import pickle
import traceback
#import torch


drop_channels = ['EEG A2-A1', 'ECG ECG']
chOrder_standard = ['CPU']
#chOrder_standard = ['EEG Fp1', 'EEG Fp2', 'EEG F3', 'EEG F4', 'EEG F7', 'EEG F8', 'EEG T3', 'EEG T4', 'EEG C3', 'EEG C4', 'EEG T5', 'EEG T6', 'EEG P3', 'EEG P4', 'EEG O1', 'EEG O2', 'EEG Fz', 'EEG Cz', 'EEG Pz']


def read_txt_signal(file_path: str) -> list[float]:
    with open(file_path, 'r', encoding='utf-8') as f:
        signal = f.read()
        signal = signal.split("\n")[:-1]
        for i in range(len(signal)):
            signal[i] = float(signal[i])
    return signal

# 각 파일에서 길이 2000 샘플 단위로 패치 분할
# 마지막 패치가 2000보다 짧으면 0으로 패딩
def signal_to_patches(signal: list[float], sr = 2000) -> list[list[float]]:

    # Hardcoded sampling rate
    sr = 2000

    patches = [signal[i:i+sr] for i in range(0, len(signal), sr)]

    # 길이가 정확히 2000의 배수가 아닌 경우
    if len(patches[-1]) < sr:                               # 마지막 조각이 부족
        patches[-1] += [0.0] * (sr - len(patches[-1]))      # 패딩 추가

    return patches

# patches의 각 patch에 대한 정보
# len(return_val['magnitude']) == 1001
def time_to_frequency(patch: list[float], sr=2000) -> list[complex]:
    """
    길이 2000인 신호 패치의 주파수 도메인 값 추출
    """
    
    # 1D 신호를 tensor로 변환
    waveform = torch.tensor(patch, dtype=torch.float32)  # [2000]
    
    # Hann window 적용 (leakage 방지)
    window = torch.hann_window(waveform.numel(), periodic=True, device=waveform.device)
    waveform = waveform * window

    # rFFT 적용 (실수 입력 전용 FFT) => 출력 크기: [n//2 + 1]
    fft_result = torch.fft.rfft(waveform)  # [1001] complex

    # 주파수 도메인 특성 추출
    magnitude = torch.abs(fft_result)     # 크기
    phase = torch.angle(fft_result)       # 위상
    real = fft_result.real               # 실수부
    imag = fft_result.imag               # 허수부
    
    # Nyquist 주파수까지만 사용 (대칭성 때문)
    n_freq = len(patch) // 2 + 1  # 1001개
    magnitude = magnitude[:n_freq]
    phase = phase[:n_freq]
    real = real[:n_freq]
    imag = imag[:n_freq]
    
    return {
        'magnitude': magnitude,  # [1001]

        # Not used
        # Reference - NeuroLM: A Universal Multi-task Foundation Model for Bridging the Gap between Language and EEG Signals
        'phase': phase,         # [1001]
        'real': real,          # [1001]
        'imag': imag,          # [1001]
        'complex': fft_result[:n_freq]  # [1001] complex
    }

# signals: ndarray, shape(n_channels, n_times)
# mne.io.Raw.get_data()의 첫 번째 반환
def BuildEvents(signals, times):
    fs = 2000.0
    [numChan, numPoints] = signals.shape
    numEvents = 29

    features = np.zeros([numEvents, numChan, int(fs) * 4])
    i = 0
    for i in range(numEvents):
        start = i * 400
        end = (i + 2) * 400
        features[i, :] = signals[:, start:end]
    return features

# https://mne.tools/stable/generated/mne.io.read_raw_edf.html#mne.io.read_raw_edf
# https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.get_data
def readEDF(fileName):
    # Rawdata: mne.io.Raw
    Rawdata = mne.io.read_raw_edf(fileName, preload=True)
    print("[*] Rawdata.ch_names: ", Rawdata.ch_names)
    print("[*] Rawdata: ", Rawdata)

    if drop_channels is not None:
        useless_chs = []
        for ch in drop_channels:
            if ch in Rawdata.ch_names:
                useless_chs.append(ch)
        Rawdata.drop_channels(useless_chs)
    # 안전한 재정렬: 표준 목록이 있고 길이가 같을 때만 시도하되,
    # 모든 표준 채널명이 실제 채널 목록에 존재하는지 확인합니다.
    if chOrder_standard is not None and len(chOrder_standard) == len(Rawdata.ch_names):
        missing = [c for c in chOrder_standard if c not in Rawdata.ch_names]
        if len(missing) == 0:
            Rawdata.reorder_channels(chOrder_standard)
        else:
            # 엄격한 예외 대신 경고를 출력하고 원본 채널 순서를 유지합니다.
            print(f"[!] 채널 재정렬 건너뜀 - 파일에 없는 표준 채널: {missing}")
            print(f"    actual channels: {Rawdata.ch_names}")
    else:
        # 길이가 다르거나 chOrder_standard가 None이면 재정렬하지 않고 경고
        if chOrder_standard is not None:
            print(f"[!] 재정렬 검사: 표준 리스트 길이({len(chOrder_standard)})와 파일 채널 길이({len(Rawdata.ch_names)})가 다릅니다. 재정렬하지 않습니다.")

    Rawdata.filter(l_freq=0.1, h_freq=75.0)
    Rawdata.notch_filter(50.0)
    Rawdata.resample(200, n_jobs=5)

    _, times = Rawdata[:]

    # 실제 신호 수집하는 부분
    # -60 (마지막 60초) * 200 (초당 샘플링 개수)
    
    # get_data returns:
    # data: ndarray, shape (n_channels, n_times);   Copy of the data in the given range.
    # times: ndarray, shape (n_times,);             Only returned if return_times=True, default is False. Times associated with the data samples. 

    # signals는 channels x `# of samples` 형태의 ndarray
    signals = Rawdata.get_data(units='uV')[:, -60 * 200:]

    Rawdata.close()
    return [signals, times, Rawdata]


def load_up_objects(fileList, Features, Labels, OutDir):
    print("[*] fileList: " , fileList)
    for fname in fileList:
        print("[*] fname: ")
        print("\t%s" % fname)
        
        DUMMY_LABEL = 1
        label = DUMMY_LABEL

        try:
            [signals, times, Rawdata] = readEDF(fname)
        except Exception as e:
            # NotImplementedError (non-EDF files) 등 모든 예외를 받아서 파일 이름과 함께 출력하고 건너뜁니다.
            print("[*] Exception Happens while processing:", fname)
            print("Exception type:", type(e).__name__)
            print("Exception message:", e)
            print("Stack trace:")
            traceback.print_exc()
            print("skipping this file and continuing...\n")
            continue
        signals = BuildEvents(signals, times)

        for idx, signal in enumerate(signals):
            # 픽클에는 파일의 실제 채널명을 저장합니다(표준 목록 대신).
            # Rawdata는 readEDF에서 close되었지만 ch_names는 여전히 사용할 수 있습니다.
            sample = {
                "X": signal,
                "ch_names": [name.upper().split(' ')[-1] for name in Rawdata.ch_names],
                "y": label,
            }
            print(signal.shape)

            save_pickle(
                sample,
                os.path.join(
                    OutDir, fname.split("/")[-1].split(".")[0] + "-" + str(idx) + ".pkl"
                ),
            )

    return Features, Labels


def save_pickle(object, filename):
    with open(filename, "wb") as f:
        pickle.dump(object, f)


root = "../dummy_dataset/edf_samples"
out_dir = '../dummy_dataset/processed'

train_out_dir = os.path.join(out_dir, "train")
eval_out_dir = os.path.join(out_dir, "eval")
test_out_dir = os.path.join(out_dir, "test")
if not os.path.exists(train_out_dir):
    os.makedirs(train_out_dir)
if not os.path.exists(eval_out_dir):
    os.makedirs(eval_out_dir)
if not os.path.exists(test_out_dir):
    os.makedirs(test_out_dir)

edf_files = []
for dirName, subdirList, fileList in os.walk(root):
    for fname in fileList:
        # 대소문자 무관하게 .edf 확장자만 수집
        if fname.lower().endswith('.edf'):
            edf_files.append(os.path.join(dirName, fname))
edf_files.sort()
print("[*] edf_files: ")
print(edf_files)

edf_files = [ edf_files[0] ]


train_files = edf_files[:len(edf_files)//5*3]
eval_files = edf_files[len(edf_files)//5*3:len(edf_files)//5*4]
test_files = edf_files[len(edf_files)//5*4:]

fs = 200
TrainFeatures = np.empty(
    (0, 4, fs * 30)
)  # 0 for lack of intialization, 22 for channels, fs for num of points
TrainLabels = np.empty([0, 1])
load_up_objects(
    train_files, TrainFeatures, TrainLabels, train_out_dir
)

fs = 200
EvalFeatures = np.empty(
    (0, 19, fs * 4)
)  # 0 for lack of intialization, 22 for channels, fs for num of points
EvalLabels = np.empty([0, 1])
load_up_objects(
    eval_files, EvalFeatures, EvalLabels, eval_out_dir
)

fs = 200
TestFeatures = np.empty(
    (0, 19, fs * 4)
)  # 0 for lack of intialization, 22 for channels, fs for num of points
TestLabels = np.empty([0, 1])
load_up_objects(
    test_files, TestFeatures, TestLabels, test_out_dir
)