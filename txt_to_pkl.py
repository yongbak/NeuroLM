"""
by Yongbak
Copy from prepare_workload.py and modify to read .txt signal files
"""

import mne
import numpy as np
import os
import pickle
import traceback
import torch

from augmentor import TimeSeriesAugmentor as TA
from constants import (
    NUM_OF_SAMPLES_PER_TOKEN,
    GAUSSIAN_NOISE_MEAN,
    GAUSSIAN_NOISE_STD,
    AMPLITUDE_SCALING_MIN,
    AMPLITUDE_SCALING_MAX
)

drop_channels = None
chOrder_standard = ['DEVICE']


def read_txt_signal(file_path: str) -> list[float]:
    print("[*] read_txt_signal")
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

# signals: ndarray, shape(n_channels, n_times)
# mne.io.Raw.get_data()의 첫 번째 반환
def BuildEvents(signals, times, fs=2000.0):
    
    # signals는 readTXT에서 2000 Hz로 리샘플링되어 오므로 fs를 2000으로 설정

    [numChan, numPoints] = signals.shape

    # 기본 윈도우: 20초 길이, 10초 스텝 (겹침 50%) — 필요하면 인자화할 수 있습니다.
    window_sec = NUM_OF_SAMPLES_PER_TOKEN / fs
    step_sec = window_sec * 0.5
    window_samples = int(round(window_sec * fs))
    step_samples = int(round(step_sec * fs))

    # 신호가 윈도우보다 짧으면 한 개의 패치로 제로패딩해서 반환
    if numPoints < window_samples:
        features = np.zeros((1, numChan, window_samples), dtype=signals.dtype)
        features[0, :, :numPoints] = signals
        return features

    # 시작 인덱스 리스트 생성
    starts = list(range(0, numPoints - window_samples + 1, step_samples))
    # 마지막 구간이 완전히 덮이지 않으면 마지막 윈도우를 포함
    if len(starts) == 0 or (starts[-1] + window_samples) < numPoints:
        starts.append(max(0, numPoints - window_samples))

    features = np.zeros((len(starts), numChan, window_samples), dtype=signals.dtype)
    for idx, s in enumerate(starts):
        features[idx, :] = signals[:, s : s + window_samples]

    return features

# Refers to readEDF
def readTXT(filePath):
    
    sr = 2000.0
    notch = 60.0

    signal = read_txt_signal(filePath)     # list[float]
    signal = np.asarray(signal, dtype=np.float32)

    ch_names = ['DEVICE']                   # Hardcoded channel name for our purpose
    ch_types = ['misc']

    if drop_channels is not None:
        # Unreachable code for device
        pass

    info = mne.create_info(ch_names=ch_names, sfreq=sr, ch_types=ch_types)
    raw = mne.io.RawArray(signal[np.newaxis, :], info)

    # MNE의 기본 picks(None -> "data_or_ica")가 채널을 못찾는 경우가 있어
    # 명시적으로 모든 채널 인덱스를 지정합니다 (단일 채널도 포함).
    nchan = raw.info.get('nchan', raw.get_data().shape[0])
    picks = np.arange(nchan)
    
    # 필터 매개변수를 조정하여 메모리 오버플로우 방지
    # 더 높은 고역통과 주파수와 더 낮은 저역통과 주파수 사용
    
    raw.filter(l_freq=0.5, h_freq=None, picks=picks, 
           filter_length='auto', l_trans_bandwidth='auto')
    raw.notch_filter(notch, picks=picks)

    raw.resample(sr, n_jobs=5)

    # times: ndarray, shape (n_times,);             Only returned if return_times=True, default is False. Times associated with the data samples. 
    _, times = raw[:]
        
    # signals는 channels x `# of samples` 형태의 ndarray
    # data: ndarray, shape (n_channels, n_times);   Copy of the data in the given range.
    signals = raw.get_data()  # 단위 지정 제거

    return [signals, times, ch_names]

def load_up_augmented_objects(fileList, Features, Labels, OutDir, augment_factor=1):
    for fname in fileList:
        
        DUMMY_LABEL = 1
        label = DUMMY_LABEL

        try:
            # readEDF은 (signals, times, ch_names)를 반환합니다.
            [signals, times, ch_names] = readTXT(fname)
        except Exception as e:
            # NotImplementedError (non-EDF files) 등 모든 예외를 받아서 파일 이름과 함께 출력하고 건너뜁니다.
            print("[*] Exception Happens while processing:", fname)
            print("Exception type:", type(e).__name__)
            print("Exception message:", e)
            print("Stack trace:")
            traceback.print_exc()
            print("skipping this file and continuing...\n")
            continue

        # 구간별로 서로 다른 노이즈 추가, 매우 작은 노이즈
        gaussian_noised_signals = TA.gaussian_noise(signals, GAUSSIAN_NOISE_MEAN, GAUSSIAN_NOISE_STD)
        gaussian_noised_signals = BuildEvents(gaussian_noised_signals, times)

        for f in range(augment_factor):
            for idx, signal in enumerate(gaussian_noised_signals):
                # 픽클에는 파일의 실제 채널명을 저장합니다(표준 목록 대신).
                sample = {
                    "X": signal,
                    "ch_names": [name.upper().split(' ')[-1] for name in ch_names],
                    "y": label,
                }
                #print(signal.shape)

                print(
                    os.path.join(
                        OutDir, fname.split("/")[-1].split(".")[0] + "-" + str(idx) + ".pkl"
                    )
                )

                save_pickle(
                    sample,
                    os.path.join(
                        OutDir, "gaussian_noise_{f}_" + fname.split("/")[-1].split(".")[0] + "-" + str(idx) + ".pkl"
                    ),
                )

        # 구간별로 서로 다른 진폭변조, 매우 작은 변조
        amplitude_manipulated_signals = TA.amplitude_scaling(signals, AMPLITUDE_SCALING_MIN, AMPLITUDE_SCALING_MAX)
        amplitude_manipulated_signals = BuildEvents(amplitude_manipulated_signals, times)

        for f in range(augment_factor):
            for idx, signal in enumerate(amplitude_manipulated_signals):
                # 픽클에는 파일의 실제 채널명을 저장합니다(표준 목록 대신).
                sample = {
                    "X": signal,
                    "ch_names": [name.upper().split(' ')[-1] for name in ch_names],
                    "y": label,
                }
                #print(signal.shape)

                print(
                    os.path.join(
                        OutDir, fname.split("/")[-1].split(".")[0] + "-" + str(idx) + ".pkl"
                    )
                )

                save_pickle(
                    sample,
                    os.path.join(
                        OutDir, "amplitude_manipulated_{f}_" +fname.split("/")[-1].split(".")[0] + "-" + str(idx) + ".pkl"
                    ),
                )

    return Features, Labels

def load_up_objects(fileList, Features, Labels, OutDir):
    for fname in fileList:
        print("[*] fname: ")
        print("\t%s" % fname)
        
        DUMMY_LABEL = 1
        label = DUMMY_LABEL

        try:
            # readEDF은 (signals, times, ch_names)를 반환합니다.
            [signals, times, ch_names] = readTXT(fname)
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
            sample = {
                "X": signal,
                "ch_names": [name.upper().split(' ')[-1] for name in ch_names],
                "y": label,
            }
            print(signal.shape)

            print(
                os.path.join(
                    OutDir, fname.split("/")[-1].split(".")[0] + "-" + str(idx) + ".pkl"
                )
            )

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


root = "../datasets/PMD_samples"
out_dir = '../datasets/processed/PMD_samples'

train_out_dir = os.path.join(out_dir, "train")
eval_out_dir = os.path.join(out_dir, "val")
test_out_dir = os.path.join(out_dir, "test")
if not os.path.exists(train_out_dir):
    os.makedirs(train_out_dir)
if not os.path.exists(eval_out_dir):
    os.makedirs(eval_out_dir)
if not os.path.exists(test_out_dir):
    os.makedirs(test_out_dir)

csv_files = []
for dirName, subdirList, fileList in os.walk(root):
    for fname in fileList:
        # 대소문자 무관하게 .csv 확장자만 수집
        if fname.lower().endswith('.csv'):
            csv_files.append(os.path.join(dirName, fname))
csv_files.sort()
print("[*] csv_files: ")
print(csv_files)

# 클래스별로 파일 분류 (b: 정상, cc/s/m: 비정상)
class_files = {'b': [], 'cc': [], 's': [], 'm': []}
for fpath in csv_files:
    fname = os.path.basename(fpath)
    class_label = fname.split('_')[1]
    if class_label in class_files:
        class_files[class_label].append(fpath)

# 각 클래스에서 8:1:1 비율로 분할
train_files = []
eval_files = []
test_files = []

for class_label, files in class_files.items():
    n = len(files)
    train_end = (n * 8) // 10
    eval_end = (n * 9) // 10
    
    train_files.extend(files[:train_end])
    eval_files.extend(files[train_end:eval_end])
    test_files.extend(files[eval_end:])
    
    print(f"[*] Class '{class_label}': {len(files)} files total -> train: {len(files[:train_end])}, eval: {len(files[train_end:eval_end])}, test: {len(files[eval_end:])}")

print(f"[*] Total: train={len(train_files)}, eval={len(eval_files)}, test={len(test_files)}")

fs = 2000  # Match the actual sampling rate used in readTXT and BuildEvents
TrainFeatures = np.empty(
    (0, 1, fs * 4)  # 1 channel (DEVICE), 4 seconds window
)  # 0 for lack of intialization, 1 for channel, fs * 4 for 4-second window
TrainLabels = np.empty([0, 1])
load_up_objects(
    train_files, TrainFeatures, TrainLabels, train_out_dir
)
load_up_augmented_objects(
    train_files, TrainFeatures, TrainLabels, train_out_dir
)

fs = 2000  # Match the actual sampling rate used in readTXT and BuildEvents
EvalFeatures = np.empty(
    (0, 1, fs * 4)  # 1 channel (DEVICE), 4 seconds window
)  # 0 for lack of intialization, 1 for channel, fs * 4 for 4-second window
EvalLabels = np.empty([0, 1])
load_up_objects(
    eval_files, EvalFeatures, EvalLabels, eval_out_dir
)
load_up_augmented_objects(
    eval_files, EvalFeatures, EvalLabels, eval_out_dir
)

fs = 2000  # Match the actual sampling rate used in readTXT and BuildEvents
TestFeatures = np.empty(
    (0, 1, fs * 4)  # 1 channel (DEVICE), 4 seconds window
)  # 0 for lack of intialization, 1 for channel, fs * 4 for 4-second window
TestLabels = np.empty([0, 1])
load_up_objects(
    test_files, TestFeatures, TestLabels, test_out_dir
)
load_up_augmented_objects(
    test_files, TestFeatures, TestLabels, test_out_dir
)