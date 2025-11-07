"""
by Wei-Bang Jiang
https://github.com/935963004/NeuroLM
"""

import mne
import numpy as np
import os
import pickle


drop_channels = ['EEG A2-A1', 'ECG ECG']
chOrder_standard = ['EEG Fp1', 'EEG Fp2', 'EEG F3', 'EEG F4', 'EEG F7', 'EEG F8', 'EEG T3', 'EEG T4', 'EEG C3', 'EEG C4', 'EEG T5', 'EEG T6', 'EEG P3', 'EEG P4', 'EEG O1', 'EEG O2', 'EEG Fz', 'EEG Cz', 'EEG Pz']


def BuildEvents(signals, times):
    fs = 200.0
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
    Rawdata = mne.io.read_raw_edf(fileName, preload=True)

    if drop_channels is not None:
        useless_chs = []
        for ch in drop_channels:
            if ch in Rawdata.ch_names:
                useless_chs.append(ch)
        Rawdata.drop_channels(useless_chs)
    # 모든 표준 채널명이 실제 채널 목록에 존재하는지 확인합니다.
    if chOrder_standard is not None and len(chOrder_standard) == len(Rawdata.ch_names):
        Rawdata.reorder_channels(chOrder_standard)
    if Rawdata.ch_names != chOrder_standard:
        raise ValueError

    Rawdata.filter(l_freq=0.1, h_freq=75.0)
    Rawdata.notch_filter(50.0)
    Rawdata.resample(200, n_jobs=5)

    

    # 실제 신호 수집하는 부분
    # -60 (마지막 60초) * 200 (초당 샘플링 개수)
    
    # get_data returns:
    # data: ndarray, shape (n_channels, n_times);   Copy of the data in the given range.
    # times: ndarray, shape (n_times,);             Only returned if return_times=True, default is False. Times associated with the data samples. 
    _, times = Rawdata[:]

    # signals는 channels x `# of samples` 형태의 ndarray
    signals = Rawdata.get_data(units='uV')[:, -60 * 200:]

    Rawdata.close()
    return [signals, times, Rawdata]


def load_up_objects(fileList, Features, Labels, OutDir):
    for fname in fileList:
        print("\t%s" % fname)
        if fname[-5] == '1':
            label = 0
        elif fname[-5] == '2':
            label = 1
        try:
            # readEDF은 (signals, times, RawData)를 반환합니다.
            [signals, times, Rawdata] = readEDF(fname)
        except (ValueError, KeyError):
            print("something funky happened in " + fname)
            continue
        signals = BuildEvents(signals, times)

        for idx, signal in enumerate(signals):
            sample = {
                "X": signal,
                "ch_names": [name.upper().split(' ')[-1] for name in chOrder_standard],
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


root = "/home/v-weibjiang/EEGworkload/physionet.org/files/eegmat/1.0.0"
out_dir = '../EEGWorkload'
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
        if fname[-4:] == ".edf":
            edf_files.append(os.path.join(dirName, fname))
edf_files.sort()

train_files = edf_files[:52]
eval_files = edf_files[52:62]
test_files = edf_files[62:]

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
