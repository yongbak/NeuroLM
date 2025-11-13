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


def prepare_TUEV_dataset(root, is_instruct=False, eeg_max_len=-1, text_max_len=-1):
    train_files = os.listdir(os.path.join(root, "processed_train"))
    val_files = os.listdir(os.path.join(root, "processed_eval"))
    test_files = os.listdir(os.path.join(root, "processed_test"))

    # prepare training and test data loader
    train_dataset = TUEVLoader(
        os.path.join(
            root, "processed_train"), train_files, is_instruct=is_instruct, eeg_max_len=eeg_max_len, text_max_len=text_max_len
    )
    test_dataset = TUEVLoader(
        os.path.join(
            root, "processed_test"), test_files, is_instruct=is_instruct, is_val=True, eeg_max_len=eeg_max_len, text_max_len=text_max_len
    )
    val_dataset = TUEVLoader(
        os.path.join(
            root, "processed_eval"), val_files, is_instruct=is_instruct, is_val=True, eeg_max_len=eeg_max_len, text_max_len=text_max_len
    )
    print(len(train_files), len(val_files), len(test_files))
    return train_dataset, test_dataset, val_dataset


def prepare_TUAB_dataset(root, is_instruct=False, eeg_max_len=-1, text_max_len=-1):
    train_files = os.listdir(os.path.join(root, "train"))
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test data loader
    train_dataset = TUABLoader(os.path.join(root, "train"), train_files, is_instruct=is_instruct, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    test_dataset = TUABLoader(os.path.join(root, "test"), test_files, is_instruct=is_instruct, is_val=True, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    val_dataset = TUABLoader(os.path.join(root, "val"), val_files, is_instruct=is_instruct, is_val=True, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    print(len(train_files), len(val_files), len(test_files))
    return train_dataset, test_dataset, val_dataset


def prepare_TUSL_dataset(root, is_instruct=False, eeg_max_len=-1, text_max_len=-1):
    train_files = os.listdir(os.path.join(root, "train"))
    val_files = os.listdir(os.path.join(root, "eval"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test data loader
    train_dataset = TUSLLoader(os.path.join(root, "train"), train_files, is_instruct=is_instruct, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    test_dataset = TUSLLoader(os.path.join(root, "test"), test_files, is_instruct=is_instruct, is_val=True, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    val_dataset = TUSLLoader(os.path.join(root, "eval"), val_files, is_instruct=is_instruct, is_val=True, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    print(len(train_files), len(val_files), len(test_files))
    return train_dataset, test_dataset, val_dataset


def prepare_HMC_dataset(root, is_instruct=False, eeg_max_len=-1, text_max_len=-1):
    train_files = os.listdir(os.path.join(root, "train"))
    val_files = os.listdir(os.path.join(root, "eval"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test data loader
    train_dataset = HMCLoader(os.path.join(root, "train"), train_files, is_instruct=is_instruct, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    test_dataset = HMCLoader(os.path.join(root, "test"), test_files, is_instruct=is_instruct, is_val=True, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    val_dataset = HMCLoader(os.path.join(root, "eval"), val_files, is_instruct=is_instruct, is_val=True, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    print(len(train_files), len(val_files), len(test_files))
    return train_dataset, test_dataset, val_dataset


def prepare_Workload_dataset(root, is_instruct=False, eeg_max_len=-1, text_max_len=-1):
    train_files = os.listdir(os.path.join(root, "train"))
    val_files = os.listdir(os.path.join(root, "eval"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test data loader
    train_dataset = WorkloadLoader(os.path.join(root, "train"), train_files, is_instruct=is_instruct, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    test_dataset = WorkloadLoader(os.path.join(root, "test"), test_files, is_instruct=is_instruct, is_val=True, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    val_dataset = WorkloadLoader(os.path.join(root, "eval"), val_files, is_instruct=is_instruct, is_val=True, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
    print(len(train_files), len(val_files), len(test_files))
    return train_dataset, test_dataset, val_dataset


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


def txt_to_full_pickle(txt_file_path, output_pkl_path, sampling_rate=2000.0, notch=60.0):
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
    
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(sample, f)
    
    print(f"✅ Saved full signal pickle: {output_pkl_path}")
    print(f"   Shape: {signals.shape}, Duration: {signals.shape[1]/sampling_rate:.2f}s")
    
    return signals.shape


if __name__ == "__main__":
    # Example usage
    txt_file = "/home/yongbak/research/NeuroLM/datasets/PMD-Dataset/data/s0_b_2024_07.csv"
    output_pkl = "/home/yongbak/research/NeuroLM/datasets/processed/tmp/s0_b_2024_07.pkl"
    txt_to_full_pickle(txt_file, output_pkl)