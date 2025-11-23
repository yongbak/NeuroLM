import numpy as np
# Time series augmentations
class TimeSeriesAugmentor:
    # Safe augmentations for malicious signals
    @staticmethod
    def gaussian_noise(signal, mean=0, std=0.01):
        if type(signal) == list:
            signal = np.array(signal)
        noise = np.random.normal(mean, std, signal.shape)
        return signal + noise

    @staticmethod
    def amplitude_scaling(signal, scale_min=0.9, scale_max=1.1):
        if type(signal) == list:
            signal = np.array(signal)
        scale = np.random.uniform(scale_min, scale_max)
        return signal * scale
