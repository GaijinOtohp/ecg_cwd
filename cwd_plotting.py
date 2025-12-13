import numpy as np
import matplotlib.pyplot as plt

from nn_structures import RLFramework
from cwd_rl_custom import CWDRLCustom, SignalSegment
from cwd_utils import PeakSample, pub_rescale_signal, pub_compute_distribution, pub_scan_peaks

def rl_peak_scan(rescaled_samples: list[float], sampling_rate: int, rl_framework: RLFramework):
    signal_segments_list: list[SignalSegment] = CWDRLCustom.pub_segment_the_main_samples(rescaled_samples, sampling_rate, 0.5)
    peak_list: list[PeakSample] = []
    for segment in signal_segments_list:
        # Get the cwd_rl features of the selected segment
        segment_features, _, _ = pub_compute_distribution(segment._segment_samples, 10)
        # Predict the initial state of the selected segment
        predicted_cwd_rl_output = rl_framework._exploitation_model._model.predict(np.array([segment_features]), verbose=0)
        predicted_cwd_rl_output = predicted_cwd_rl_output[0]
        # Scan the peaks
        dim_list = rl_framework._dimensions_list
        at = dim_list[0]._min + (predicted_cwd_rl_output[0] * (dim_list[0]._max - dim_list[0]._min))
        art = dim_list[1]._min + (predicted_cwd_rl_output[1] * (dim_list[1]._max - dim_list[1]._min))
        temp_peak_list = pub_scan_peaks(segment._segment_samples,
                                    segment._starting_index,
                                    sampling_rate,
                                    at,
                                    art)
        peak_list.extend(temp_peak_list)

    # Remove duplicates from the peak_list and order it by index
    unique_peak_list = {}
    for peak in peak_list:
        unique_peak_list[peak._index] = peak
    sorted_unique_peak_list: list[PeakSample] = sorted(list(unique_peak_list.values()), key=lambda x: x._index)

    return sorted_unique_peak_list


def plot_rl_peak_scan(samples: list[float], sampling_rate: int, rl_framework: RLFramework):
    # Rescale samples to be in an amplitude interval of 1
    global_amp_interval = 1
    rescaled_samples = pub_rescale_signal(samples, global_amp_interval)

    # Scan the peaks of the signal
    sorted_unique_peak_list = rl_peak_scan(rescaled_samples, sampling_rate, rl_framework)

    # Plot the results
    x_signal = [i / sampling_rate for i in range(len(samples))]
    y_signal = samples
    x_marks = [peak_sample._index / sampling_rate for peak_sample in sorted_unique_peak_list]
    y_marks = [samples[peak_sample._index] for peak_sample in sorted_unique_peak_list]

    plt.plot(x_signal, y_signal, label="Peak scan")
    plt.plot(x_marks, y_marks, ".", markersize=6, color="blue")
    plt.legend()
    plt.show()