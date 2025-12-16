from multiprocessing import Process, Manager
import numpy as np
import itertools

from cwd_signals_selector import AnnoSignal
from nn_structures import RLFramework
from cwd_structures import LSTMTrainingData
from cwd_utils import PeakInterval, pub_rescale_signal, pub_approximate_indexes_to_intervals
from cwd_rl_custom import SignalSegment
from cwd_plotting import rl_peak_selection


class LSTMDataBuilderMemory:

    def __init__(self):
        self._output_labels = ["p_(", "p", "p_)", "qrs_(", "qrs", "qrs_)", "t_(", "t", "t_)", "other"]

        self._sampling_rate: int

        self._global_amp_interval: float

        self._arg_tan_normalizer: float = np.pi

        self._latest_classified_peak_index = 0
        self._latest_p_peak_index = 0
        self._latest_qrs_peak_index = 0
        self._latest_t_peak_index = 0

        self._p_count = 0
        self._qrs_count = 0
        self._t_count = 0

        self._probing_intervals = [0, 0, 0, 0]


def pub_generate_lstm_data(selected_anno_signals_dict: dict[str, AnnoSignal], rl_framework: RLFramework):
    shared_lstm_training_data_dict: dict[str, LSTMTrainingData] = Manager().dict()

    # Process each signal in a different process
    signal_processes = []
    for signal_key, anno_signal in selected_anno_signals_dict.items():
        # Create and start the process for generating lstm data of the selected signal
        signal_process = Process(target=pr_generate_signal_lstm_data, args=(anno_signal, signal_key, shared_lstm_training_data_dict, rl_framework))
        signal_processes.append(signal_process)
        signal_process.start()

    for process in signal_processes:
        process.join()
    return dict(shared_lstm_training_data_dict)


def pub_extract_right_triangle_branches(samples: list[float], sampling_rate: int):
    av_opposite = 0.0
    av_adjacent = 0.0
    av_tangent = 0.0
    for i in range(1, len(samples)):
        opposite = samples[i] - samples[0]
        adjacent = i / sampling_rate
        tangent = opposite / adjacent

        av_opposite += opposite
        av_adjacent += adjacent
        av_tangent += tangent
    if len(samples) > 1:
        av_opposite /= (len(samples) - 1)
        av_adjacent /= (len(samples) - 1)
        av_tangent /= (len(samples) - 1)
    av_hypotenuse = np.sqrt(av_opposite ** 2 + av_adjacent ** 2)

    return av_opposite, av_adjacent, av_hypotenuse, av_tangent


def pr_update_peak_memory_params(latest_x_peak_index: int, xx_interval_av: int, x_count: int, new_x_peak_index: int):
    latest_x_peak_interval = new_x_peak_index - latest_x_peak_index
    xx_interval_av = int(xx_interval_av * x_count + latest_x_peak_interval) // (x_count + 1)

    return new_x_peak_index, xx_interval_av, x_count + 1


def pr_get_peak_outputs(lstm_data_builder_memory: LSTMDataBuilderMemory, peak_interv_list: list[PeakInterval]):
    outputs: list[float] = [0.0] * len(lstm_data_builder_memory._output_labels)

    for peak_interv in peak_interv_list:
        for i, label in enumerate(lstm_data_builder_memory._output_labels):
            if peak_interv._name == label:
                outputs[i] = 1.0

                if label == "p":
                    lstm_data_builder_memory._latest_p_peak_index, lstm_data_builder_memory._probing_intervals[1], lstm_data_builder_memory._p_count = \
                    pr_update_peak_memory_params(lstm_data_builder_memory._latest_p_peak_index, lstm_data_builder_memory._probing_intervals[1], lstm_data_builder_memory._p_count, peak_interv._peak_index)
                elif label == "qrs":
                    lstm_data_builder_memory._latest_qrs_peak_index, lstm_data_builder_memory._probing_intervals[2], lstm_data_builder_memory._qrs_count = \
                    pr_update_peak_memory_params(lstm_data_builder_memory._latest_qrs_peak_index, lstm_data_builder_memory._probing_intervals[2], lstm_data_builder_memory._qrs_count, peak_interv._peak_index)
                elif label == "t":
                    lstm_data_builder_memory._latest_t_peak_index, lstm_data_builder_memory._probing_intervals[3], lstm_data_builder_memory._t_count = \
                    pr_update_peak_memory_params(lstm_data_builder_memory._latest_t_peak_index, lstm_data_builder_memory._probing_intervals[3], lstm_data_builder_memory._t_count, peak_interv._peak_index)

        if peak_interv._name != "other":
            lstm_data_builder_memory._latest_classified_peak_index = peak_interv._peak_index

    return outputs


def pr_get_max_min_amp_from_tan_indexes(short_range_index: int, mean_tan: float, span_samples: list[float], sampling_rate: int):
    max_amp_index = 0
    max_amp_from_tan = float('-inf')
    min_amp_index = 0
    min_amp_from_tan = float('inf')

    for i in range(1, min(short_range_index, len(span_samples))):
        val_int_tan = span_samples[0] + mean_tan * (i / sampling_rate)
        if span_samples[i] - val_int_tan > max_amp_from_tan:
            max_amp_from_tan = span_samples[i] - val_int_tan
            max_amp_index = i
        if span_samples[i] - val_int_tan < min_amp_from_tan:
            min_amp_from_tan = span_samples[i] - val_int_tan
            min_amp_index = i

    return max_amp_index, min_amp_index


def pr_get_side_range_features(short_range_index: int, span_samples: list[float], lstm_data_builder_memory: LSTMDataBuilderMemory):
    _, _, _, span_tan = pub_extract_right_triangle_branches(span_samples, lstm_data_builder_memory._sampling_rate)

    span_max_amp_index, span_min_amp_index = pr_get_max_min_amp_from_tan_indexes(short_range_index, span_tan, span_samples, lstm_data_builder_memory._sampling_rate)

    span_max_amp_tan = (span_samples[span_max_amp_index] - span_samples[0]) / (span_max_amp_index / lstm_data_builder_memory._sampling_rate)
    span_min_amp_tan = (span_samples[span_min_amp_index] - span_samples[0]) / (span_min_amp_index / lstm_data_builder_memory._sampling_rate)
    span_amp_interv_to_global = (span_samples[span_max_amp_index] - span_samples[span_min_amp_index]) / lstm_data_builder_memory._global_amp_interval
    span_max_amp_bran_to_glob = np.sqrt((span_samples[span_max_amp_index] - span_samples[0]) ** 2 +
                                        (span_max_amp_index / lstm_data_builder_memory._sampling_rate) ** 2) / lstm_data_builder_memory._global_amp_interval
    span_min_amp_bran_to_glob = np.sqrt((span_samples[span_min_amp_index] - span_samples[0]) ** 2 +
                                        (span_min_amp_index / lstm_data_builder_memory._sampling_rate) ** 2) / lstm_data_builder_memory._global_amp_interval
    
    span_max_amp_atan_div = 0
    span_min_amp_atan_div = 0
    if span_tan != 0:
        span_max_amp_atan_div = (np.atan(span_max_amp_tan) - np.atan(span_tan)) / lstm_data_builder_memory._arg_tan_normalizer
        span_min_amp_atan_div = (np.atan(span_min_amp_tan) - np.atan(span_tan)) / lstm_data_builder_memory._arg_tan_normalizer

    return [span_max_amp_atan_div, span_min_amp_atan_div, span_amp_interv_to_global, span_max_amp_bran_to_glob, span_min_amp_bran_to_glob]


def pr_get_surrounding_range_features(long_range_index: int, short_range_index: int, peak_index: int, lstm_data_builder_memory: LSTMDataBuilderMemory, rescaled_samples: list[float]):
    x_peak_pre_samples = rescaled_samples[peak_index - long_range_index : peak_index + 1]
    x_peak_pre_samples.reverse()
    x_peak_post_samples = rescaled_samples[peak_index : peak_index + long_range_index + 1]

    x_peak_pre_features = [0 for _ in range(5)]
    x_peak_post_features = [0 for _ in range(5)]
    if len(x_peak_pre_samples) > 0:
        x_peak_pre_features = pr_get_side_range_features(short_range_index, x_peak_pre_samples, lstm_data_builder_memory)
    if len(x_peak_post_samples) > 0:
        x_peak_post_features = pr_get_side_range_features(short_range_index, x_peak_post_samples, lstm_data_builder_memory)

    return x_peak_pre_features + x_peak_post_features


def pr_get_peak_features(lstm_data_builder_memory: LSTMDataBuilderMemory, peak_index: int, rescaled_samples: list[float], signal_segments_list: list[SignalSegment]):
    features: list[float] = []

    interval_features: list[float]
    for i_long_interval_segm in range(2, 5 + 1):
        for i_short_interval_segm in range(1, i_long_interval_segm):
            interval_features = pr_get_surrounding_range_features(int(lstm_data_builder_memory._probing_intervals[0] * i_long_interval_segm // 10),
                                                                  int(lstm_data_builder_memory._probing_intervals[0] * i_short_interval_segm // 10),
                                                                  peak_index,
                                                                  lstm_data_builder_memory,
                                                                  rescaled_samples)
            features.extend(interval_features)

    peak_segment = [segment for segment in signal_segments_list if segment._starting_index <= peak_index <= segment._ending_index][0]
    segment_interval = peak_segment._segment_max - peak_segment._segment_min

    rhythm_features: list[float] = [0, 0, 0]
    if lstm_data_builder_memory._probing_intervals[1] != 0:
        rhythm_features[0] = (peak_index - lstm_data_builder_memory._latest_p_peak_index) / lstm_data_builder_memory._probing_intervals[1]
    if lstm_data_builder_memory._probing_intervals[2] != 0:
        rhythm_features[1] = (peak_index - lstm_data_builder_memory._latest_qrs_peak_index) / lstm_data_builder_memory._probing_intervals[2]
    if lstm_data_builder_memory._probing_intervals[3] != 0:
        rhythm_features[2] = (peak_index - lstm_data_builder_memory._latest_t_peak_index) / lstm_data_builder_memory._probing_intervals[3]

    rhythm_features.append((rescaled_samples[peak_index] - peak_segment._segment_min) / segment_interval)
    rhythm_features.append(rescaled_samples[peak_index] / lstm_data_builder_memory._global_amp_interval)
    rhythm_features.append(segment_interval / lstm_data_builder_memory._global_amp_interval)

    features.extend(rhythm_features)

    return features


def pr_generate_signal_lstm_data(anno_signal: AnnoSignal, signal_key: str, shared_lstm_training_data_dict: dict[str, LSTMTrainingData], rl_framework: RLFramework):
    lstm_training_data = LSTMTrainingData(anno_signal._fs)
    
    # Rescale samples to be in an amplitude interval of 1
    global_amp_interval = 1
    rescaled_samples = pub_rescale_signal(anno_signal._signal, global_amp_interval)
    sampling_rate = anno_signal._fs

    # Scan the peaks of each segment using the peak scanner ----------------------------------------------------------------------
    sorted_unique_peak_list, signal_segments_list = rl_peak_selection(rescaled_samples, sampling_rate, rl_framework)
    #-----------------------------------------------------------------------------------------------------------------------------
    peak_indexes = [peak._index for peak in sorted_unique_peak_list]

    # Combine each scanned sample in peak_indexes to its closest annotation in approx_interv_list
    # Create the intervals covering 40% from the peaks in both sides
    approx_interv_list: list[PeakInterval] = pub_approximate_indexes_to_intervals(anno_signal._annotation, 40, rescaled_samples, sampling_rate)
    # Sort the intervals in a dictionary with the indecies of the scanned peaks
    peaks_to_closest_interval: list[(int, PeakInterval)] = [(closest_index, PeakInterval(peak_index=closest_index, name=peak_interv._name)) for closest_index, peak_interv in
                                                            [(min(covered_indexes_dist, key=lambda x: x[0])[1] if len(covered_indexes_dist) > 0 else peak_interv._peak_index, peak_interv) for covered_indexes_dist, peak_interv in
                                                             [([(np.sqrt(((peak_index - peak_interv._peak_index) / sampling_rate) ** 2 + (rescaled_samples[peak_index] - rescaled_samples[peak_interv._peak_index]) ** 2), peak_index) for peak_index in covered_indexes],
                                                               peak_interv) for covered_indexes, peak_interv in
                                                               [([peak_index for peak_index in peak_indexes if peak_interv._starting <= peak_index <= peak_interv._ending], peak_interv) for peak_interv in approx_interv_list]]]]
    peaks_to_closest_interval.sort(key=lambda x: x[0])
    sorted_interv_dict_buff: dict[int, list[PeakInterval]] = {}
    for key, group in itertools.groupby(peaks_to_closest_interval, key=lambda x: x[0]):
        sorted_interv_dict_buff[key] = [item[1] for item in group]

    sorted_interv_dict: dict[int, list[PeakInterval]] = {peak_index: [PeakInterval(peak_index=peak_index, name="other")] for peak_index in peak_indexes}

    # Combine other peaks with the true ones
    for peak_index in sorted_interv_dict_buff.keys():
        sorted_interv_dict[peak_index] = sorted_interv_dict_buff[peak_index]
    
    sorted_interv_dict = dict(sorted(sorted_interv_dict.items()))

    # Create the training samples--------------------------------------------------------------------------------------------------
    lstm_data_builder_memory = LSTMDataBuilderMemory()
    lstm_data_builder_memory._sampling_rate = sampling_rate
    lstm_data_builder_memory._probing_intervals[0] = sampling_rate
    lstm_data_builder_memory._global_amp_interval = global_amp_interval

    for peak_index in sorted_interv_dict.keys():
        # Set features of the peak before the outputs (the outputs updates lstm_data_builder_memory for the next peak)
        lstm_training_data._input_sequences.append(pr_get_peak_features(lstm_data_builder_memory, peak_index, rescaled_samples, signal_segments_list))
        # Set the output
        lstm_training_data._output_sequences.append(pr_get_peak_outputs(lstm_data_builder_memory, sorted_interv_dict[peak_index]))
        lstm_training_data._index_sequences.append(peak_index)

    # Append the training_data for the current signal into training_data_dict
    shared_lstm_training_data_dict[signal_key] = lstm_training_data