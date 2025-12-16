import numpy as np
import matplotlib.pyplot as plt

from nn_structures import RLFramework
from cwd_rl_custom import CWDRLCustom, SignalSegment
from cwd_utils import PeakSample, PeakInterval, pub_rescale_signal, pub_compute_distribution, pub_scan_peaks
from cwd_structures import CWDFramework, time_error_tolerance


def rl_peak_selection(rescaled_samples: list[float], sampling_rate: int, rl_framework: RLFramework):
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

    return sorted_unique_peak_list, signal_segments_list


def plot_rl_peak_selection(samples: list[float], sampling_rate: int, rl_framework: RLFramework):
    # Rescale samples to be in an amplitude interval of 1
    global_amp_interval = 1
    rescaled_samples = pub_rescale_signal(samples, global_amp_interval)

    # Scan the peaks of the signal
    sorted_unique_peak_list, _ = rl_peak_selection(rescaled_samples, sampling_rate, rl_framework)

    # Plot the results
    x_signal = [i / sampling_rate for i in range(len(samples))]
    y_signal = samples
    x_marks = [peak_sample._index / sampling_rate for peak_sample in sorted_unique_peak_list]
    y_marks = [samples[peak_sample._index] for peak_sample in sorted_unique_peak_list]

    plt.plot(x_signal, y_signal, label="Signal")
    plt.plot(x_marks, y_marks, ".", markersize=6, color="blue", label="Peak selection")
    # Add axis labels
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (mV)")
    plt.legend()
    plt.show()


def plot_lstm_peak_classification(samples: list[float], sampling_rate: int, cwd_framework: CWDFramework):
    from cwd_lstm_data_generator import LSTMDataBuilderMemory, pr_get_peak_features, pr_get_peak_outputs
    from cwd_validation_tools import RefFloat
    from cwd_signals_selector import MyAnnotation

    # Rescale samples to be in an amplitude interval of 1
    global_amp_interval = 1
    rescaled_samples = pub_rescale_signal(samples, global_amp_interval)

    # Scan the peaks of the signal-----------------------------------------------------------------------------
    sorted_unique_peak_list, signal_segments_list = rl_peak_selection(rescaled_samples, sampling_rate, cwd_framework._cwd_rl_framework)

    # Annotate the selected peaks------------------------------------------------------------------------------
    annotation_list: list[MyAnnotation] = []

    lstm_input_segment = []
    lstm_data_builder_memory = LSTMDataBuilderMemory()
    lstm_data_builder_memory._sampling_rate = sampling_rate
    lstm_data_builder_memory._probing_intervals[0] = sampling_rate
    lstm_data_builder_memory._global_amp_interval = global_amp_interval

    # latest_classified_ref_output and latest_annotation are used for rectifying the index of the latest classified peak
    latest_classified_ref_output: list[RefFloat] = [RefFloat(0)] * cwd_framework._cwd_lstm_model._output_dim
    latest_annotation: list[MyAnnotation] = [MyAnnotation("no_label", -1)] * cwd_framework._cwd_lstm_model._output_dim
    thresholds_list = [threshold_item._threshold for threshold_item in cwd_framework._cwd_lstm_model._output_thresholds]
    labels_list = list(time_error_tolerance.keys())

    for peak_sample in sorted_unique_peak_list:
        # Get the features of the selected peaks
        lstm_input_segment.insert(0, pr_get_peak_features(lstm_data_builder_memory, peak_sample._index, rescaled_samples, signal_segments_list))
        if len(lstm_input_segment) < 2:
            lstm_input_segment.append([0] * cwd_framework._cwd_lstm_model._input_dim)
        else:
            lstm_input_segment = lstm_input_segment[0:2]

        # Predict the labels of the peak_sample
        y_pred_prob = cwd_framework._cwd_lstm_model._model.predict(np.array([lstm_input_segment]), verbose=0)
        y_pred_prob = y_pred_prob[0][0]
        # Convert the prediction values to references (allowing for varying their values from latest_classified_ref_output)
        predicted_ref_output = [RefFloat(out_val) for out_val in y_pred_prob]

        matched_intervs_list: list[PeakInterval] = []

        for i_out in range(len(thresholds_list)):
            if predicted_ref_output[i_out]._value >= thresholds_list[i_out]:
                peak_label = labels_list[i_out]
                # Check if this is a new peak classification
                if latest_classified_ref_output[i_out]._value == 0 and peak_label != "other":
                    matched_intervs_list.append(PeakInterval(name=peak_label, peak_index=peak_sample._index))

                    # Create a new annotation
                    annotation_list.append(MyAnnotation(peak_label, peak_sample._index))
                    latest_annotation[i_out] = annotation_list[-1]
                if predicted_ref_output[i_out]._value >= latest_classified_ref_output[i_out]._value:
                    latest_classified_ref_output[i_out] = predicted_ref_output[i_out]
                    # Update the latest annotation with the new index
                    if peak_label != "other":
                        latest_annotation[i_out]._index = peak_sample._index
            else:
                latest_classified_ref_output[i_out] = RefFloat(0)
        
        # Update lstm_data_builder_memory
        pr_get_peak_outputs(lstm_data_builder_memory, matched_intervs_list)

    # Plot the results----------------------------------------------------------------------------------------------
    x_signal = [i / sampling_rate for i in range(len(samples))]
    y_signal = samples
    x_marks = [my_annotation._index / sampling_rate for my_annotation in annotation_list]
    y_marks = [samples[my_annotation._index] for my_annotation in annotation_list]

    plt.plot(x_signal, y_signal, label="Signal")
    plt.plot(x_marks, y_marks, ".", markersize=6, color="blue", label="Peak classification")
    # Add labels to each point
    for i, my_annotation in enumerate(annotation_list):
        plt.annotate(my_annotation._symbol, # The text label
                     (x_marks[i], y_marks[i]), # The point location (xy)
                     textcoords="offset points", # How to position the text
                     xytext=(0, 10), # Distance from the point (x, y)
                     ha='center') # Horizontal alignment
    # Add axis labels
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (mV)")
    plt.legend()
    plt.show()