import numpy as np
from multiprocessing import shared_memory
import pickle
import math

from cwd_signals_selector import MyAnnotation
from nn_structures import ValidationMetricsItem


class PeakInterval:

    def __init__(self, name="", starting=0, ending=0, peak_index=0):
        self._name = name
        self._starting = starting
        self._ending = ending
        self._peak_index = peak_index


class PeakSample:

    def __init__(self, index=0, value=0.0):
        self._index: int = index
        self._value: float = value

        self._prev_tan: float
        self._next_tan: float
        self._deviation_angle: float # Argument

        self._prev_mag: float
        self._next_mag: float


def pub_mean_min_max(samples: list[float]):
    mean = 0.0
    min = float("inf")
    max = float("-inf")
    for sample in samples:
        mean += sample / len(samples)
        if sample < min:
            min = sample
        if sample > max:
            max = sample

    return mean, min, max


def pub_amplitude_interval(samples: list[float]):
    _, min, max = pub_mean_min_max(samples)
    return max - min


def pub_compute_distribution(segment_samples: list[float], resolution: int):
    # Get the segment characteristics
    _, min, max = pub_mean_min_max(segment_samples)
    # Compute the step of the distribution based on the resolution
    step = (max - min) / resolution

    # Compute the distribution
    distribution = [0 for _ in range(resolution)]
    distribution_normalized = [0 for _ in range(resolution)]

    if step != 0:
        for sample in segment_samples:
            bar_index = int((sample - min) // step)
            if bar_index == len(distribution):
                bar_index -= 1
            distribution[bar_index] += 1
    for i in range(len(distribution)):
        distribution_normalized[i] = distribution[i] / len(segment_samples)

    return distribution_normalized, min, step


def pub_rescale_signal(samples: list[float], new_amp_interval: float):
    mean, min, max = pub_mean_min_max(samples)
    amp_interval = max - min
    scaling_ratio = new_amp_interval / amp_interval
    # Normalize the samples
    filtered_samples = []

    for sample in samples:
        filtered_samples.append(sample * scaling_ratio)

    return filtered_samples


def pub_approximate_indexes_to_intervals(annotation: list[MyAnnotation], tolerance: float, full_signal: list[float], sampling_rate: int):
    from cwd_utils import PeakInterval

    # There might be two peaks on the same index
    # They should have the same interval but with their unique label
    intervals_dict: dict[int, PeakInterval] = {}
    intervals_list: list[PeakInterval] = []

    for anno in annotation:
        index_interval = PeakInterval(name=anno._symbol, peak_index=anno._index)
        # Check if the interval of the current peak already exists
        if index_interval._peak_index in intervals_dict:
            # If yes then just clown the peak interval but with the different naming
            index_interval._starting = intervals_dict[index_interval._peak_index]._starting
            index_interval._ending = intervals_dict[index_interval._peak_index]._ending
        else:
            # Create a new interval for the peak
            # Get the index of the next and previous peaks if exists
            prev_peak_index_array = [anno._index for anno in annotation if anno._index < index_interval._peak_index]
            next_peak_index_array = [anno._index for anno in annotation if anno._index > index_interval._peak_index]
            if len(prev_peak_index_array) > 0:
                # Compute the distance of the tolerance and find its index
                prev_dist_amp_tol = np.sqrt(((index_interval._peak_index - max(prev_peak_index_array)) / sampling_rate) ** 2 +
                                            (full_signal[index_interval._peak_index] - full_signal[max(prev_peak_index_array)]) ** 2) * \
                                    tolerance / 100
                # Compute the starting index using prev_dist_amp_tol and the ending index "index_interval._peak_index"
                # by computing distances from the latest peak in prev_peak_index_array
                dist_diff = float("inf")
                for i_peak_index in range(max(prev_peak_index_array), index_interval._peak_index):
                    i_dist_amp = np.sqrt(((index_interval._peak_index - i_peak_index) / sampling_rate) ** 2 +
                                         (full_signal[index_interval._peak_index] - full_signal[i_peak_index]) ** 2)
                    if np.abs(i_dist_amp - prev_dist_amp_tol) < dist_diff:
                        index_interval._starting = i_peak_index
                        dist_diff = np.abs(i_dist_amp - prev_dist_amp_tol)
            else:
                index_interval._starting = index_interval._peak_index - int(tolerance * (index_interval._peak_index - 0) / 100)

            if len(next_peak_index_array) > 0:
                # Compute the distance of the tolerance and find its index
                next_dist_amp_tol = np.sqrt(((min(next_peak_index_array) - index_interval._peak_index) / sampling_rate) ** 2 +
                                            (full_signal[min(next_peak_index_array)] - full_signal[index_interval._peak_index]) ** 2) * \
                                    tolerance / 100
                # Compute the ending index using next_dist_amp_tol and the starting index "index_interval._peak_index"
                # by computing distances from the earliest peak in next_peak_index_array
                dist_diff = float("inf")
                for i_peak_index in range(index_interval._peak_index + 1, min(next_peak_index_array) + 1):
                    i_dist_amp = np.sqrt(((i_peak_index - index_interval._peak_index) / sampling_rate) ** 2 +
                                         (full_signal[i_peak_index] - full_signal[index_interval._peak_index]) ** 2)
                    if np.abs(i_dist_amp - next_dist_amp_tol) < dist_diff:
                        index_interval._ending = i_peak_index
                        dist_diff = np.abs(i_dist_amp - next_dist_amp_tol)
            else:
                index_interval._ending = index_interval._peak_index + int(tolerance * ((len(full_signal) - 1) - index_interval._peak_index) / 100)

            # Add the new interval in the dictionary
            intervals_dict[index_interval._peak_index] = index_interval

        # Add the corner's interval to the list
        intervals_list.append(index_interval)
    
    return intervals_list


def pr_mag_tan(begining_sample: PeakSample, ending_sample: PeakSample, sampling_rate: int):
    x_diff = (ending_sample._index - begining_sample._index) / sampling_rate
    y_diff = ending_sample._value - begining_sample._value

    mag = np.sqrt(x_diff ** 2 + y_diff ** 2)
    tan = y_diff / x_diff

    return mag, tan


def pub_scan_peaks(samples: list[float], scan_starting_index: int, sampling_rate: int, at: float, art: float):
    from cwd_utils import PeakSample

    peak_list: list[PeakSample] = []
    amplitude_interval_value = pub_amplitude_interval(samples)

    if amplitude_interval_value == 0:
        return peak_list
    
    temp_peak_list: list[PeakSample] = []
    latest_peak: PeakSample = None

    for i in range(len(samples)):
        # Set current sample
        # samples is an excerpt from a longer list of samples.
        # That's why "scan_starting_index" is added as the padding of the excerpt
        temp_peak_list.append(PeakSample(index=scan_starting_index + i, value=samples[i]))

        if i == 0:
            latest_peak = temp_peak_list[0]

        # Get the index of the last peak without the padding
        last_peak_shift_indx = latest_peak._index - scan_starting_index

        # Compute _prev_mag and _prev_tan of the current sample
        if i - last_peak_shift_indx > 0:
            temp_peak_list[i]._prev_mag, temp_peak_list[i]._prev_tan = pr_mag_tan(latest_peak, temp_peak_list[i], sampling_rate)

        # Check if the current sample is two indexes ahead of the latest peak
        if i - last_peak_shift_indx > 1:
            # Update _next_mag, _next_tan, and _deviation_angle of the samples between the latest state and the current sample
            for j in range(last_peak_shift_indx + 1, i):
                temp_peak_list[j]._next_mag, temp_peak_list[j]._next_tan = pr_mag_tan(temp_peak_list[j], temp_peak_list[i], sampling_rate)
                temp_peak_list[j]._deviation_angle = (np.atan(temp_peak_list[j]._next_tan) - np.atan(temp_peak_list[j]._prev_tan)) * 180 / np.pi

            # Select the samples with the angle deviation that exceeds at
            # and both of _prev_mag and _next_mag exceeds amplitude_interval * art
            latest_peaks_buff = temp_peak_list[last_peak_shift_indx + 1 : i]
            selected_peaks = [peak_samp for peak_samp in latest_peaks_buff if peak_samp._next_mag > amplitude_interval_value * art and
                                                                              peak_samp._prev_mag > amplitude_interval_value * art and
                                                                              np.abs(peak_samp._deviation_angle) > at]
            
            # Check if there are any selected samples that fulfills the conditions
            if len(selected_peaks) > 0:
                # Select the one with the largest segments
                peak_list.append(max(selected_peaks, key=lambda peak: peak._next_mag + peak._prev_mag))
                latest_peak = peak_list[-1]

                # If new peak is created
                # then update all previous_mag and _prev_tan of the new peak's next samples
                last_peak_shift_indx = latest_peak._index - scan_starting_index
                for j in range(last_peak_shift_indx + 1, i + 1):
                    temp_peak_list[j]._prev_mag, temp_peak_list[j]._prev_tan = pr_mag_tan(latest_peak, temp_peak_list[j], sampling_rate)

    return peak_list

def get_object_from_shared_memory(shm_name: str):
    sm = shared_memory.SharedMemory(name=shm_name)
    return pickle.loads(sm.buf.tobytes()), sm


def put_object_in_shared_memory(obj, sm: shared_memory.SharedMemory):
    binary_obj = pickle.dumps(obj, pickle.HIGHEST_PROTOCOL)
    sm.buf[:len(binary_obj)] = binary_obj


def segment_inputs_and_outputs(input_array, output_array, segment_length):
    input_segments = []
    output_segments = []

    total_length = len(input_array)
    num_segments = total_length // segment_length

    for i in range(num_segments):
        start_idx = i * segment_length
        end_idx = start_idx + segment_length

        input_segment = input_array[start_idx:end_idx]
        output_segment = output_array[start_idx:end_idx]

        input_segments.append(input_segment)
        output_segments.append(output_segment)

    return (np.array(input_segments), np.array(output_segments))


def segment_inputs_and_outputs_redundantly(input_array, output_array, timesteps): # shape of input_array and output_array is (samples, features)
    input_segments = []
    output_segments = []

    empty_input = [0] * len(input_array[0]) # returns zeros of shape (features)
    empty_output = [0] * len(output_array[0]) # returns zeros of shape (labels)
    for i in range(len(input_array)):
        input_segment = [empty_input] * timesteps
        output_segment = [empty_output] * timesteps

        for j in range(timesteps):
            input_segment[j] = input_array[i - j] if 0 <= i - j else empty_input
            output_segment[j] = output_array[i - j] if 0 <= i - j < len(output_array) else empty_output

        input_segments.append(input_segment) # shape of input_segments is (samples, timesteps, features)
        output_segments.append(output_segment) # shape of output_segments is (samples, timesteps, labels)
        #output_segments.append(output_array[i]) # shape of output_segments is (samples, labels)

    return (np.array(input_segments), np.array(output_segments))


def sort_dataset_sequences_to_flat_batches(input_seqs, output_seqs, batch_size): # shape of input_seqs and output_seqs is (sequences, samples, timesteps, features)
    input_segments = []
    output_segments = []

    for i_seq_batch in range(math.ceil(len(input_seqs) / batch_size)):

        total_samples = sum([len(seq) for index, seq in enumerate(input_seqs) if i_seq_batch * batch_size <= index < (i_seq_batch + 1) * batch_size])

        empty_input_segment = np.zeros(input_seqs[0][0].shape) # returns zeros of shape (timesteps, features)
        empty_output_segment = np.zeros(output_seqs[0][0].shape) # returns zeros of shape (timesteps, labels)
        for i_samp in range(total_samples):
            i_seq = i_seq_batch * batch_size + i_samp % batch_size
            i_seq_samp = i_samp // batch_size

            input_segment = empty_input_segment
            output_segment = empty_output_segment

            if i_seq < len(input_seqs) and i_seq_samp < len(input_seqs[i_seq]):
                input_segment = input_seqs[i_seq][i_seq_samp]
                output_segment = output_seqs[i_seq][i_seq_samp]

            input_segments.append(input_segment) # shape of input_segments is (samples, timesteps, features)
            output_segments.append(output_segment) # shape of output_segments is (samples, timesteps, labels)

    return (np.array(input_segments), np.array(output_segments))


def extract_validation_sub_metrics(validation_metrics_item: ValidationMetricsItem):
    tp = validation_metrics_item._true_positives
    tn = validation_metrics_item._true_negatives
    fp = validation_metrics_item._false_positives
    fn = validation_metrics_item._false_negatives

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)

    print(f"accuracy: {accuracy:.2f}\t sensitivity: {sensitivity:.2f}\t specificity: {specificity:.2f}\t PPV: {ppv:.2f}\t NPV: {npv:.2f}")
