from time import time
import numpy as np

from nn_structures import RLFramework
from cwd_signals_selector import AnnoSignal
from cwd_utils import PeakInterval, pub_rescale_signal, pub_approximate_indexes_to_intervals, pub_compute_distribution, pub_scan_peaks, pub_amplitude_interval, get_object_from_shared_memory, put_object_in_shared_memory
from circular_queue import CircularQueue


class SignalSegment:
        def __init__(self, starting_index = 0, ending_index = 0, segment_samples = []):
            self._starting_index = starting_index
            self._ending_index = ending_index
            self._segment_samples = segment_samples
            self._segment_mean = 0
            self._segment_min = 0
            self._segment_max = 0
            

class CWDRLCustom:

    _penalty_mouvement = 1

    _penalty_false_negative = 5 # not selecting a peak
    _penalty_false_positive = 1 # over selecting a peak

    _reward_true_positive = 1
    _reward_selecting_all_peaks = 5

    _pfrt_increment = 0.5 # pfrt -> peak over the fiducial points ratio threshold
    _pfrt_reset = 2
    _pfrt_hold_reset = 50


    def __init__(self, rl_framework: RLFramework):
        from reinforcement_learning import Environment

        self._approx_interv_list: list[PeakInterval] = []
        self._rescaled_samples: list[float] = []
        self._sampling_rate = 0.0

        self._at_circular_queue: CircularQueue
        self._art_circular_queue: CircularQueue

        self._signal_segments_list: list[SignalSegment] = []
        self._selected_segment = -1

        self._done = False

        self._pfrt = 2 # pfrt -> peak over the fiducial points ratio threshold
        self._pfrt_hold = 50

        # Create the environment for the reinforcement learning model
        learning_rate = 0.1
        discount = 0.95
        self._env = Environment(rl_framework._dimensions_list, learning_rate, discount, self.pr_compute_reward, self.pr_check_if_done)


    def pr_compute_reward(self, observation, movements, env):
        reward = 0
        bad_state = False

        # Scan corners of the selected window
        dim_list = self._env._dimensions_list
        at = dim_list[0]._min + (observation[0] * dim_list[0]._step)
        art = dim_list[1]._min + (observation[1] * dim_list[1]._step)
        temp_peak_list = pub_scan_peaks(self._signal_segments_list[self._selected_segment]._segment_samples,
                                    self._signal_segments_list[self._selected_segment]._starting_index,
                                    self._sampling_rate,
                                    at,
                                    art)
        # Take only the indexes of the corners
        temp_peak_indexes = [peak._index for peak in temp_peak_list]
        # Create a list of the intervals holding the peaks in temp_peak_indexes
        temp_intervals_list: list[PeakInterval] = []

        # Compute the reward of current step
        # Get the intervals of the true peaks of the selected window only
        window_approx_intervals = [interval for interval in self._approx_interv_list if
                                   self._signal_segments_list[self._selected_segment]._starting_index < interval._starting and
                                   interval._ending < self._signal_segments_list[self._selected_segment]._ending_index]
        
        # Compute reward value of the current step
        reward -= CWDRLCustom._penalty_mouvement
        for peak_index in temp_peak_indexes:
            # Check if the scanned peak is included in true peak interval
            chosen_interval = [interval for interval in window_approx_intervals if interval._starting <= peak_index and
                                                                                   peak_index <= interval._ending]
            if len(chosen_interval) > 0:
                # If yes then check if this interval has not included a previous peak
                if chosen_interval[0] not in temp_intervals_list:
                    reward += CWDRLCustom._reward_true_positive
                    temp_intervals_list.append(chosen_interval[0])
                else:
                    reward -= CWDRLCustom._penalty_false_positive
            else:
                reward -= CWDRLCustom._penalty_false_positive
        # Compute the reward of the false negatives
        for interval in window_approx_intervals:
            if len([peak_index for peak_index in temp_peak_indexes if interval._starting <= peak_index <= interval._ending]) == 0:
                reward -= CWDRLCustom._penalty_false_positive
                bad_state = True

        if len(window_approx_intervals) > 0:
            all_peaks_detected = len(temp_peak_indexes) / len(window_approx_intervals) < self._pfrt
        else:
            all_peaks_detected = False
        if not bad_state and (all_peaks_detected or len(window_approx_intervals) == 0) or (self._pfrt - self._pfrt_reset) >= 40:
            if all_peaks_detected:
                reward += CWDRLCustom._reward_selecting_all_peaks
            self._done = True
        
        return reward, bad_state
    

    def pr_check_if_done(self, new_state):
        done = False
        # ----Done when emprovement was almost stable----
        self._at_circular_queue.enqueue(new_state[0])
        self._art_circular_queue.enqueue(new_state[1])

        if pub_amplitude_interval(self._at_circular_queue._nodes_list) < 2 and len(self._at_circular_queue._nodes_list) > 4 and \
           pub_amplitude_interval(self._art_circular_queue._nodes_list) < 2 and len(self._art_circular_queue._nodes_list) > 4 and \
           not self._done:
            self._env.agent_reset()
            self._at_circular_queue = CircularQueue(5)
            self._art_circular_queue = CircularQueue(5)

        if self._done:
            done = True
            self._done = False
            self._pfrt_hold = CWDRLCustom._pfrt_hold_reset
        elif self._pfrt_hold > 0:
            self._pfrt_hold -= 1
        else:
            self._pfrt += CWDRLCustom._pfrt_increment
            self._pfrt_hold = CWDRLCustom._pfrt_hold_reset

        return done
    

    def pub_deep_fit_rl_data(self, representative_anno_segment: AnnoSignal, rl_framework_shm_name: str, lock):
        at_dim = self._env._dimensions_list[0]
        art_dim = self._env._dimensions_list[1]
        # Include the signal infos
        # Rescale samples to be in an amplitude interval of 1
        self._rescaled_samples = pub_rescale_signal(representative_anno_segment._signal, 1)
        self._sampling_rate = representative_anno_segment._fs

        # Create the intervals covering 40% from the peaks in both sides
        self._approx_interv_list = pub_approximate_indexes_to_intervals(representative_anno_segment._annotation, 40, self._rescaled_samples, self._sampling_rate)

        # Initialize the conditions of finishing the episodes
        self._at_circular_queue = CircularQueue(5)
        self._art_circular_queue = CircularQueue(5)

        # Create the global inputs and outputs data lists
        global_inputs_list = []
        global_outputs_list = []

        # Segment the samples of signal according to their distribution
        self._signal_segments_list = CWDRLCustom.pub_segment_the_main_samples(self._rescaled_samples, self._sampling_rate, 0.5)
        max_episodes = 3
        for self._selected_segment in range(len(self._signal_segments_list)):
            self._env.q_table_reset()
            self._pfrt = CWDRLCustom._pfrt_reset

            # Get the features of the selected segment
            segment_features, _, _ = pub_compute_distribution(self._signal_segments_list[self._selected_segment]._segment_samples, 10)

            # Iterate through episodes
            for _ in range(max_episodes):                
                # Predict the initial state of the selected segment
                #lock.acquire()
                rl_framework, _ = get_object_from_shared_memory(rl_framework_shm_name)
                episode_predicted_output = rl_framework._exploration_model._model.predict(np.array([segment_features]), verbose=0)
                #lock.release()
                episode_predicted_output = episode_predicted_output[0]

                # Start training with the new initial state in the new episode
                (_, _, best_state, _) = self._env.deep_train(episode_predicted_output)

                # Check if the predicted state of the exploration model is greater than the improvementThreshold
                improvement_threshold = 0.001
                best_at = best_state[0] / at_dim._size
                best_art = best_state[1] / art_dim._size
                if (np.abs(episode_predicted_output[0] - best_at) + np.abs(episode_predicted_output[1] - best_art)) > improvement_threshold:
                    # Train the Exploration model with the new action sequence
                    #lock.acquire()
                    rl_framework, rl_framework_shm = get_object_from_shared_memory(rl_framework_shm_name)
                    start_time = time()
                    rl_framework._exploration_model._model.fit(np.array([segment_features]), np.array([[best_at, best_art]]), batch_size=4, epochs=1, verbose=0)
                    end_time = time()
                    rl_framework._exploration_model._last_training_elapsed_time_seconds = end_time - start_time
                    put_object_in_shared_memory(rl_framework, rl_framework_shm)
                    #lock.release()
            # Include the segment_features in global_inputs_list and get its best_state as output
            global_inputs_list.append(segment_features)
            # Get the best state of the environment
            acceptable_states_with_action = [(params["reward"], state_with_action) for state_with_action, params in self._env.get_q_table_dict().items() if params["bad_state"] == False]
            # Check if there was no best state found
            if len(acceptable_states_with_action) == 0:
                # Then just take the whole Q table
                acceptable_states_with_action = [(params["reward"], state_with_action) for state_with_action, params in self._env.get_q_table_dict().items()]
            # Take the state with the highest reward
            best_state_with_action = max(acceptable_states_with_action, key=lambda x: x[0])[1]
            best_state = best_state_with_action[:-1]

            best_at = best_state[0] / at_dim._size
            best_art = best_state[1] / art_dim._size
            global_outputs_list.append([best_at, best_art])

        return global_inputs_list, global_outputs_list
    
    
    def pub_segment_the_main_samples(global_samples: list, sampling_rate: int, distribution_bar_threshold: float):
        import math
        import pywt
        from cwd_utils import pub_mean_min_max

        signal_segments_list = []
        # The segment should be at least 0.2 seconds long
        segment_init_len = math.ceil(0.2 * sampling_rate)
        # The segment could extend to the next and previous segments up to 0.1 seconds
        segment_extension = math.ceil(0.1 * sampling_rate)

        _, haar_dwt_lvl_1 = pywt.dwt(np.array(global_samples), 'haar') # take details only from the transform coefficients
        abs_haar_dwt_lvl_1 = [abs(x) for x in haar_dwt_lvl_1]
        dwt_down_scale = 2
        dwt_seg_init_len = int(segment_init_len // dwt_down_scale)
        dwt_seg_extension = int(segment_extension // dwt_down_scale)

        i_dwt_global = 0
        while i_dwt_global < len(abs_haar_dwt_lvl_1):
            dwt_pref_extension = 0
            dwt_suff_extension = 0
            dwt_segment_buffer = []
            # The segment should be up to 1 second
            for i_dwt_segment in [i for i in range(sampling_rate) if (i_dwt_global + i) < len(abs_haar_dwt_lvl_1)]:
                dwt_segment_buffer.append(abs_haar_dwt_lvl_1[i_dwt_global + i_dwt_segment])
                # The segment should be at least of length dwtSegmentInitialLen
                if len(dwt_segment_buffer) < dwt_seg_init_len:
                    continue
                # Compute the distribution of the derivative
                distribution, _, _ = pub_compute_distribution(dwt_segment_buffer, 10)
                # Check if the distribution of the derivative is not equiprobable
                segment_exceeded_limit = False
                for bar in distribution:
                    if bar >= distribution_bar_threshold:
                        # Remove the sample that caused the distortion on the distribution
                        dwt_segment_buffer.pop()
                        segment_exceeded_limit = True
                        break
                if segment_exceeded_limit:
                    break
            buff_end_index_before_extension = i_dwt_global + len(dwt_segment_buffer)

            # Extend the segment
            # The extension should be in the limits of the derivative's interval
            _, dwt_min, dwt_max = pub_mean_min_max(dwt_segment_buffer)
            dwt_max_pref_ext = dwt_seg_extension if i_dwt_global - dwt_seg_extension >= 0 else i_dwt_global
            dwt_max_suff_ext = dwt_seg_extension if (buff_end_index_before_extension + dwt_seg_extension) < len(abs_haar_dwt_lvl_1) else (len(abs_haar_dwt_lvl_1) - buff_end_index_before_extension) - 1

            for i in range(1, dwt_max_pref_ext + 1):
                val = abs_haar_dwt_lvl_1[i_dwt_global - i]
                if dwt_min <= val <= dwt_max:
                    dwt_pref_extension = i
                else:
                    break
            for i in range(1, dwt_max_suff_ext + 1):
                val = abs_haar_dwt_lvl_1[buff_end_index_before_extension + i]
                if dwt_min <= val <= dwt_max:
                    dwt_suff_extension = i
                else:
                    break
            
            # Include the new segment in signal_segments_list
            new_segment = SignalSegment(starting_index=(i_dwt_global - dwt_pref_extension) * dwt_down_scale,
                                        ending_index=(buff_end_index_before_extension + dwt_suff_extension + 1) * dwt_down_scale - 1) # Add the samples of the gape between two segments [ (.. + 1 * dwtDownScale) - 1 ]
            new_segment._segment_samples = global_samples[new_segment._starting_index : new_segment._ending_index + 1]
            new_segment._segment_mean, new_segment._segment_min, new_segment._segment_max = pub_mean_min_max(new_segment._segment_samples)

            signal_segments_list.append(new_segment)

            # Move i_dwt_global according to the new segment
            i_dwt_global = buff_end_index_before_extension

        return signal_segments_list
            