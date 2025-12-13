import pickle
from multiprocessing import Process, Manager
from time import time

from cwd_signals_selector import AnnoSignal, MyAnnotation
from cwd_structures import TrainingData
from cwd_utils import put_object_in_shared_memory, get_object_from_shared_memory


def pub_generate_rl_data(selected_anno_signals_dict: dict[str, AnnoSignal], rl_framework_shm_name: str, lock):
    # Create a shared dictionary to store the RL data
    # Get previously generated data
    shared_rl_data_dict: dict[str, TrainingData] = Manager().dict(pr_get_generated_rl_data())

    # Only signals with non-generated RL data are processed
    # Process each signal in a different process
    start_time = time()
    signal_processes = []
    for signal_key, anno_signal in selected_anno_signals_dict.items():
        if signal_key not in shared_rl_data_dict:
            # Get representative segments for long signals
            representative_anno_segments = pub_get_representative_signal_segments(anno_signal)

            # Create and start the process for generating rl data of the selected signal
            signal_process = Process(target=pr_generate_signal_rl_data, args=(representative_anno_segments, signal_key, shared_rl_data_dict, rl_framework_shm_name, lock))
            signal_processes.append(signal_process)
            signal_process.start()

    for process in signal_processes:
        process.join()
        
    end_time = time()
    #lock.acquire()
    rl_framework, rl_framework_shm = get_object_from_shared_memory(rl_framework_shm_name)
    rl_framework._last_exploration_elapsed_time_seconds = end_time - start_time
    put_object_in_shared_memory(rl_framework, rl_framework_shm)
    #lock.release()
    return dict(shared_rl_data_dict)


def pr_generate_signal_rl_data(representative_anno_segments: list[AnnoSignal], signal_key: str, shared_rl_data_dict: dict[str, TrainingData], rl_framework_shm_name: str, lock):
    # Create a shared memory for the training_data
    manager = Manager()
    training_data = TrainingData()
    training_data._input_sequences = manager.list()
    training_data._output_sequences = manager.list()

    # Process each segment in a different process
    segment_processes = []
    for anno_segmen in representative_anno_segments:
        segment_process = Process(target=pr_generate_segment_rl_data, args=(anno_segmen, training_data, rl_framework_shm_name, lock))
        segment_processes.append(segment_process)
        segment_process.start()
    
    for process in segment_processes:
        process.join()

    training_data._input_sequences = list(training_data._input_sequences)
    training_data._output_sequences = list(training_data._output_sequences)
    if len(training_data._input_sequences) == 0 or len(training_data._output_sequences) == 0:
        return

    # Save the generated RL data to the database
    from db_helper import DbHelper
    
    sql_command = "INSERT INTO rl_dataset (signal_key, training_data) VALUES (?, ?)"
    command_args = [signal_key, pickle.dumps(training_data, pickle.HIGHEST_PROTOCOL)]
    DbHelper.insert(sql_command, command_args)

    #lock.acquire()
    shared_rl_data_dict[signal_key] = training_data
    #lock.release()


def pr_generate_segment_rl_data(representative_anno_segment: AnnoSignal, training_data: str, rl_framework_shm_name: str, lock):
    # Generate RL data for this signal
    from cwd_rl_custom import CWDRLCustom

    # Load the RL framework and training data from shared memory
    #lock.acquire()
    rl_framework, _ = get_object_from_shared_memory(rl_framework_shm_name)
    cwd_rl = CWDRLCustom(rl_framework)
    #lock.release()

    global_inputs_list, global_outputs_list = cwd_rl.pub_deep_fit_rl_data(representative_anno_segment, rl_framework_shm_name, lock)

    # Set the training data to the shared memory
    #lock.acquire()
    training_data._input_sequences.extend(global_inputs_list)
    training_data._output_sequences.extend(global_outputs_list)
    #lock.release()
    

def pr_get_generated_rl_data():
    from db_helper import DbHelper

    sql_command = "SELECT signal_key, training_data FROM rl_dataset"
    rows = DbHelper.query(sql_command)
    
    rl_data_dict: dict[str, TrainingData] = dict()
    for row in rows:
        signal_key = row[0]
        training_data = pickle.loads(row[1])
        rl_data_dict[signal_key] = training_data

    return rl_data_dict


def pub_get_representative_signal_segments(anno_signal: AnnoSignal, region_length_seconds = 300, segment_length_seconds = 10):
    # Proposing that for each 5 min (300 seconds) of the signal should have a representative segment of 10 seconds
    import random

    region_length_samples = int(region_length_seconds * anno_signal._fs)

    segment_length_samples = segment_length_seconds * anno_signal._fs
    representative_anno_segments = []

    # Create the segments by taking random 10 seconds segments from each part of the original signal
    for region_start in range(0, len(anno_signal._signal), region_length_samples):
        # Get the possible limits of the segment
        region_end = min(region_start + region_length_samples, len(anno_signal._signal))
        start_index = random.randint(region_start, region_end - segment_length_samples)  # Ensure at least 10 seconds for segment
        end_index = start_index + segment_length_samples
        sel_segment = anno_signal._signal[start_index : end_index]

        # Translate and select the annotation of the selected segment
        sel_anno = [MyAnnotation(anno._symbol, anno._index - start_index) for anno in anno_signal._annotation
                    if start_index <= anno._index and anno._index < end_index]

        representative_anno_segments.append(AnnoSignal(sel_segment, anno_signal._fs, sel_anno))

    return representative_anno_segments
