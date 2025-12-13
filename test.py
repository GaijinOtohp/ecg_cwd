from multiprocessing import Manager, shared_memory
import pickle
import numpy as np
from time import time

from cwd_signals_selector import AnnoSignal, pub_get_selected_signals
from cwd_structures import CWDFramework, TrainingData, LSTMTrainingData, SelSignalParams, selected_training_signals_params, time_error_tolerance, selected_validation_signals_params
from cwd_nn import pub_create_cwd_framework, pub_fit, pub_initialize_cwd_framework
from cwd_rl_data_generator import pub_generate_rl_data
from db_helper import DbHelper
from cwd_lstm_data_generator import pub_generate_lstm_data
from cwd_validation_tools import roc_thresholds_tune, validate_lstm_model
from cwd_utils import segment_inputs_and_outputs_redundantly, sort_dataset_sequences_to_flat_batches, extract_validation_sub_metrics


class Test:

    def __init__(self):
        # Get a CWD framework with the name "test_framework" from the database if exists, or create a new one
        model_id, cwd_framework = self.get_cwd_framework("test_framework_1")

        # Traing the model
        self.train_cwd_framework(model_id, cwd_framework, selected_training_signals_params)

        # Validate
        self.validate_cwd_framework(model_id, cwd_framework, selected_validation_signals_params)

        # Print validation result
        for class_label, validation_metrics_item in zip(time_error_tolerance.keys(), cwd_framework._cwd_lstm_model._validation_data._validation_metrics):
            print(f"{class_label}:")
            extract_validation_sub_metrics(validation_metrics_item)


    def get_cwd_framework(self, framework_name: str):
        print(f'Getting CWD framework with the name {framework_name} from database...')
        model_id, cwd_framework = pub_initialize_cwd_framework(framework_name)
        if model_id == -1:
            print(f'No CWD framework exist with the name {framework_name}.\nCreating a new one...')
            model_id, cwd_framework = pub_create_cwd_framework(framework_name)
        print("CWD framework obtained.")
        return model_id, cwd_framework
    

    def train_cwd_framework(self, model_id: int, cwd_framework: CWDFramework, selected_training_signals_params: dict[str, SelSignalParams]):
        print("Getting selected training annotated signals...")
        selected_training_anno_signals_dict: dict[str, AnnoSignal] = pub_get_selected_signals("./", selected_training_signals_params)
        print("Selected annotated training signals obtained.")

        #__________________________________________________________________________________________________________#
        print("Getting RL training data...")
        # Since multiprocessing cannot share objects, we need to use a shared memory for the rl_data_dict and rl_framework
        manager = Manager()
        # Create a shared memory for the RL framework
        binary_rl_framework = pickle.dumps(cwd_framework._cwd_rl_framework, pickle.HIGHEST_PROTOCOL)
        rl_framework_shm = shared_memory.SharedMemory(create=True, size=len(binary_rl_framework))
        rl_framework_shm.buf[:len(binary_rl_framework)] = binary_rl_framework
        rl_framework_shm = shared_memory.SharedMemory(name=rl_framework_shm.name)
        # Create a lock for the shared memory
        lock = manager.Lock()
        rl_training_data_dict: dict[str, TrainingData] = pub_generate_rl_data(selected_training_anno_signals_dict, rl_framework_shm.name, lock)
        cwd_framework._cwd_rl_framework = pickle.loads(rl_framework_shm.buf.tobytes())
        print("RL training data obtained.")

        print("Training exploitation RL model...")
        rl_training_inputs = [input for training_data in rl_training_data_dict.values() for input in training_data._input_sequences]
        rl_training_outputs = [output for training_data in rl_training_data_dict.values() for output in training_data._output_sequences]
        
        pub_fit(cwd_framework._cwd_rl_framework._exploitation_model, rl_training_inputs, rl_training_outputs)
        print("Exploitation RL model trained.")

        print("Getting LSTM training data...")
        lstm_training_data_dict = None
        sql_command = "SELECT training_data_seqs FROM lstm_dataset WHERE framework_name = ?"
        command_args = [cwd_framework._name]
        rows = DbHelper.query(sql_command, command_args)
        for row in rows:
            lstm_training_data_dict = pickle.loads(row[0])
        if lstm_training_data_dict is None:
            lstm_training_data_dict: dict[str, LSTMTrainingData] = pub_generate_lstm_data(selected_training_anno_signals_dict, cwd_framework._cwd_rl_framework)
            print("Saving LSTM training data to the database...")
            sql_command = "INSERT INTO lstm_dataset (framework_name, training_data_seqs) VALUES (?, ?)"
            command_args = [cwd_framework._name, pickle.dumps(lstm_training_data_dict, pickle.HIGHEST_PROTOCOL)]
            DbHelper.insert(sql_command, command_args)
            print("LSTM training data saved to the database.")
        print("LSTM training data obtained.")

        print("Training LSTM model...")
        # One timestep for each 106 features input and each 10 classification output
        lstm_flat_all_sequence_inputs = [input for training_data in lstm_training_data_dict.values() for input in training_data._input_sequences]
        lstm_flat_all_sequence_outputs = [output for training_data in lstm_training_data_dict.values() for output in training_data._output_sequences]
        lstm_all_segment_timestep_inputs, lstm_all_segment_timestep_outputs = segment_inputs_and_outputs_redundantly(lstm_flat_all_sequence_inputs, lstm_flat_all_sequence_outputs, 2)

        # The following could be used for batching the samples across sequences instead of moving the batch window on all sequences in one line
        '''
        lstm_sequences_segment_timestep_inputs = []
        lstm_sequences_segment_timestep_outputs = []
        for training_data in lstm_training_data_dict.values():
            sequence_segment_timestep_inputs, sequence_segment_timestep_outputs = segment_inputs_and_outputs_redundantly(training_data._input_sequences, training_data._output_sequences, 2)
            lstm_sequences_segment_timestep_inputs.append(sequence_segment_timestep_inputs)
            lstm_sequences_segment_timestep_outputs.append(sequence_segment_timestep_outputs)

        lstm_training_inputs, lstm_training_outputs = sort_dataset_sequences_to_flat_batches(lstm_sequences_segment_timestep_inputs, lstm_sequences_segment_timestep_outputs, batch_size=4)
        '''

        pub_fit(cwd_framework._cwd_lstm_model, lstm_all_segment_timestep_inputs, lstm_all_segment_timestep_outputs, epochs=200, batch_size=4)
        print("LSTM model trained.")

        #__________________________________________________________________________________________________________#
        print("Tuning thresholds using ROC...")
        roc_thresholds_tune(cwd_framework._cwd_lstm_model, lstm_training_data_dict, list(time_error_tolerance.values()))
        print("Tuned thresholds using ROC.")

        #__________________________________________________________________________________________________________#
        print("Saving the trained CWD framework...")
        sql_command = "UPDATE cwd_frameworks SET framework_object = ? WHERE _id = ?"
        command_args = [pickle.dumps(cwd_framework, pickle.HIGHEST_PROTOCOL), model_id] # use pickle for serializing the model object
        model_id = DbHelper.update(sql_command, command_args)
        print("Trained CWD framework saved.")


    def validate_cwd_framework(self, model_id: int, cwd_framework: CWDFramework, selected_validating_signals_params: dict[str, SelSignalParams]):
        start_time = time()
        print("Getting selected validation annotated signals...")
        selected_validating_anno_signals_dict: dict[str, AnnoSignal] = pub_get_selected_signals("./", selected_validating_signals_params)
        print("Selected annotated validation signals obtained.")
        
        #__________________________________________________________________________________________________________#
        print("Getting LSTM validation data...")
        lstm_validation_data_dict: dict[str, LSTMTrainingData] = pub_generate_lstm_data(selected_validating_anno_signals_dict, cwd_framework._cwd_rl_framework)
        print("LSTM validation data obtained.")

        #__________________________________________________________________________________________________________#
        print("Getting prediction for all signals...")
        lstm_predictions_dict: dict[str, list[list[np.array]]] = dict()

        for signal_key, lstm_training_data in lstm_validation_data_dict.items():
            X_test, _ = segment_inputs_and_outputs_redundantly(lstm_training_data._input_sequences, lstm_training_data._output_sequences, 2)

            y_pred_probs = cwd_framework._cwd_lstm_model._model.predict(X_test, verbose=0)

            lstm_predictions_dict[signal_key] = [y_pred_prob[0] for y_pred_prob in y_pred_probs]
        print("Prediction for all signals obtained.")
        
        #__________________________________________________________________________________________________________#
        print("Validating the framework...")
        thresholds_list = [threshold_item._threshold for threshold_item in cwd_framework._cwd_lstm_model._output_thresholds]
        validation_metrics, confusion_matrix = validate_lstm_model(lstm_validation_data_dict, lstm_predictions_dict, thresholds_list, list(time_error_tolerance.values()))
        cwd_framework._cwd_lstm_model._validation_data._validation_metrics = validation_metrics
        cwd_framework._cwd_lstm_model._validation_data._confusion_matrix = confusion_matrix
        print("Validation complete.")
        end_time = time()
        cwd_framework._cwd_lstm_model._last_validation_elapsed_time_seconds = end_time - start_time
        training_size = cwd_framework._cwd_lstm_model._validation_data._training_size
        validation_size = len([input for training_data in lstm_validation_data_dict.values() for input in training_data._input_sequences])
        cwd_framework._cwd_lstm_model._validation_data._data_size = training_size + validation_size

        #__________________________________________________________________________________________________________#
        print("Saving the validated CWD framework...")
        sql_command = "UPDATE cwd_frameworks SET framework_object = ? WHERE _id = ?"
        command_args = [pickle.dumps(cwd_framework, pickle.HIGHEST_PROTOCOL), model_id] # use pickle for serializing the model object
        model_id = DbHelper.update(sql_command, command_args)
        print("Validated CWD framework saved.")


if __name__ == "__main__":
    test = Test()