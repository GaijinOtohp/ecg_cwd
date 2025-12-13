import numpy as np
import statistics

from cwd_structures import LSTMTrainingData
from nn_structures import NNModel, ValidationMetricsItem
from cwd_utils import segment_inputs_and_outputs_redundantly

class RefFloat:
    def __init__(self, init_val: float):
        self._value: float = init_val


def get_nearby_positive_predicted_peaks(i_out: int, time_tol: list[float], sampling_rate: int, i_peak: int, peak_time_index: int, predi_output_index_pair_list: list[(list[RefFloat], int)], thresholds_list: list[float]):
    nearby_positive_peaks: list[(list[RefFloat], float)] = []

    tolerance = time_tol[i_out] / 1000.0 * sampling_rate # dividing by 1000d to convert the deviation from milliseconds to seconds
    for i_pred in range(i_peak, -1, -1):
        if abs(predi_output_index_pair_list[i_pred][1] - peak_time_index) <= tolerance:
            if predi_output_index_pair_list[i_pred][0][i_out]._value >= thresholds_list[i_out]:
                time_error_millis = (predi_output_index_pair_list[i_pred][1] - peak_time_index) / sampling_rate * 1000.0 # multiplying by 1000d to convert the deviation from seconds to milliseconds
                nearby_positive_peaks.append((predi_output_index_pair_list[i_pred][0], time_error_millis))
        else:
            break
    for i_pred in range(i_peak + 1, len(predi_output_index_pair_list)):
        if abs(predi_output_index_pair_list[i_pred][1] - peak_time_index) <= tolerance:
            if predi_output_index_pair_list[i_pred][0][i_out]._value >= thresholds_list[i_out]:
                time_error_millis = (predi_output_index_pair_list[i_pred][1] - peak_time_index) / sampling_rate * 1000.0 # multiplying by 1000d to convert the deviation from seconds to milliseconds
                nearby_positive_peaks.append((predi_output_index_pair_list[i_pred][0], time_error_millis))
        else:
            break

    return nearby_positive_peaks


def validate_lstm_model(lstm_training_data_dict: dict[str, LSTMTrainingData], lstm_predictions_dict: dict[str, list[list[np.array]]], thresholds_list: list[float], time_tol: list[float]):
    validation_metrics = [ValidationMetricsItem() for _ in range(len(thresholds_list))]
    confusion_matrix = [[0 for _ in range(len(thresholds_list))] for _ in range(len(thresholds_list))]

    prediction_time_error: list[list[float]] = [[] for _ in range(len(thresholds_list))]

    # The following is used for rectifying predicted outputs
    latest_classified_ref_output: list[RefFloat] = []
    predi_output_index_pair_list: list[(list[RefFloat], int)]= []

    for signal_key, lstm_training_data in lstm_training_data_dict.items():
        # Rectify predicted outputs---------------------------------------------------------------------------------------------------------

        # Pair the outputs as reference with their indexes
        predi_output_index_pair_list = [([RefFloat(out_val) for out_val in predi_output], peak_time_index) for predi_output, peak_time_index in zip(lstm_predictions_dict[signal_key], lstm_training_data._index_sequences)]

        latest_classified_ref_output = predi_output_index_pair_list[0][0][:]

        for (predicted_ref_output, _) in predi_output_index_pair_list:
            #for predi_ref_out_val, latest_clas_ref_out_val, out_val_threshold in zip(predicted_ref_output, latest_classified_ref_output, thresholds_list):
            for i_out in range(len(thresholds_list)):
                if predicted_ref_output[i_out]._value >= thresholds_list[i_out]:
                    if predicted_ref_output[i_out]._value >= latest_classified_ref_output[i_out]._value:
                        latest_classified_ref_output[i_out]._value = 0
                        latest_classified_ref_output[i_out] = predicted_ref_output[i_out]
                    else:
                        predicted_ref_output[i_out]._value = 0
                else:
                    latest_classified_ref_output[i_out] = RefFloat(0)

        # Set the validation measurements of the prediction for the current sequence--------------------------------------------------------
        for i_peak, (predicted_ref_output, peak_time_index) in enumerate(predi_output_index_pair_list):
            true_output = lstm_training_data._output_sequences[i_peak]
            for i_out in range(len(thresholds_list)):
                # Check if the output should be positive or negative
                if true_output[i_out] == 1:
                    # Get the nearby positive predicted peaks for the current output class
                    nearby_positive_peaks = get_nearby_positive_predicted_peaks(i_out, time_tol, lstm_training_data._sampling_rate, i_peak, peak_time_index, predi_output_index_pair_list, thresholds_list)

                    # Include the deviations of the class
                    prediction_time_error[i_out].extend([time_error for (predi_positive_output, time_error) in nearby_positive_peaks])

                    # Check if there is any positive forcasted output
                    if len(nearby_positive_peaks) > 0:
                        # True Positive
                        validation_metrics[i_out]._true_positives += 1
                    else:
                        validation_metrics[i_out]._false_negatives += 1
                else:
                    if predicted_ref_output[i_out]._value < thresholds_list[i_out]:
                        validation_metrics[i_out]._true_negatives += 1
                    else:
                        validation_metrics[i_out]._false_positives += 1

        # Set the confusion matrix----------------------------------------------------------------------------------------------------------
        for i_peak, peak_time_index in enumerate(lstm_training_data._index_sequences):
            true_output = lstm_training_data._output_sequences[i_peak]
            for col in range(len(thresholds_list)):
                if true_output[col] == 1:
                    for row in range(len(thresholds_list)):
                        nearby_positive_peaks = get_nearby_positive_predicted_peaks(row, time_tol, lstm_training_data._sampling_rate, i_peak, peak_time_index, predi_output_index_pair_list, thresholds_list)

                        if len(nearby_positive_peaks) > 0:
                            confusion_matrix[col][row] += 1

    # Compute te deviation mean and standard deviation of the predictions-------------------------------------------------------------------
    for i_out in range(len(thresholds_list)):
        validation_metrics[i_out]._class_deviation_tolerance = time_tol[i_out]
        validation_metrics[i_out]._class_deviation_mean = statistics.mean(prediction_time_error[i_out]) if len(prediction_time_error[i_out]) > 0 else float('inf')
        validation_metrics[i_out]._class_deviation_std = statistics.stdev(prediction_time_error[i_out]) if len(prediction_time_error[i_out]) > 1 else float('inf')

    return validation_metrics, confusion_matrix


def roc_thresholds_tune(model_framework: NNModel, lstm_training_data_dict: dict[str, LSTMTrainingData], time_tol: list[float]):
    # Get the predictions for all signals--------------------------------------------------------------------------------
    lstm_predictions_dict: dict[str, list[list[np.array]]] = dict()

    for signal_key, lstm_training_data in lstm_training_data_dict.items():
        X_test, _ = segment_inputs_and_outputs_redundantly(lstm_training_data._input_sequences, lstm_training_data._output_sequences, 2)

        y_pred_probs = model_framework._model.predict(X_test, verbose=0)

        lstm_predictions_dict[signal_key] = [y_pred_prob[0] for y_pred_prob in y_pred_probs]
    
    # Compute ROC---------------------------------------------------------------------------------------------------------
    # validate the model for each threshold
    for threshold in [val / 100.0 for val in range(0, 100, 1)]:
        # Set the new output thresholds
        for output_threshold_item in model_framework._output_thresholds:
            output_threshold_item._threshold = threshold
        # Validate the model using the new thresholds
        thresholds_list = [threshold for _ in range(len(model_framework._output_thresholds))]
        validation_metrics, _ = validate_lstm_model(lstm_training_data_dict, lstm_predictions_dict, thresholds_list, time_tol)
        # Set the ROC values for the current threshold
        for i_out in range(len(thresholds_list)):
            model_framework._output_thresholds[i_out]._roc[threshold]["true_positives"] = validation_metrics[i_out]._true_positives
            model_framework._output_thresholds[i_out]._roc[threshold]["false_positives"] = validation_metrics[i_out]._false_positives

    # Set the best thresholds based on the ROC values------------------------------------------------
    for i_out in range(len(model_framework._output_thresholds)):
        model_framework._output_thresholds[i_out]._threshold = max(model_framework._output_thresholds[i_out]._roc.keys(),
                                                                    key=lambda thr: model_framework._output_thresholds[i_out]._roc[thr]["true_positives"] - model_framework._output_thresholds[i_out]._roc[thr]["false_positives"])
