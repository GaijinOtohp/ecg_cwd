from tensorflow.keras.models import Sequential


class OutputThresholdItem:

    def __init__(self):
        self._high_output_av: 1
        self._low_output_av: 0
        self._threshold: 0.5
        self._roc = dict()

        for threshold in [val / 100.0 for val in range(0, 100, 1)]:
            self._roc[threshold] = {"true_positives": 0, "false_positives": 0}


class ValidationMetricsItem:
    def __init__(self):
        self._true_positives = 0
        self._true_negatives = 0
        self._false_positives = 0
        self._false_negatives = 0

        self._class_deviation_tolerance = 0.0
        self._class_deviation_mean = 0.0
        self._class_deviation_std = 0.0

        self._mase = 0.0 # Mean Absolute Scaled Error
        self._mae = 0.0  # Mean Absolute Error
        self._mae_naive = 0.0  # Mean Absolute Error for Naive forcast
        self._i_samples = 0


class ValidationParams:

    def __init__(self, output_dim):
        self._data_size = 0
        self._training_size = 0
        
        self._validation_metrics = [ValidationMetricsItem() for _ in range(output_dim)]
        self._confusion_matrix = [[0 for _ in range(output_dim)] for _ in range(output_dim)]


class NNModel:

    def __init__(self, input_dim, output_dim, model_architecture_func, model_path = ""):
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._model_path = model_path
        self._validation_data = ValidationParams(output_dim)

        self._model: Sequential = model_architecture_func(input_dim, output_dim)

        self._output_thresholds = [OutputThresholdItem() for _ in range(output_dim)]

        self._last_training_elapsed_time_seconds = 0.0
        self._last_validation_elapsed_time_seconds = 0.0


class RLFramework:

    def __init__(self, input_dim, output_dim, model_architecture_func, model_path = "", dimensions_list = []):
        from reinforcement_learning import Dimension
        self._dimensions_list: list[Dimension] = dimensions_list

        self._exploitation_model = NNModel(input_dim, output_dim, model_architecture_func, model_path + "_exploitation")
        self._exploration_model = NNModel(input_dim, output_dim, model_architecture_func, model_path + "_exploration")

        self._last_exploration_elapsed_time_seconds = 0.0
