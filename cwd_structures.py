class CWDFramework:

    def __init__(self):
        from nn_structures import RLFramework
        from nn_structures import NNModel

        self._name: str

        self._cwd_rl_framework: RLFramework

        self._cwd_lstm_model: NNModel


class TrainingData:

    def __init__(self):
        self._input_sequences: list[list[float]] = []
        self._output_sequences: list[list[float]] = []


class LSTMTrainingData:

    def __init__(self, sampling_rate: int = -1):
        self._input_sequences: list[list[float]] = []
        self._output_sequences: list[list[float]] = []

        self._index_sequences: list[int] = []
        self._sampling_rate: int = sampling_rate


class SelSignalParams:

    def __init__(self, record_name="", signal_index=0, starting_second=0, ending_second=0):
        self._record_name = record_name
        self._signal_index = signal_index
        self._starting_second = starting_second
        self._ending_second = ending_second


time_error_tolerance = {
    "p_(": 10.2,
    "p": 75,
    "p_)": 12.7,
    "qrs_(": 6.5,
    "qrs": 75,
    "qrs_)": 11.6,
    "t_(": 75,
    "t": 75,
    "t_)": 30.6,
    "other": 0
}


selected_training_signals_params = {
    "sel30/0/27-57": SelSignalParams("sel30", 0, 27, 57),
    "sel32/0/22-52": SelSignalParams("sel32", 0, 22, 52),
    "sel42/1/0-25.2": SelSignalParams("sel42", 1, 0, 25.2),
    "sel43/1/0-23": SelSignalParams("sel43", 1, 0, 23),
    "sel47/0/0-30": SelSignalParams("sel47", 0, 0, 30),
    "sel48/0/0-30": SelSignalParams("sel48", 0, 0, 30),
    "sel52/0/0-30": SelSignalParams("sel52", 0, 0, 30),
    "sele0104/1/0-30": SelSignalParams("sele0104", 1, 0, 30),
    "sele0106/1/0-30": SelSignalParams("sele0106", 1, 0, 30),
    "sele0110/0/0-30": SelSignalParams("sele0110", 0, 0, 30),
    "sele0111/1/0-30": SelSignalParams("sele0111", 1, 0, 30),
    "sele0121/0/0-30": SelSignalParams("sele0121", 0, 0, 30),
    "sele0122/0/0-30": SelSignalParams("sele0122", 0, 0, 30),
    "sele0124/0/25-55": SelSignalParams("sele0124", 0, 25, 55),
    "sele0126/0/0-30": SelSignalParams("sele0126", 0, 0, 30),
    "sele0129/1/0-30": SelSignalParams("sele0129", 1, 0, 30),
    "sele0133/1/0-30": SelSignalParams("sele0133", 1, 0, 30),
    "sele0136/1/0-30": SelSignalParams("sele0136", 1, 0, 30),
    "sele0170/0/0-30": SelSignalParams("sele0170", 0, 0, 30),
    "sele0210/1/0-30": SelSignalParams("sele0210", 1, 0, 30),
    "sele0211/0/0-30": SelSignalParams("sele0211", 0, 0, 30),
    "sele0405/0/0-30": SelSignalParams("sele0405", 0, 0, 30),
    "sele0411/0/0-30": SelSignalParams("sele0411", 0, 0, 30),
    "sele0509/0/0-30": SelSignalParams("sele0509", 0, 0, 30),
    "sele0603/0/0-30": SelSignalParams("sele0603", 0, 0, 30),
    "sele0604/1/0-30": SelSignalParams("sele0604", 1, 0, 30),
    "sele0607/0/0-30": SelSignalParams("sele0607", 0, 0, 30),
    "sel100/0/10-40": SelSignalParams("sel100", 0, 10, 40),
    "sel103/0/0-30": SelSignalParams("sel103", 0, 0, 30),
    "sel114/0/0-30": SelSignalParams("sel114", 0, 0, 30),
    "sel116/0/60-90": SelSignalParams("sel116", 0, 60, 90),
    "sel117/0/0-30": SelSignalParams("sel117", 0, 0, 30),
    "sel123/0/0-30": SelSignalParams("sel123", 0, 0, 30),
    "sel14046/0/0-30": SelSignalParams("sel14046", 0, 0, 30),
    "sel15814/0/7-37": SelSignalParams("sel15814", 0, 7, 37),
    "sel16272/0/0-30": SelSignalParams("sel16272", 0, 0, 30),
    "sel16273/0/0-22": SelSignalParams("sel16273", 0, 0, 22),
    "sel16420/0/0-20.5": SelSignalParams("sel16420", 0, 0, 20.5),
    "sel16483/0/0-30": SelSignalParams("sel16483", 0, 0, 30),
    "sel16539/0/0-30": SelSignalParams("sel16539", 0, 0, 30),
    "sel16773/1/0-30": SelSignalParams("sel16773", 1, 0, 30),
    "sel16795/0/0-30": SelSignalParams("sel16795", 0, 0, 30),
    "sel17152/0/15-45": SelSignalParams("sel17152", 0, 15, 45),
    "sel301/0/0-30": SelSignalParams("sel301", 0, 0, 30),
    "sel306/0/0-30": SelSignalParams("sel306", 0, 0, 30),
    "sel307/0/0-30": SelSignalParams("sel307", 0, 0, 30),
    "sel308/0/0-30": SelSignalParams("sel308", 0, 0, 30),
    "sel310/0/32-62": SelSignalParams("sel310", 0, 32, 62),
    "sel811/0/0-30": SelSignalParams("sel811", 0, 0, 30),
    "sel840/0/4-34": SelSignalParams("sel840", 0, 4, 34),
    "sel847/0/0-30": SelSignalParams("sel847", 0, 0, 30),
    "sel872/0/0-30": SelSignalParams("sel872", 0, 0, 30),
    "sel873/0/21-51": SelSignalParams("sel873", 0, 21, 51)
}


selected_validation_signals_params = {
    "sel33/0/0-30": SelSignalParams("sel33", 0, 0, 30),
    "sele0114/1/0-30": SelSignalParams("sele0114", 1, 0, 30),
    "sele0166/1/0-30": SelSignalParams("sele0166", 1, 0, 30),
    "sele0406/0/0-30": SelSignalParams("sele0406", 0, 0, 30),
    "sele0606/0/0-30": SelSignalParams("sele0606", 0, 0, 30),
    "sele0609/0/0-30": SelSignalParams("sele0609", 0, 0, 30),
    "sele0612/0/0-30": SelSignalParams("sele0612", 0, 0, 30),
    "sele0704/0/6-36": SelSignalParams("sele0704", 0, 6, 36),
    "sel213/0/1-31": SelSignalParams("sel213", 0, 1, 31),
    "sel223/0/0-30": SelSignalParams("sel223", 0, 0, 30),
    "sel231/0/0-30": SelSignalParams("sel231", 0, 0, 30),
    "sel14172/1/0-30": SelSignalParams("sel14172", 1, 0, 30),
    "sel16786/0/0-30": SelSignalParams("sel16786", 0, 0, 30),
    "sel17453/1/0-30": SelSignalParams("sel17453", 1, 0, 30),
    "sel302/0/0-30": SelSignalParams("sel302", 0, 0, 30),
    "sel803/0/0-30": SelSignalParams("sel803", 0, 0, 30),
    "sel808/0/0-30": SelSignalParams("sel808", 0, 0, 30),
    "sel820/0/2-32": SelSignalParams("sel820", 0, 2, 32),
    "sel821/0/0-30": SelSignalParams("sel821", 0, 8, 38),
    "sel871/0/0-30": SelSignalParams("sel871", 0, 0, 30)
}
