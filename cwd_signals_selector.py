from pathlib import Path
import wfdb

from cwd_structures import SelSignalParams


class MyAnnotation:

    def __init__(self, symbol: str, index: int):
        self._symbol = symbol
        self._index = index


class AnnoSignal:
    def __init__(self, signal: list[float], fs: int, annotation: list[MyAnnotation]):
        self._signal = signal
        self._fs = fs
        self._annotation = annotation


def pr_convert_annotation(index: int, symbol: str, num: int):
    if symbol in ["(", ")"]:
        if num == 0:
            return MyAnnotation("p_" + symbol, index)
        elif num == 1:
            return MyAnnotation("qrs_" + symbol, index)
        elif num == 2:
            return MyAnnotation("t_" + symbol, index)
    elif symbol not in ["p", "t"]:
        return MyAnnotation("qrs", index)
    else:
        return MyAnnotation(symbol, index)
    

def pub_get_selected_signals(path: str, selected_signals_params_dict: dict[str, SelSignalParams]):
    files = list(Path(path).rglob(f'*{".dat"}'))

    selected_files_dict = {signal_key: file for signal_key, params in selected_signals_params_dict.items() for file in files if file.stem == params._record_name}

    selected_anno_signals_dict: dict[str, AnnoSignal] = dict()
    
    for signal_key, file in selected_files_dict.items():
        record = wfdb.rdsamp(file.parent / file.stem)
        sel_signal_params = selected_signals_params_dict[signal_key]

        starting_index = int(sel_signal_params._starting_second * record[1]["fs"])
        ending_index = int(sel_signal_params._ending_second * record[1]["fs"])

        signal = record[0][starting_index: ending_index, sel_signal_params._signal_index]
        annotation = wfdb.rdann(str(file.parent / file.stem), 'pu' + str(sel_signal_params._signal_index))

        my_annotation = [pr_convert_annotation(sam - starting_index, sym, num) for (sam, sym, num) in zip(annotation.sample, annotation.symbol, annotation.num)
                         if starting_index < sam < ending_index]

        selected_anno_signals_dict[signal_key] = AnnoSignal(signal, record[1]["fs"], my_annotation)

    return selected_anno_signals_dict