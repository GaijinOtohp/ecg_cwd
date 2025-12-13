from time import time
import numpy as np
import pickle
import tensorflow as tf

from tensorflow.keras.layers import Dense, LSTM, Bidirectional
from tensorflow.keras.models import Sequential

from nn_structures import NNModel
from db_helper import DbHelper
from cwd_structures import CWDFramework


def pub_fit(model_framework: NNModel, training_inputs: list[list[float]], training_outputs: list[list[float]], batch_size=4, epochs=1000):
    # Create early stopping parameter
    callback = tf.keras.callbacks.EarlyStopping(
                                                    monitor='loss',
                                                    min_delta=0.0001,
                                                    patience=15,
                                                    verbose=0,
                                                    mode='auto',
                                                    baseline=None,
                                                    restore_best_weights=False,
                                                    start_from_epoch=0
                                                )
    # Train the model
    training_inputs = np.array(training_inputs)
    training_outputs = np.array(training_outputs)
    start_time = time()
    model_framework._model.fit(training_inputs, training_outputs, batch_size=batch_size, epochs=epochs, callbacks=[callback], verbose=1)
    end_time = time()
    model_framework._last_training_elapsed_time_seconds = end_time - start_time
    model_framework._validation_data._training_size = len(training_inputs)


def pub_create_cwd_framework(name: str):
    from cwd_structures import time_error_tolerance

    cwd_framework = CWDFramework()

    cwd_framework._name = name
    cwd_framework._cwd_rl_framework = pr_create_cwd_rl_framework(model_path="./ai_models/" + name + "/")
    cwd_framework._cwd_lstm_model = NNModel(input_dim=106, output_dim=10, model_architecture_func=pr_create_cwd_lstm_model,
                                                model_path="./ai_models/" + name + "/lstm_model")
    
    # Set the prediction tolerance
    for output_index, tolerance in zip(range(10), time_error_tolerance.values()):
        cwd_framework._cwd_lstm_model._validation_data._validation_metrics[output_index]._class_deviation_tolerance = tolerance
    
    # Save the CWD framework
    sql_command = "INSERT INTO cwd_frameworks (framework_name, framework_object) VALUES (?, ?)"
    command_args = [name, pickle.dumps(cwd_framework, pickle.HIGHEST_PROTOCOL)] # use pickle for serializing the model object
    model_id = DbHelper.insert(sql_command, command_args)

    return model_id, cwd_framework


def pub_initialize_cwd_framework(framework_name: str):
    sql_command = "SELECT _id, framework_object FROM cwd_frameworks WHERE framework_name = ?"
    command_args = [framework_name]
    rows = DbHelper.query(sql_command, command_args)

    framework_id = -1
    cwd_framework: CWDFramework = None
    for row in rows:
        framework_id = row[0]
        cwd_framework = pickle.loads(row[1])

    return framework_id, cwd_framework


def pr_create_cwd_rl_framework(model_path: str):
    from reinforcement_learning import Dimension
    from nn_structures import RLFramework

    dimensions_list: list[Dimension] = []
    dimensions_list.append(Dimension(name="AT", size=30, min=1, max=25))
    dimensions_list.append(Dimension(name="ART", size=60, min=0, max=0.3))

    CWD_RL_framework = RLFramework(input_dim=10, output_dim=2, model_architecture_func=pr_create_cwd_rl_model,
                                    model_path=model_path + "rl_frmwrk", dimensions_list=dimensions_list)

    return CWD_RL_framework


def pr_create_cwd_rl_model(input_dim, output_dim):
    hidden_dim = int(input_dim * 2 / 3 + output_dim)

    model = Sequential([
        Dense(units=hidden_dim, activation='elu', input_shape=(input_dim,)), # The input layer connected to the 1st hidden layer. The 1st layer has "ELU" as activation function
        Dense(units=output_dim) # The 2nd hidden layer connected to the output layer. The output layer has no activation function
    ])

    model.compile(optimizer='sgd',
                    loss='mae')

    return model


def pr_create_cwd_lstm_model(input_dim: int, output_dim: int):
    hidden_dim = int(input_dim * 2 / 3 + output_dim)

    model = Sequential()
    model.add(Bidirectional(LSTM(units=hidden_dim, activation='tanh', return_sequences=True, input_shape=(2, input_dim)), merge_mode='concat'))
    model.add(Dense(units=output_dim, activation='linear'))

    model.compile(optimizer='adam',
                    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))

    return model