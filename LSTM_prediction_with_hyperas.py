from hyperopt import Trials, STATUS_OK, tpe
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.python.keras.layers import Dense, LSTM
from tensorflow.python.keras.models import Sequential

import numpy as np
import pandas as pd

from hyperas import optim
from hyperas.distributions import choice, uniform

numer_step = 0

def data():
    def split_sequence(sequence, n_steps):
        X, y = list(), list()
        for i in range(len(sequence)):
            # find the end of this pattern
            end_ix = i + n_steps
            # check if we are beyond the sequence
            if end_ix > len(sequence)-1:
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)
    
    original_data = pd.read_csv("./train.csv")
    n_features = len(original_data.columns)-1
    original_data = original_data[original_data.columns[1:]]
    n_step = int(input("Step number?(Ex-3): "))
    numer_step = n_step
    n_split_ratio = int(input("Split number(Bigger)(Ex-8)?: "))/10

    original_train_data = original_data.values[:int(original_data.shape[0]/24 * n_split_ratio*24)]
    original_val_data = original_data.values[int(original_data.shape[0]/24 * n_split_ratio*24):]

    original_train_data = (original_train_data - original_train_data.min(axis=0)) / \
                            (original_train_data.max(axis=0) - original_train_data.min(axis=0) + 1e-7)
    original_val_data = (original_val_data - original_val_data.min(axis=0)) / \
                            (original_val_data.max(axis=0) - original_val_data.min(axis=0) + 1e-7)

    x_train, y_train = split_sequence(original_train_data, n_step)
    x_val, y_val = split_sequence(original_val_data, n_step)

    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], n_features))
    x_val = x_val.reshape((x_val.shape[0], x_val.shape[1], n_features))
    
    return x_train, y_train, x_val, y_val, n_step, n_features
    
    
def create_model(x_train, y_train, x_val, y_val, n_step, n_features):
    callbacks_list = [
        EarlyStopping(
            monitor='val_loss',
            patience = 5,
        ),
        ReduceLROnPlateau(
            monitor = 'val_loss',
            factor = 0.3,
            patience = 2,
        )
    ]
    
    a = x_train.shape[0]
    b = x_val.shape[0]
    bigger = a if a > b else b
    gcd = 1
    for i in range(1, int(bigger**0.5)+1):
        if (a%i == 0) and (b%i == 0) and (gcd < i):
            gcd = i
            
    model = Sequential()
    model.add(LSTM({{choice([64, 128, 256])}}, return_sequences=True, batch_input_shape=(gcd, n_step, n_features), stateful=True))
    #if {{choice(['add', 'no'])}} == 'add':
    #    model.add(LSTM({{choice([64, 128, 256])}}, stateful=True, return_sequences=True))
    model.add(Dense(n_features))
    model.compile(optimizer='adam', loss='mae', metrics=['acc'])
    
    for epoch_idx in range(200):
        hist = model.fit(x_train, y_train, epochs=1, validation_data=(x_val, y_val), callbacks=callbacks_list, shuffle=False, verbose=0, batch_size=gcd)
        model.reset_states()
    
    return {'loss': np.amin(hist.history['val_loss']), 'status': STATUS_OK, 'model': model}
    
    
def prediction(model):
    def split_sequence(sequence, n_steps):
        X, y = list(), list()
        for i in range(len(sequence)):
            # find the end of this pattern
            end_ix = i + n_steps
            # check if we are beyond the sequence
            if end_ix > len(sequence)-1:
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)
    
    original_data = pd.read_csv("./test.csv")
    n_features = len(original_data.columns)-1
    original_data = original_data[original_data.columns[1:]]

    original_test_data = original_data.values
    original_test_data = (original_test_data - original_test_data.min(axis=0)) / \
                            (original_test_data.max(axis=0) - original_test_data.min(axis=0) + 1e-7)

    x_test, y_test = split_sequence(original_test_data, number_step)
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], n_features))
    
    n_samples = x_test.shape[0]
    gcd = 1
    for i in range(1, int(n_samples**0.5)+1):
        if (n_samples % i == 0) and (gcd < i):
            gcd = i
            
    prediction_values = model.predict(x_test, batch_size=gcd)
    prediction_values = prediction_values*(x_test.max(axis=0) - x_test.min(axis=0) + 1e-7) \
                            + x_test.min(axis=0)
    print(prediction_values)
    
    
if __name__ == '__main__':
    import gc; gc.collect()
    best_run, best_model = optim.minimize(model=create_model, data=data, algo=tpe.suggest, max_evals=10, trials=Trials(), keep_temp=True)
    prediction(best_model)
    print(best_run)
    