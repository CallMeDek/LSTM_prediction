from tensorflow.keras.layers import Dropout
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.python.keras.layers import Dense, LSTM
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.utils import multi_gpu_model
from tensorflow.python.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd


np.random.seed(0)


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
    
    
callbacks_list = [
        EarlyStopping(
            monitor='val_acc',
            patience = 5,
        ),
        ReduceLROnPlateau(
            monitor = 'val_acc',
            factor = 0.3,
            patience = 3,
        )
    ]   


original_train_val_data = pd.read_csv("./train.csv")
n_features = len(original_train_val_data.columns)-1
original_train_val_data = original_train_val_data[original_train_val_data.columns[1:]]
n_step = int(input("Step number?(Ex-3): "))
n_split_ratio = int(input("Split number(Bigger)(Ex-8)?: "))/10

original_test_data = pd.read_csv("./test.csv")
n_test_features = len(original_test_data.columns)-1
original_test_data = original_test_data[original_test_data.columns[1:]]

original_train_data_values = original_train_val_data.values[:int(original_train_val_data.shape[0]/24 * n_split_ratio*24)]
original_val_data_values = original_train_val_data.values[int(original_train_val_data.shape[0]/24 * n_split_ratio*24):]
original_test_data_values = original_test_data.values

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(original_train_data_values)

original_train_data_normalized = scaler.transform(original_train_data_values)
original_val_data_normalized = scaler.transform(original_val_data_values)
original_test_data_normalized = scaler.transform(original_test_data_values)

x_train, y_train = split_sequence(original_train_data_normalized, n_step)
x_val, y_val = split_sequence(original_val_data_normalized, n_step)
x_test, y_test = split_sequence(original_test_data_normalized, n_step)

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], n_features))
x_val = x_val.reshape((x_val.shape[0], x_val.shape[1], n_features))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], n_features))

a = x_train.shape[0]
b = x_val.shape[0]
bigger = a if a > b else b
gcd = 1
for i in range(1, int(bigger**0.5)+1):
    if (a%i == 0) and (b%i == 0) and (gcd < i):
        gcd = i

model = Sequential()
model.add(LSTM(64, input_shape=(n_step, n_features), return_sequences=True))
model.add(LSTM(128))
model.add(Dense(n_features))  
model.compile(optimizer='rmsprop', loss='mae', metrics=['acc'])
for epoch_idx in range(30):
    hist = model.fit(x_train, y_train, epochs=1, verbose=1, validation_data=(x_val, y_val), callbacks=callbacks_list, shuffle=False)
    model.reset_states()
    
no_temp_prediction_values_normalized = model.predict(x_test)
no_temp_prediction_values = scaler.inverse_transform(no_temp_prediction_values_normalized)

y_train = y_train[:, 0]
y_val = y_val[:, 0]        
        
model = Sequential()
model.add(LSTM(64, input_shape=(n_step, n_features), return_sequences=True))
model.add(LSTM(128))
model.add(Dense(n_features))
model.add(Dense(64))
model.add(Dense(256))
model.add(Dropout(0.21))
model.add(Dense(128))
model.add(Dense(256))
model.add(Dropout(0.21))
model.add(Dense(1))
model.compile(optimizer='rmsprop', loss='mae', metrics=['acc'])

for epoch_idx in range(30):
    hist = model.fit(x_train, y_train, epochs=1, verbose=1, validation_data=(x_val, y_val), callbacks=callbacks_list, shuffle=False)
    model.reset_states()
    
temp_prediction_values_normalized = model.predict(x_test)
temp_prediction_values = temp_prediction_values_normalized * (original_train_data_values[:, 0].max() - original_train_data_values[:, 0].min() + 1e-7) + original_train_data_values[:, 0].min()

with open('./24_test_result.csv', 'w') as file:
    file.write("PD(No_temp),temp,PD(temp)\n")
    for row_no_temp, row_temp in zip(no_temp_prediction_values, temp_prediction_values):
        file.write(f"{row_no_temp[0]},{row_no_temp[1]},{row_temp[0]}\n")