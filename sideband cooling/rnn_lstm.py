# Recusive Neural Network for extrapolating optimal pulse times
# For improved computational speed and extrapolation beyond computatable datasets

# See TS's guide on RNNs
# https://www.tensorflow.org/guide/keras/rnn

import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, Embedding

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from tqdm import tqdm

from sideband_cooling import simulation

# Predict optimal pulse time given eta, pi/omega0, initial nbar, and previous optimized pulse times

# %% Generate dataset
N_rand = 100

omegas = 2*np.pi * np.linspace(1/(2 * 5e-6), 1/(2 * 20e-6), 50)
omega_rands = np.random.choice(omegas, N_rand)

# Assume eta and initial nbar are computable from confining harmonic frequency omega_z
omega_zs = 2*np.pi * np.linspace(0.5, 3, 50) * 1e6
omega_z_rands = np.random.choice(omega_zs, N_rand)

lam = 355e-9
hbar = 1.05457e-34
m = 170.936 * 1.6605e-27
eta_rands = np.sqrt(2) * 2*np.pi/lam * np.sqrt(hbar/2/m/omega_z_rands)

initial_nbar_rands = 19.6e6/2/(omega_z_rands/2/np.pi)

N_pulses_rands = np.random.choice(np.arange(5, 40), N_rand)

strategy = 'optimal'
cooled_nbars = []
pulse_schedules = []
for omega_z, initial_nbar, omega, N_pulses in tqdm(zip(omega_z_rands, initial_nbar_rands, omega_rands, N_pulses_rands), total=len(omega_z_rands)):
    sim = simulation.SidebandCooling(omega_z, initial_nbar, omega)
    cooled_nbar, pulse_schedule, cooling_time, cooled_dist = sim.cool(strategy, N_pulses, 1)
    cooled_nbars.append(cooled_nbar)
    pulse_schedules.append(pulse_schedule[0])
    sim.reset_current_dist()

# Simulate and save data
pi_times = np.pi / omega_rands
dataX = np.column_stack((eta_rands, pi_times, initial_nbar_rands, N_pulses_rands))
dataY = np.array(pulse_schedules)

# Save data into files
with open('data/dataX_'+strategy+'_nbar.npy', 'wb') as f:
    np.save(f, dataX)
with open('data/dataY_'+strategy+'_nbar.npy', 'wb') as f:
    np.save(f, dataY)

# %% Load and preprocess data

strategy = 'optimal'
# load data
with open('data/dataX_'+strategy+'_nbar.npy', 'rb') as f:
    dataX = np.load(f)
with open('data/dataY_'+strategy+'_nbar.npy', 'rb') as f:
    dataY = np.load(f)

# %% Group pulse times into batches of 10 with the "label" of batch being the pulse just after the batch
batch_size = 10
pulse_times = dataX[:,1]
features = []
labels = []
for i in range(batch_size, len(pulse_times)):
    features.append(pulse_times[i - batch_size: i])
    labels.append(pulse_times[i])
    
ptimesX = np.array(features)
ptimesY = np.array(labels)

ind = 10
plt.plot(np.arange(batch_size), ptimesX[ind], label='features')
plt.scatter(batch_size, ptimesY[ind], label='label')
plt.grid()
plt.legend()
plt.show()

# %%


# Preprocess data
X_train, X_test, y_train, y_test = train_test_split(ptimesX, ptimesY, test_size=0.33, random_state=42)

scalerX = preprocessing.StandardScaler().fit(X_train)
scalery = preprocessing.StandardScaler().fit(y_train.reshape(-1,1))
X_train = scalerX.transform(X_train)
y_train = (scalery.transform(y_train.reshape(-1,1))).reshape(-1,)
X_test = scalerX.transform(X_test)
y_test = (scalery.transform(y_test.reshape(-1,1))).reshape(-1,)

# %%

model = Sequential()

# Embedding layer
model.add(
    Embedding(input_dim=1000,
              input_length=10,
              output_dim=100,
              trainable=False,
              mask_zero=True))

# Masking layer for pre-trained embeddings
model.add(Masking(mask_value=0.0))

# Recurrent layer
model.add(LSTM(64, return_sequences=False, 
               dropout=0.1, recurrent_dropout=0.1))

# Fully connected layer
model.add(Dense(64, activation='relu'))

# Dropout for regularization
model.add(Dropout(0.5))

# Output layer
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(
    optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
# %%

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Create callbacks
callbacks = [EarlyStopping(monitor='val_loss', patience=5), 
             ModelCheckpoint('../models/model.h5', save_best_only=True, save_weights_only=False)]

# %%

history = model.fit(X_train,  y_train, 
                    batch_size=10, epochs=150,
                    callbacks=callbacks,
                    verbose=0,
                    validation_data=(X_test, y_test))