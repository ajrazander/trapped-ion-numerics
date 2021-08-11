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
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
from tqdm import tqdm

from sideband_cooling import simulation

# Predict optimal pulse time given eta, pi/omega0, initial nbar, and previous optimized pulse times

# %% Generate dataset
N_rand = 100

omegas = 2*np.pi * np.linspace(1/(2 * 1e-6), 1/(2 * 15e-6), 50)
omega_rands = np.random.choice(omegas, N_rand)

# Assume eta and initial nbar are computable from confining harmonic frequency omega_z
omega_zs = 2*np.pi * np.linspace(0.75, 3, 50) * 1e6
omega_z_rands = np.random.choice(omega_zs, N_rand)

lam = 355e-9
hbar = 1.05457e-34
m = 170.936 * 1.6605e-27
eta_rands = np.sqrt(2) * 2*np.pi/lam * np.sqrt(hbar/2/m/omega_z_rands)

initial_nbar_rands = 19.6e6/2/(omega_z_rands/2/np.pi)

N_pulses_rands = np.random.choice(np.arange(5, 10), N_rand)

strategy = 'optimal'
cooled_nbars = []
pulse_schedules = []
for omega_z, initial_nbar, omega, N_pulses in tqdm(zip(omega_z_rands, initial_nbar_rands, omega_rands, N_pulses_rands), total=len(omega_z_rands)):
    sim = simulation.SidebandCooling(omega_z, initial_nbar, omega)
    cooled_nbar, pulse_schedule, cooling_time, cooled_dist = sim.cool(strategy, N_pulses, 1)
    cooled_nbars.append(cooled_nbar)
    pulse_schedules.append(pulse_schedule)
    sim.reset_current_dist()

# Simulate and save data
pi_times = np.pi / omega_rands
dataX = np.column_stack((eta_rands, pi_times, initial_nbar_rands, N_pulses_rands))
dataY = np.array(pulse_schedules)

# %% Save data into files
with open('data/dataX_'+strategy+'_nbar.npy', 'wb') as f:
    np.save(f, dataX)
with open('data/dataY_'+strategy+'_nbar.npy', 'wb') as f:
    np.save(f, dataY)

# %% Load data

strategy = 'optimal'
# load data
with open('data/dataX_'+strategy+'_nbar.npy', 'rb') as f:
    dataX = np.load(f)
with open('data/dataY_'+strategy+'_nbar.npy', 'rb') as f:
    dataY = np.load(f, allow_pickle=True)

# %% Group pulse times into groups with the "label" of group being the pulse just after the group
batch_size = 4  # must be less than minimal N_pulses_rands

features = []
labels = []
for i in range(len(dataY)):
    for j in range(batch_size, len(dataY[i])):
        features.append(dataY[i][j - batch_size: j])
        labels.append(dataY[i][j])
    
ptimesX = np.array(features)
ptimesY = np.array(labels)

ind = 0
plt.plot(np.arange(batch_size), ptimesX[ind], label='features')
plt.scatter(batch_size, ptimesY[ind], label='label')
plt.grid()
plt.legend()
plt.show()

# %% Scale data

scalerX = preprocessing.MinMaxScaler(feature_range=(0,1))
ptimesX_scaled = scalerX.fit_transform(ptimesX)
scalerY = preprocessing.MinMaxScaler(feature_range=(0,1))
ptimesY_scaled = scalerY.fit_transform(ptimesY.reshape(-1,1))

# %% Split into test and train

X_train, X_test, y_train, y_test = train_test_split(ptimesX_scaled, ptimesY_scaled, test_size=0.33, random_state=42)
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
# y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1]))

# X_train = tf.reshape(X_train, (1, X_train.shape[1]))
# y_train = tf.reshape(y_train, (1, y_train.shape[1]))


# %%

model = Sequential()

# Recurrent layer
model.add(LSTM(10, input_shape=(1, batch_size)))

# Fully connected layer
# model.add(Dense(10, activation='relu'))
# 
# Dropout for regularization
model.add(Dropout(0.5))

# Output layer
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

model.summary()

# %%
model.fit(X_train,  y_train, batch_size=1, epochs=100, verbose=2)

# %%

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# "unscale" predictions
train_pred = scalerY.inverse_transform(train_pred)
test_pred = scalerY.inverse_transform(test_pred)
y_train = scalerY.inverse_transform(y_train)
y_test = scalerY.inverse_transform(y_test)

print('Train score', np.sqrt(mean_squared_error(y_train, train_pred)))
print('Test score', np.sqrt(mean_squared_error(y_test, test_pred)))


# %% Plot predictions
	
ind = 20

plt.plot(np.arange(batch_size), ptimesX_scaled[ind])
plt.scatter(batch_size, ptimesY_scaled[ind], color='b')

pred = model.predict(np.reshape(ptimesX_scaled[ind], (1,1,batch_size)))
plt.scatter(batch_size, pred, color='r')
plt.grid()
plt.show()
