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
N_rand_samples = 1000000

omegas = 2*np.pi * np.linspace(1/(2 * 1e-6), 1/(2 * 15e-6), 100)
omega_rands = np.random.choice(omegas, N_rand_samples)

# Assume eta and initial nbar are computable from confining harmonic frequency omega_z
omega_zs = 2*np.pi * np.linspace(0.75, 4, 100) * 1e6
omega_z_rands = np.random.choice(omega_zs, N_rand_samples)

lam = 355e-9
hbar = 1.05457e-34
m = 170.936 * 1.6605e-27
eta_rands = np.sqrt(2) * 2*np.pi/lam * np.sqrt(hbar/2/m/omega_z_rands)

initial_nbar_rands = 19.6e6/2/(omega_z_rands/2/np.pi)

N_pulses_rands = np.random.choice(np.arange(10, 100), N_rand_samples)

strategy = 'fixed'
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


with open('data/dataX_'+strategy+'_nbar.npy', 'wb') as f:
    np.save(f, dataX)
with open('data/dataY_'+strategy+'_nbar.npy', 'wb') as f:
    np.save(f, dataY)
    

# %% Load data

strategy = 'fixed'
# load data
with open('data/dataX_'+strategy+'_nbar.npy', 'rb') as f:
    dataX = np.load(f)
with open('data/dataY_'+strategy+'_nbar.npy', 'rb') as f:
    dataY = np.load(f, allow_pickle=True)
    
# %% view portion of data

end = 200

dataX0 = dataX[:,0][:end]
dataX1 = dataX[:,1][:end]
dataX2 = dataX[:,2][:end]
dataX3 = dataX[:,3][:end]
dataY0 = [x[0] for x in dataY[:end]]

plt.scatter(dataY0, dataX0)
plt.ylabel('eta')
plt.xlabel('pulse time')
plt.grid()
plt.show()

plt.scatter(dataY0, dataX1)
plt.ylabel('pi times')
plt.xlabel('pulse time')
plt.grid()
plt.show()

plt.scatter(dataY0, dataX2)
plt.ylabel('initial nbars')
plt.xlabel('pulse time')
plt.grid()
plt.show()

plt.scatter(dataY0, dataX3)
plt.ylabel('N pulses')
plt.xlabel('pulse time')
plt.grid()
plt.show()
# %% Pad pulse times with zeros so number of pulses is uniform (same size as first NN layer)

N_pulse_total = 100
dataY_padded = np.zeros((len(dataY), N_pulse_total))
    
for i in tqdm(range(len(dataY))):
    if len(dataY[i]) < N_pulse_total:
        len_dif = N_pulse_total - len(dataY[i])
        dataY_padded[i] = np.append(dataY[i], [0]*len_dif).astype(np.float64)
    
# %% Scale data

# Preprocess data
X_train, X_test, y_train, y_test = train_test_split(dataX, dataY_padded, test_size=0.33, random_state=42)

# scalerX = preprocessing.StandardScaler().fit(X_train)
# scalery = preprocessing.StandardScaler().fit(y_train)
scalerX = preprocessing.MinMaxScaler().fit(X_train)
scalery = preprocessing.MinMaxScaler().fit(y_train)
X_train = scalerX.transform(X_train)
y_train = (scalery.transform(y_train))
X_test = scalerX.transform(X_test)
y_test = (scalery.transform(y_test))


# %%

def make_model(lr_rate):
    model = Sequential()
    
    # Recurrent layer
    model.add(Dense(N_pulse_total, activation='relu', input_shape=(1, len(X_train[0]))))
    
    # Fully connected layer
    model.add(Dense(N_pulse_total * 2, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(N_pulse_total * 2, activation='relu'))
    # model.add(Dropout(0.2))
    # model.add(Dense(N_pulse_total, activation='relu'))
    # model.add(Dropout(0.2))
    # model.add(Dense(30, activation='relu'))
    
    
    # Output layer
    model.add(Dense(N_pulse_total))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate = lr_rate)
    
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])
    
    return model


# %% Hyper parameter search
lr_rates = [0.01, 0.02]

accuracies = []
losses = []
for lr_rate in tqdm(lr_rates):
    # Compile the model
    model = make_model(lr_rate)

    model.summary()
    
    history = model.fit(X_train,  y_train, batch_size=50, epochs=300, verbose=1)
    losses.append(history.history['loss'])
    accuracies.append(history.history['accuracy'])
# %%

# history = model.fit(X_train,  y_train, batch_size=1, epochs=300, verbose=1)

# %%
for accuracy, lr_rate in zip(accuracies, lr_rates):
    plt.plot(accuracy, label='lr rate '+str(lr_rate))
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.ylim([-0.01,1.01])
plt.grid()
plt.show()
    
for loss, lr_rate in zip(losses, lr_rates):
    plt.plot(loss, label='lr rate '+str(lr_rate))
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.yscale('log')
plt.grid()
plt.show()

# %%

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# "unscale" predictions
train_pred = scalery.inverse_transform(train_pred)
test_pred = scalery.inverse_transform(test_pred)
y_train = scalery.inverse_transform(y_train)
y_test = scalery.inverse_transform(y_test)

ind = 110
plt.plot(y_test[ind], label='test')
plt.plot(test_pred[ind], label='pred')
plt.ylim([1e-8, 1e-4])
plt.legend()
plt.grid()
plt.show()

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
