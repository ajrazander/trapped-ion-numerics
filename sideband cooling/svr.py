# Support Vector Regression
# For improved computational speed

import numpy as np

import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn import svm, preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV

from sideband_cooling import simulation
 

# Predict optimal pulse time (FIXED METHOD)
# eta, pi/omega0, initial nbar, and number of pulses

# %% Generate dataset
N_rand = 200

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

N_pulses_rands = np.random.choice(np.arange(5, 50), N_rand)

strategy = 'fixed'
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
with open('data/dataX_fixed_nbar.npy', 'wb') as f:
    np.save(f, dataX)
with open('data/dataY_fixed_nbar.npy', 'wb') as f:
    np.save(f, dataY)

# %% Load and preprocess data

# load data
with open('data/dataX_fixed_nbar.npy', 'rb') as f:
    dataX = np.load(f)
with open('data/dataY_fixed_nbar.npy', 'rb') as f:
    dataY = np.load(f)
    
# Preprocess data
X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.33, random_state=42)

scalerX = preprocessing.StandardScaler().fit(X_train)
scalery = preprocessing.StandardScaler().fit(y_train.reshape(-1,1))
X_train = scalerX.transform(X_train)
y_train = (scalery.transform(y_train.reshape(-1,1))).reshape(-1,)
X_test = scalerX.transform(X_test)
y_test = (scalery.transform(y_test.reshape(-1,1))).reshape(-1,)


# %% Use Cross-validation to tune svr hyperparameters
svr = svm.SVR()

param_grid = [{'C': [1, 10, 100, 1000, 5000, 10000], 'gamma': [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06], 'kernel': ['rbf']}]
svr_grid = GridSearchCV(svr, param_grid, scoring='r2')
svr_grid.fit(X_train, y_train)

means = svr_grid.cv_results_['mean_test_score']
stds = svr_grid.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, svr_grid.cv_results_['params']):
    print("%0.4f (+/-%0.04f) for %r"
          % (mean, std * 2, params))
print("Best parameters set found on development set:\n")
print(svr_grid.best_params_,"\n")
print("Grid scores on development set:\n")

# %% Save otpimal hyperparameters and train

gamma_opt = svr_grid.best_params_['gamma']
C_opt = svr_grid.best_params_['C']

# train SVR on training data
svr = svm.SVR(C=C_opt, gamma=gamma_opt)
svr.fit(X_train, y_train)

# compute predictions on test data
preds = svr.predict(X_test)

# %% Analyze model

# convert y's back to time
preds_times = scalery.inverse_transform(preds)
y_test_times = scalery.inverse_transform(y_test)

plt.figure(figsize=(7,6.5))
plt.scatter(y_test_times, preds_times)
plt.plot(preds_times,preds_times, '-r')
plt.xlabel('exact')
plt.ylabel('prediction')
plt.title('SVR pred vs actual')
plt.grid()
plt.show()

rel_errors = np.abs((preds_times - y_test_times)/y_test_times)

plt.figure(figsize=(7,3))
plt.scatter(y_test_times, rel_errors)
plt.title('Relative error between exact and prediction')
plt.grid()
plt.show()

print('root mean square', np.sqrt(np.mean((y_test_times - preds_times)**2)))
print('average relative error', np.mean(rel_errors), '+/-', np.std(rel_errors))
correlation_matrix = np.corrcoef(y_test_times, preds_times)
correlation_xy = correlation_matrix[0,1]
r_squared = correlation_xy**2
print('r squared', r_squared)