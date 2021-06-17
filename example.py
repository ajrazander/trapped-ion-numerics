import numpy as np

import matplotlib.pyplot as plt

from sideband_cooling import simulation

# %% Example usage of simulation package

# define constants
omega_z = 2*np.pi * 1e6
initial_nbar = 19.6e6 / 2 / (omega_z / (2*np.pi))
omega = 2*np.pi/(2 * 5e-6)

sim = simulation.SidebandCooling(omega_z, initial_nbar, omega)

strategy = 'classic'
N_pulses = 10

cooled_nbar, pulse_schedule, cooling_time, cooled_dist = sim.cool(strategy, N_pulses, 1)

# %% Plot results

print('initial nbar', initial_nbar, ' now cooled to', cooled_nbar)

ns_partial = np.arange(0, 100)

plt.plot(ns_partial, cooled_dist[:len(ns_partial)])
plt.xlabel('harmonic state', fontsize=12)
plt.ylabel('probability', fontsize=12)
plt.title('cold ion harmonic distribution', fontsize=16)
plt.yscale('log')
plt.grid()
plt.show()

# %% Plot nbar as a function of number of pulses

from tqdm import tqdm

# define constants
omega_z = 2*np.pi * 1e6
initial_nbar = 19.6e6 / 2 / (omega_z / (2*np.pi))
omega = 2*np.pi/(2 * 5e-6)

sim = SidebandCooling(omega_z, initial_nbar, omega)

strategy = 'fixed'
list_pulses = np.arange(2, 12)

cooled_nbars = []
pulse_schedules = []
cooling_times = []
cooled_dists = []
for N_pulses in tqdm(list_pulses):
    cooled_nbar, pulse_schedule, cooling_time, cooled_dist = sim.cool(strategy, N_pulses, 1)
    cooled_nbars.append(cooled_nbar)
    pulse_schedules.append(pulse_schedule)
    cooling_times.append(cooling_time)
    cooled_dists.append(cooled_dist)
    sim.reset_current_dist()
    
# %% Plot results

plt.plot(list_pulses, cooled_nbars)
plt.xlabel('number of pulses', fontsize=12)
plt.ylabel(r'$\bar{n}$', fontsize=12)
plt.title(strategy+r' strategy with final $\bar{n}=$'+str(cooled_nbars[-1])[:4], fontsize=16)
plt.grid()
plt.show()

# %% Apply mutli-order cooling

# define constants
omega_z = 2*np.pi * 1e6
initial_nbar = 19.6e6 / 2 / (omega_z / (2*np.pi))
omega = 2*np.pi/(2 * 5e-6)

sim = SidebandCooling(omega_z, initial_nbar, omega)

strategy = 'fixed'
pulses_1st_order = np.arange(2, 12)
pulses_2nd_order = pulses_1st_order[::-1]

cooled_nbars = []
pulse_schedules = []
cooling_times = []
cooled_dists = []
for p1s, p2s in tqdm(zip(pulses_1st_order, pulses_2nd_order), total=len(pulses_1st_order)):
    cooled_nbar2, pulse_schedule2, cooling_time2, cooled_dist2 = sim.cool(strategy, p2s, 2)
    cooled_nbar, pulse_schedule, cooling_time, cooled_dist = sim.cool(strategy, p1s, 1)
    cooled_nbars.append(cooled_nbar)
    pulse_schedules.append(pulse_schedule)
    cooling_times.append(cooling_time)
    cooled_dists.append(cooled_dist)
    sim.reset_current_dist()

# %% Plot results

plt.plot(pulses_1st_order, cooled_nbars)
plt.xlabel('number of 1st order pulses', fontsize=12)
plt.ylabel(r'$\bar{n}$', fontsize=12)
plt.title(strategy+r' strategy with minimum $\bar{n}=$'+str(np.min(cooled_nbars))[:4], fontsize=16)
plt.grid()
plt.show()

plt.plot(pulses_2nd_order, cooled_nbars)
plt.xlabel('number of 2nd order pulses', fontsize=12)
plt.ylabel(r'$\bar{n}$', fontsize=12)
plt.title(strategy+r' strategy with minimum $\bar{n}=$'+str(np.min(cooled_nbars))[:4], fontsize=16)
plt.grid()
plt.show()
