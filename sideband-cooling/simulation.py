# Resolved Sideband Cooling Calculation and Simulation

# Import packages
import numpy as np

from scipy.special import eval_genlaguerre
from scipy.optimize import fmin_l_bfgs_b

import matplotlib.pyplot as plt



class SidebandCooling():
    """Abstract Class for containing sideband simulation functionality"""
    
    
    def __init__(self):
        self.ns = np.arange(0, 1000)
    
    
    def get_thermal_dist(self, nbar):
        return np.exp(self.ns * np.log(nbar) - 
                      (self.ns + 1) * np.log(nbar + 1))
    
    
    def get_rsb_omegas(self, eta, omega0, order):
        prefactor = eta**order * np.exp(-eta**2/2) * omega0 * 
                        eval_genlaguerre(self.ns[order:]-order, order, eta**2)
        ns_sqrt = np.ones(self.ns[order:].shape)
        for i in range(order):
            ns_sqrt *= np.sqrt(self.ns[order:]-i)
        rsb_omegas = prefactor / ns_sqrt
        return np.abs(rsb_omegas)
    
    
    def get_bsb_omegas(self, eta, omega0, order):
        prefactor = eta**order * np.exp(-eta**2/2) * omega0 * eval_genlaguerre(self.ns, order, eta**2)
        ns_sqrt = np.ones(self.ns.shape)
        for i in range(order):
            ns_sqrt *= np.sqrt(self.ns+1+i)
        bsb_omegas = prefactor / ns_sqrt
        return np.abs(bsb_omegas)


# %% support functions


def rabi_rsb(ts, ns, eta, prob_ns, omega0, order):
    flops = np.zeros(ts.shape)
    omegas = np.append([0]*order, rsb_omegas(ns, eta, omega0, order))
    for i, t in enumerate(ts):
        flops[i] = np.sum(prob_ns * np.sin(omegas * t / 2)**2)
    return flops

def rabi_bsb(ts, ns, eta, prob_ns, omega0, order):
    flops = np.zeros(ts.shape)
    omegas = bsb_omegas(ns, eta, omega0, order)
    for i, t in enumerate(ts):
        flops[i] = np.sum(prob_ns * np.sin(omegas * t / 2)**2)
    return flops

def sinc_squared(x, x0, width, signal, background):
    return signal * np.sinc((x-x0)/width/np.pi)**2 + background

def double_poisson(x, mean_dark, mean_bright, w1):
    return w1*poisson.pmf(x, mean_dark) + (1-w1)*poisson.pmf(x, mean_bright)

# %% Supporting functions for graph theory numerical computations

def compute_weights(time, rsb_1st_omegas):
    a_ns = np.sin(rsb_1st_omegas[1:]*time * 0.5)**2
    b_ns = np.cos(rsb_1st_omegas*time * 0.5)**2
    return np.diag(b_ns, k=0) + np.diag(a_ns, k=1)

def compute_n_bar(time, *params):
    ns, rsb_1st_omegas, pulses, dist = params
    # compute weights for a given number of pulses
    weights = compute_weights(time, rsb_1st_omegas)
    weights = np.linalg.matrix_power(weights, pulses)
    # compute the final nbar
    return np.sum(ns*np.matmul(weights, dist))

# %% Comparison between different sideband cooling strategies: classic, fixed, and numerically optimized

# compute eta
lam = 355e-9
hbar = 1.05457e-34
m = 170.936 * 1.6605e-27
omega_z = 2*np.pi * 0.670e6
eta = 1.54 * 2*np.pi/lam * np.sqrt(hbar/2/m/omega_z) # 1.54 coefficient to account for Raman beam angle not being 45
initial_nbar = 15.36

# initialize other experiment parameters
omega0 = 2*np.pi / (2 * 7.7e-6)
num_pulses = np.arange(1, 61)

ns = np.arange(0, 500)
print('fraction of population under consideration', np.sum(prob_n(ns, initial_nbar)))
print('nbar contribution excluded from simulation', initial_nbar - np.sum(ns*prob_n(ns, initial_nbar)))


# %% The classic method

def classic(initial_nbar, num_pulses, rsb_1st_omegas, ns=ns):
    pulse_times_all = []
    total_times = []
    nbar_mins = []
    thermal_dist = prob_n(ns, initial_nbar)
    with np.errstate(divide='ignore'):
        pi_times = np.pi / rsb_1st_omegas
    for pulses in tqdm(num_pulses):
        pulse_times = []
        weights = np.eye(len(ns))
        time = 0
        for p in range(pulses, 0, -1):
            weights = np.matmul(compute_weights(pi_times[p], rsb_1st_omegas), weights)
            time += pi_times[p]
            pulse_times.append(pi_times[p])
        nbar = np.sum(ns*np.matmul(weights, thermal_dist))
        nbar_mins.append(nbar)
        total_times.append(time)
        pulse_times_all.append(pulse_times)

    total_times = np.array(total_times)
    nbar_mins = np.array(nbar_mins)
    
    return total_times, nbar_mins, pulse_times_all

print('\n--- getting classic method data ---\n', flush=True)
rsb_1st_omegas = np.append(0, rsb_omegas(ns, eta, omega0, 1))

total_times_classic, nbar_mins_classic, pulse_times_classic = classic(initial_nbar, num_pulses, rsb_1st_omegas)

# compute final harmonic distributions
final_dists_classic = []
for pulses in tqdm(pulse_times_classic):
    weights = np.eye(len(ns))
    for p in pulses:
        weights = np.matmul(compute_weights(p, rsb_1st_omegas), weights)
    final_dists_classic.append(np.matmul(weights, prob_n(ns, initial_nbar)))

final_dists_classic = np.array(final_dists_classic)

# %% save full simulation data
data_full = [total_times_classic, nbar_mins_classic, pulse_times_classic, final_dists_classic]
np.savez('sim_classic_data.npz', *data_full)

# %% The fixed pulse method

def fixed(initial_nbar, num_pulses, rsb_1st_omegas, ns=ns):
    pulse_times_all = []
    total_times = []
    nbar_mins = []
    thermal_dist = prob_n(ns, initial_nbar)
    for pulses in tqdm(num_pulses):
        # organize inputs into tuple
        params = (ns, rsb_1st_omegas, pulses, thermal_dist)

        t0s = [14.8e-6]
        res = fmin_l_bfgs_b(compute_n_bar, t0s, bounds=[(7e-7, 25e-6), ], pgtol=1e-14, fprime=None, factr=1e2, args=params, approx_grad=1, epsilon=1e-17)
        nbar_mins.append(res[1])
        total_times.append(res[0][0]*pulses)
        pulse_times_all.append(list(res[0])*pulses)
        
    pulse_times_all = np.array(pulse_times_all, dtype='object')
    total_times = np.array(total_times)
    nbar_mins = np.array(nbar_mins)
    
    return total_times, nbar_mins, pulse_times_all

print('\n--- getting fixed pulse method data ---\n', flush=True)
rsb_1st_omegas = np.append(0, rsb_omegas(ns, eta, omega0, 1))

total_times_fixed, nbar_mins_fixed, pulse_times_fixed = fixed(initial_nbar, num_pulses, rsb_1st_omegas)

# compute final harmonic dist
final_dists_fixed = []
for pulses in tqdm(pulse_times_fixed):
    weights = np.eye(len(ns))
    for p in pulses:
        weights = np.matmul(compute_weights(p, rsb_1st_omegas), weights)
    final_dists_fixed.append(np.matmul(weights, prob_n(ns, initial_nbar)))
final_dists_fixed = np.array(final_dists_fixed)


# %% save full simulation data
data_full = [total_times_fixed, nbar_mins_fixed, pulse_times_fixed, final_dists_fixed]
np.savez('sim_fixed_data.npz', *data_full)

# %% Numerically optmized pulse strategy

def compute_n_bar_topt(ts, *params):
    ns, rsb_1st_omegas, thermal_dist = params
    weights = np.eye(len(ns))
    for time in ts:
        weights = np.matmul(compute_weights(time, rsb_1st_omegas), weights)
    # compute the final nbar
    return np.sum(ns*np.matmul(weights, thermal_dist))

def optimum(initial_nbar, t0s, num_pulses, rsb_1st_omegas, ns=ns):
    pulse_times_all = []
    total_times = []
    nbar_mins = []
    thermal_dist = prob_n(ns, initial_nbar)
    params = (ns, rsb_1st_omegas, thermal_dist)
    for i, pulses in enumerate(tqdm(num_pulses)):
        
        res = fmin_l_bfgs_b(compute_n_bar_topt, t0s[i], bounds=[(8e-7, 60e-6)]*pulses, pgtol=1e-14, factr=1e2, fprime=None, args=params, approx_grad=1, epsilon=1e-17)
        pulse_times_all.append(res[0])
        total_times.append(np.sum(res[0]))
        nbar_mins.append(res[1])
        
    pulse_times_all = np.array(pulse_times_all, dtype='object')
    total_times = np.array(total_times)
    nbar_mins = np.array(nbar_mins)
    
    return total_times, nbar_mins, pulse_times_all


print('\n--- getting numerically optimium method data ---\n', flush=True)
rsb_1st_omegas = np.append(0, rsb_omegas(ns, eta, omega0, 1))

t0s = pulse_times_fixed
total_times_opt, nbar_mins_opt, pulse_times_opt = optimum(initial_nbar, t0s, num_pulses, rsb_1st_omegas)
# compute final harmonic dist
final_dists_opt = []
for pulses in tqdm(pulse_times_opt):
    weights = np.eye(len(ns))
    for p in pulses:
        weights = np.matmul(compute_weights(p, rsb_1st_omegas), weights)
    final_dists_opt.append(np.matmul(weights, prob_n(ns, initial_nbar)))
final_dists_opt = np.array(final_dists_opt)

# %% save full simulation data
data_full = [total_times_opt, nbar_mins_opt, pulse_times_opt, final_dists_opt]
np.savez('sim_opt_data.npz', *data_full)


# %% load data

container = np.load('sim_classic_data.npz', allow_pickle=True)
data_full = [container[key] for key in container]
total_times_classic, nbar_mins_classic, pulse_times_classic, final_dists_classic = data_full

container = np.load('sim_fixed_data.npz', allow_pickle=True)
data_full = [container[key] for key in container]
total_times_fixed, nbar_mins_fixed, pulse_times_fixed, final_dists_fixed = data_full

container = np.load('sim_opt_data.npz', allow_pickle=True)
data_full = [container[key] for key in container]
total_times_opt, nbar_mins_opt, pulse_times_opt, final_dists_opt = data_full

# %% Compare strategies

# plt.figure(figsize=(14,5))
# plt.plot(total_times_classic*1e6, nbar_mins_classic, '-r')
# plt.plot(total_times_fixed*1e6, nbar_mins_fixed, '-b')
# plt.plot(total_times_opt*1e6, nbar_mins_opt, '-m')
# plt.ylabel(r'final $\bar{n}$')
# plt.xlabel(r'total cooling time ($\mu$s)')
# plt.yscale('log')
# plt.legend(['classic', 'fixed', 'optimal'])
# plt.grid()
# plt.show()

plt.figure(figsize=(10,5))
plt.plot(num_pulses, nbar_mins_classic, '-r')
plt.plot(num_pulses, nbar_mins_fixed, '-b')
plt.plot(num_pulses, nbar_mins_opt, '-m')
plt.xlabel('number of pulses')
plt.ylabel(r'final $\bar{n}$')
plt.yscale('log')
plt.legend(['classic', 'fixed', 'optimal'])
plt.grid()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(num_pulses, nbar_mins_classic, '-r')
plt.plot(num_pulses, nbar_mins_fixed, '-b')
plt.plot(num_pulses, nbar_mins_opt, '-m')
plt.xlabel('number of pulses')
plt.ylabel(r'final $\bar{n}$')
plt.legend(['classic', 'fixed', 'optimal'])
plt.grid()
plt.show()

# plt.figure(figsize=(10,5))
# plt.plot(num_pulses, (nbar_mins_classic - nbar_mins_opt), '-r')
# plt.plot(num_pulses, (nbar_mins_fixed - nbar_mins_opt), '-b')
# plt.xlabel('number of pulses')
# plt.ylabel(r'absolute difference from optimal method')
# plt.yscale('log')
# plt.legend(['classic', 'fixed', 'optimal'])
# plt.grid()
# plt.show()

# %%

def compute_weights_2nd(time, rsb_2nd_omegas):
    c_ns = np.sin(rsb_2nd_omegas[2:]*time/2)**2
    b_ns = np.cos(rsb_2nd_omegas*time/2)**2
    return np.diag(b_ns, k=0) + np.diag(c_ns, k=2)

def compute_weights_3rd(time, rsb_3rd_omegas):
    c_ns = np.sin(rsb_3rd_omegas[3:]*time/2)**2
    b_ns = np.cos(rsb_3rd_omegas*time/2)**2
    return np.diag(b_ns, k=0) + np.diag(c_ns, k=3)

# any order. Every seq includes every order
def compute_grouped_seqs(total_pulses, num_orders):
    order_list = []
    for i in range(num_orders):
        order_list.append(i+1)
    
    seqs_unsorted = np.array(list(combinations_with_replacement(order_list, total_pulses)))
    seqs_final = []
    for seq in seqs_unsorted:
        if len(np.unique(seq)) == num_orders:
            seqs_final.append(np.sort(seq)[::-1])
    return np.array(seqs_final)


def compute_nbar_multi_order(times, *params):
    ns, rsb_1st_omegas, rsb_2nd_omegas, rsb_3rd_omegas, dist, num_1pulses, num_2pulses, num_3pulses = params
    
    times_1, times_2, times_3 = times
    
    # times at the BEGINNING of the times array are applied first
    weights_1st = np.linalg.matrix_power(compute_weights(times_1, rsb_1st_omegas), num_1pulses)
    weights_2nd = np.linalg.matrix_power(compute_weights_2nd(times_2, rsb_2nd_omegas), num_2pulses)
    weights_3rd = np.linalg.matrix_power(compute_weights_3rd(times_3, rsb_3rd_omegas), num_3pulses)
    
    weights = np.matmul(weights_1st, np.matmul(weights_2nd, weights_3rd))
    # compute and return nbar
    return np.sum(ns*np.matmul(weights, dist))

# %% Multi order simulation

def fixed_multi_order(initial_nbar, seqs, rsb_1st_omegas, rsb_2nd_omegas, rsb_3rd_omegas, ns=ns):
    pulse_times_all = []
    total_times = []
    nbar_mins = []
    thermal_dist = prob_n(ns, initial_nbar)
    for i, seq in enumerate(seqs):
        num_1pulses = np.sum(seq==1)
        num_2pulses = np.sum(seq==2)
        num_3pulses = np.sum(seq==3)
        
        # organize inputs into tuple
        params = (ns, rsb_1st_omegas, rsb_2nd_omegas, rsb_3rd_omegas,
                  thermal_dist, num_1pulses, num_2pulses, num_3pulses)

        t0s = [2*np.pi/omega0, 2*np.pi/omega0*1.5, 2*np.pi/omega0*1.75]
        res = fmin_l_bfgs_b(compute_nbar_multi_order, t0s, bounds=[(7e-7, 40e-6), (7e-7, 50e-6), (7e-7, 70e-6)], pgtol=1e-14, fprime=None, factr=1e2, args=params, approx_grad=1, epsilon=1e-17)
        
        pulse_timings = [res[0][2]]*num_3pulses + [res[0][1]]*num_2pulses + [res[0][0]]*num_1pulses
        total_times.append(np.sum(pulse_timings))
        pulse_times_all.append(pulse_timings)
        nbar_mins.append(res[1])
        
    pulse_times_all = np.array(pulse_times_all)
    total_times = np.array(total_times)
    nbar_mins = np.array(nbar_mins)
    
    return total_times, nbar_mins, pulse_times_all

print('\n--- getting multi order data ---\n', flush=True)
pulse_times_fixed_multi = []
total_times_fixed_multi = []
nbar_mins_fixed_multi = []
best_seqs_fixed_multi = []
final_dist_fixed_multi = []

rsb_1st_omegas = np.append([0], rsb_omegas(ns, eta, omega0, 1))
rsb_2nd_omegas = np.append([0,0], rsb_omegas(ns, eta, omega0, 2))
rsb_3rd_omegas = np.append([0,0,0], rsb_omegas(ns, eta, omega0, 3))

for pulses in tqdm(num_pulses):
    seqs = compute_grouped_seqs(pulses, 3)
    total_times, nbar_mins, pulse_times = fixed_multi_order(initial_nbar, seqs, rsb_1st_omegas, rsb_2nd_omegas, rsb_3rd_omegas, ns=ns)
    # extract smallest nbar from different seqs
    min_arg = np.argmin(nbar_mins)
    pulse_times_fixed_multi.append(pulse_times[min_arg])
    total_times_fixed_multi.append(total_times[min_arg])
    nbar_mins_fixed_multi.append(nbar_mins[min_arg])
    best_seqs_fixed_multi.append(seqs[min_arg])
    # compute final harmonic dist
    for p_times in pulse_times_fixed_multi:
        weights = np.eye(len(ns))
        for t, s in zip(p_times, best_seqs_fixed_multi[0]):
            if s == 3:
                weights = compute_weights_3rd(t, rsb_3rd_omegas) @ weights
            elif s == 2:
                weights = compute_weights_2nd(t, rsb_2nd_omegas) @ weights
            elif s == 1:
                weights = compute_weights(t, rsb_1st_omegas) @ weights
            else:
                print('Error: sequence label error')
        final_dist_fixed_multi.append(weights @ prob_n(ns, initial_nbar))

pulse_times_fixed_multi = np.array(pulse_times_fixed_multi)
total_times_fixed_multi = np.array(total_times_fixed_multi)
nbar_mins_fixed_multi = np.array(nbar_mins_fixed_multi)
best_seqs_fixed_multi = np.array(best_seqs_fixed_multi)
final_dist_fixed_multi = np.array(final_dist_fixed_multi)

print('\npulse timings', np.unique(pulse_times_fixed_multi))
print('nbar', nbar_mins_fixed_multi)

# %% save full simulation data
data_full = [pulse_times_fixed_multi, total_times_fixed_multi, nbar_mins_fixed_multi, best_seqs_fixed_multi, final_dist_fixed_multi]
np.savez('sim_multifixed_data.npz', *data_full)


# %% load data
container = np.load('sim_multifixed_data.npz', allow_pickle=True)
data_full = [container[key] for key in container]
pulse_times_fixed_multi, total_times_fixed_multi, nbar_mins_fixed_multi, best_seqs_fixed_multi, final_dist_fixed_multi = data_full

# %%

plt.plot(np.arange(149), final_dist_fixed_multi[150], alpha=0.5, width=0.8)
plt.yscale('log')
plt.show()

plt.plot(np.arange(149), final_dist_fixed[150], alpha=0.5, width=0.8)
plt.yscale('log')
plt.show()

# plt.plot(num_pulses, final_dist_fixed)
# plt.plot(num_pulses, final_dist_fixed_multi)
# plt.yscale('log')
# plt.grid()
# plt.show()

