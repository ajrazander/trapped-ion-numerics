# Resolved Sideband Cooling Simulation

# Import packages
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

from scipy.special import eval_genlaguerre
from scipy.optimize import fmin_l_bfgs_b, Bounds, minimize, broyden1

from itertools import combinations_with_replacement

import matplotlib.pyplot as plt
%matplotlib inline

from tqdm import tqdm


# %% support functions
def prob_n(ns, n_bar):
    return np.exp(ns * np.log(n_bar) - (ns+1) * np.log(n_bar + 1))

def rsb_omegas(ns, eta, omega0, order):
    prefactor = eta**order * np.exp(-eta**2/2) * omega0 * eval_genlaguerre(ns[order:]-order, order, eta**2)
    ns_sqrt = np.ones(ns[order:].shape)
    for i in range(order):
        ns_sqrt *= np.sqrt(ns[order:]-i)
    rsb_omegas = prefactor / ns_sqrt
    return np.abs(rsb_omegas)

def bsb_omegas(ns, eta, omega0, order):
    prefactor = eta**order * np.exp(-eta**2/2) * omega0 * eval_genlaguerre(ns, order, eta**2)
    ns_sqrt = np.ones(ns.shape)
    for i in range(order):
        ns_sqrt *= np.sqrt(ns+1+i)
    bsb_omegas = prefactor / ns_sqrt
    return np.abs(bsb_omegas)

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


# %% Comparison between different sideband cooling strategies: classic, fixed, and numerically optimized

# compute eta
lam = 355e-9
hbar = 1.05457e-34
m = 170.936 * 1.6605e-27
omega_z = 2*np.pi * 0.670e6
eta = 1.54 * 2*np.pi/lam * np.sqrt(hbar/2/m/omega_z) # 1.56 coefficient to account for Raman beam angle
# eta = 0.19 # direction fit on RSB
initial_nbar = 15.36  # manually input measured initial nbar

# initialize other experiment parameters
omega0 = 2*np.pi / (2 * 7.7e-6)
num_pulses = np.arange(5, 40)

ns = np.arange(0, 750)
print('fraction of population under consideration', np.sum(prob_n(ns, initial_nbar)))
print('nbar contribution excluded from simulation', initial_nbar - np.sum(ns*prob_n(ns, initial_nbar)))

# %% save full simulation data
data_full = [omega_z, initial_nbar, eta, omega0, np.max(ns), num_pulses]
np.savez('sim_multiorder_data_inputs.npz', *data_full)


# %% load data
# container = np.load('sim_multiorder_data_inputs.npz', allow_pickle=True)
# data_full = [container[key] for key in container]
# omega_z, initial_nbar, eta, omega0, ns_max, num_pulses = data_full

# ns = np.arange(0, ns_max)

# %% Supporting functions for graph theory numerical computations

def compute_weights(time, rsb_1st_omegas):
    a_ns = np.sin(rsb_1st_omegas[1:]*time/2)**2
    b_ns = np.cos(rsb_1st_omegas*time/2)**2
    return np.diag(b_ns, k=0) + np.diag(a_ns, k=1)

def compute_n_bar(time, *params):
    ns, rsb_1st_omegas, pulses, dist = params
    # compute weights for a given number of pulses
    weights = compute_weights(time, rsb_1st_omegas)
    weights = np.linalg.matrix_power(weights, pulses)
    # compute the final nbar
    return np.sum(ns*np.matmul(weights, dist))

def fixed(initial_n, num_pulses, rsb_1st_omegas, ns=ns):
    pulse_times_all = []
    total_times = []
    nbar_mins = []
    pop_vec_thermal = prob_n(ns, initial_n)
    for pulses in tqdm(num_pulses):
        # organize inputs into tuple
        params = (ns, rsb_1st_omegas, pulses, pop_vec_thermal)

        t0s = [2*np.pi/omega0]
        res = fmin_l_bfgs_b(compute_n_bar, t0s, bounds=[(7e-7, 25e-6), ], pgtol=1e-14, fprime=None, factr=1e2, args=params, approx_grad=1, epsilon=1e-17)
        nbar_mins.append(res[1])
        total_times.append(res[0][0]*pulses)
        pulse_times_all.append(list(res[0])*pulses)
        
    pulse_times_all = np.array(pulse_times_all)
    total_times = np.array(total_times)
    nbar_mins = np.array(nbar_mins)
    return total_times, nbar_mins, pulse_times_all

print('\n--- getting 1st order data ---\n', flush=True)

rsb_1st_omegas = np.append(0, rsb_omegas(ns, eta, omega0, 1))
# t0s = pulse_times_classic_full[i]
t0s = [[2*np.pi/omega0] * pulses for pulses in num_pulses]
total_times, nbar_mins, pulse_times_fixed = fixed(initial_nbar, num_pulses, rsb_1st_omegas)
# compute final harmonic dist
final_dists = []
for pulses in pulse_times_fixed:
    weights = np.eye(len(ns))
    for p in pulses:
        weights = np.matmul(compute_weights(p, rsb_1st_omegas), weights)
    final_dists.append(np.matmul(weights, prob_n(ns, initial_nbar)))

# convert to numpy array
pulse_times_fixed = np.array(pulse_times_fixed)
total_times_fixed = np.array(total_times)
nbar_mins_fixed = np.array(nbar_mins)
final_dists_fixed = np.array(final_dists)

print('\npulse timings', pulse_times_fixed)
print('nbar', nbar_mins_fixed)

# %% save full simulation data
data_full = [pulse_times_fixed, total_times_fixed, nbar_mins_fixed, final_dists_fixed]
np.savez('sim_multiorder_data_fixed.npz', *data_full)


# %% load data
# container = np.load('sim_multiorder_data_fixed.npz', allow_pickle=True)
# data_full = [container[key] for key in container]
# pulse_times_fixed, total_times_fixed, nbar_mins_fixed, final_dists_fixed = data_full


# %% The optimum method with 2nd and 1st order pulses

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
    ns, rsb_1st_omegas, rsb_2nd_omegas, dist, seq = params
    num_1pulses = np.sum(seq==1)
    num_2pulses = np.sum(seq==2)
    num_3pulses = np.sum(seq==3)
    
    times_1, times_2, times_3 = times
    
    # times at the BEGINNING of the times array are applied first
    weights_1st = np.linalg.matrix_power(compute_weights(times_1, rsb_1st_omegas), num_1pulses)
    weights_2nd = np.linalg.matrix_power(compute_weights_2nd(times_2, rsb_2nd_omegas), num_2pulses)
    weights_3rd = np.linalg.matrix_power(compute_weights_3rd(times_3, rsb_3rd_omegas), num_3pulses)
    
    weights = np.matmul(weights_1st, np.matmul(weights_2nd, weights_3rd))
    # compute and return nbar
    return np.sum(ns*np.matmul(weights, dist))


# %%

def fixed_multi_order(initial_n, seqs, rsb_1st_omegas, rsb_2nd_omegas, rsb_3rd_omegas, ns=ns):
    pulse_times_all = []
    total_times = []
    nbar_mins = []
    pop_vec_thermal = prob_n(ns, initial_n)
    for i, seq in enumerate(tqdm(seqs)):
        # organize inputs into tuple
        params = (ns, rsb_1st_omegas, rsb_2nd_omegas, pop_vec_thermal, seq)

        t0s = [2*np.pi/omega0, 2*np.pi/omega0*2, 2*np.pi/omega0*3]
        res = fmin_l_bfgs_b(compute_nbar_multi_order, t0s, bounds=[(10e-6, 40e-6), (10e-6, 90e-6), (10e-6, 300e-6)], pgtol=1e-14, fprime=None, factr=1e2, args=params, approx_grad=1, epsilon=1e-17)
        
        num_1pulses = np.sum(seq==1)
        num_2pulses = np.sum(seq==2)
        num_3pulses = np.sum(seq==3)
        
        pulse_timings = [res[0][2]]*num_3pulses + [res[0][1]]*num_2pulses + [res[0][0]]*num_1pulses
        total_times.append(np.sum(pulse_timings))
        pulse_times_all.append(pulse_timings)
        # if len(nbar_mins) > 0:
        #     if res[1] > nbar_mins[-1]:  # stop computation if most recent nbar is higher than previous
        #         return total_times, nbar_mins, pulse_times_all
        nbar_mins.append(res[1])
        
    pulse_times_all = np.array(pulse_times_all)
    total_times = np.array(total_times)
    nbar_mins = np.array(nbar_mins)
    
    return total_times, nbar_mins, pulse_times_all

print('\n--- getting 2nd order data ---\n', flush=True)
pulse_times_fixed_multi = []
total_times_fixed_multi = []
nbar_mins_fixed_multi = []
best_seqs_fixed_multi = []
final_dists_fixed_multi = []

rsb_1st_omegas = np.append(0, rsb_omegas(ns, eta, omega0, 1))
rsb_2nd_omegas = np.append([0,0], rsb_omegas(ns, eta, omega0, 2))
rsb_3rd_omegas = np.append([0,0,0], rsb_omegas(ns, eta, omega0, 3))

for pulses in num_pulses:
    seqs = compute_grouped_seqs(pulses, 3)
    seqs_small = [seq for seq in seqs if np.mean(seq)>2 and np.mean(seq)<2.5]
    total_times_all, nbar_mins_all, pulse_times_fixed_multi_all = fixed_multi_order(initial_nbar, seqs_small, rsb_1st_omegas, rsb_2nd_omegas, rsb_3rd_omegas, ns=ns)
    # extract smallest nbar from different seqs
    min_arg = np.argmin(nbar_mins_all)
    pulse_times_fixed_multi.append(pulse_times_fixed_multi_all[min_arg])
    total_times_fixed_multi.append(total_times_all[min_arg])
    nbar_mins_fixed_multi.append(nbar_mins_all[min_arg])
    best_seqs_fixed_multi.append(seqs_small[min_arg])
    # compute final harmonic dist
    dist = np.zeros(len(ns))
    for p_times in pulse_times_fixed_multi:
        weights = np.eye(len(ns))
        for t, s in zip(p_times, seqs_small[min_arg]):
            if s == 3:
                weights = compute_weights_3rd(t, rsb_3rd_omegas) @ weights
            elif s == 2:
                weights = compute_weights_2nd(t, rsb_2nd_omegas) @ weights
            else:
                weights = compute_weights(t, rsb_1st_omegas) @ weights
        dist = weights @ prob_n(ns, initial_nbar)
    final_dists_fixed_multi.append(dist)

pulse_times_fixed_multi = np.array(pulse_times_fixed_multi)
total_times_fixed_multi = np.array(total_times_fixed_multi)
nbar_mins_fixed_multi = np.array(nbar_mins_fixed_multi)
best_seqs_fixed_multi = np.array(best_seqs_fixed_multi)

print('average seq value for best seqs', [np.mean(seq) for seq in best_seqs_fixed_multi])
# print('\npulse timings', np.unique(pulse_times_fixed_multi))
print('nbar', nbar_mins_fixed_multi)

# %% save full simulation data
data_full = [pulse_times_fixed_multi, best_seqs_fixed_multi, total_times_fixed_multi, nbar_mins_fixed_multi, final_dists_fixed_multi]
np.savez('sim_multiorder_data_fixedmulti.npz', *data_full)


# %% load data
# container = np.load('sim_multiorder_data_fixedmulti.npz', allow_pickle=True)
# data_full = [container[key] for key in container]
# pulse_times_fixed_multi, best_seqs_fixed_multi, total_times_fixed_multi, nbar_mins_fixed_multi, final_dists_fixed_multi = data_full


# %%

container = np.load('sim_multiorder_data_inputs.npz', allow_pickle=True)
data_full = [container[key] for key in container]
omega_z, initial_nbar, eta, omega0, ns_max, num_pulses = data_full
ns = np.arange(0, ns_max)

container = np.load('sim_multiorder_data_fixed.npz', allow_pickle=True)
data_full = [container[key] for key in container]
pulse_times_fixed, total_times_fixed, nbar_mins_fixed, final_dists_fixed = data_full
# nbars_fixed = np.sum(ns * final_dists_fixed, axis=0)

container = np.load('sim_multiorder_data_fixedmulti.npz', allow_pickle=True)
data_full = [container[key] for key in container]
pulse_times_fixed_multi, best_seqs_fixed_multi, total_times_fixed_multi, nbar_mins_fixed_multi, final_dists_fixed_multi = data_full
# nbars_fixed_multi = np.sum(ns * final_dists_fixed_multi, axis=1)

fig = plt.figure(constrained_layout=True, figsize=(6,5))
gs = fig.add_gridspec(2, 2)
ax0 = fig.add_subplot(gs[:,0])
ax1a = fig.add_subplot(gs[0,1])
ax1b = fig.add_subplot(gs[1,1])

ax0.plot(num_pulses, nbar_mins_fixed)
ax0.plot(num_pulses, nbar_mins_fixed_multi)
ax0.set_ylabel(r'final $\bar{n}$')
ax0.set_xlabel('num pulses')
# ax0.set_xticks([0,5,10,15])
ax0.legend(['first order', 'multi order'])
ax0.grid()

end = 150
ax1a.plot(ns[:end], final_dists_fixed[-1][:end])
ax1a.set_yscale('log')
ax1a.set_ylabel('probability')
ax1a.set_xlabel('harmonic state (n)')
ax1a.set_ylim([1e-7, 1])
ax1a.grid()

ax1b.plot(ns[:end], final_dists_fixed_multi[-1][:end])
ax1b.set_yscale('log')
ax1b.set_ylabel('probability')
ax1b.set_xlabel('harmonic state (n)')
ax1b.set_ylim([1e-7, 1])
ax1b.grid()

plt.show()

fig.savefig('multiorder_comparison.pdf')


# %% method comparison

lam = 355e-9
hbar = 1.05457e-34
m = 170.936 * 1.6605e-27
omega_z = 2*np.pi * 0.670e6
eta = 1.54 * 2*np.pi/lam * np.sqrt(hbar/2/m/omega_z) # 1.56 coefficient to account for Raman beam angle
initial_nbar = 15.36  # manually input measured initial nbar

# initialize other experiment parameters
omega0 = 2*np.pi / (2 * 7.7e-6)
num_pulses = np.arange(5, 15)

ns = np.arange(0, 500)
print('fraction of population under consideration', np.sum(prob_n(ns, initial_nbar)))
print('nbar contribution excluded from simulation', initial_nbar - np.sum(ns*prob_n(ns, initial_nbar)))

# %% save full simulation data
data_full = [omega_z, initial_nbar, eta, omega0, np.max(ns), num_pulses]
np.savez('sim_method_comp_data_inputs.npz', *data_full)


# %% load data
# container = np.load('sim_multiorder_data_inputs.npz', allow_pickle=True)
# data_full = [container[key] for key in container]
# omega_z, initial_nbar, eta, omega0, ns_max, num_pulses = data_full

# ns = np.arange(0, ns_max)


# %% The classic method

def classic(initial_n, num_pulses, rsb_1st_omegas, ns=ns):
    pulse_times_all = []
    total_times = []
    nbar_mins = []
    pop_vec_thermal = prob_n(ns, initial_n)
    with np.errstate(divide='ignore'):
        pi_times = np.pi / rsb_1st_omegas
    for pulses in num_pulses:
        pulse_times = []
        weights = np.eye(len(ns))
        time = 0
        for p in range(pulses, 0, -1):
            weights = np.matmul(compute_weights(ns, pi_times[p], rsb_1st_omegas), weights)
            time += pi_times[p]
            pulse_times.append(pi_times[p])
        nbar = np.sum(ns*np.matmul(weights, pop_vec_thermal))
        nbar_mins.append(nbar)
        total_times.append(time)
        pulse_times_all.append(pulse_times)

    total_times = np.array(total_times)
    nbar_mins = np.array(nbar_mins)
    
    return total_times, nbar_mins, pulse_times_all

print('\n--- getting classic method data ---\n', flush=True)
total_times_classic = []
nbar_mins_classic = []
pulse_times_classic = []
final_dists_classic = []
for i, (initial_n, eta) in enumerate(tqdm(zip(initial_ns, etas))):
    rsb_1st_omegas = np.append(0, rsb_omegas(ns, eta, omega0, 1))
    total_times, nbar_mins, pulse_times = classic(initial_n, num_pulses_classic[i,:], rsb_1st_omegas)
    # compute final harmonic dist
    final_dists = []
    for pulses in pulse_times:
        weights = np.eye(len(ns))
        for p in pulses:
            weights = np.matmul(compute_weights(ns, p, rsb_1st_omegas), weights)
        final_dists.append(np.matmul(weights, prob_n(ns, initial_n)))
    pulse_times_classic.append(pulse_times)
    total_times_classic.append(total_times)
    nbar_mins_classic.append(nbar_mins)
    final_dists_classic.append(final_dists)

# convert to numpy array
pulse_times_classic = np.array(pulse_times_classic)
total_times_classic = np.array(total_times_classic)
nbar_mins_classic = np.array(nbar_mins_classic)
final_dists_classic = np.array(final_dists_classic)


# %% save full simulation data
data_full = [pulse_times_classic, total_times_classic, nbar_mins_classic, final_dists_classic]
np.savez('sim_method_comp_classic.npz', *data_full)


# %% load data
# container = np.load('sim_method_comp_classic.npz', allow_pickle=True)
# data_full = [container[key] for key in container]
# pulse_times_classic, total_times_classic, nbar_mins_classic, final_dists_classic = data_full


# %% Supporting functions for graph theory numerical computations

def fixed(initial_n, num_pulses, rsb_1st_omegas, ns=ns):
    pulse_times_all = []
    total_times = []
    nbar_mins = []
    pop_vec_thermal = prob_n(ns, initial_n)
    for pulses in tqdm(num_pulses):
        # organize inputs into tuple
        params = (ns, rsb_1st_omegas, pulses, pop_vec_thermal)

        t0s = [2*np.pi/omega0]
        res = fmin_l_bfgs_b(compute_n_bar, t0s, bounds=[(7e-7, 25e-6), ], pgtol=1e-14, fprime=None, factr=1e2, args=params, approx_grad=1, epsilon=1e-17)
        nbar_mins.append(res[1])
        total_times.append(res[0][0]*pulses)
        pulse_times_all.append(list(res[0])*pulses)
        
    pulse_times_all = np.array(pulse_times_all)
    total_times = np.array(total_times)
    nbar_mins = np.array(nbar_mins)
    return total_times, nbar_mins, pulse_times_all

print('\n--- getting 1st order data ---\n', flush=True)

rsb_1st_omegas = np.append(0, rsb_omegas(ns, eta, omega0, 1))
# t0s = pulse_times_classic_full[i]
t0s = [[2*np.pi/omega0] * pulses for pulses in num_pulses]
total_times, nbar_mins, pulse_times_fixed = fixed(initial_nbar, num_pulses, rsb_1st_omegas)
# compute final harmonic dist
final_dists = []
for pulses in pulse_times_fixed:
    weights = np.eye(len(ns))
    for p in pulses:
        weights = np.matmul(compute_weights(p, rsb_1st_omegas), weights)
    final_dists.append(np.matmul(weights, prob_n(ns, initial_nbar)))

# convert to numpy array
pulse_times_fixed = np.array(pulse_times_fixed)
total_times_fixed = np.array(total_times)
nbar_mins_fixed = np.array(nbar_mins)
final_dists_fixed = np.array(final_dists)

print('\npulse timings', pulse_times_fixed)
print('nbar', nbar_mins_fixed)

# %% save full simulation data
data_full = [pulse_times_fixed, total_times_fixed, nbar_mins_fixed, final_dists_fixed]
np.savez('sim_method_comp_data_fixed.npz', *data_full)

# %% load data
# container = np.load('sim_method_comp_data_fixed.npz', allow_pickle=True)
# data_full = [container[key] for key in container]
# pulse_times_fixed, total_times_fixed, nbar_mins_fixed, final_dists_fixed = data_full

# %% Numerically optmized pulse strategy

def compute_n_bar_topt(ts, *params):
    ns, rsb_1st_omegas, pop_vec_thermal = params
    weights = np.eye(len(ns))
    for time in ts:
        weights = np.matmul(compute_weights(ns, time, rsb_1st_omegas), weights)
    # compute the final nbar
    return np.sum(ns*np.matmul(weights, pop_vec_thermal))

def optimum(initial_n, t0s, num_pulses, rsb_1st_omegas, ns=ns):
    pulse_times_all = []
    total_times = []
    nbar_mins = []
    pop_vec_thermal = prob_n(ns, initial_n)
    params = (ns, rsb_1st_omegas, pop_vec_thermal)
    for i, pulses in enumerate(tqdm(num_pulses, leave=False)):
        # print(len(t0s[i]*pulses), len([(8e-7, 8e-6)]*pulses))
        res = fmin_l_bfgs_b(compute_n_bar_topt, t0s[i], bounds=[(8e-7, 40e-6)]*pulses, pgtol=1e-14, factr=1e2, fprime=None, args=params, approx_grad=1, epsilon=1e-17)
        
        pulse_times_all.append(res[0])
        total_times.append(np.sum(res[0]))
        nbar_mins.append(res[1])
    
    return total_times, nbar_mins, pulse_times_all

print('\n--- getting numerically optimium method data ---\n', flush=True)
pulse_times_opt = []
total_times_opt = []
nbar_mins_opt = []
final_dists_opt = []
for i, initial_n in enumerate(tqdm(initial_ns)):
    rsb_1st_omegas = np.append(0, rsb_omegas(ns, eta, omega0, 1))
    # t0s = pulse_times_classic_full[i]
    t0s = pulse_times_fixed_full[i]
    total_times, nbar_mins, pulse_times = optimum(initial_n, t0s, num_pulses_opt[i,:], rsb_1st_omegas)
    # compute final harmonic dist
    final_dists = []
    for pulses in pulse_times:
        weights = np.eye(len(ns))
        for p in pulses:
            weights = np.matmul(compute_weights(ns, p, rsb_1st_omegas), weights)
        final_dists.append(np.matmul(weights, prob_n(ns, initial_n)))
    pulse_times_opt.append(pulse_times)
    total_times_opt.append(total_times)
    nbar_mins_opt.append(nbar_mins)
    final_dists_opt.append(final_dists)
    

# convert to numpy array
pulse_times_opt = np.array(pulse_times_opt)
total_times_opt = np.array(total_times_opt)
nbar_mins_opt = np.array(nbar_mins_opt)
final_dists_opt = np.array(final_dists_opt)


# %% save full simulation data
data_full = [pulse_times_opt, total_times_opt, nbar_mins_opt, final_dists_opt]
np.savez('sim_method_comp_opt.npz', *data_full)


   # %% load data
# container = np.load('sim_method_comp_opt.npz', allow_pickle=True)
# data_full = [container[key] for key in container]
# pulse_times_opt, total_times_opt, nbar_mins_opt, final_dists_opt = data_full


# %%

container = np.load('sim_multiorder_data_inputs.npz', allow_pickle=True)
data_full = [container[key] for key in container]
omega_z, initial_nbar, eta, omega0, ns_max, num_pulses = data_full
ns = np.arange(0, ns_max)

container = np.load('sim_method_comp_classic.npz', allow_pickle=True)
data_full = [container[key] for key in container]
pulse_times_classic, total_times_classic, nbar_mins_classic, final_dists_classic = data_full

container = np.load('sim_method_comp_data_fixed.npz', allow_pickle=True)
data_full = [container[key] for key in container]
pulse_times_fixed, total_times_fixed, nbar_mins_fixed, final_dists_fixed = data_full

container = np.load('sim_method_comp_opt.npz', allow_pickle=True)
data_full = [container[key] for key in container]
pulse_times_opt, total_times_opt, nbar_mins_opt, final_dists_opt = data_full


fig = plt.figure(igsize=(5,5))


plt.plot(num_pulses, nbar_min_classic, '-r')
plt.plot(num_pulses, nbar_mins_fixed, '-b')
plt.plot(num_pulses, nbar_mins_opt, 'darkgreen')
plt.set_ylabel(r'final $\bar{n}$')
plt.set_xlabel('num pulses')
plt.set_xticks([0,5,10,15])
plt.legend(['classic', 'fixed', 'optimal'])
plt.grid()


plt.show()

fig.savefig('method_comp.pdf')