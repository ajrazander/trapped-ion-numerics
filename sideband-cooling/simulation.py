# Resolved Sideband Cooling Calculation and Simulation

# Import packages
import numpy as np

from scipy.special import eval_genlaguerre
from scipy.optimize import fmin_l_bfgs_b

import matplotlib.pyplot as plt



class SidebandCooling():
    """Abstract Class for containing sideband simulation functionality"""
    
    
    """
    Parameters
    omega_z: the frequency of the confining harmonic potential
    initial_nbar: the average harmonic state of the ion before cooling
    eta_angle: the angle (degrees) between the two counterpropogating Raman beams (typically 90 degrees)
    omega: the carrier Rabi oscillaiton frequency
    """
    def __init__(self, omega_z, initial_nbar, omega, eta_angle=90, n_max=1000):
        # range of harmonic states used throughout simulation
        self.ns = np.arange(0, n_max)
        
        # compute Lab-Dicke parameter "eta" (values for Yb171+ Raman transition)
        lam = 355e-9
        hbar = 1.05457e-34
        m = 170.936 * 1.6605e-27
        self.eta = 2*np.sin(eta_angle*np.pi/360) * 2*np.pi/lam * np.sqrt(hbar/2/m/omega_z)
        
        # initial average harmonic state of the confined ion
        self.initial_nbar = initial_nbar
        
        self.omega = omega

    
    def comp_thermal_dist(self, nbar):
        return np.exp(self.ns * np.log(nbar) - (self.ns + 1) * np.log(nbar + 1))
    
    
    def comp_rsb_omegas(self, order):
        prefactor = self.eta**order * np.exp(-self.eta**2/2) * self.omega * eval_genlaguerre(self.ns[order:]-order, order, self.eta**2)
        ns_sqrt = np.ones(self.ns[order:].shape)
        for i in range(order):
            ns_sqrt *= np.sqrt(self.ns[order:]-i)
        rsb_omegas = prefactor / ns_sqrt
        rsb_omegas = np.append([0]*order, rsb_omegas)
        return np.abs(rsb_omegas)
    
    
    def comp_bsb_omegas(self, order):
        prefactor = self.eta**order * np.exp(-eta**2/2) * self.omega * eval_genlaguerre(self.ns, order, self.eta**2)
        ns_sqrt = np.ones(self.ns.shape)
        for i in range(order):
            ns_sqrt *= np.sqrt(self.ns+1+i)
        bsb_omegas = prefactor / ns_sqrt
        return np.abs(bsb_omegas)
    
    
    def comp_weights(self, time, rsb_omegas, order):
        a_ns = np.sin(rsb_omegas[order:] * time/2)**2
        b_ns = np.cos(rsb_omegas * time/2)**2
        return np.diag(b_ns, k=0) + np.diag(a_ns, k=order)
    
    
    def cool(self, strategy, N_pulses):
        self.cooled_nbar = 0
        self.pulse_schedule = [0]*N_pulses
        self.cooling_time = 0
        self.cooled_dist = np.array([0]*self.ns)
        
        if strategy == 'classic':
            self.cooled_nbar, self.pulse_schedule, self.cooling_time, self.cooled_dist = self.classic(N_pulses)
        elif strategy == 'fixed':
            self.cooled_nbar, self.pulse_schedule, self.cooling_time, self.cooled_dist = self.fixed(N_pulses)
        elif strategy == 'optimal':
            self.cooled_nbar, self.pulse_schedule, self.cooling_time, self.cooled_dist = self.optimal(N_pulses)
        else:
            print('Strategy selection error. The strategy you selected is not an option. Please try \`classic\`, \`fixed\`, or \`optimal\`')
        
        return self.cooled_nbar, self.pulse_schedule, self.cooling_time, self.cooled_dist


    """
    Parameters
    N_pulses: the number of sideband cooling pulses
    
    Returns
    cooled_nbar: the harmonic distributions average state after sideband cooling
    pulse_schedule: the order and timing of the sideband cooling pulses
    cooling_time: the total length of time it took to sideband cool (excluding optimal pumping)
    cooled_dist: the harmonic distribution after sideband cooling
    """
    def classic(self, N_pulses):
        pulse_schedule = []
        cooled_nbar = 0
        cooling_time = 0
        
        # compute initial harmonic distribution
        thermal_dist = self.comp_thermal_dist(self.initial_nbar)
        
        # compute 1st order Rabi oscillation frequency
        # Already includes 0 frequency for transition from n=0 to n=-1
        rsb_1st_omegas = self.comp_rsb_omegas(1)
        
        # compute "pi" time for each transition frequency
        with np.errstate(divide='ignore'):
            pi_times = np.pi / rsb_1st_omegas

        # compute nbar and save total cooling time and pulse schedule
        weights = np.eye(len(self.ns))
        for i in range(N_pulses, 0, -1):
            weights = np.matmul(self.comp_weights(pi_times[i], rsb_1st_omegas, 1), weights)
            pulse_schedule.append(pi_times[i])
        cooling_time = np.sum(pulse_schedule)
        
        # compute cooled distribution
        cooled_dist = np.matmul(weights, thermal_dist)
        
        # compute nbar of cooled distribution
        cooled_nbar = np.sum(self.ns * cooled_dist)
        
        return cooled_nbar, pulse_schedule, cooling_time, cooled_dist
    
    
    def comp_nbar_fixed(self, time, *params):
        rsb_1st_omegas, N_pulses, dist = params
        # compute weights for a given number of pulses
        weights = self.comp_weights(time, rsb_1st_omegas, 1)
        weights = np.linalg.matrix_power(weights, N_pulses)
        # compute the final nbar
        return np.sum(self.ns*np.matmul(weights, dist))
    
    
    def fixed(self, N_pulses):
        pulse_schedule = []
        cooled_nbar = 0
        cooling_time = 0
        
        # compute initial harmonic distribution
        thermal_dist = self.comp_thermal_dist(self.initial_nbar)
        
        # compute 1st order Rabi oscillation frequency
        # Already includes 0 frequency for transition from n=0 to n=-1
        rsb_1st_omegas = self.comp_rsb_omegas(1)
        
        # organize inputs into tuple for optimization
        params = (rsb_1st_omegas, N_pulses, thermal_dist)
        
        # bounded gradient descent optimization
        t0 = [2*np.pi/self.omega]
        res = fmin_l_bfgs_b(self.comp_nbar_fixed, t0, bounds=[(t0[0]*0.01, t0[0]*5),], pgtol=1e-14, fprime=None, factr=1e2, args=params, approx_grad=1, epsilon=1e-17)
        
        # save minimum nbar, optimal pulse time and total sideband cooling time (excluding optimal pumping)
        cooled_nbar = res[1]
        pulse_schedule = list(res[0])*N_pulses
        cooling_time = np.sum(pulse_schedule)
        
        # compute cold distribution
        weights = self.comp_weights(res[0], rsb_1st_omegas, 1)
        weights = np.linalg.matrix_power(weights, N_pulses)
        cooled_dist = np.matmul(weights, thermal_dist)
        
        return cooled_nbar, pulse_schedule, cooling_time, cooled_dist
    
    
    def comp_nbar_opt(self, ts, *params):
        rsb_1st_omegas, dist = params
        weights = np.eye(len(self.ns))
        for time in ts:
            weights = np.matmul(self.comp_weights(time, rsb_1st_omegas, 1), weights)
        # compute the final nbar
        return np.sum(self.ns*np.matmul(weights, dist))


    def optimal(self, N_pulses):
        pulse_schedule = []
        cooled_nbar = 0
        cooling_time = 0
        
        # compute initial harmonic distribution
        thermal_dist = self.comp_thermal_dist(self.initial_nbar)
        
        # compute 1st order Rabi oscillation frequency
        # Already includes 0 frequency for transition from n=0 to n=-1
        rsb_1st_omegas = self.comp_rsb_omegas(1)
        
        # organize inputs into tuple for optimization
        params = (rsb_1st_omegas, thermal_dist)
        
        # bounded gradient descent optimization
        t0s = [2*np.pi/self.omega] * N_pulses
        res = fmin_l_bfgs_b(self.comp_nbar_opt, t0s, bounds=[(t0s[0]*0.01, t0s[0]*20)]*N_pulses, pgtol=1e-14, factr=1e2, fprime=None, args=params, approx_grad=1, epsilon=1e-17)
        
        # save minimum nbar, optimal pulse time and total sideband cooling time (excluding optimal pumping)
        cooled_nbar = res[1]
        pulse_schedule = res[0]
        cooling_time = np.sum(pulse_schedule)
        
        # check if optimiation hit bounds
        lowerbound_bool = np.isclose([t0s[0]*0.01] * N_pulses, pulse_schedule)
        upperbound_bool = np.isclose([t0s[0]*20] * N_pulses, pulse_schedule)
        
        if np.sum(lowerbound_bool) > 0:
            print('Bound error. Lower bound limited optimization. Decrease lower bound.')
        elif np.sum(upperbound_bool) > 0:
            print('Bound error. Upper bound limited optimization. Increase upper bound.')
        
        # compute cold distribution
        weights = np.eye(len(self.ns))
        for time in pulse_schedule:
            weights = np.matmul(self.comp_weights(time, rsb_1st_omegas, 1), weights)
        cooled_dist = np.matmul(weights, thermal_dist)
        
        return cooled_nbar, pulse_schedule, cooling_time, cooled_dist
        
# %% Example usage

# define constants
omega_z = 2*np.pi * 1e6
initial_nbar = 19.6e6 / 2 / (omega_z / (2*np.pi))
omega = 2*np.pi/(2 * 5e-6)

sim = SidebandCooling(omega_z, initial_nbar, omega)

strategy = 'classic'
N_pulses = 10

cooled_nbar, pulse_schedule, cooling_time, cooled_dist = sim.cool(strategy, N_pulses)

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

strategy = 'optimal'
list_pulses = np.arange(2, 12)

cooled_nbars = []
pulse_schedules = []
cooling_times = []
cooled_dists = []
for N_pulses in tqdm(list_pulses):
    cooled_nbar, pulse_schedule, cooling_time, cooled_dist = sim.cool(strategy, N_pulses)
    cooled_nbars.append(cooled_nbar)
    pulse_schedules.append(pulse_schedule)
    cooling_times.append(cooling_time)
    cooled_dists.append(cooled_dist)

# %% Plot results

plt.plot(list_pulses, cooled_nbars)
plt.xlabel('number of pulses', fontsize=12)
plt.ylabel(r'$\bar{n}$', fontsize=12)
plt.title(strategy+' strategy', fontsize=16)
plt.grid()
plt.show()




# %% Multi order pulses

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