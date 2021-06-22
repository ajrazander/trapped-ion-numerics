# Resolved Sideband Cooling Calculation and Simulation

# Import packages
import numpy as np

from scipy.special import eval_genlaguerre
from scipy.optimize import fmin_l_bfgs_b
    
class SidebandCooling():
    """Abstract Class for containing sideband simulation functionality"""    
    

    """
    Parameters
    omega_z: the frequency of the confining harmonic potential
    initial_nbar: the average harmonic state of the ion before cooling
    eta_angle: the angle (degrees) between the two counterpropogating Raman beams (typically 90 degrees)
    omega: the carrier Rabi oscillaiton frequency
    """
    def __init__(self, omega_z, initial_nbar, omega, eta_angle=90, n_max=500):
        # range of harmonic states used throughout simulation
        self.ns = np.arange(0, n_max)
        
        # compute Lab-Dicke parameter "eta" (values for Yb171+ Raman transition)
        lam = 355e-9
        hbar = 1.05457e-34
        m = 170.936 * 1.6605e-27
        self.eta = 2*np.sin(eta_angle*np.pi/360) * 2*np.pi/lam * np.sqrt(hbar/2/m/omega_z)
        
        # initial average harmonic state of the confined ion
        self.initial_nbar = initial_nbar
        
        # initial thermal distribution
        self.current_dist = self.comp_thermal_dist(self.initial_nbar)
        
        # Rabi carrier frequency. Necessary to calculate red sideband frequencies
        self.omega = omega


    def comp_thermal_dist(self, nbar):
        return np.exp(self.ns * np.log(nbar) - (self.ns + 1) * np.log(nbar + 1))


    def comp_rsb_omegas(self, rsb_order):
        prefactor = self.eta**rsb_order * np.exp(-self.eta**2/2) * self.omega * eval_genlaguerre(self.ns[rsb_order:]-rsb_order, rsb_order, self.eta**2)
        ns_sqrt = np.ones(self.ns[rsb_order:].shape)
        for i in range(rsb_order):
            ns_sqrt *= np.sqrt(self.ns[rsb_order:]-i)
        rsb_omegas = prefactor / ns_sqrt
        rsb_omegas = np.append([0]*rsb_order, rsb_omegas)
        return np.abs(rsb_omegas)


    def comp_bsb_omegas(self, bsb_order):
        prefactor = self.eta**bsb_order * np.exp(-self.eta**2/2) * self.omega * eval_genlaguerre(self.ns, bsb_order, self.eta**2)
        ns_sqrt = np.ones(self.ns.shape)
        for i in range(bsb_order):
            ns_sqrt *= np.sqrt(self.ns+1+i)
        bsb_omegas = prefactor / ns_sqrt
        return np.abs(bsb_omegas)
    
    
    def get_current_dist(self):
        return self.current_dist


    def reset_current_dist(self):
        self.current_dist = self.comp_thermal_dist(self.initial_nbar)
        

    def comp_weights(self, time, rsb_omegas, rsb_order):
        a_ns = np.sin(rsb_omegas[rsb_order:] * time/2)**2
        b_ns = np.cos(rsb_omegas * time/2)**2
        return np.diag(b_ns, k=0) + np.diag(a_ns, k=rsb_order)


    def cool(self, strategy, N_pulses, rsb_order):
        if strategy == 'classic':
            self.cooled_nbar, self.pulse_schedule, self.cooling_time, self.cooled_dist = self.classic(N_pulses, rsb_order)
        elif strategy == 'fixed':
            self.cooled_nbar, self.pulse_schedule, self.cooling_time, self.cooled_dist = self.fixed(N_pulses, rsb_order)
        elif strategy == 'optimal':
            self.cooled_nbar, self.pulse_schedule, self.cooling_time, self.cooled_dist = self.optimal(N_pulses, rsb_order)
        else:
            print('Strategy selection error. The strategy you selected is not an option. Please try \`classic\`, \`fixed\`, or \`optimal\`')
        
        self.current_dist = self.cooled_dist
        return self.cooled_nbar, self.pulse_schedule, self.cooling_time, self.current_dist


    """
    Parameters
    N_pulses: the number of sideband cooling pulses
    
    Returns
    cooled_nbar: the harmonic distributions average state after sideband cooling
    pulse_schedule: the order and timing of the sideband cooling pulses
    cooling_time: the total length of time it took to sideband cool (excluding optimal pumping)
    cooled_dist: the harmonic distribution after sideband cooling
    """
    def classic(self, N_pulses, rsb_order):
        pulse_schedule = []
        cooled_nbar = 0
        cooling_time = 0
        
        # compute 1st order Rabi oscillation frequency
        # Already includes 0 frequency for transition from n=0 to n=-1
        rsb_omegas = self.comp_rsb_omegas(rsb_order)
        
        # compute "pi" time for each transition frequency
        with np.errstate(divide='ignore'):
            pi_times = np.pi / rsb_omegas

        # compute nbar and save total cooling time and pulse schedule
        weights = np.eye(len(self.ns))
        for i in range(N_pulses, 0, -1):
            weights = np.matmul(self.comp_weights(pi_times[i], rsb_omegas, 1), weights)
            pulse_schedule.append(pi_times[i])
        cooling_time = np.sum(pulse_schedule)
        
        # compute cooled distribution
        cooled_dist = np.matmul(weights, self.current_dist)
        
        # compute nbar of cooled distribution
        cooled_nbar = np.sum(self.ns * cooled_dist)
        
        return cooled_nbar, pulse_schedule, cooling_time, cooled_dist
    
    
    def comp_nbar_fixed(self, time, *params):
        rsb_omegas, N_pulses, dist, rsb_order = params
        # compute weights for a given number of pulses
        weights = self.comp_weights(time, rsb_omegas, rsb_order)
        weights = np.linalg.matrix_power(weights, N_pulses)
        # compute the final nbar
        return np.sum(self.ns*np.matmul(weights, dist))
    
    
    def fixed(self, N_pulses, rsb_order):
        pulse_schedule = []
        cooled_nbar = 0
        cooling_time = 0
    
        # compute 1st order Rabi oscillation frequency
        # Already includes 0 frequency for transition from n=0 to n=-1
        rsb_omegas = self.comp_rsb_omegas(rsb_order)
        
        # organize inputs into tuple for optimization
        params = (rsb_omegas, N_pulses, self.current_dist, rsb_order)
        
        # bounded gradient descent optimization
        t0 = [rsb_order * 2*np.pi/self.omega]
        res = fmin_l_bfgs_b(self.comp_nbar_fixed, t0, bounds=[(t0[0]*0.01, t0[0]*5),], pgtol=1e-14, fprime=None, factr=1e2, args=params, approx_grad=1, epsilon=1e-17)
        
        # save minimum nbar, optimal pulse time and total sideband cooling time (excluding optimal pumping)
        cooled_nbar = res[1]
        pulse_schedule = list(res[0])*N_pulses
        cooling_time = np.sum(pulse_schedule)
        
        # compute cold distribution
        weights = self.comp_weights(res[0], rsb_omegas, rsb_order)
        weights = np.linalg.matrix_power(weights, N_pulses)
        cooled_dist = np.matmul(weights, self.current_dist)
        
        return cooled_nbar, pulse_schedule, cooling_time, cooled_dist
    
    
    def comp_nbar_opt(self, ts, *params):
        rsb_omegas, dist, rsb_order = params
        weights = np.eye(len(self.ns))
        for time in ts:
            weights = np.matmul(self.comp_weights(time, rsb_omegas, rsb_order), weights)
        # compute the final nbar
        return np.sum(self.ns*np.matmul(weights, dist))


    def optimal(self, N_pulses, rsb_order):
        pulse_schedule = []
        cooled_nbar = 0
        cooling_time = 0
        
        # compute 1st order Rabi oscillation frequency
        # Already includes 0 frequency for transition from n=0 to n=-1
        rsb_omegas = self.comp_rsb_omegas(rsb_order)
        
        # organize inputs into tuple for optimization
        params = (rsb_omegas, self.current_dist, rsb_order)
        
        # bounded gradient descent optimization
        t0s = [rsb_order * 2*np.pi/self.omega] * N_pulses
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
            weights = np.matmul(self.comp_weights(time, rsb_omegas, rsb_order), weights)
        cooled_dist = np.matmul(weights, self.current_dist)
        
        return cooled_nbar, pulse_schedule, cooling_time, cooled_dist

