# Simulation of ion positions

import numpy as np

import matplotlib.pyplot as plt

from scipy.integrate import odeint
from scipy.optimize import minimize, curve_fit

from numba import jit
import time

# %% Constants for Yb171+
m = 170.936323 * 1.66054e-27  # ion mass
e = 1.60217662e-19  # electron charge
epsilon_0 = 8.8541878128e-12

# Trap parameters
r0 = 250e-6 * np.sqrt(2)  # distance to rf electrodes
z0 = 670e-6  # distance to dc electrodes
kappa = 0.35  # geometric factor
Omega_t = 2*np.pi * 21e6  # rf frequency

# %% Ion crystal positions

# number of ions
N = 4

# harmonic frequencies
wx = 2 * np.pi * 0.4e6
wy = 2 * np.pi * 0.4e6
wz = 2 * np.pi * 1.2e6

@jit(nopython=True, fastmath=True)
def potential_energy(positions):
    # wx, wy, wz, N, m, e, epsilon_0 = params
    xs = positions[:N]
    ys = positions[N:2*N]
    zs = positions[2*N:3*N]
    
    # harmonic energy
    harm_energy = np.sum(1/2 * m * (wx**2 * xs**2 + wy**2 * ys**2 + wz**2 * zs**2))

    # electronic interaction
    interaction = 0
    for i in range(N):
        for j in range(N):
            if j != i:
                interaction += 1/np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2 + (zs[i]-zs[j])**2)
    interaction = e**2/4/np.pi/epsilon_0 * interaction
        
    return harm_energy + interaction

# xs_0 = np.random.choice(range(-N,N), N) * 1e-6
xs_0 = np.arange(0, N) * 1e-6
# ys_0 = np.random.choice(range(-N,N), N) * 1e-6
ys_0 = np.array([0] * N) * 1e-6
zs_0 = np.array([0] * N) * 1e-6

pos = np.append(np.append(xs_0, ys_0), zs_0)

# params = [wx, wy, wz, N, m, e, epsilon_0]
# pot_en = potential_energy(pos, *params)
pot_en = potential_energy(pos)
print('Potential Energy', pot_en)

plt.plot(xs_0 * 1e6, ys_0 * 1e6, '.', color='royalblue', markersize=12, markeredgecolor='darkblue')
plt.xlabel('x distance (micron)')
plt.ylabel('y distance (micron)')
plt.title('Initial ion position guess')
plt.grid()
plt.show()

# %% Minimize potential energy to determine ion crystal positions

# run bad guess through optimization with on a few iterations
pos_better = minimize(potential_energy, pos, method='COBYLA', options={'tol':1e-30, 'maxiter':500})

# Get fine results with better initial best and many iterations
pos_better = minimize(potential_energy, pos_better.x, method='COBYLA', options={'tol':1e-30, 'maxiter':80000})

# Get fine results with better initial best and many iterations
res = minimize(potential_energy, pos_better.x, method='COBYLA', options={'tol':1e-30, 'maxiter':80000})


# %% Plot radial plane
xs_f = res.x[:N]
ys_f = res.x[N:2*N]
zs_f = res.x[2*N:3*N]

plt.plot(xs_f * 1e6, ys_f * 1e6, '.', markersize=16, color='royalblue', markeredgecolor='k')
plt.title('Ion equilibrium position')
plt.xlabel('x distance (micron)')
plt.ylabel('y distance (micron)')
plt.ylim(-30, 30)
plt.xlim(-30, 30)
plt.grid()
plt.gca().set_aspect('equal')
plt.show()

# %% Rotate crystal

# angle is in degrees
def rotate_crystal(xs, ys, angle):
    xs_new = np.cos(angle/180*np.pi) * xs - np.sin(angle/180*np.pi) * ys
    ys_new = np.cos(angle/180*np.pi) * ys + np.sin(angle/180*np.pi) * xs
    return xs_new, ys_new

xs_f, ys_f = rotate_crystal(xs_f, ys_f, -4)

plt.figure(figsize=(6,6))
plt.scatter(xs_f*1e6, ys_f*1e6, marker='o',s = 100)
plt.title('Ion equilibrium position')
plt.xlabel('x position in micron')
plt.ylabel('y position in micron')
plt.ylim(-30,30)
plt.xlim(-30,30)
plt.grid()
plt.show()

# %% Transverse ("drumhead") mode calculation

def normal_modes(xs, ys, *constants):
    e, epsilon_0, m = constants
    scale = e**2/4/np.pi/epsilon_0/m
    
    # Add harmonic terms to matrix
    matrix = -wz**2 * np.diag(np.ones(N))

    # Add interaction diagonal terms
    for i in range(N):
        diag = 0
        for j in range(N):
            if i != j:
                diag += scale / np.sqrt((xs[i]-xs[j])**2+(ys[i]-ys[j])**2)**3
        matrix[i,i] += diag

#     Add interaction cross terms 
    for i in range(N):
        for j in range(N):
            cross_term = 0
            if i != j:
                cross_term = scale / np.sqrt((xs[i]-xs[j])**2+(ys[i]-ys[j])**2)**3
            matrix[i,j] -= cross_term
    return matrix


# Compute normal modes and frequencies
constants = (e, epsilon_0, m)
z_vals, z_vecs = np.linalg.eig(normal_modes(xs_f, ys_f, *constants))

z_freqs = np.round(np.sqrt(np.abs(z_vals))/2/np.pi)
z_freqs, np.round(z_vecs, 3)

adjusted_freqs = (z_freqs - np.max(z_freqs))/1e3
plt.vlines(adjusted_freqs, [0]*N, [2]*N)
plt.title('Mode frequency - COM frequency')
plt.xlabel('kHz')
plt.show()

# %% J_ij interaction rate calculations

def compute_Jijs(eig_vecs, eig_vals, mu, wzs, *constants):
    omega, m, hbar, dk = constants
    prefactor = omega**2 * hbar * dk**2 / 2 / m
    num_modes = eig_vecs.shape[1]
    j_ijs = np.zeros((num_modes, num_modes))
    for i in range(num_modes):
        for j in range(num_modes):
            mode_sum = 0
            if j != i:
                mode_sum = np.sum(eig_vecs[i,:]*eig_vecs[j,:] / (mu**2 - eig_vals**2))
            j_ijs[i,j] = mode_sum
    return prefactor * j_ijs

mu = 2*np.pi * 100e3 + np.max(z_freqs)*2*np.pi
omega = 2*np.pi / (2*8.2e-6)
hbar = 1.054571817e-34
dk = np.sqrt(2) * 2*np.pi/ 355e-9

constants = (omega, m, hbar, dk)
j_ijs = compute_Jijs(z_vecs, z_freqs*2*np.pi, mu, z_freqs*2*np.pi, *constants)

# Plot radial plane
i = 0
interaction_rates = j_ijs[i,:]

plt.figure(figsize=(6,6))
plt.scatter(xs_f*1e6, ys_f*1e6, c=interaction_rates/1e3, cmap='jet', marker='o',s = 150)
plt.title('Ion equilibrium positions')
plt.xlabel(r'x position ($\mu$m)')
plt.ylabel(r'y position ($\mu$m)')
plt.ylim(-30,30)
plt.xlim(-30,30)
plt.text(xs_f[i]*1e6 - 2, ys_f[i]*1e6 + 2, 'ith ion')
plt.grid()
plt.colorbar(label=r'$J_{ij}$ (kHz) from the ith ion')
plt.show()


# %% Compare distance to J_ij rate

r_ij = np.zeros((N, N))
for i in range(len(xs_f)):
    for j in range(len(ys_f)):
        if j!=i:
            r_ij[i,j] = np.sqrt((xs_f[i] - xs_f[j])**2 + (ys_f[i]-ys_f[j])**2)

def poly(rs, a, b, c):
    return a/rs**b + c

i=3
r_ijs_no_zeros = np.delete(r_ij[i,:]*1e6, i)
j_ijs_no_zeros = np.delete(j_ijs[i,:]/1e3, i)

try:
    guess = [1e3, 2, 0]
    popt, pcov = curve_fit(poly, r_ijs_no_zeros, j_ijs_no_zeros, p0=guess, absolute_sigma=True)
    print('fit parameters', popt)
    print('fit uncertainties', np.sqrt(np.diag(pcov)))
    
    rs_fit = np.linspace(np.min(r_ijs_no_zeros), np.max(r_ijs_no_zeros), 100)
    
    plt.plot(rs_fit, poly(rs_fit, *popt), '-b')
    plt.title('power '+str(popt[1])[:4]+' decay')
except RuntimeError:
    print('Runtime Error in fit')
    

plt.scatter(r_ijs_no_zeros, j_ijs_no_zeros)
plt.xlabel('radial distance (micron)')
plt.ylabel('J_ij (kHz)')
plt.grid()
plt.show()

# %%  Calculate U(t) for N ions (U(t) = exp[sum of J_ij sig_x^i sig_x^j])
from scipy.linalg import expm

N = 2

sigma_x = np.array([[0,1],[1,0]])
iden = np.eye(2,2)

# compute sigma_x for each of N ions. will generate N 2^Nx2^N matrices, so make sure N is small
sigma_xs = []
for i in range(N):
    if i==0:
        kron_matrix = np.kron(iden, sigma_x)
    elif i==1:
        kron_matrix = np.kron(sigma_x, iden)
    else:
        kron_matrix = np.kron(iden, iden)
    for j in range(2, N):
        if i==j:
            kron_matrix = np.kron(sigma_x, kron_matrix)
        else:
            kron_matrix = np.kron(iden, kron_matrix)
    sigma_xs.append(kron_matrix)

# compute Hamiltonian
H_eff = np.zeros((2**N, 2**N))
for i in range(N):
    for j in range(i+1, N):
        H_eff += j_ijs[i,j] * (sigma_xs[i] @ sigma_xs[j])

# compute exponentiated operators
ts = np.linspace(0, 150, 200)*1e-6
state1 = np.array(np.append([1], [0]*(2**(N)-1)))
# state1 = np.array([1,0,0,0])
# state2 = np.array([0,1,0,0])
# state3 = np.array([0,0,1,0])
# state4 = np.array([0,0,0,1])
state4 = np.array(np.append([0]*(2**(N)-1), [1]))
probs = []
vecs = []
state = state1
for t in ts:
    U_t = expm(-1.0j * H_eff * t)
    vecs.append((U_t @ state1))
    probs.append(np.abs(np.transpose(state) @ U_t @ state)**2)

plt.plot(ts*1e6, probs)
plt.plot(ts*1e6, np.cos(j_ijs[0,1]*ts)**2)
plt.plot(ts*1e6, np.sin(j_ijs[0,1]*ts)**2)
plt.xlabel(r'time ($\mu$s)')
plt.ylabel('probability')
plt.grid()
plt.show()

plt.plot(ts*1e6, np.abs(vecs)**2)
plt.xlabel(r'time ($\mu$s)')
plt.ylabel('probability')
plt.grid()
plt.show()


# %% Phase-space trajectories for MS gate




# %% Pulse shaping for MS gate


