# Simulation of ion positions

import numpy as np

import matplotlib.pyplot as plt

from scipy.integrate import odeint
from scipy.optimize import minimize, curve_fit

from numba import jit
import time

# %% Constants for Yb171+

# Constants
epsilon_0 = 8.8541878128e-12
m = 170.936323 * 1.66054e-27  # ion mass
e = 1.60217662e-19  # electron charge

# Trap parameters
r0 = 250e-6 * np.sqrt(2)  # distance to rf electrodes
z0 = 670e-6  # distance to dc electrodes
kappa = 0.35  # geometric factor
Omega_t = 2*np.pi * 21e6  # rf frequency


# %% Axial confinement (z-direction)

z_i = 1e-9  # initial ion position
vz_i = 0  # initial ion speed

U0 = 24.0

omega_z = np.sqrt((2*e*kappa*U0) / (m*z0**2))
print('axial harmonic frequency', omega_z/2/np.pi)

# U is a vector such that z=U[0] and y=U[1]; returns [z', y']
def dU_dt(U, t):
    return [U[1], -omega_z**2*U[0]]

U_initial = [z_i, vz_i]
ts = np.linspace(0, 10, 800)*1e-6
Us = odeint(dU_dt, U_initial, ts)
zs = Us[:,0]

plt.plot(ts, zs)
plt.xlabel("t")
plt.ylabel("z")
plt.show()

plt.plot(ts * 1e6, np.max(zs) * np.cos(omega_z * ts) * 1e9, '-b')
plt.plot(ts * 1e6, zs * 1e9, '.r')
plt.legend(['theory', 'ODE'])
plt.xlabel(r'time ($\mu$s')
plt.ylabel('distance (nanometers)')
plt.title('z-axis motion')
plt.grid()
plt.show()

# %% Radial confinement

x_i = 1e-6  # initial ion position
vx_i = 0  # initial ion speed

V0 = 340.0

a = (4*e*kappa*U0) / (m*z0**2*Omega_t**2)
qx = (2*e*V0) / (Omega_t**2*m*r0**2)

def dU_dt(U, chi):
    return [U[1], (2*qx*np.cos(2 * chi) - a)*U[0]]

U_i = [x_i, vx_i]
chis = Omega_t * ts / 2
Us = odeint(dU_dt, U_i, chis)
xs = Us[:,0]

omega_x = np.sqrt(e/m * (qx*V0/4/r0**2 - kappa*U0/z0**2))

print('expected secular frequency (MHz)', omega_x/2/np.pi*1e-6)

plt.plot(ts * 1e6, np.max(xs) * np.cos(omega_x * ts) * 1e9, '-b')
plt.plot(ts * 1e6, xs * 1e9, '-r')
plt.xlabel(r'time ($\mu$s)')
plt.ylabel('distance (nanometers)')
plt.title('x-axis motion')
plt.grid()
plt.show()

# %% Ion crystal positions

# number of ions
N = 7

# harmonic frequencies
wx = 2 * np.pi * 0.35e6
wy = wx*1
wz = 2 * np.pi * 0.895e6
print('alpha', wz/wx)

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

params = [wx, wy, wz, N, m, e, epsilon_0]
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

xs_f, ys_f = rotate_crystal(xs_f, ys_f, 2)

plt.figure(figsize=(6,6))
plt.scatter(xs_f*1e6, ys_f*1e6, marker='o',s = 100)
plt.title('Ion equilibrium position')
plt.xlabel('x position in micron')
plt.ylabel('y position in micron')
plt.ylim(-30,30)
plt.xlim(-30,30)
plt.grid()
plt.show()
