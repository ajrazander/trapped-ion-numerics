# Simulation of ion positions

import numpy as np

import matplotlib.pyplot as plt

from scipy.integrate import odeint
from scipy.optimize import minimize, curve_fit, fmin_ncg

from qiskit.opflow import Zero

import networkx as nx
import networkx.algorithms.isomorphism as iso

from numba import jit
import time

def sort_eig(evals, evecs):
    idx = evals.argsort()[::-1]   
    evals = evals[idx]
    evecs = evecs[:,idx]
    
    return evals, evecs
# %% Constants for Yb171+

# Constants
m = 170.936323 * 1.66054e-27  # ion mass
e = 1.60217662e-19  # electron charge
epsilon_0 = 8.8541878128e-12

# Trap parameters
r0 = 250e-6 * np.sqrt(2)  # distance to rf electrodes
z0 = 670e-6  # distance to dc electrodes
kappa = 0.35  # geometric factor
Omega_t = 2*np.pi * 21e6  # rf frequency

 
# %% Ion crystal positions

# Number of ions
N = 7

# Harmonic frequencies
wx = 2 * np.pi * 0.4e6
wy = 2 * np.pi * 0.4e6
wz = 2 * np.pi * 1.1e6

# @jit(nopython=True, fastmath=True)
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
ys_0 = np.arange(0, N) * 1e-6
# ys_0 = np.array([0]*(N//2) + [1]*(N//2))
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

# # Get fine results with better initial best and many iterations
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
plt.ylim(-20, 20)
plt.xlim(-20, 20)
for i in range(N):
    plt.text(xs_f[i]*1e6 - 3, ys_f[i]*1e6 + 1, str(i))
plt.grid()
plt.gca().set_aspect('equal')
plt.show()

# %% Rotate crystal

# angle is in degrees
def rotate_crystal(xs, ys, angle):
    xs_new = np.cos(angle/180*np.pi) * xs - np.sin(angle/180*np.pi) * ys
    ys_new = np.cos(angle/180*np.pi) * ys + np.sin(angle/180*np.pi) * xs
    return xs_new, ys_new

xs_f, ys_f = rotate_crystal(xs_f, ys_f, -1)

plt.plot(xs_f*1e6, ys_f*1e6, '.', markersize=16, color='royalblue', markeredgecolor='k')
plt.title('Ion equilibrium position')
plt.xlabel('x position in micron')
plt.ylabel('y position in micron')
plt.ylim(-20,20)
plt.xlim(-20,20)
for i in range(N):
    plt.text(xs_f[i]*1e6 - 3, ys_f[i]*1e6 + 1, str(i))
plt.grid()
plt.gca().set_aspect('equal')
plt.show()

# %% Transverse ("drumhead") mode calculatiimport networkx.algorithms.isomorphism as isoon
# From Phil Richerme's paper:
# Physical Review A 94, 032320 (2016)
# "Two-dimensional ion crystals in radio-frequency traps for quantum simulation"

def normal_modes(xs, ys, wz, get_matrix=False, *constants):
    e, epsilon_0, m = constants
    scale = e**2/4/np.pi/epsilon_0 / m * 2
    
    # Add harmonic terms to matrix
    matrix = wz**2 * np.diag(np.ones(N))

    # Add Coulomb interaction (diagonal terms)
    for i in range(N):
        diag_element = 0
        for j in range(N):
            if i != j:
                diag_element += scale / ((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)**(3/2)
        matrix[i,i] -= diag_element

    # Add Coulomb interaction (cross terms)
    for i in range(N):
        for j in range(N):
            cross_term = 0
            if i != j:
                cross_term = scale / ((xs[i]-xs[j])**2+(ys[i]-ys[j])**2)**(3/2)
            matrix[i,j] += cross_term
    e_vals, e_vecs = np.linalg.eigh(matrix)
    if get_matrix:
        return e_vals, e_vecs, matrix
    else:
        return e_vals, e_vecs

# constants = (e, epsilon_0, m)
# matrix = normal_modes(xs_f, ys_f, wz, *constants)

constants = (e, epsilon_0, m)
z_vals, z_vecs, z_mat = normal_modes(xs_f, ys_f, wz, True, *constants)
# z_vals, z_vecs = sort_eig(z_vals, z_vecs)

z_freqs = np.round(np.sqrt(np.abs(z_vals))/2/np.pi)
z_freqs, np.round(z_vecs, 3)

adjusted_freqs = (z_freqs - np.max(z_freqs))/1e3
plt.vlines(0-z_freqs, [0]*N, [2]*N, color='red')
plt.vlines(0, 0, 2, color='k')
plt.vlines(z_freqs, [0]*N, [2]*N, color='blue')
plt.grid()
plt.title('Mode frequencies (red, carrier, and blue)')
plt.xlabel('MHz')
plt.show()

# Plot relative distance of modes from COM
# plt.plot([0,1,1,2], np.sort(z_freqs) / np.max(z_freqs))
# plt.grid()
# plt.show()

print('mode frequencies (MHz)', z_freqs)

# %% Plot  distance between modes

mus = np.linspace(-0.8, 0.6, 100) * 1e6 + np.max(z_freqs)
B_coefs = np.array([1/(mus**2 - freq**2) for freq in z_freqs])
B_coefs_rel_com = np.array([coef/coef[0] for coef in B_coefs])

# plt.plot(z_freqs, B_coefs)
# plt.grid()
# plt.show()

plt.plot(mus, B_coefs_rel_com.T)
plt.legend([str(np.round(freq/1e6, 3)) for freq in z_freqs])
plt.ylim([-10, 10])
plt.grid()
plt.show()

plt.plot(z_freqs, B_coefs_rel_com[:,0])
# plt.legend([str(np.round(freq/1e6, 3)) for freq in z_freqs])
plt.ylim([-10, 10])
plt.grid()
plt.show()

# %% Compute Spectral Decomposition (See chapter 1 ex 5)
# aka Build up normal mode "matricies"

from scipy.sparse import csgraph
# The MS J_ij terms are usually in a sum over mode. Here's let's rewrite that
# in terms of matrices Bs

Bs = []
B_sum =  np.zeros((N,N))
for i in range(N):
    v = z_vecs[:,i]
    v_reshaped = np.reshape(v, (-1,1))  # give column vectors 2 dimensions
    B = v_reshaped @ v_reshaped.T
    # B = []
    # for b in v:
    #     B.append(v * b)
    # # Replace diagonal with zeros
    # B = np.vstack(B)
    # np.fill_diagonal(B, 0)
    Bs.append(B)
    B_sum += z_vals[i] * B

evals_bs = [np.round(np.linalg.eigh(csgraph.laplacian(Bs[i]))[0], 4) for i in range(N)]
evecs_bs = [np.round(np.linalg.eigh(csgraph.laplacian(Bs[i]))[1], 4) for i in range(N)]

# jij = np.round(Bs[0], 5)
# evals_b, evecs_b = np.linalg.eig(csgraph.laplacian(jij))
# evals_b, evecs_b = sort_eig(evals_b, evecs_b)
# print(np.round(evals_b, 3))
# print(np.round(evecs_b, 3))

# %% Plot each subgraph from each mode

for B in Bs:
    evals, evecs = np.linalg.eigh(csgraph.laplacian(B))
    evals, evecs = sort_eig(evals, evecs)
    print(np.round(np.real(evals), 3))
    print(np.round(np.real(evecs), 3))
    # plt.plot(evecs)
    # plt.legend([str(np.round(e, 3)) for e in evals])
    # plt.grid()
    # plt.show()
    
    plt.scatter(evecs[:,-2], evecs[:,-3], color='b')
    plt.show()
    
    # G1 = nx.from_numpy_matrix(B.T)
    # nx.draw(G1, with_labels=True, pos=nx.spring_layout(G1))
    # plt.show()


# %% Target graph
# target_adjm = np.array([[0,0,1,1],[0,0,1,1],[1,1,0,0], [1,1,0,0]])  # 4 ions
# target_adjm = np.array([[0,0,1,1,0],[0,0,0,1,1],[1,0,0,0,1],[1,1,0,0,0],
#                      [0,1,1,0,0]])  # 5 ions
# target_adjm = np.array([[0,0,0,1,1,1],[0,0,1,0,1,1],[0,1,0,1,0,1], 
#                       [1,0,1,0,0,1], [1,1,0,0,0,1], [1,1,1,1,1,0]])  # 6 ions NN

# target_adjm = np.array([[0,0,0,0,0,1],[0,0,0,0,0,1],[0,0,0,0,0,1], 
#                       [0,0,0,0,0,1], [0,0,0,0,0,1], [1,1,1,1,1,0]])  # 6 ions star

# target_adjm = np.array([[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1], 
#                       [1,1,1,1,1,1], [1,1,1,1,1,1], [1,1,1,1,1,1]])  # 6 all to all

target_adjm = np.array([[0,0,1,0,1,1,0],[0,0,0,1,1,0,1],[1,0,0,1,1,0,0],
                      [0,1,1,0,1,0,0],[1,1,1,1,0,1,1],[1,0,0,0,1,0,1],
                      [0,1,0,0,1,1,0]])  # 7 ions NN

# target_adjm = np.array([[0,0,0,0,1,0,0], [0,0,0,0,0,0,0], [0,0,0,0,1,0,0],
#                       [0,0,0,0,0,0,0], [0,0,1,0,0,0,1], [0,0,0,0,0,0,1],
#                       [0,0,0,0,1,1,0]])  # 7 ions super obscure looking graph

# target_adjm = np.array([[0,0,0,1,0,0,1],[0,0,1,0,0,1,0],[0,1,0,0,0,1,0],
#                       [1,0,0,0,0,0,1],[0,0,0,0,0,0,0],[0,1,1,0,0,0,0],
#                       [1,0,0,1,0,0,0]])  # 7 ions NNN (no center) (2 triples)

# target_adjm = np.array([[0,1,0,0,0,0,0],[1,0,0,0,0,0,0],[0,0,0,0,0,0,1],
#                       [0,0,0,0,0,1,0],[0,0,0,0,0,0,0],[0,0,0,1,0,0,0],
#                       [0,0,1,0,0,0,0]])  # 7 ions NNNN (no center) (3 paris)


# target_adjm = np.array([[0,1,1,1,1,1,1],[1,0,1,1,1,1,1],[1,1,0,1,1,1,1],
#                       [1,1,1,0,1,1,1],[1,1,1,1,0,1,1],[1,1,1,1,1,0,1],
#                       [1,1,1,1,1,1,0]])  # 7 ions all-to all

# target_adjm = np.array([[0,1,0,0,1,0,1,0,0,0], [1,0,0,0,0,0,1,0,0,1],
#                         [0,0,0,1,1,1,1,0,1,0], [0,0,1,0,0,1,0,1,0,0],
#                         [1,0,1,0,0,1,1,0,0,0], [0,0,1,1,1,0,0,0,0,0],
#                         [1,1,1,0,1,0,0,0,1,0], [0,0,0,1,0,0,0,0,1,1],
#                         [0,1,1,1,0,0,1,1,0,1], [0,1,0,0,0,0,0,1,1,0]])  # 10 ions

Lt = csgraph.laplacian(target_adjm)
t_vals, t_vecs = np.linalg.eigh(Lt)
# t_vals, t_vecs = np.linalg.eigh(target_adjm)

Ts = []
T_sum = np.zeros((N,N))
for i in range(N):
    v = np.reshape(t_vecs[:,i], (-1,1))
    Ts.append(v@v.T)
    T_sum += t_vals[i] * (v@v.T)
t_vals, t_vecs = sort_eig(t_vals, t_vecs)
print(np.round(t_vals, 3))
print(np.round(t_vecs, 3))
plt.plot(t_vecs)
plt.legend([str(np.round(t, 3)) for t in t_vals])
plt.grid()
plt.show()

plt.scatter(t_vecs[:,-2], t_vecs[:,-3], color='b')
plt.show()

G2 = nx.from_numpy_matrix(target_adjm)
nx.draw(G2, with_labels=True, pos=nx.spring_layout(G2))

# %% Look at Similarity transforms between target graph and sum of Bs

# Compute transform matrix for known working example
Ps = []
for i in range(N):
    v = np.reshape(evecs[:,i], (-1,1))  # convert to 2D 
    t = np.reshape(t_vecs[:,i], (-1,1))  # convert to 2D
    # check for eigenvalues of zero
    diag_coefs = []
    for i in range(N):
        if np.round(evals[i], 3) != 0 or np.round(t_vals[i], 3) != 0:
            diag_coefs += [np.sqrt(np.abs(evals[i] / t_vals[i]))]
        else:
            diag_coefs += [0]
    P = t_vecs.T @ np.diag(diag_coefs) @ evecs
    # Pinv = np.linalg.inv(P)
    Ps.append(P)

Diag_jij = np.linalg.inv(z_vecs) @ Lt @ z_vecs
print('diag matrix', np.round(Diag_jij, 2))
# t = Pinv @ Ljij @ P

# %% Compute spectra of B matrices (D. Speilman 2019 book)


# lbs = []  # list of laplacians of Bs
# for b in Bs:
#     lbs.append(csgraph.laplacian(b))

# evals, evecs = np.linalg.eig(csgraph.laplacian(0*lbs[0] - 2*lbs[1] - 2*lbs[2] - 4*lbs[3]))
# jij = -7*Bs[0] - 5*Bs[1] - 4*Bs[2] - 4*Bs[3] - 2*Bs[4] - 2*Bs[5] - 0*Bs[6]  # 7 ions NN
# jij = -2*Bs[0] - 2*Bs[1] - 0*Bs[2] - 0*Bs[3] - 0*Bs[4] - 0*Bs[5] - 2*Bs[6]  # 7 ions NNNN no center
# jij = -3*Bs[0] - 3*Bs[1] - 0*Bs[2] - 0*Bs[3] - 0*Bs[4] - 0*Bs[5] - 3*Bs[6]  # 7 ions NNN no center
# jij = -0*Bs[0] - 0*Bs[1] - 3*Bs[2] - 3*Bs[3] - 3*Bs[4] - 3*Bs[5] - 0*Bs[6]  # 7 ions NNN no center
# jij = 0*Bs[0] - Bs[1] - Bs[2] - Bs[3] - Bs[4] - Bs[5] # 6 ions
# jij = 1*Bs[0] # 6 ions
# jij = 6*Bs[0] + 4.618*Bs[1] + 4.618*Bs[2] + 2.382*Bs[3] + 2.382*Bs[4] + 0*Bs[5] # 6 ions
# jij = 0*Bs[0] - 4.618*Bs[1] - 4.618*Bs[2] - 2.382*Bs[3] - 2.382*Bs[4] - 6*Bs[5] # 6 ions
# jij = 1*Bs[0] - 1*Bs[1] - 1*Bs[2] - 1*Bs[3] - 1*Bs[4] - 1*Bs[5] # 6 ions Star
# jij = 0*Bs[0] - 1*Bs[1] - 1*Bs[2] - 4*Bs[3] - 4*Bs[4] # 5 ions
# jij = (0*Bs[0] - 1.418*Bs[1] - 1.772*Bs[2] - 3.536*Bs[3] 
# - 4*Bs[4] - 4.14*Bs[5] - 5.136*Bs[6] - 5.279*Bs[7] - 6.163*Bs[8] - 6.556*Bs[9])  # 10 ions ~NN

jij = np.sum([-np.diag(Diag_jij)[i] * Bs[i] for i in range(N)], axis=0)

np.fill_diagonal(jij, 0)
jij = np.round(jij, 5)
Ljij = csgraph.laplacian(jij)
evals, evecs = np.linalg.eigh(Ljij)
# evals, evecs = np.linalg.eigh(jij)
# evals, evecs = sort_eig(evals, evecs)
print(np.round(evals, 3))
print(np.round(evecs, 3))
plt.plot(evecs)
plt.legend([str(np.round(e, 3)) for e in evals])
plt.grid()
plt.show()

plt.scatter(evecs[:,-2], evecs[:,-3], color='b')
plt.show()

G1 = nx.from_numpy_matrix(jij.T)
nx.draw(G1, with_labels=True, pos=nx.spring_layout(G1))
plt.show()


#%% Analyze matrix used to compute normal modes

# m_vals, m_vecs = np.linalg.eigh(csgraph.laplacian(z_mat/1e12))
# m_vals, m_vecs = sort_eig(m_vals, m_vecs)
# print(np.round(m_vals, 3))
# print(np.round(m_vecs, 3))
# plt.plot(m_vecs)
# plt.legend([str(np.round(t, 3)) for t in m_vals])
# plt.grid()
# plt.show()

# plt.scatter(m_vecs[:,-2], m_vecs[:,-3], color='b')
# plt.show()

# G = nx.from_numpy_matrix(z_mat)
# nx.draw(G, with_labels=True, pos=nx.spring_layout(G))


# %% J_ij interaction rate calculations using eigenvector matrices

def compute_Jijs(Bs, eig_vals, mu, wzs, *constants):
    omega, m, hbar, dk = constants
    prefactor = omega**2 * hbar * dk**2 / 2 / m
    num_modes = len(eig_vals)
    j_ijs = np.zeros((num_modes, num_modes))
    for i in range(num_modes):
        j_ijs += 1  / (mu**2 - eig_vals[i]**2) * Bs[i]
    return prefactor * j_ijs

mu = 2*np.pi * 10e3 + np.max(z_freqs)*2*np.pi
omega = 2*np.pi  * 110e3
hbar = 1.054571817e-34
dk = np.sqrt(2) * 2*np.pi/ 355e-9

constants = (omega, m, hbar, dk)
j_ijs = compute_Jijs(Bs, z_freqs*2*np.pi, mu, z_freqs*2*np.pi, *constants)

# Plot radial plane
i = 4
interaction_rates = j_ijs[i,:]

plt.figure(figsize=(6,6))
plt.scatter(xs_f*1e6, ys_f*1e6, c=interaction_rates/1e3, cmap='jet', marker='o', s=150)
plt.title('Ion equilibrium positions')
plt.xlabel(r'x position ($\mu$m)')
plt.ylabel(r'y position ($\mu$m)')
plt.ylim(-30,30)
plt.xlim(-30,30)
for i in range(N):
    plt.text(xs_f[i]*1e6 - 2, ys_f[i]*1e6 + 2, str(i)+'th ion')
plt.grid()
plt.colorbar(label=r'$J_{ij}$ (kHz) from the ith ion')
plt.show()

j_ijs_normed = j_ijs / np.max(j_ijs)
plt.imshow(j_ijs_normed)
plt.title('Normalized J_ijs')
plt.colorbar()
plt.show()

print('Normalized J_ijs', j_ijs_normed)

# %% J_ij interaction rate calculations using eigenvector elements

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

mu = 2*np.pi * 50e3 + np.max(z_freqs)*2*np.pi
omega = 2*np.pi  * 110e3
hbar = 1.054571817e-34
dk = np.sqrt(2) * 2*np.pi/ 355e-9

constants = (omega, m, hbar, dk)
j_ijs = compute_Jijs(z_vecs, z_freqs*2*np.pi, mu, z_freqs*2*np.pi, *constants)

# Plot radial plane
i = 2
interaction_rates = j_ijs[i,:]

plt.figure(figsize=(6,6))
plt.scatter(xs_f*1e6, ys_f*1e6, c=interaction_rates/1e3, cmap='jet', marker='o', s=150)
plt.title('Ion equilibrium positions')
plt.xlabel(r'x position ($\mu$m)')
plt.ylabel(r'y position ($\mu$m)')
plt.ylim(-30,30)
plt.xlim(-30,30)
plt.text(xs_f[i]*1e6 - 2, ys_f[i]*1e6 + 2, 'ith ion')
plt.grid()
plt.colorbar(label=r'$J_{ij}$ (kHz) from the ith ion')
plt.show()

j_ijs_normed = j_ijs / np.max(j_ijs)
plt.imshow(j_ijs_normed)
plt.title('Normalized J_ijs')
plt.colorbar()
plt.show()

print('Normalized J_ijs', j_ijs_normed)


# %% Compare distance to J_ij rate

r_ij = np.zeros((N, N))
for i in range(len(xs_f)):
    for j in range(len(ys_f)):
        if j!=i:
            r_ij[i,j] = np.sqrt((xs_f[i] - xs_f[j])**2 + (ys_f[i]-ys_f[j])**2)

def poly(rs, a, b, c):
    return a/rs**b + c

i=1
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

# N must be > 2
N = 3

sigma_x = np.array([[0,1],[1,0]])
iden = np.eye(2,2)

# compute sigma_x (tensor) sigma_x pair-wise interactions for all ion pairs
def sigma_x_ij(i,j):
    if i == 0 or j == 0:    
        sigma_x_ij = sigma_x
    else:
        sigma_x_ij = iden
    for k in range(1, N):
        if k == i:
            sigma_x_ij = np.kron(sigma_x, sigma_x_ij)
        elif k == j:
            sigma_x_ij = np.kron(sigma_x, sigma_x_ij)
        else:
            sigma_x_ij = np.kron(iden, sigma_x_ij)
    return sigma_x_ij
            
        
# compute Hamiltonian
H_eff = np.zeros((2**N, 2**N))
for i in range(N):
    for j in range(i+1, N):
        H_eff += j_ijs[i,j] * sigma_x_ij(i,j)

# Display Hamiltonian matrix
plt.imshow(H_eff, cmap='jet')
plt.title('Effective Hamiltonian')
plt.colorbar()
plt.show()

# Plot energy spectrum
e_vals, e_vecs = np.linalg.eig(H_eff)
e_vals = np.round(np.real(e_vals), 5)
e_vecs = np.round(e_vecs, 5)

normed_evals = (e_vals - np.min(e_vals)) / (np.max(e_vals) - np.min(e_vals))

plt.hlines(normed_evals,0,1)
plt.ylabel('Normalized Energy spectrum')
plt.title('Effective Hamiltonian Spectrum')
plt.grid()
plt.show()

# Ferromagnetic ground state
# 0001, 0010, 0100, 0111, 1000, 1011, 1101, 1110

# Antiferromagnetic ground state
# 0000, 0011, 0101, 0110, 1001, 1010, 1100, 1111
# 0001, 0010, 0100, 1000, 1001, 1011, 1100, 1101, 1110

# %% Evolve H_eff over time

# compute exponentiated operators
ts = np.linspace(0, 3000, 1000)*1e-6
state = (Zero^Zero^Zero).to_matrix()
probs = []
vecs = []

probs = [np.abs(np.transpose(state) @ expm(-1.0j * t * H_eff) @ state)**2 for t in ts]
plt.plot(ts*1e3, probs)
plt.xlabel('time (ms)')
plt.ylabel('prob 0000')
plt.grid()
plt.show()

probs_full = [np.abs(expm(-1.0j * t * H_eff) @ state)**2 for t in ts]
probs_full = np.array(probs_full)
# state_labels = ['0000','0001','0010','0011','0100','0101','0110','0111','1000','1001','1010','1011','1100','1101','1110','1111']
state_labels = ['000', '001', '010', '011', '100', '101', '110', '111']

# display all states evolving over time
plt.plot(ts*1e3, probs_full)
plt.xlabel('time (ms)')
plt.ylabel('probability')
plt.legend(state_labels)
plt.grid()
plt.show()

# display brightness of ions. compute sigma_z expectation value

bright_full = probs_full[:,1] + probs_full[:,2] + 2*probs_full[:,3] + probs_full[:,4] + 2*probs_full[:,5] + 2 * probs_full[:,6] + 3 * probs_full[:,7]

plt.plot(ts*1e3, bright_full)
plt.xlabel('time (ms)')
plt.ylabel('probability')
plt.grid()
plt.show()

# plt.plot(ts*1e6, probs)
# plt.plot(ts*1e6, np.cos(j_ijs[0,1]*ts)**2)
# plt.plot(ts*1e6, np.sin(j_ijs[0,1]*ts)**2)
# plt.xlabel(r'time ($\mu$s)')
# plt.ylabel('probability')
# plt.grid()
# plt.show()

# plt.plot(ts*1e6, np.abs(vecs)**2)
# plt.xlabel(r'time ($\mu$s)')
# plt.ylabel('probability')
# plt.grid()
# plt.show()

# %% Phase-space trajectories for MS gate



