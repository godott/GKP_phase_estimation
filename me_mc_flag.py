"""
- This script runs a simulation of the Phase Estimation GKP prepartion protocol with the
options to insert qubit noise, systematic Kerr error and pulse shape.
"""

#######################################

# Imports

#######################################

# Python library
import random
import pickle
import argparse
import sys
import re

# Plotting related
import matplotlib
matplotlib.use("Agg")
from matplotlib import cm
import matplotlib.pyplot as plt

# Numerics related
import numpy as np
from qutip import * 
about()

########################################

# Parse input error configuration 

########################################

argParser = argparse.ArgumentParser(description='Error configuration')
argParser.add_argument('-l', '--locations', nargs='+', type=int, dest='locations', default=[])
argParser.add_argument('-e', '--errors', nargs='+', type=int, dest='errors', default=[])
argParser.add_argument('-m', '--measurements', type=str, dest='meas')
argParser.add_argument('-k', '--kerr', type=int, dest='kerr_scale')
argParser.add_argument('-i', '--ifnonlinear', type=str, dest='if_non_linear')
argParser.add_argument('-n', '--noise', type=str, dest='if_noise')
argParser.add_argument('-p', '--pulse', type=str, dest='pulse_shape')

args = argParser.parse_args()

error_locations = args.locations
location_str = ''.join([str(e) for e in error_locations])

errors = args.errors
error_str = ''.join([str(e) for e in errors])

kerr_scale = args.kerr_scale
measurement_result = args.meas

# Python so not good at parsing boolean numbers

if args.if_non_linear == "False":
    if_non_linear = False

elif args.if_non_linear == "True":
    if_non_linear = True

if args.if_noise == "False":
    if_noise = False

elif args.if_non_linear == "True":
    if_noise = True

pulse_shape =args.pulse_shape

if len(error_locations) != len(errors):

    print("Input error, dimension does not match")
    sys.exit(0)

# Initialize configuration data structure

pe_configuration = [-1] * 16

for i in range(len(error_locations)):
    pe_configuration[error_locations[i]] = errors[i]

########################################

# Constants

########################################

rounds = 4
save_path = './'

initial_prob = 1.0

n_c = 100 # Cavity level
n_q = 2 # Qubit level

hbar = 1.0

Delta = 2E3  # Detuning, MHz
g = 100 # MHz
kerr_coe = 2E-3 # Kerr coeff, MHz

if kerr_scale == 10:
    g = 56 # MHz
    kerr_coe = 2E-4 # Kerr coeff, MHz
    k = np.sqrt(1E-4) # photon loss rate, MHz
    gamma1 = np.sqrt(6.66E-3) # qubit decay rate, MHz
    gamma_phi = np.sqrt(5.0E-3) # qubit dephasing rate, MHz
    p_depol = 0.005

if kerr_scale == 100:
    g = 32 # MHz
    kerr_coe = 2E-5 # Kerr coeff, MHz
    k = np.sqrt(1E-5) # photon loss rate, MHz
    gamma1 = np.sqrt(6.66E-4) # qubit decay rate, MHz
    gamma_phi = np.sqrt(5.0E-4) # qubit dephasing rate, MHz
    p_depol = 0.001

if kerr_scale == 0:
    g = 32 # MHz
    kerr_coe = 0. # Kerr coeff, MHz
    k = 0. # photon loss rate, MHz
    gamma1 = 0. # qubit decay rate, MHz
    gamma_phi = 0. # qubit dephasing rate, MHz
    p_depol = 0.


phi = ((g ** 4) / (Delta ** 3))

if kerr_scale == 0:
    phi = 0.0

chi = g ** 2 / Delta - g ** 4 / Delta ** 3

T = np.pi / chi 
T_evo_single_displacement = 0.14 # Microsecond

# Numerical parameters 

times = np.linspace(0.0, T, 2000) # Microsecond
xvec = np.linspace(-10, 10, 500) # Axis for plotting the wigner function
maxd = 18.0
leng = np.linspace(-maxd, maxd, 10001) # x axis
dq = 2 * maxd / 10000

# Applied phase at each round

psi_list = [0, np.pi * 0.5] * 4

#######################################################

# Helper functions 

#######################################################

# a helper function to find out the eigenfunctions of an oscillators evaluated at certain points

def osc_eigen(N, pnts):
    pnts = np.asarray(pnts)
    lpnts = len(pnts)
    A= np.zeros((N, lpnts))
    A[0, :] = np.exp(-pnts **2 / 2.0) / np.pi ** 0.25
    if N == 1:
        return A
    else:
        A[1, :] = np.sqrt(2) * pnts * A[0 , :]
        for k in range(2, N):
            A[k, :] = np.sqrt(2. / k ) * pnts * A[k - 1, :] -  np.sqrt((k - 1.0) / k) * A[k-2, : ]
            
        return A
    
def convert_to_fock(state, N, x_axis):
    
    A = osc_eigen(N, x_axis)
    ret = np.dot(state.T, A[0]) * basis(N, 0)
    for i in range(1, N):
        coe = np.dot(state.T, A[i])
        ret = ret + coe * basis(N, i)
    
    return ret.unit()

def Hd1_coeff(t, args):
     
    if pulse_shape == "gaussian":

        coe = - 2.09562 * np.sqrt(2*np.pi) * chi / np.pi
        gaussian_mu = 0.5 * np.pi / chi
        gaussian_sigma = 0.1 * np.pi / chi
        expo = np.exp( - 0.5 * (t - gaussian_mu) ** 2 / gaussian_sigma ** 2)

        return coe * expo

    elif pulse_shape == "square":

        return - np.sqrt(2 * np.pi) * chi / 4

    else:
        
        print("Pulse shape not supported!")
        sys.exit(0)

###############################################

# Operators and initial states

###############################################

# Initial states, all in density matrices

q_psi0 = fock_dm(n_q, 0)

Delta_q = 0.2
q = np.linspace(-10, 10, 4000)

sqz_vacuum_q = np.exp(- q ** 2 / (2 * Delta_q **2)) /((np.pi * Delta_q **2) ** (0.25))
N = np.dot(sqz_vacuum_q.T, sqz_vacuum_q)
sqz_vacuum_q = 1 / np.sqrt(N) * sqz_vacuum_q

sqz_vacuum = convert_to_fock(sqz_vacuum_q, n_c, q)
c_psi0 = Qobj(sqz_vacuum) 
c_psi0_dm = c_psi0 * c_psi0.dag()

psi0 = tensor(q_psi0, q_psi0, c_psi0) # Initial state

##################################################

# Main functions

##################################################

def pe_protocol(input_dm, rounds, prob):

    # Input density matrix --- only cavity
    dm = input_dm
    
    if if_noise:
        dm = free_evolve(dm)

    # Cavity + ancilla Transmon
    dm = tensor(fock_dm(n_q, 0), dm)

    # Hadamard gate
    hadamard = tensor(snot(), qeye(n_c))
    dm = hadamard * dm * hadamard.dag()

    # Flag qubit
    dm = tensor(fock_dm(n_q, 0), dm)
    
    # CNOT gate
    
    # Qutip instrinsic cnot complains dimension incompatible,
    # so refine one here
    p0_np = np.array([[1,0],[0,0]], dtype=np.complex)
    p1_np = np.array([[0,0],[0,1]], dtype=np.complex)
    x_np = np.array([[0,1],[1,0]], dtype=np.complex)
    id_np = np.array([[1,0],[0,1]], dtype=np.complex)
    cn_np = np.kron(id_np, p0_np) + np.kron(x_np, p1_np)

    cn = Qobj(cn_np, dims=[[2,2],[2,2]])

    cnot_ancilla = tensor(cn, qeye(n_c))
    dm = cnot_ancilla * dm * cnot_ancilla.dag()

    # Insert error
    
    dm, prob = insert_error_first_half(dm, rounds, prob)

    # Control displacement

    dm = control_displacement_evolve(dm, rounds, if_non_linear)

    # CNOT gate

    cnot_ancilla = tensor(cn, qeye(n_c))
    dm = cnot_ancilla * dm * cnot_ancilla.dag()

    # Flag qubit measurement error

    dm, prob = insert_flag_measurement_error(dm, rounds, prob)

    # Measure flag qubit

    proj0 = tensor(fock(n_q, 0) * fock(n_q, 0).dag(), qeye(n_q), qeye(n_c))
    proj1 = tensor(fock(n_q, 1) * fock(n_q, 1).dag(), qeye(n_q), qeye(n_c))
    
    print("flag measurement prob", (dm * proj0).tr())

    prob = prob * (dm * proj0).tr()
    dm = (proj0 * dm * proj0).unit()
    dm = dm.ptrace([1,2])

    # Compensate the phase in the control displacement

    rz_m = np.array([[np.exp(-1j * 0.287854), 0], [0, np.exp(1j * 0.287854)]])
    rz_B_q = Qobj(rz_m)
    Rz_B = tensor(rz_B_q, qeye(n_c))
    dm = Rz_B * dm * Rz_B.dag()
    
    # Doing the rz gate and flipping the phase 

    dm, prob = insert_rz_error(dm, rounds, prob)

    # Hadamard gate 

    hadamard = tensor(snot(), qeye(n_c))
    dm = hadamard * dm * hadamard.dag()
    
    # Ancilla measurement error

    dm, prob = insert_ancilla_measurement_error(dm, rounds, prob) 
    
    # Measure ancilla qubit

    proj0 = tensor(fock(n_q, 0) * fock(n_q, 0).dag(), qeye(n_c))
    proj1 = tensor(fock(n_q, 1) * fock(n_q, 1).dag(), qeye(n_c))
    
    if measurement_result[rounds] == '0':
        
        print("ancilla measurement prob", (dm*proj0).tr())

        prob = (dm*proj0).tr() * prob
        dm = (proj0 * dm * proj0).unit()
        dm = dm.ptrace(1)

    else:

        print("ancilla measurement prob", (dm*proj1).tr())

        prob = (dm*proj1).tr() * prob
        dm = (proj1 * dm * proj1).unit()
        dm = dm.ptrace(1)

    if if_noise:
        dm = free_evolve(dm)

    file_save_title = "loc{}_error{}_meas{}_{}_{}pulse_noise{}_round{}_kerr{}_prob{}".format(location_str,
                                                                                             error_str,
                                                                                             measurement_result,
                                                                                             if_non_linear,
                                                                                             pulse_shape,
                                                                                             if_noise,
                                                                                             rounds,
                                                                                             kerr_scale,
                                                                                             prob)

    qsave(dm, file_save_title)
    wig_plot(dm, file_save_title)

    return dm, prob

def free_evolve(input_dm):
    
    # T_evo needs to be figured out, depend on the Hadamard(20 ns)
    # and CNOT (250 ns)

    T_evo = T_evo_single_displacement # Micro second

    evo_times = np.linspace(0.0, T_evo, 1000)

    a = destroy(n_c)
    H0 = qeye(n_c)
    H = H0

    c_ops = [k * a] # Collapse operator

    result = mesolve(H, input_dm, evo_times, c_ops, [], options=Options(nsteps=2000), progress_bar=True)
    final_dm = result.states[-1]

    return final_dm

def control_displacement_evolve(input_dm, rounds, if_non_linear):

    T_evo = T
    evo_times = np.linspace(0.0, T_evo, 2000)
    a = tensor(qeye(n_q), qeye(n_q), destroy(n_c))
    sz = tensor(qeye(n_q), sigmaz(), qeye(n_c))
    sm = tensor(qeye(n_q), destroy(n_q), qeye(n_c))
    sz_flag = tensor(sigmaz(), qeye(n_q), qeye(n_c))
    sm_flag = tensor(destroy(n_q), qeye(n_q), qeye(n_c))
    
    if if_noise:
        c_ops = [k * a, gamma1 * sm, gamma_phi * sz, gamma1 * sm_flag, gamma_phi * sz_flag] # Collapse operators
    else:
        c_ops = []

    if not if_non_linear:
        Hs = chi * a.dag() * a * sz 
    else:
        Hs = chi * a.dag() * a * sz - phi * (a.dag() * a) * (a.dag() * a) * sz - 0.5 * kerr_coe * (a.dag() * a) * (a.dag() * a)

    Hd1 = a + a.dag()
    H = [Hs, [Hd1, Hd1_coeff]]

    result = mesolve(H, input_dm, evo_times, c_ops, [], options=Options(nsteps=2000), progress_bar=True)
    final_dm = result.states[-1]

    return final_dm

def insert_error_first_half(input_dm, rounds, prob):
    
    dm = input_dm
    location = rounds * 4 + 0

    xx = tensor(sigmax(), sigmax(), qeye(n_c))
    xy = tensor(sigmax(), sigmay(), qeye(n_c))
    xz = tensor(sigmax(), sigmaz(), qeye(n_c))
    yx = tensor(sigmay(), sigmax(), qeye(n_c))
    yy = tensor(sigmay(), sigmay(), qeye(n_c))
    yz = tensor(sigmay(), sigmaz(), qeye(n_c))
    zx = tensor(sigmaz(), sigmax(), qeye(n_c))
    zy = tensor(sigmaz(), sigmay(), qeye(n_c))
    zz = tensor(sigmaz(), sigmaz(), qeye(n_c))
    xi = tensor(sigmax(), qeye(n_q), qeye(n_c))
    yi = tensor(sigmay(), qeye(n_q), qeye(n_c))
    zi = tensor(sigmaz(), qeye(n_q), qeye(n_c))
    ix = tensor(qeye(n_q), sigmax(), qeye(n_c))
    iy = tensor(qeye(n_q), sigmay(), qeye(n_c))
    iz = tensor(qeye(n_q), sigmaz(), qeye(n_c))

    if rounds == 0:
        p1_z = p_depol / 15 * (1 - p_depol / 10) + p_depol ** 2 / 450 + (1 - p_depol / 15) * p_depol / 15
        p1_x = p_depol / 15
    
    if rounds > 0:
        p1_z = 2 * p_depol / 3 * (1 - p_depol / 10) + 2 * p_depol / 3 * p_depol / 30 + (1 - 2 * p_depol / 3) * p_depol / 15
        p1_x = 2 * p_depol / 3

    p_l = p_depol / 15 * (p1_x * p1_z + p1_x * (1 - p1_z) + (1 - p1_x) * p1_z + (1 - p1_x) * (1 - p1_z))
    p_iz = p_depol / 15 * (p1_x * p1_z + p1_x * (1 - p1_z) + (1 - p1_x) * (1 - p1_z)) + (1 - p_depol) * (1 - p1_x) * p1_z
    p_xi = p_depol / 15 * (p1_x * p1_z + p1_x * (1 - p1_z) + (1 - p1_x) * (1 - p1_z)) + (1 - p_depol) * (p1_x * (1-p1_z))
    p_xz = p_depol / 15 *  ((1 - p1_x) * (1 - p1_z) + p1_x * (1 - p1_z) + (1 - p1_x) * p1_z) + (1 - p_depol) * p1_x * p1_z
    p_id = 1.0 - 12 * p_l - p_iz - p_xi - p_xz

    # For the error location after the first CNOT, we build a table
    # build the probability table

    e_table = [xx, xy, yx, yy, yz, zx, zy, zz, yi, zi, ix, iy, iz, xi, xz]
    prob_table = [p_l] * 12 + [p_iz, p_xi, p_xz]

    if pe_configuration[location] == -1:

        prob = p_id * prob
        print("no error in first half")

    else:
         
        error_index = pe_configuration[location]
        error_operator = e_table[error_index]
        dm = error_operator * dm * error_operator.dag()
        prob = prob_table[error_index] * prob
        
        print("has error in first half")
    
    return dm, prob

def insert_flag_measurement_error(input_dm, rounds, prob):

    dm = input_dm
    p_meas = 8 * p_depol / 15 * (1 - 2 * p_depol/ 3) + 2 * p_depol / 3 * (1 - 8 * p_depol / 15)
    print("flag error rate", p_meas)

    location = rounds * 4 + 1
    
    if pe_configuration[location] != -1:
        sx = tensor(sigmax(), qeye(n_q), qeye(n_c))
        dm = sx * dm * sx.dag()
    
        prob = prob * p_meas
        print("has flag measurement error")

    else:
        print("no flag measurement error")
        prob = prob * (1 - p_meas)

    return dm, prob

def insert_rz_error(input_dm, rounds, prob):
    
    dm = input_dm

    p_meas = 8 * p_depol / 15
    location = rounds * 4 + 2
    
    print("Rz error rate", p_meas)

    if pe_configuration[location] != -1:

        Rz = tensor(rz(-psi_list[rounds]), qeye(n_c))
        dm = Rz * dm * Rz.dag()
        prob = prob * p_meas
        print("rz error")

    else:
        Rz = tensor(rz(psi_list[rounds]), qeye(n_c))
        dm = Rz * dm * Rz.dag()
        prob = prob * (1 - p_meas)
    
        print("no rz error")

    return dm, prob

def insert_ancilla_measurement_error(input_dm, rounds, prob):

    dm = input_dm
    location =  rounds * 4 + 3

    p_meas = ((8 * p_depol / 15 * (1 - p_depol / 15) * (1 - 2 * p_depol / 3)
              + p_depol / 15 * (1 - 8 * p_depol / 15) * (1 - 2 * p_depol /3 )
              + (1 - 8 * p_depol / 15) * (1 - p_depol / 15) * 2 * p_depol /3)
             )

    print("ancilla measurement error rate", p_meas)

    if pe_configuration[location] != -1:

        sx = tensor(sigmax(), qeye(n_c))
        dm = sx * dm * sx.dag()
        prob = prob * p_meas
        print("has ancilla measurement error")

    else:
        prob = prob * (1 - p_meas)
        print("no ancilla measurement error")

    return dm, prob

def wig_plot(state0, title):

    match = re.compile(".*(round\d).*").search(title)
    match2 = re.compile(".*(meas\d*).*").search(title)

    if match:

        round_string = str(int(match.group(1)[-1]) + 1) + " round:"

    if match2:

        title_string = round_string + match2.group(1)[-4:]

    wig0 = wigner(state0, xvec, xvec)

    nrm = matplotlib.colors.Normalize(vmin=-0.32, vmax=0.32) 

    plt.cla()
    fig, axes = plt.subplots(1, 1, figsize=(2.4 * 3,2.4 * 3))

    cont0 = axes.contourf(xvec, xvec, wig0, 100, cmap=cm.RdBu, norm=nrm, levels=np.linspace(-0.4,0.4,80))
    lbl0 = axes.set_title(title_string, fontsize=22)

    cb0 = fig.colorbar(cont0, ax=axes)

    plt.tight_layout()
    plt.xlabel("q", fontsize=18)
    plt.xlabel("p", fontsize=18)
    plt.savefig(title+".png", dpi=300)

def arg_max(state, prob):
    """
    function to calculate the arg max (THE LOVE<3)
    """

    psi = state.full()
    q = np.linspace(-18, 18, 10001)
    leng = np.linspace(-18, 18, 10001)
    n = len(psi)
    A = osc_eigen(n, leng)
    xpsi_dm = np.dot(A.T, np.dot(psi, A))

    # xpsi = np.dot(psi.T,A)

    xpsi = np.diagonal(xpsi_dm)

    FT_list = []
    for p_value in leng:
        p_slice = np.exp(1j * leng * p_value) 
        FT_list.append(p_slice)
        
    FT = np.stack(FT_list) 
    
    ppsi = np.matmul(FT, xpsi)
    N = np.dot(ppsi, np.conjugate(ppsi.T))
    ppsi = 1/np.sqrt(N) * ppsi

    den_p = np.multiply(ppsi, np.conjugate(ppsi))

    mask_one = np.ones_like(q) # mask on the q space which is 1 everywhere
    mask_zero = np.zeros_like(q) 
    delta_condition = np.sqrt(np.pi) / 6.0
    int_mask_1 = np.where((q % np.sqrt(np.pi) < delta_condition),
                          mask_one, mask_zero)
    int_mask = np.where((q % np.sqrt(np.pi) > np.sqrt(np.pi) - delta_condition),
                        mask_one, int_mask_1)
    
    den = np.multiply(ppsi, np.conjugate(ppsi))
    
    argmax_center = -100.
    cor_p_max = -100.

    for ag in np.linspace(-np.sqrt(np.pi)/2.0, np.sqrt(np.pi)/2.0, 1000):
        """
        Compute the argmax
        """

        r_mask_p = np.roll(int_mask, int(ag / dq))

        cor_p = np.dot(r_mask_p, den.T)

        if cor_p > cor_p_max:

            argmax_center = ag
            cor_p_max = cor_p

    delta_space = np.linspace(0, np.sqrt(np.pi)/2.0, 1000)
    
    if abs(argmax_center) > np.sqrt(np.pi)/6.0:

        print("Fail! Argmax too large:", argmax_center)
        err_title = "\nloc{}_error{}_meas{}_kerr{}_prob{:.9f} argmax:{}".format(location_str,
                                                                                error_str,
                                                                                measurement_result,
                                                                                kerr_scale,
                                                                                prob,
                                                                                argmax_center)

        with open("unaccepted_states.txt", "a+") as f_un:
            f_un.write(err_title)
        sys.exit(0)
    
    else:

        epsilon_noisy = []
        
        for delta_condition in delta_space:

            int_mask_1 = np.where((q % np.sqrt(np.pi) < delta_condition),
                                  mask_one, mask_zero)
            int_mask = np.where((q % np.sqrt(np.pi) > np.sqrt(np.pi) - delta_condition),
                                mask_one, int_mask_1)
            
            # equivalent to shift back using argmax
            r_mask_p = np.roll(int_mask, int(argmax_center / dq))

            # r_mask_p = int_mask

            cor_p = np.dot(r_mask_p, den_p.T)

            epsilon_noisy.append(cor_p)

    file_save_title = "sqrt_over_2_minus_delta_one_minus_epsilon_loc{}_error{}_meas{}_{}_{}pulse_noise{}_round{}_kerr{}_prob{:.9f}".format(location_str,
                                                                  error_str,
                                                                  measurement_result,
                                                                  if_non_linear,
                                                                  pulse_shape,
                                                                  if_noise,
                                                                  rounds,
                                                                  kerr_scale,
                                                                  prob)

    with open(save_path + file_save_title + ".txt", "w+") as f:
        
        for i in range(len(delta_space)):
            f.write("{} {}\n".format(np.sqrt(np.pi) / 2.0 - delta_space[i], epsilon_noisy[i].real))

def plot_xbasis_dm(state, title):

    psi = state.full()
    n = len(psi)
    maxd = 18
    leng = np.linspace(-maxd, maxd, 10001)
    A = osc_eigen(n, leng)
    xpsi_dm = np.dot(A.T, np.dot(psi, A))
    # xpsi = np.dot(psi.T,A)
    xpsi = np.diagonal(xpsi_dm)

    N = np.dot(xpsi, np.conjugate(xpsi.T))
    xpsi = 1/np.sqrt(N) * xpsi

    plt.cla()
    fig, axes = plt.subplots(1, 1, figsize=(4.5, 4.5))
    axes.plot(leng, np.multiply(xpsi, np.conjugate(xpsi)))
    axes.set_xlim((-10,10))
    plt.title("4 rounds: 1111", fontsize=18)
    plt.xlabel(r"$q$ basis", fontsize=18)
    plt.ylabel(r"Wave function density", fontsize=18)
    plt.savefig(title + '.png', dpi=600, figsize=((200, 240)), bbox_inches='tight')
    
def plot_pbasis_dm(state, title):

    psi = state.full()
    n = len(psi)
    maxd = 18
    leng = np.linspace(-maxd, maxd, 10001)
    A = osc_eigen(n, leng)
    xpsi_dm = np.dot(A.T, np.dot(psi, A))
    # xpsi = np.dot(psi.T,A)
    xpsi = np.diagonal(xpsi_dm)

    FT_list = []
    for p_value in leng:
        p_slice = np.exp(1j * leng * p_value) 
        FT_list.append(p_slice)
        
    FT = np.stack(FT_list) 
    
    ppsi = np.matmul(FT, xpsi)
    N = np.dot(ppsi, np.conjugate(ppsi.T))
    ppsi = 1/np.sqrt(N) * ppsi

    plt.cla()
    fig, axes = plt.subplots(1, 1, figsize=(4.5, 4.5))
    axes.plot(leng, np.multiply(ppsi, np.conjugate(ppsi)))
    axes.set_xlim((-16,16))
    plt.title("4 rounds: 1111", fontsize=18)
    plt.xlabel(r"$q$ basis", fontsize=18)
    plt.ylabel(r"Wave function density", fontsize=18)
    plt.savefig(title + '.png', dpi=600, figsize=((200, 240)), bbox_inches='tight')
    plt.savefig(title + '_p.png')

def analyze_state(state):

    psi = state.full()
    n = len(psi)
    maxd = ceil(18)
    leng = np.linspace(-maxd, maxd, 10001)
    A = osc_eigen(n, leng)
    xpsi_dm = np.dot(A.T, np.dot(psi, A))
    # xpsi = np.dot(psi.T,A)
    xpsi = np.diagonal(xpsi_dm)

    FT_list = []
    for p_value in leng:
        p_slice = np.exp(1j * leng * p_value) 
        FT_list.append(p_slice)
        
    FT = np.stack(FT_list) 
    
    ppsi = np.matmul(FT, xpsi)
    N = np.dot(ppsi, np.conjugate(ppsi.T))
    ppsi = 1/np.sqrt(N) * ppsi

    den_p = np.multiply(ppsi, np.conjugate(ppsi))

    delta_space = np.logspace(-3, -0.053, 1000)
    
    mask_one = np.ones_like(leng)
    mask_zero = np.zeros_like(leng)

    epsilon_plot = []

    for ii, delta_condition in enumerate(delta_space[700:-1]):

        int_mask_1 = np.where((leng % (np.sqrt(np.pi)) < delta_condition),
                              mask_one, mask_zero)
        int_mask = np.where((leng % (np.sqrt(np.pi)) > (np.sqrt(np.pi) - delta_condition)),
                            mask_one, int_mask_1)

        r_mask_p = np.roll(int_mask, int(np.argmax(den_p) - len(q)/2.0))

        # r_mask_p = int_mask

        cor_p = np.dot(r_mask_p, den_p.T)
        epsilon_plot.append(cor_p) 
     
        plt.cla()

    plt.title(str(len(measurement_result)) + " rounds:" + measurement_result, fontsize=18, **csfont)
    plt.plot(leng, den_p)
    plt.xlabel(r"$p$ basis")
    plt.ylabel(r"Wave function density")
    plt.savefig(save_path + "p_axis" + measurement_result + ".png", dpi=600, figsize=((200,240)), bbox_inches='tight')

    plt.cla()

    plt.title(str(len(measurement_result)) + " rounds:" + measurement_result, fontsize=18, **csfont)
    plt.plot(np.sqrt(np.pi) / 2 - delta_space[700:-1], epsilon_plot)
    plt.xlabel(r"$\frac{\sqrt{\pi}}{2} - \delta$",fontsize=18, **csfont)
    plt.ylabel(r"$1 - \epsilon$",fontsize=18, **csfont)
    plt.savefig(save_path + "delta_epsilon" + measurement_result + ".png", dpi=600, figsize=((200,240)), bbox_inches='tight')

    """
    with open(save_path + "/delta_epsilon" + x_string + ".txt", "w+") as f:
        for i in range(len(epsilon_plot)):
            f.write(str(np.sqrt(pi) / 2 - delta_space[i+llb])+ " "+ str(epsilon_plot[i].real)+ "\n")
    """
    return

############################

# Main code

############################

dm = c_psi0_dm

for i in range(4):
    dm, initial_prob = pe_protocol(dm, i, initial_prob)

final_prob = initial_prob
arg_max(dm, final_prob)
plot_xbasis_dm(dm, "x_basis")
plot_pbasis_dm(dm, "p_basis")
