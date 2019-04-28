from math import *
import cmath
import numpy as np 
import sys
import cmath
import scipy.integrate
import scipy.misc
import scipy.linalg as la
import scipy.fftpack as ft
import os
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re

import operator as op

import functools
from functools import reduce
import itertools

import argparse
import pickle
import time 

from collections import defaultdict

################################################################################

# Program input:
# - delta_condition: the interval we are interested in our goodness condition
# - epsilon_condition: the probability threshold for the cavity quantum states

################################################################################

argParser = argparse.ArgumentParser(description='Delta-Epsilon condition')
argParser.add_argument('--lub',type=int,  dest='lub')
argParser.add_argument('--llb',type=int,  dest='llb')
argParser.add_argument('--protocol', type=str, default="u", dest='protocol')
argParser.add_argument('--strat', type=str, default="na", dest='strat')
argParser.add_argument('--rounds', type=int, default=8, dest='rounds')
args = argParser.parse_args()

lub = args.lub
llb = args.llb
protocol = args.protocol
rounds = args.rounds
strat = args.strat

# Constants

delta_space = np.logspace(-4, -0.053, 600)

M_D = rounds if protocol == 'u' else 4
M_D2 = rounds - M_D

########################################################################

# Parameters for integral on displacement axis for the cavity state

########################################################################

n = 102001 # Number of integration intervals on the x-axis
lb = -35.0 # Lower-bound
ub = 35.0 # Upper-bound
Delta = 0.2 # The Gaussian width defined in the GKP states

dq = (ub - lb) / (n - 1.) # The infinitesimal interval for integration

q = np.linspace(lb, ub, n)
q_large = q * (2 * np.pi ) / ((dq) **2 * n)

"""
q_sum = np.linspace(lb * 2, ub * 2, 2 * n + 1) # x axis for SUM gate simulation

FT_list = []

for p_value in q:
    p_slice = np.exp(1j * q * p_value) 
    FT_list.append(p_slice)
    
FT = np.stack(FT_list)  # Density matrix for the Fourier transformation
"""

##########################################################

# Masks for the integral of delta-epsilon condition

##########################################################

mask_one = np.ones_like(q)
mask_zero = np.zeros_like(q)

"""
# Mask for the general case where peaks are multiples of sqrt pi

int_mask_1 = np.where((q % sqrt(pi) < delta_condition) , mask_one, mask_zero)
int_mask = np.where((q % sqrt(pi) > sqrt(pi) - delta_condition) , mask_one, int_mask_1)

#  Mask for logical 0 in q space or logical 1 in p space

int_mask_2 = np.where((q % (2 * sqrt(pi)) < delta_condition) , mask_one, mask_zero)
int_mask_2pi = np.where((q % (2 * sqrt(pi)) > sqrt(pi) - delta_condition) , mask_one, int_mask_1)
"""

#####################################################################################

# The GKP states

#####################################################################################

GKP_x = np.zeros_like(q)

for t in range(-11, 12):
    # GKP_x = GKP_x +np.exp(-2 * (0.1*Delta)**2 * t**2) * np.exp(- 0.5 * (q - 2 * t * sqrt(pi))**2 / Delta**2)
    GKP_x = GKP_x + np.exp(-2 * pi * Delta ** 2 * t**2) * np.exp(- 0.5 * (q - 2 * t * sqrt(pi))**2 / Delta**2)

# GKP_x = np.exp(-0.5j*sqrt(pi)*q)*GKP_x

N = np.dot(GKP_x, np.conj(GKP_x).T) 
GKP_x = GKP_x / np.sqrt(N) 

den_x = np.multiply(GKP_x, np.conj(GKP_x))

# Calculation of the photon number 

# print("normalization:", np.dot(GKP_x, np.conj(GKP_x)))
# print("photon #:", np.dot(GKP_x, q**2 * np.conj(GKP_x)))
# print(np.dot(int_mask, den_x.T))
# print(np.dot(den_x, np.conj(np.roll(den_x, int(sqrt(pi)/dq))).T))
# plt.plot(q, den_x)
# plt.show()

p = q 
dp = dq
# GKP_p = np.matmul(FT, GKP_x)

# N = np.dot(GKP_p, np.conj(GKP_p)) 
# GKP_p = GKP_p / np.sqrt(N.real)
# den_p = np.multiply(GKP_p, np.conj(GKP_p))

# Calculation of the photon number

# print("photon #:", np.dot(GKP_p, p**2 * np.conj(GKP_p)))
# print(np.dot(int_mask, den_p.T))
# print(np.dot(den_p, np.conj(np.roll(den_p, int(0.5*sqrt(pi)/dq))).T))
# plt.plot(q, den_p)
# plt.xlim(-16, 16)
# fock_gkp = convert_to_fock(GKP_x, 100, q)
# wigsave_2D(fock_gkp, "teststate.png")
# plt.show()

#############################################################

# SUM gate simulation

#############################################################

"""

GKP_sum = np.zeros_like(q_sum)

sum_basis = np.roll(np.array(list(np.zeros(n)) + list(GKP_x)), -int(n/2))
GKP_tile = np.tile(sum_basis, (n, 1))

GKP_sum = np.sum(GKP_tile * sum_basis.T + sum_basis.T, axis = 0)
N = np.dot(GKP_sum, np.conj(GKP_sum).T) 
GKP_sum = GKP_sum / np.sqrt(N) 

den_sum = np.multiply(GKP_sum, np.conj(GKP_sum))
plt.plot(q_sum, den_sum)
plt.show()

FTsum_list = []
for p_value in q_sum:
    p_slice = exp(1j * q_sum * p_value) 
    FTsum_list.append(p_slice)

FTsum = np.stack(FTsum_list) 

p = q 
dp = dq
GKP_sump = np.zeros_like(q_sum)

GKP_sump = np.matmul(FTsum, GKP_sum)

N = np.dot(GKP_sump, np.conj(GKP_sump)) 
GKP_sump = GKP_sump / np.sqrt(N.real)
den_sump = np.multiply(GKP_sump, np.conj(GKP_sump))
# plt.plot(q_sum, den_sump)

"""

###############################################

# Helper functions for identical permutations

###############################################

class unique_element:
    def __init__(self,value,occurrences):
        self.value = value
        self.occurrences = occurrences

def perm_unique(elements):
    eset=set(elements)
    listunique = [unique_element(i,elements.count(i)) for i in eset]
    u=len(elements)
    return perm_unique_helper(listunique,[0]*u,u-1)

def perm_unique_helper(listunique,result_list,d):
    if d < 0:
        yield tuple(result_list)
    else:
        for i in listunique:
            if i.occurrences > 0:
                result_list[d]=i.value
                i.occurrences-=1
                for g in  perm_unique_helper(listunique,result_list,d-1):
                    yield g
                i.occurrences+=1

def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer//denom

########################################################################

# Helper function for generating the peaks

########################################################################

def find_psi(x):

    return_psi = [0]
    for i in range(1, len(x)):

        return_psi.append(pe_dict[x[:i]])

    return return_psi

def peak(t):
    return np.exp(- 0.5 * (q - t * sqrt(pi))**2 / Delta**2)

def peak_large(t):
    return np.exp(- 0.5 * (q_large - t * sqrt(pi))**2 / Delta**2)

def shift(k, x, phi):
    str_k = [0]*k + [1]*(len(x)-k)
    str_permu = list(perm_unique(str_k))
    rt_coe = 0 
    for i in range(len(str_permu)):
        phase = 1.0
        for m in range(len(str_permu[i])):
            if str_permu[i][m] == 1:
                phase = phase * np.exp(1j * (x[m] * pi + phi[m]))
        rt_coe += phase
    return rt_coe

"""
def shift(k, x, phi):
    str_k = [0]*k + [1]*(8-k)
    str_permu = list(perm_unique(str_k))
    rt_coe = 0 
    for i in range(len(str_permu)):
        phase = 1.0
        for m in range(len(str_permu[i])):
            if str_permu[i][m] == 1:
                phase = phase * np.exp(1j * (x[m] * pi + phi[m]))
        rt_coe += phase
    return rt_coe
"""

def shift1(k, x, phi):
    
    return_phase = 0.0

    for k_D2 in range((k + 1)/2):

        k_D = k - 2 * k_D2
        
        if k_D < 0 or k_D > M_D or k_D2 > M_D2: continue # Safety sanity check

        str_k_D2 = [1] * k_D2 + [0] * (M_D2 - k_D2)
        str_k_D  = [1] * k_D  + [0] * (M_D  - k_D)

        str_permu_k_D2 =  list(perm_unique(str_k_D2))
        str_permu_k_D  =  list(perm_unique(str_k_D))

        phase_D2 = 0.0 
        phase_D =0.0
        phase_k = 0.0

        for i in range(len(str_permu_k_D2)):
            phase_term = 1.0
            for m in range(len(str_permu_k_D2[i])):
                if str_permu_k_D2[i][m] == 1:
                    phase_term = phase_term * np.exp(1j * (x[m + M_D] * pi
                                                     + phi[m + M_D]))

            phase_D2 += phase_term 
        
        if k_D == 0:

            phase_D = 1.0

        else:
            for i in range(len(str_permu_k_D)):
                phase_term = 1.0
                for m in range(len(str_permu_k_D[i])):
                    if str_permu_k_D[i][m] == 1:
                        phase_term = phase_term * np.exp(1j * (x[m] * pi + phi[m]))

                phase_D += phase_term 

        phase_k = phase_D2 * phase_D

        return_phase += phase_k

    return return_phase

def all_x(m):
    
    return_list = []
    for k in range(m + 1):
        str_k = [0]*k + [1]*(m-k)
        str_permu = list(perm_unique(str_k))

        return_list += str_permu
    
    return return_list

##################################################

# Constatnts for adaptive & nonadaptive Phase Estimation

##################################################

psi_space = np.linspace(-pi, pi, 2000)

psi_list = [0, 0.5 * pi] * 4

x = []
psi = [0.0]

###############################################################

# Helper functions for calculating the probability, see notes

################################################################

def total_prob(x, psi):
    
    M = len(x)
    prob = 0
    for shift in range(M+1):
           
        comb = list(itertools.combinations(range(M), shift))
        prob_one_shift = 0
        for candi in comb:
            amp = -np.exp(1j*pi)
            for ind in candi:
                amp = amp * np.exp(1j*(psi[ind] + x[ind]*pi))
            prob_one_shift += amp
        
        prob += prob_one_shift.conj()*prob_one_shift
        
        
    return prob.real/2**(2*M)

def total_prob_1(x, psi):

    """
    Total probability for D2 protocol
    """

    M = len(x)
    
    prob = 0.0

    for shift in range(M_D + 2 * M_D2 +1):
        
        prob_shift = 0.0
        for md2 in range(M_D2 + 1):

            if shift < 2 * md2:
                continue

            comb_md2 = list(itertools.combinations(range(M_D2), md2))
            prob_md2 = 0.0

            for candi_md2 in comb_md2:
                amp = -np.exp(1j*pi)
                for ind in candi_md2:
                    amp = amp * np.exp(1j*(psi[ind + M_D] + x[ind + M_D]*pi))
                prob_md2 += amp

            md = shift - 2 * md2 

            comb_md = list(itertools.combinations(range(M_D), md))
            prob_md = 0.0

            for candi_md in comb_md:
                amp = -np.exp(1j*pi)
                for ind in candi_md:
                    amp = amp * np.exp(1j*(psi[ind] + x[ind]*pi))
                prob_md += amp

            prob_shift += (prob_md * prob_md2) * (prob_md * prob_md2).conj()

        prob += prob_shift

    return prob.real/2**(2*M)

def meas_prob(x, psi, psi_now):

    return total_prob(x + [0], psi + [psi_now]) 

def barbara_prob(x, psi):
   
    theta_tilde = find_theta_tilde(x, psi)
    prob_theta_tilde = 1.0
    for i in range(len(x)):
        prob_theta_tilde = prob_theta_tilde * (np.cos((psi[i] + theta_tilde) * 0.5 + x[i] * np.pi * 0.5) ** 2)

    return prob_theta_tilde

#####################################################################

# Phase estimation

#####################################################################

def find_theta_tilde(x, psi):

    int_1 = 0.0
    d_theta = psi_space[1] - psi_space[0]

    for theta_dummy in psi_space:
        term = 1.0
        for i in range(len(x)):
            term = term * np.cos((psi[i] + theta_dummy) * 0.5 + x[i] * np.pi * 0.5) ** 2

        int_1 = int_1 + np.exp(1j * theta_dummy) * term 

    int_1 = int_1 * d_theta

    theta_tilde = np.angle(int_1)

    return theta_tilde

#####################################################################

# Adaptive optimizations

#####################################################################

def pe_optimize(x, psi):

    """
    Barbara's adaptive optimization
    """
    
    max_int = 0.0
    psi_max = 0.0
    
    for psi_now in psi_space:
        
        int_1 = 1.0
        int_2 = 1.0
        for theta in psi_space:
                 
            term = 1.0
            for i in range(len(x)):
                term = term * cos((psi[i] + theta)/2+x[i]*pi/2)**2
            int_1 = int_1 + np.exp(1j*theta) * term * (cos(psi_now + theta)/2)**2
            int_2 = int_2 + np.exp(1j*theta) * term * (cos(psi_now + theta)/2+0.5*pi)**2
            
        int_1 = abs(int_1 * 2*pi/ 599)  
        int_2 = abs(int_2 * 2*pi/ 599) 
        
        if int_1 + int_2 > max_int: 
            psi_max = psi_now
            max_int = int_1 + int_2
            
    return psi_max

def dm_optimize(x, psi):

    state = find_state(x, psi)
    state_shift = np.multiply(np.exp(-1j*np.sqrt(2*np.pi))*p, state)
    psi_now = np.angle(np.dot(state_shift, state.T)) + np.pi/2

    return psi_now

######################################################

# Data analysis

######################################################

epsilon_delta_dict = {}
den_p_dict = {}

def find_state(x, psi):

    result_state = np.zeros_like(q)
    
    if protocol == "u":

        for i in range(len(x)+1):

            result_state = result_state + shift(i, x, psi) * peak(2 * i- len(x)) 
    
    elif protocol == "u2":

        for i in range(M_D + 2 * M_D2 + 1):

            result_state = (result_state
                            + shift1(i, x, psi)
                            * peak(2 * (i - (M_D + 2 * M_D2)/2)))

    N = np.dot(result_state, np.conj(result_state).T) 

    result_state = result_state / np.sqrt(N)
    
    rs_p = np.matmul(FT, result_state)
    N = np.dot(rs_p, np.conj(rs_p).T)
    rs_p = rs_p / np.sqrt(N)

    return rs_p

def analyze_state(x, psi, ifplot = False):

    result_state = np.zeros_like(q)
    rs_fft = np.zeros_like(q_large)
    
    if protocol == "u":

        for i in range(M_D + 1):

            result_state = result_state + shift(i, x, psi) * peak(2 * (i - 4)) 
            rs_fft = rs_fft + shift(i, x, psi) * peak_large(2 * (i - 4))
    
    elif protocol == "u2":

        for i in range(M_D + 2 * M_D2 + 1):

            result_state = (result_state
                            + shift1(i, x, psi)
                            * peak(2 * (i - (M_D + 2 * M_D2)/2)))

            rs_fft = (rs_fft
                      + shift1(i, x, psi)
                      * peak(2 * (i - (M_D + 2 * M_D2)/2)))

    est_phase = find_theta_tilde(x, psi)

    # Result state 

    N = np.dot(result_state, np.conj(result_state).T) 
    result_state = result_state / np.sqrt(N)
    den_result = np.multiply(result_state, np.conj(result_state))

    result_state_1 = np.exp(-0.5j * est_phase/ np.sqrt(np.pi) * q) * result_state
    
    # Result state large

    N = np.dot(rs_fft, np.conj(rs_fft).T) 
    rs_fft = rs_fft/ np.sqrt(N)

    # rs_fft_barbara_shift = np.exp(-0.5j * est_phase/ np.sqrt(np.pi) * q_large) * rs_fft # Barbara shift

    """

    # q mask
    ov_q = np.dot(den_result, np.conj(np.roll(den_result, int(sqrt(pi)/dq))).T)
    r_mask_q = int_mask_2pi
    cor_q = np.dot(r_mask_q, den_result.T)

    """
    
    # Fourier tranform for Barbara shift

    """

    rs_fft_2 = np.multiply(np.exp(np.pi * 1j * np.arange(n)), rs_fft_barbara_shift.T)
    rs_p = ft.fft(rs_fft_2)
    rs_p = np.multiply(np.exp(np.pi * 1j * np.arange(n) * (2 * pi)/ (n * (dq)**2)), rs_p.T)
    N = np.dot(rs_p, np.conj(rs_p).T)
    rs_p = rs_p / np.sqrt(N)

    den_p = np.multiply(rs_p, np.conj(rs_p))

    """

    # Fourier transform for unshifted

    rs_fft_unshifted = np.multiply(np.exp(np.pi * 1j * np.arange(n)), rs_fft.T)
    rs_p_unshifted = ft.fft(rs_fft_unshifted)
    rs_p_unshifted = np.multiply(np.exp(np.pi * 1j * np.arange(n) * (2 * pi)/ (n * (dq)**2)), rs_p_unshifted.T)
    N = np.dot(rs_p_unshifted, np.conj(rs_p_unshifted).T)
    rs_p_unshifted = rs_p_unshifted / np.sqrt(N)

    den_p_unshifted = np.multiply(rs_p_unshifted, np.conj(rs_p_unshifted))
    
    den_p = den_p_unshifted

    # den_p_unshifted = den_p

    ov_p = np.dot(den_p, np.conj(np.roll(den_p, int(0.5 * sqrt(pi)/dq))).T)

    # photon_q = np.dot(result_state, q**2 * np.conj(result_state))
    # photon_p = np.dot(rs_p, p**2 * np.conj(rs_p))

    # photon_number = 0.5 * (photon_q + photon_p) + 0.5

    # with open("photon_{}_{}_{}.txt".format(protocol, strat, rounds), "w+") as f_photon:
        
    # f_photon.write("{}\n".format(photon_number))

    if protocol == "u":

        prob = total_prob(x, psi).real

    elif protocol == "u2":

        prob = total_prob_1(x, psi).real
    
    for delta_condition in delta_space[llb: lub]:

        int_mask_1 = np.where((q % sqrt(pi) < delta_condition),
                              mask_one, mask_zero)
        int_mask = np.where((q % sqrt(pi) > sqrt(pi) - delta_condition),
                            mask_one, int_mask_1)
        
        r_mask_p = np.roll(int_mask, int(np.argmax(den_p) - len(q)/2.0))

        # r_mask_p = int_mask

        cor_p = np.dot(r_mask_p, den_p.T)

        for epsilon_condition in np.logspace(-2.5, 0, 1500):

            if cor_p.real > 1 - epsilon_condition:

                if (delta_condition, epsilon_condition) not in epsilon_delta_dict:

                    epsilon_delta_dict[(delta_condition, epsilon_condition)] = prob * 1.21

                else:

                    epsilon_delta_dict[(delta_condition, epsilon_condition)] += prob * 1.21

    if ifplot:
        
        x_string = ''.join([str(t) for t in x])
        psi_string = '_'.join(['{0:.2f}'.format(t) for t in psi])

        save_title = "{0}_{1}_prob{2:.8f}_x{3}_psi{4}".format(strat,
                                                          protocol,
                                                          prob,
                                                          x_string,
                                                          psi_string)

        plt.cla()
        plt.title(x_string)
        plt.plot(q, den_result)    
        plt.xlabel(r"$x$ basis", fontsize=25)
        plt.ylabel(r"Wave function density", fontsize=25)
        plt.savefig(save_title + "_x.png", dpi=600, figsize=((200,240)))
        
        plt.cla()
     
        plt.title(x_string)
        plt.plot(q, den_p)
        plt.xlabel(r"$p$ basis", fontsize=25)
        plt.ylabel(r"Wave function density", fontsize=25)
        plt.savefig(save_title + "_p_barbara_shift.png", dpi=600, figsize=((200,240)))

        plt.cla()

        plt.title(x_string)
        plt.plot(q, den_p_unshifted)
        plt.xlabel(r"$p$ basis", fontsize=25)
        plt.ylabel(r"Wave function density", fontsize=25)
        plt.savefig(save_title + "_p.png", dpi=600, figsize=((200,240)))
    
    return

with open("pe.p", "rb") as f_pe: # with open("pe_dm.p", "rb") as f_pe:

    pe_dict = pickle.load(f_pe)


with open("select_str.p", "rb") as f_select:

    select_x_list = pickle.load(f_select)

select_x_list = [list(x) for x in select_x_list]

sel_x = []

for x_term in select_x_list:
    sel_x.append([int(x) for x in x_term]) 

# pe_dict = {}

def run_pe(x, psi):

    if len(x) == 0:
        run_pe([0], psi)
        run_pe([1], psi)
    
    elif len(x) == rounds:       
        
        if x in sel_x:
            analyze_state(x, psi, ifplot = False)
        # est_phase(x, psi)

        return
    
    elif strat == "a":

        # Terhal adaptive

        # psi_now = dm_optimize(x, psi)
        # pe_dict[tuple(x)] = psi_now

        psi_now = pe_dict[tuple(x)]

        run_pe(x + [0], psi + [psi_now])
        run_pe(x + [1], psi + [psi_now])
    
    elif strat == "na":

        # Non-adaptive

        M = len(x)
        run_pe(x + [0], psi + [psi_list[M]])
        run_pe(x + [1], psi + [psi_list[M]])
    
    return

x = []
run_pe(x, psi)

# create plot

"""
ax = plt.subplot()
index = list(np.linspace(0, 0.1, 51) + [1.1])
bar_width = 0.2
opacity = 0.8

rects1 = ax.bar(index, my_bin, bar_width, alpha=opacity, color='black', label='PE8')

ax.set_ylabel('Probability', fontsize=6)
ax.set_ylim(0,0.6)
ax.set_aspect(1.8)
"""

# print len(all_x(8))
# with open("pe_dm.p", "wb+") as f_pe:
    # pickle.dump(pe_dict, f_pe, protocol=2)

with open("delta_epsilon_{}_{}_{}_perfect_shift_measurement_error.txt".
          format(strat, protocol, rounds),
          "w+") as f:

    for my_key in epsilon_delta_dict:

        delta_write = my_key[0]
        epsilon_write = my_key[1]
        prob_write = epsilon_delta_dict[my_key]

        f.write("{} {} {}\n".
                format(delta_write, epsilon_write, prob_write))

"""
with open("den_p_{}_{}_{}_{}.p".format(n, strat, protocol, rounds), "w+") as f_den:

    pickle.dump(den_p_dict, f_den, protocol=2)
"""
