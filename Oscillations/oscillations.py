#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as pl
from numba import autojit
import time
import sys

@autojit
def stochastic(t, eta, amplitude, frequency):
    """
    Create time series of stochastic oscillations for a given damping rate
    (eta), amplitude and frequency. From De Ridder et al. 2006
    Usage:
    t - array: time stamps given in units of seconds
    eta - float: damping rate
    amplitude - float: amplitude of oscillations
    frequency - float: frequency of oscillations
    Author: grd349
    Edited by jsk389
    """

    # Compute cadence from time stamps
    dt = (t.max()-t.min()) / float(len(t))

    # Compute time between kicks for a given damping rate
    dtkick = 1.0 / eta / 100.0

    # If time between kicks is less than cadence set equal to cadence
    if dtkick < cadence:
        dtkick = cadence

    # Standard deviation of white noise component
    sigmae = amplitude * np.sqrt(eta * dtkick)

    N_noise = np.round((t.max() - t.min()) / dtkick + 1).astype(int)

    # Compute white noise components
    bnoise = sigmae * np.random.randn(N_noise)
    cnoise = sigmae * np.random.randn(N_noise)

    bn, cn = np.zeros(N_noise), np.zeros(N_noise)

    # Amplitudes
    coeff = np.exp(-eta * dtkick)
    for i in range(N_noise):
        bn[i] = coeff * bn[i-1] + bnoise[i]
        cn[i] = coeff * cn[i-1] + cnoise[i]

    # Generate signal
    N_time = len(t)
    output = np.zeros(N_time)
    n = np.floor(t / dtkick).astype(int)

    #output = np.exp(-eta * (t - (n*dtkick))) * (\
    #         bn * np.sin(2.0*np.pi*frequency*t) + \
    #         cn * np.cos(2.0*np.pi*frequency*t))
    for i in range(N_time):
        first = bn[n[i]] * np.sin(2.0 * np.pi * frequency * t[i])
        second = cn[n[i]] * np.cos(2.0 * np.pi * frequency * t[i])
        output[i] = np.exp(-eta * (t[i] - (n[i] * dtkick))) * \
                          (first + second)
    
    return output

@autojit
def lorentzian(t, linewidth, amplitude, frequency):
    """
    It is much easier to think of oscillation parameters in terms of the
    Lorentzian profile that is seen in the power spectrum. Therefore 
    generate oscillations with respect to supplied Lorentzian profile
    parameters

    Usage:
    t - array: time stamps
    linewidth - array: linewidth of Lorentzian profile, linked to eta
                       through eta = linewidth * pi
    amplitude - array: amplitude of Lorentzian
    frequency - array: central frequency of Lorentzian (Hertz)
    """
    eta = linewidth * np.pi
    y = stochastic(t, eta, amplitude, frequency)
    return y

if __name__=="__main__":
    # Run quick example
    cadence = 40.0
    days = 100.0 * 1.0 * 73.0
    npts = days * 24.0 * 3600.0 / cadence
    linewidth = 1.0e-6
    amplitude = 100.0
    frequency = 200e-6
    t = np.linspace(0, npts*cadence, npts)
    

    s = time.time()
    y = lorentzian(t, linewidth, amplitude, frequency)
    print("Time taken for dataset of length {0} days is {1} s".format(int(days), time.time()-s))
    
