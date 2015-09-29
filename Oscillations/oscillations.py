#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as pl

"""
Create time series of stochastic oscillations for a given damping rate
(eta), amplitude and frequency. From De Ridder et al. 2006
"""

def stochastic(t, eta, amplitude, frequency):
    """
    Usage:
    t - array: time stamps given in units of seconds
    eta - float: damping rate
    amplitude - float: amplitude of oscillations
    frequency - float: frequency of oscillations
    """

    # Compute cadence from time stamps
    cadence = t[1] - t[0]

    # Compute time between kicks for a given damping rate
    dtkick = 1.0 / eta / 100.0

    # If time between kicks is less than cadence set equal to cadence
    if dtkick < cadence:
        dtkick = cadence

    # Standard deviation of white noise component
    sigmae = amplitude * np.sqrt(eta * dtkick)

    npts_noise = np.round((t.max() - t.min()) / dtkick + 1)

    # Compute white noise components
    bnoise = sigmae * np.random.randn(npts_noise)
    cnoise = sigmae * np.random.randn(npts_noise)

    bn = np.zeros(npts_noise)
    cn = np.zeros(npts_noise)

    # Amplitudes
    coeff = np.exp(-eta * dtkick)
    for i in np.arange(npts_noise):
        bn[i] = coeff * bn[i-1] + bnoise[i]
        cn[i] = coeff * cn[i-1] + cnoise[i]

    # Generate signal
    npts_time = t.shape[0]
    output = np.zeros(npts_time)
    for i in np.arange(npts_time):
        n = np.floor(t[i] / dtkick)
        output[i] = np.exp(-eta * (t[i] - (n * dtkick))) * \
	        (bn[n] * np.sin(6.28318 * frequency * t[i]) + \
		        cn[n] * np.cos(6.28318 * frequency * t[i]))
    
    return output

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
    days = 100.0 * 1.0
    npts = days * 24.0 * 3600.0 / cadence
    linewidth = 1.0e-6
    amplitude = 100.0
    frequency = 200e-6
    time = np.linspace(0, npts*cadence, npts)

    y = lorentzian(time, linewidth, amplitude, frequency)

    pl.figure(1)
    pl.plot(time, y, 'k')
    pl.xlabel(r'Time (s)')
    pl.ylabel(r'Normalised Flux (ppm)')
    pl.show()
