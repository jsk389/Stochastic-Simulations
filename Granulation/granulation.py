#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as pl

"""
Create time series of granulation for a given rms amplitude (sigma) and 
characteristic timescale (tau). From De Ridder et al. 2006.
"""

def granulation(time, sigma, tau):
    """
    Create granulation time series
    Usage:
    time - array: time stamps
    sigma - float: rms amplitude (ppm)
    tau - float: characteristic timescale (s)
    Author: grd349
    Edited by jsk389
    """
    
    # Compute cadence from time series - assuming regular sampling
    dt = (time.max()-time.min()) / float(len(time))

    # Calculate white noise component
    temp = sigma * np.sqrt(dt / tau) * \
                np.random.randn(len(time))

    anoise = np.zeros(len(time))
    anoise[0] = temp[0]
    
    # Generate signal
    for i in range(len(time)):
        anoise[i] = np.exp(-dt/tau) * anoise[i-1] + temp[i]

    return anoise

if __name__=="__main__":

    # Quick demo script
    dt = 40.0 # cadence
    n_days = 365.0
    N = (n_days * 86400.0) / dt    
    time = np.linspace(0, N*dt, N)

    sigma = 52 #(ppm)
    tau = 206 # s

    # Generate time series
    gran = granulation(time, sigma, tau)

    # Plot
    pl.plot(time, gran, 'k')
    pl.xlabel(r'Time (s)')
    pl.ylabel(r'Normalised Flux (ppm)')
    pl.show()

