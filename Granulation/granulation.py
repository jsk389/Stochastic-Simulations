#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as pl

"""
Create time series of granulation for a given rms amplitude (sigma) and 
characteristic timescale (tau). From De Ridder et al. 2006.
"""

def granulation(t, sigma, tau):
    """
    Create granulation time series
    Usage:
    t - array: time stamps
    sigma - float: rms amplitude (ppm)
    tau - float: characteristic timescale (s)
    """
    
    # Compute cadence from time series - assuming regular sampling
    cadence = t[1]-t[0]

    # Calculate white noise component
    tempnoise = sigma * np.sqrt(cadence / tau) * \
                np.random.randn(t.shape[0])

    anoise = np.zeros(t.shape[0])
    anoise[0] = tempnoise[0]
    
    # Generate signal
    for i in np.arange(t.shape[0])
        anoise[i] = np.exp(-cadence/tau) * anoise[i-1] + tempnoise[i]

    return anoise

if __name__=="__main__":

    # Quick demo script
    cadence = 40.0
    days = 365.0 * 1.0
    npts = days * 24.0 * 3600.0 / cadence    
    t = np.arange(0, npts*cadence, cadence)

    sigma = 52 #(ppm)
    tau = 206 # s

    # Generate time series
    gran = granulation(t, sigma, tau)

    # Plot
    pl.plot(t, gran, 'k')
    pl.xlabel(r'Time (s)')
    pl.ylabel(r'Normalised Flux (ppm)')
    pl.show()

