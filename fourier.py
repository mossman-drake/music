import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import math

def fourier_single_point(signal_y, signal_x, frequency):
    exponent = signal_x * -2j * math.pi * frequency
    fourier_internals = np.multiply(signal_y, np.exp(exponent))
    return integrate.trapezoid(fourier_internals, dx=signal_x[1]-signal_x[0])

#Signal in Time-Domain
SAMPLES_PER_SEC = 48000
# Time values for 1.5s of a 48kHz sample
x_vals = np.linspace(start=0, stop=1.5, num=int(SAMPLES_PER_SEC*1.5))
# Single-Frequency signal at 314 Hz
sample = np.sin(314*(2*np.pi)*x_vals)

# Frequency Domain
min_val = 0
max_val = 400
step = 0.1
frequencies = np.linspace(start=min_val, stop=max_val, num=int((max_val-min_val)/step+1))

fourier_transform = [fourier_single_point(sample, x_vals, f) for f in frequencies]
plt.plot(frequencies, np.abs(fourier_transform), alpha=0.5)
plt.show()
