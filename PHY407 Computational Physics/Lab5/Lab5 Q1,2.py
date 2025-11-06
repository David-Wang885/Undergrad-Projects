import pylab as plt
import numpy as np
import math
import cmath

# 1
x_temp, y = np.loadtxt("sp500c.csv", float, delimiter=',', skiprows=1, unpack=True)  # load text from file
# 1 (a) create an array of number to fill up the gap created by weekend and holidays
x = np.arange(0, len(x_temp), 1)

# 1(c) keep only first 10% of the Fourier coefficients are non-zero
c = np.fft.rfft(y)  # Fourier transformation
c_temp = np.zeros(len(c), complex)
for j in range(len(c)//10):  # keep only first 10% of Fourier coefficients
    c_temp[j] = c[j]
z_temp = np.fft.irfft(c_temp)  # Inverse Fourier transformation
z = np.zeros(len(x)-1)
for i in range(len(z)):
    z[i] = z_temp[i]
x_z = x[:-1]

# 1(d) taking low-pass filter, taking away all the components with period smaller than 6 months
n = x_temp[-1] // 182
c_low = np.zeros(len(c), complex)
for j in range(int(n)):
    c_low[j] = c[j]
z_low = np.fft.irfft(c_low)
z_low_plot = np.zeros(len(x)-1)
for i in range(len(z)):
    z_low_plot[i] = z_low[i]

# 1(e) taking high-pass filter, taking away all the components with period more than 1 week
n = x_temp[-1] // 7
c_high = np.zeros(len(c), complex)
for j in range(int(n), len(c)):
    c_high[j] = c[j]
z_high = np.fft.irfft(c_high)
z_high_plot = np.zeros(len(x)-1)
for i in range(len(z)):
    z_high_plot[i] = z_high[i]


# plotting all the graphs
plt.plot(x, y, label='Original data')
plt.plot(x_z, z, label='FFT')
plt.plot(x_z, z_low_plot, label='Low pass filter of 6 months')
plt.plot(x_z, z_high_plot, label='High pass filter of 1 week')
plt.title('S&P 500 stock index from late 2014 until 2019')
plt.xlabel('Business day')
plt.ylabel('Closing value')
plt.legend()
plt.savefig('Question 1.pdf')
plt.show()


# 2
time, L_S, A, B, C, D, E = np.loadtxt("msl_rems.csv", float, delimiter=',', skiprows=1, unpack=True)
dt = time[1] - time[0]


def cosine(array, t):
    """Return the amplitude, period and the phase of the given data points"""
    c_arr = np.fft.fft(array)  # get the fft
    f_temp = np.fft.fftfreq(len(array), t)  # get the frequencies
    peak = 0
    while abs(c_arr)[peak] != max(abs(c_arr)):
        peak += 1
    f = f_temp[peak]
    amplitude = 2 * max(abs(c_arr)/len(array))
    phase = max(np.arctan2(-c_arr.imag, c_arr.real))
    return amplitude, 1/f, phase


def cosine_wave(t, amplitude, period, phase):
    """Generate a sine wave with known amplitude, period, and phase"""
    return amplitude * np.cos((t/period + phase) * 2 * np.pi)


test = cosine_wave(np.arange(0, 100, 0.1), 1, 0.6, 0)
amp, per, pha = cosine(test, 0.1)
print(amp, per, pha)

# Question 2 we solve the part when getting amplitude, period and phase in the
# last minute so the rest are incomplete. We've tried our best.


