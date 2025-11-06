import numpy as np
import scipy.optimize as sp
import matplotlib.pyplot as plt


# Define constants
L = 0.99157  # distance between two pivots in meter


# Import and adjust data
reading_up, period_up = np.loadtxt('Up', skiprows=1, unpack=True)
reading_down, period_down = np.loadtxt('Down', skiprows=1, unpack=True)
reading = (43.408 - reading_up) / 100


# Plot the readings
plt.plot(reading, period_up, "*-", label="Up")
plt.plot(reading, period_down, "*-", label="Down")
plt.title('Reading on the fine adjustment vs period of 8 oscillation')
plt.xlabel('Reading (cm)')
plt.ylabel('Period (s)')
plt.legend()
plt.savefig('Reading on the fine adjustment vs period of 8 oscillation.pdf')
plt.show()


# Define function for fitting
def f(a, b, x):
    return a*x + b


# Fit the data and output them
p_opt1, p_cov1 = sp.curve_fit(f, reading, period_up)
p_var1 = np.diag(p_cov1)
p_opt2, p_cov2 = sp.curve_fit(f, reading, period_down)
p_var2 = np.diag(p_cov2)

plt.plot(reading_up, f(p_opt1[0], p_opt1[1], reading_up), label="Up")
plt.plot(reading_up, f(p_opt2[0], p_opt2[1], reading_up), label="Down")
plt.title('Reading on the fine adjustment vs period of 8 oscillation by curve fitting')
plt.xlabel('Reading (cm)')
plt.ylabel('Period (s)')
plt.legend()
plt.savefig('Reading on the fine adjustment vs period of 8 oscillation by curve fitting.pdf')
plt.show()


# Define system of equation for fsolve and output the Kater Period and 'g'
def solve(x):
    f1 = p_opt1[0] * x[0] + p_opt1[1] - x[1]
    f2 = p_opt2[0] * x[0] + p_opt2[1] - x[1]
    return [f1, f2]


x = sp.fsolve(solve, np.array([0, 16]))
kp = x[1]
g = (2 * np.pi) ** 2 * L / kp ** 2
print("The Kater period is ", kp, " second")
print("The value of g is ", g, " m/s^2")
