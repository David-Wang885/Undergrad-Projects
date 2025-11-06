import numpy as np
import scipy.optimize as sp
import matplotlib.pyplot as plt


# Define constants
T0 = 273.15  # Temperature of an end with constant temperature


# Define chi square
def chi_sq(N, n, sigma, y, f):
    """Return the chi square of a function.
    N = number of observation
    n = number of parameter of the function
    sigma = array reading error or the standard deviation
    y = array of y value
    f = array of value calculated by the model function"""
    dof = N - n
    total = np.sum(np.square((y - f)/sigma))
    return total / dof


# Import and adjust data
T, V, R = np.loadtxt('Lab 2', skiprows=1, unpack=True)


# Define function
def f(x, s):
    return x * s


# Fitting the curve
p_opt, p_cov = sp.curve_fit(f, T, V)
p_var = np.diag(p_cov)  # Get uncertainties for constant
s_0 = p_opt  # Store constants


# Plot the graph
plt.plot(T, f(T, s_0))
plt.plot(T, V)
plt.title('Thermocouple')
plt.xlabel('Temperature difference (K)')
plt.ylabel('EMF (V)')
plt.show()


# Print out all the constant
print('S has value of ' + str(s_0) + ' and uncertainty of ' + str(p_var[0]))


# Define calibration curve function
def v(t, s):
    return s * t


# Check for chi square
print('The value of chi square is ' + str(chi_sq(len(V), 1, 0.001, V,
                                                 f(V, s_0))))

