import numpy as np
import scipy.optimize as sp
import matplotlib.pyplot as plt


# Define constants
q = 1.6 * 10 ** -19  # charge of one electron
k = 1.38 * 10 ** -23  # Boltzmann Constant
T = 298.5  # room temperature
T1_error = 0.1
R1_error = 0.1
A1_error = 0.1
A2_error = 0.001
V1_error = 0.001
V2_error = 0.1


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


# Question 1 Thermistor
# Import and adjust data
T1_raw, R1_raw = np.loadtxt('Lab1 Thermistor', skiprows=1, unpack=True)
T1 = 1 / (T1_raw + 273.15)
R1 = np.log(R1_raw)


# Define function
def f1(x, a_1, a_2, a_3, a_4):
    return a_1 + a_2 * x + a_3 * x**2 + a_4 * x**3


# Fitting the curve
p_opt1, p_cov1 = sp.curve_fit(f1, R1, T1)
p_var1 = np.diag(p_cov1)  # Get uncertainties for each constant
a1, a2, a3, a4 = p_opt1[0], p_opt1[1], p_opt1[2], p_opt1[3]  # Store constants


# Plot the graph
plt.plot(R1, f1(R1, a1, a2, a3, a4))
plt.plot(R1, T1)
plt.title('Thermistor')
plt.xlabel('Log of Resistance (Ohm)')
plt.ylabel('1/T (K)')
plt.show()


# Print out all the constant
for i in range(len(p_var1)):
    print('a' + str(i+1) + ' has value of ' + str(p_opt1[i]))


# Define calibration curve function
def t(r):
    return 1/(a1 + a2*np.log(r) + a3*np.log(r)**2 + a4*np.log(r)**3) - 273.15


# Check for chi square
print('The value of chi square is ' + str(chi_sq(len(T1), 5, 0.1, T1,
                                                 f1(R1, a1, a2, a3, a4))))


# Question 2 Diode
# Import and adjust data
I1, V1 = np.loadtxt('Lab1 Diode (forward)', skiprows=1, unpack=True)


# Define function
def f2(x, i0, k0):
    return i0 * (np.e**((q*x)/(k0*T)) - 1)


# Fitting the curve
p_opt2, p_cov2 = sp.curve_fit(f2, V1, I1, (0.06, 4.69*10**-22))
p_var2 = np.diag(p_cov2)
i_init = p_opt2[0]
k_init = p_opt2[1]


# # Plot the graph
plt.plot(V1, f2(V1, i_init, k_init))
plt.plot(V1, I1)
plt.title('Voltage vs Current')
plt.xlabel('Voltage (V)')
plt.ylabel('Current (I)')
plt.show()


# Print out all the constant
print('I0 is ' + str(i_init) + ' and k is ' + str(k_init))


# Define calibration curve function
def i(v):
    return i_init * (np.e ** ((q * v) / (k_init * T)) - 1)


# Check for chi square
print('The value of chi square is ' + str(chi_sq(18, 2, 0.001, I1, i(V1))))
