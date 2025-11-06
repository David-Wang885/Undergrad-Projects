import numpy as np
import scipy.optimize as sp
import matplotlib.pyplot as plt

# Define constants
lamb_0 = 284*10**-9  # Lambda_0 is set


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


# The Hartmann Relation
# Import and adjust data
scale_1, lamb_1_temp = np.loadtxt('Lab 3 Helium', skiprows=1, unpack=True)
lamb_1 = 1 / (lamb_1_temp * 10 ** -9 - lamb_0)  # Create y=(lambda_0-lambda)^-1


# Define function
def f1(x, a0, a1):
    return x * a0 + a1


# Fitting the curve
p_opt1, p_cov1 = sp.curve_fit(f1, lamb_1, scale_1)
p_var1 = np.diag(p_cov1)  # Get uncertainties for constant
m, b = p_opt1[0], p_opt1[1]  # Store constants


# Plot the graph
plt.plot(lamb_1, f1(lamb_1, m, b))
plt.plot(lamb_1, scale_1)
plt.title('The Hartmann Relation')
plt.xlabel('1/(Wavelength Measured - lambda_0)')
plt.ylabel('Reading of the Scale')
plt.show()


# Print out all the constant
print('m has value of ' + str(m) + ' and uncertainty of ' + str(p_var1[0]))
print('b has value of ' + str(b) + ' and uncertainty of ' + str(p_var1[1]))


# Define a function convert scale reading to wavelength
def scale_to_lambda(s):
    return m / (s - b) + lamb_0


# Check for chi square
print('The value of chi square is ' + str(chi_sq(len(scale_1), 2, 0.01, scale_1,
                                                 f1(lamb_1, m, b))))


# The Rydberg Constant
# Import and adjust data
n, scale_2 = np.loadtxt('Lab 3 Rydberg', skiprows=1, unpack=True)
lamb_2 = 1 / scale_to_lambda(scale_2)  # Create y = 1/lambda


# Define function
def f2(x, a2):
    return a2 * (1 / 4 - (1/(x ** 2)))


# Fitting the curve
p_opt2, p_cov2 = sp.curve_fit(f2, n, lamb_2)
p_var2 = np.diag(p_cov2)  # Get uncertainties for constant
rydberg = p_opt2[0]  # Store constants


# Plot the graph
plt.plot(n, f2(n, rydberg))
plt.plot(n, lamb_2)
plt.title('The Rydberg constant')
plt.xlabel('Energy level')
plt.ylabel('1 / Wavelength of the light')
plt.show()


# Print out all the constant
print('The Rydberg constant has value of ' + str(rydberg) +
      ' and uncertainty of ' + str(p_var2[0]))


# Check for chi square
print('The value of chi square is ' + str(chi_sq(len(lamb_2), 1, 1/(3*(10**-9)),
                                                 lamb_2, f2(n, rydberg))))
