import numpy as np
import scipy.optimize as sp
import matplotlib.pyplot as plt


# Define constants
length = 0.265  # in meter
f_twenty = 0.0001993
f_thirty = 0.0002989

# Import and adjust data
unloaded_r, unloaded_a = np.loadtxt('Unloaded', skiprows=1, unpack=True)
twenty_r, twenty_a = np.loadtxt('20 mg', skiprows=1, unpack=True)
thirty_r, thirty_a = np.loadtxt('30 mg', skiprows=1, unpack=True)

# Adjust the unit of the readings into meter
unloaded_r = (unloaded_r - unloaded_r[0])/100
twenty_r = (twenty_r - twenty_r[0])/100
thirty_r = (thirty_r - thirty_r[0])/100

# Plot the readings
plt.plot(unloaded_a, unloaded_r, "*-", label='Unloaded')
plt.plot(twenty_a, twenty_r, "*-", label='20mg')
plt.plot(thirty_a, thirty_r, "*-", label='30mg')
plt.title('Current verse Distance between two rods')
plt.xlabel('Current (A)')
plt.ylabel('Distance (m)')
plt.legend()
plt.savefig('Current verse Distance between two rods.pdf')
plt.show()


# Define function for fitting
def d_twenty(i, miu):
    return miu * length * i**2 / (f_twenty * 2 * np.pi)


def d_thirty(i, miu):
    return miu * length * i**2 / (f_thirty * 2 * np.pi)


# Fit the data and output them
p_opt1, p_cov1 = sp.curve_fit(d_twenty, twenty_a, twenty_r)
p_var1 = np.diag(p_cov1)
print('The estimation of miu when the weight is 20 mg is', p_opt1[0], '. The uncertainty is ', p_var1[0])

p_opt2, p_cov2 = sp.curve_fit(d_thirty, thirty_a, thirty_r)
p_var2 = np.diag(p_cov2)
print('The estimation of miu when the weight is 20 mg is', p_opt2[0], '. The uncertainty is ', p_var2[0])
