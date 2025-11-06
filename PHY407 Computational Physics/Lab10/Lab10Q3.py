import numpy as np
import random
import pylab as plt

# 3(a)

# Constant
w = 2  # Integral of the weight function
N = 10000  # Number of random points


# Function
def f_a(x):  # Return the value of f(x) for question 3(a)
    return 1 / (np.sqrt(x) * (1 + np.e ** x))


def g_a(x):  # Return the value of g(x) = f(x) / w(x) for question 3(a)
    return 1 / (1 + np.e ** x)


# Create array
I_mvm = []  # Create an array for mean value method
I_is = []  # Create an array for importance sampling


# Main loop
for _ in range(100):  # Loop 100 times for each method
    total_g = 0  # Create instance to store the summation of g(x)
    total_f = 0  # Create instance to store the summation of f(x)
    for i in range(N):  # Loop N times
        temp = random.random()  # Create random float
        total_g += g_a(temp ** 2)  # Add up all the g(x)
        total_f += f_a(temp)  # Add up all the f(x)
    I_is.append(total_g * w / N)  # Append into array
    I_mvm.append(total_f / N)


# Plot
plt.hist(I_is, 10, range=[0.8, 0.88])
plt.title('Importance Sampling')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.savefig('Importance Sampling.pdf')
plt.show()

plt.hist(I_mvm, 10, range=[0.8, 0.88])
plt.title('Mean Value Method')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.savefig('Mean Value Method.pdf')
plt.show()
