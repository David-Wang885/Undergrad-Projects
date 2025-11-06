import numpy as np
import pylab as plt
import random

# 2
# Constants
J = 1
T = 1
kB = 1
size = 20
steps = 100000
beta = kB * T


# Define functions
def e(s):  # Calculate energy for a given matrix
    vertical = s[:, 1:] * s[:, :-1]
    horizontal = s[1:, :] * s[:-1, :]
    return -J * (np.sum(vertical) + np.sum(horizontal))


# Create matrix
s = np.ones([size, size], int)  # Set up a matrix with all 1
for i in range(size):  # Randomly change some of spin to -1
    for j in range(size):
        if random.random() < 0.5:
            s[i, j] = s[i, j] * -1

# Create array
eplot = []  # Create an array to store energy
mplot = []  # Create an array to store magnetization
E = e(s)  # Calculate the original energy

# Main loop
for k in range(steps):

    # Choose the spin and the move
    i = random.randrange(size)
    j = random.randrange(size)
    if random.random() < 0.5:
        dn = 1
        dE = 0
    else:
        dn = -1
        s_temp = s.copy()
        s_temp[i, j] *= dn
        dE = e(s_temp) - e(s)

    # Decide whether to accept the move
    if dE < 0 or random.random() < np.exp(-dE*beta):
        s[i, j] *= dn
        E += dE

    # Create plot for every 2000 steps
    # if k % 2000 == 0:
    #     plt.imshow(s, cmap='binary')
    #     plt.title(str(k))
    #     plt.savefig(str(k) + '.pdf')
    #     plt.show()

    # Append information into array
    mplot.append(np.sum(s))
    eplot.append(E)

plt.plot(mplot)
plt.title('Total Magnetization vs steps for T = 1')
plt.xlabel('Steps')
plt.ylabel('Total Magnetization')
plt.savefig('Total Magnetization vs steps for T = 1.pdf')
plt.show()

