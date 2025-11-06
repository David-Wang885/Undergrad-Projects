import pylab as plt
import numpy as np


# 2(a)
# constant
m = 9.1094e-31       # mass of electron
hbar = 1.0546e-34    # planck's constant over 2*pi
e = 1.6022e-19       # charge of electron
L = 5.2918e-11       # bohr radius
V0 = 50 * e
a = 10e-11
N = 1000
h = L/N


# Function
# define potential function
def v(x):
    return V0 * x ** 2 / a ** 2


# define differential equation
def f(r, x, en):
    psi = r[0]
    phi = r[1]
    fpsi = phi
    fphi = (2 * m / hbar ** 2) * (v(x) - en) * psi
    return np.array([fpsi, fphi], float)


# function to calculate the wave function for a particular energy
def solve(en):
    psi = 0.0
    phi = 1.0
    r = np.array([psi, phi], float)

    for x in np.arange(0, L, h):
        k1 = h * f(r, x, en)
        k2 = h * f(r + 0.5 * k1, x + 0.5 * h, en)
        k3 = h * f(r + 0.5 * k2, x + 0.5 * h, en)
        k4 = h * f(r + k3, x + h, en)
        r += (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return r[0]


# Function to get the n-th excited state energy by put in ground state energy
def get(en, n):
    return en * (1 + 2 * n)


# Main program to find the energy using the secant method
E1 = 0.0
E2 = e
psi2 = solve(E1)

target = e / 1000
while abs(E1 - E2) > target:
    psi1, psi2 = psi2, solve(E2)
    E1, E2 = E2, E2 - psi2 * (E2 - E1) / (psi2 - psi1)

print("E0 =", E2 / e, "eV")
print("E1 =", get(E2 / e, 1), "eV")
print("E2 =", get(E2 / e, 2), "eV")


# 2(b)
# Function
def v_anh(x):
    return V0 * x ** 4 / a ** 4


def f_anh(r, x, en):
    psi = r[0]
    phi = r[1]
    fpsi = phi
    fphi = (2 * m / hbar ** 2) * (v_anh(x) - en) * psi
    return np.array([fpsi, fphi], float)


# Calculate the wave function for a particular energy
def solve_anh(en):
    psi = 0.0
    phi = 1.0
    r = np.array([psi, phi], float)

    for x in np.arange(0, L, h):
        k1 = h * f_anh(r, x, en)
        k2 = h * f_anh(r + 0.5 * k1, x + 0.5 * h, en)
        k3 = h * f_anh(r + 0.5 * k2, x + 0.5 * h, en)
        k4 = h * f_anh(r + k3, x + h, en)
        r += (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return r[0]


# Main program to find the energy using the secant method
E1 = 0.0
E2 = e
psi2 = solve_anh(E1)


target = e / 1000
while abs(E1 - E2) > target:
    psi1, psi2 = psi2, solve_anh(E2)
    E1, E2 = E2, E2 - psi2 * (E2 - E1) / (psi2 - psi1)

anh_ground = E2
anh_1 = get(E2, 1)
anh_2 = get(E2, 2)
print("E0 =", anh_ground / e, "eV")
print("E1 =", get(E2 / e, 1), "eV")
print("E2 =", get(E2 / e, 2), "eV")


# 2(c)
# Array setup for integration
arr_x = np.arange(-5 * a, 5 * a, h)


# Get an array of psi for different energy
def get_psi(en):
    arr_psi = []
    r_init = np.array([0.0, 1.0], float)
    for x in arr_x:
        arr_psi.append(r_init[0])
        k1 = h * f_anh(r_init, x, en)
        k2 = h * f_anh(r_init + 0.5 * k1, x + 0.5 * h, en)
        k3 = h * f_anh(r_init + 0.5 * k2, x + 0.5 * h, en)
        k4 = h * f_anh(r_init + k3, x + h, en)
        r_init += (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return arr_psi


# Get arrays for 3 energy levels
psi_g = get_psi(anh_ground)
psi_1 = get_psi(anh_1)
psi_2 = get_psi(anh_2)


# Define normalization function
def norm(arr):
    n = len(arr) // 2
    total = 0
    for i in range(1, n, 2):
        total += h * 4 * abs(arr[i]) ** 2 / 3
    for j in range(2, n, 2):
        total += h * 2 * abs(arr[j]) ** 2 / 3
    return np.array(arr) / (total * 2)


# Plot 3 graphs of different energy levels
plt.plot(arr_x, norm(psi_g))
plt.title('Wave function of ground state')
plt.xlabel('Position')
plt.ylabel('Psi in log scale')
plt.yscale('log')
plt.savefig('Wave function of ground state.pdf')
plt.show()

plt.plot(arr_x, norm(psi_1))
plt.title('Wave function of 1st excited state')
plt.xlabel('Position')
plt.ylabel('Psi in log scale')
plt.yscale('log')
plt.savefig('Wave function of 1st excited state.pdf')
plt.show()

plt.plot(arr_x, norm(psi_2))
plt.title('Wave function of 2nd excited state')
plt.xlabel('Position')
plt.ylabel('Psi in log scale')
plt.yscale('log')
plt.savefig('Wave function of 2nd excited state.pdf')
plt.show()

