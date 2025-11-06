import pylab as plt
import numpy as np
import dcst

# 1 (b)

# Constants
g = 9.81
H = 0.01
A = 0.002
miu = 0.5
sig = 0.05
L = 1
delx = 0.02
N = int(L / delx)
delt = 0.01
epsilon = delt/10

t1 = 0
t2 = 1
t3 = 4
tend = t3 + epsilon


# Define functions
def eta0(x):
    return A*np.e**(-(x-miu)**2 / sig**2)


def f(u, e):
    f1 = 1/2 * u**2 + g * e
    f2 = (H + e) * u
    return np.array([f1, f2], float)


# Create array
u_arr = np.zeros(N+1, float)
eta_arr = np.zeros(N+1, float)
up = np.zeros(N+1, float)
ep = np.zeros(N+1, float)
x_arr = np.zeros(N+1, float)

for i in range(N+1):
    eta_arr[i] = eta0(i * delx)
    x_arr[i] = i * delx

# Plot T=0s
plt.plot(x_arr, u_arr, label='T=0s')

# Main loop
t = 0.0
c = delt / (2 * delx)
d = delt / delx
while t < tend:
    u_halfs = (u_arr[1:N] + u_arr[:N-1]) / 2 - c * \
              (f(u_arr[1:N], eta_arr[1:N])[0] -
               f(u_arr[:N-1], eta_arr[:N-1])[0])
    eta_halfs = (eta_arr[1:N] + eta_arr[:N-1]) / 2 - c * \
                (f(u_arr[1:N], eta_arr[1:N])[1] -
                 f(u_arr[:N-1], eta_arr[:N-1])[1])
    f_halfs = f(u_halfs, eta_halfs)
    u_halfl = (u_arr[2:N+1] + u_arr[1:N]) / 2 - c * \
              (f(u_arr[2:N+1], eta_arr[2:N+1])[0] -
               f(u_arr[1:N], eta_arr[1:N])[0])
    eta_halfl = (eta_arr[2:N+1] + eta_arr[1:N]) / 2 - c * \
                (f(u_arr[2:N+1], eta_arr[2:N+1])[1] -
                 f(u_arr[1:N], eta_arr[1:N])[1])
    f_halfl = f(u_halfl, eta_halfl)

    u_final = np.concatenate(([0], (f_halfl[0]-f_halfs[0]), [0]))
    eta_final = np.concatenate(([f(u_arr[1], eta_arr[1])[1] -
                                 f(u_arr[0], eta_arr[0])[1]],
                                (f_halfl[1]-f_halfs[1]),
                                [f(u_arr[N], eta_arr[N])[1] -
                                 f(u_arr[N-1], eta_arr[N-1])[1]]))
    up = u_arr - d * u_final
    ep = eta_arr - d * eta_final
    up, u_arr = u_arr, up
    ep, eta_arr = eta_arr, ep
    t += delt

    if abs(t-t1) < epsilon:
        plt.plot(x_arr, u_arr, label='T=0s')
    if abs(t-t2) < epsilon:
        plt.plot(x_arr, u_arr, label='T=1s')
    if abs(t-t3) < epsilon:
        plt.plot(x_arr, u_arr, label='T=4s')


# Plot
plt.xlabel('x')
plt.ylabel('u')
plt.title('Velocity at each position at T = 0s')
plt.legend()
plt.savefig('Velocity at each position at different T with constant H.pdf')
plt.show()


# 1 (c)

# Constants
g = 9.81
A = 0.0005
miu = 0
sig = 0.1
L = 1
N = 150
delx = L / N
delt = 0.001
epsilon = delt/10

t1 = 0
t2 = 1
t3 = 2
t4 = 3
t5 = 4
tend = t5 + epsilon


# Define functions
def eta0(x):
    return A*np.e**(-(x-miu)**2 / sig**2)


def h(x):
    return 0.001 + 0.1 * np.e**(-7*x)


def f(u, e, height):
    f1 = 1/2 * u**2 + g * e
    f2 = (height + e) * u
    return np.array([f1, f2], float)


# Create array
u_arr = np.zeros(N+1, float)
eta_arr = np.zeros(N+1, float)
h_arr = np.zeros(N+1, float)
up = np.zeros(N+1, float)
ep = np.zeros(N+1, float)
x_arr = np.zeros(N+1, float)

for i in range(N+1):
    eta_arr[i] = eta0(i * delx)
    h_arr[i] = h(i * delx)
    x_arr[i] = i * delx

# Plot T=0s
plt.plot(x_arr, u_arr, label='t=0')


# Main loop
t = 0.0
c = delt / (2 * delx)
d = delt / delx
while t < tend:
    u_halfs = (u_arr[1:N] + u_arr[:N-1]) / 2 - c * \
              (f(u_arr[1:N], eta_arr[1:N], h_arr[1:N])[0] -
               f(u_arr[:N-1], eta_arr[:N-1], h_arr[:N-1])[0])
    eta_halfs = (eta_arr[1:N] + eta_arr[:N-1]) / 2 - c * \
                (f(u_arr[1:N], eta_arr[1:N], h_arr[1:N])[1] -
                 f(u_arr[:N-1], eta_arr[:N-1], h_arr[:N-1])[1])
    h_halfs = (h_arr[1:N] + h_arr[:N - 1]) / 2 - c * \
                (f(u_arr[1:N], eta_arr[1:N], h_arr[1:N])[1] -
                 f(u_arr[:N - 1], eta_arr[:N - 1], h_arr[:N-1])[1])
    f_halfs = f(u_halfs, eta_halfs, h_halfs)
    u_halfl = (u_arr[2:N+1] + u_arr[1:N]) / 2 - c * \
              (f(u_arr[2:N+1], eta_arr[2:N+1], h_arr[2:N+1])[0] -
               f(u_arr[1:N], eta_arr[1:N], h_arr[1:N])[0])
    eta_halfl = (eta_arr[2:N+1] + eta_arr[1:N]) / 2 - c * \
                (f(u_arr[2:N+1], eta_arr[2:N+1], h_arr[2:N + 1])[1] -
                 f(u_arr[1:N], eta_arr[1:N], h_arr[1:N])[1])
    h_halfl = (h_arr[2:N + 1] + h_arr[1:N]) / 2 - c * \
                (f(u_arr[2:N + 1], eta_arr[2:N + 1], h_arr[2:N + 1])[1] -
                 f(u_arr[1:N], eta_arr[1:N], h_arr[1:N])[1])
    f_halfl = f(u_halfl, eta_halfl, h_halfl)

    u_final = np.concatenate(([0], (f_halfl[0]-f_halfs[0]), [0]))
    eta_final = np.concatenate(([f(u_arr[1], eta_arr[1], h_arr[1])[1] -
                                 f(u_arr[0], eta_arr[0], h_arr[0])[1]],
                                (f_halfl[1]-f_halfs[1]),
                                [f(u_arr[N], eta_arr[N], h_arr[N])[1] -
                                 f(u_arr[N-1], eta_arr[N-1], h_arr[N-1])[1]]))
    up = u_arr - d * u_final
    ep = eta_arr - d * eta_final
    up, u_arr = u_arr, up
    ep, eta_arr = eta_arr, ep
    t += delt

    if abs(t-t1) < epsilon:
        plt.plot(x_arr, eta_arr, label='t=0')
    if abs(t-t2) < epsilon:
        plt.plot(x_arr, eta_arr, label='t=1')
    if abs(t-t3) < epsilon:
        plt.plot(x_arr, eta_arr, label='t=2')
    if abs(t-t4) < epsilon:
        plt.plot(x_arr, eta_arr, label='t=3')
    if abs(t-t5) < epsilon:
        plt.plot(x_arr, eta_arr, label='t=4')


# Plot
plt.xlabel('x')
plt.ylabel('eta')
plt.title('Tsunami simulation for different t')
plt.legend()
plt.savefig('Tsunami simulation for different t.pdf')
plt.show()


# 2 (a)

# Constants
v = 100
L = 1
d = 0.1
C = 1
sigma = 0.3
h = 10e-6
w = 0.01
N = 200
delx = L / N

phi0 = 0


# Define functions
def psi(x):
    return C * (x*(L-x))/L**2 * np.e**(-(x-d)**2 / (2*sigma**2))


# Create array
phi_arr = np.zeros(N+1)
psi_arr = np.zeros(N+1)
x_arr = np.zeros(N+1)

for i in range(N):
    psi_arr[i] = psi(i * delx)
    x_arr[i] = i*delx

# Plot t=0s
plt.plot(x_arr, phi_arr, label='t=0s')

t1 = 0.002
t2 = 0.004
t3 = 0.006
t4 = 0.012
t5 = 0.1
tend = t5 + h/100

# Main loop
t = 0.0
c = h * v**2 / delx**2

while t < tend:
    phi_arr += h * psi_arr
    psi_arr[1:N] += c*(phi_arr[2:N+1]+phi_arr[0:N-1]-2*phi_arr[1:N])

    if abs(t-t1) < h/100:
        plt.plot(x_arr, phi_arr, label='t=0.002s')
    if abs(t-t2) < h/100:
        plt.plot(x_arr, phi_arr, label='t=0.004s')
    if abs(t-t3) < h/100:
        plt.plot(x_arr, phi_arr, label='t=0.006s')
    if abs(t-t4) < h/100:
        plt.plot(x_arr, phi_arr, label='t=0.012s')
    if abs(t-t5) < h/100:
        plt.plot(x_arr, phi_arr, label='t=0.1s')
    t += h

plt.xlabel('x')
plt.ylabel('phi')
plt.title('Amplitude at each position at different t')
plt.legend()
plt.savefig('Amplitude at each position at different t.pdf')
plt.show()


# 2 (c)
# Main loop
while t < tend:
    phi_temp, psi_temp = phi_arr, psi_arr
    phi_arr += h * (psi_arr + psi_temp) / 2
    psi_arr[1:N] += c*(phi_arr[2:N+1]+phi_arr[0:N-1]-2*phi_arr[1:N] +
                       phi_temp[2:N+1]+phi_temp[0:N-1]-2*phi_temp[1:N])

    if abs(t - t1) < h / 100:
        plt.plot(x_arr, phi_arr, label='t=0.002s')
    if abs(t - t2) < h / 100:
        plt.plot(x_arr, phi_arr, label='t=0.004s')
    if abs(t - t3) < h / 100:
        plt.plot(x_arr, phi_arr, label='t=0.006s')
    if abs(t - t4) < h / 100:
        plt.plot(x_arr, phi_arr, label='t=0.012s')
    if abs(t - t5) < h / 100:
        plt.plot(x_arr, phi_arr, label='t=0.1s')
    t += h

# Plot
plt.xlabel('x')
plt.ylabel('phi')
plt.title('Amplitude at each position at different t by Crank–Nicolson method')
plt.legend()
plt.savefig('Amplitude at each position at different t by Crank–Nicolson method.pdf')
plt.show()

# 2(e)
# Constants
k = np.zeros(N+1)
for i in range(N+1):
    k[i] = i
omega = v * k[1:] * np.pi / L

# Create array
an = dcst.dst(phi_arr[1:])
bn = dcst.dst(psi_arr[1:])

# Calculation
psi_fft_t1 = dcst.idst(bn * np.sin(omega * t1) / omega)
psi_fft_t2 = dcst.idst(bn * np.sin(omega * t2) / omega)
psi_fft_t3 = dcst.idst(bn * np.sin(omega * t3) / omega)
psi_fft_t4 = dcst.idst(bn * np.sin(omega * t4) / omega)
psi_fft_t5 = dcst.idst(bn * np.sin(omega * t5) / omega)
plt.plot(x_arr[:N], psi_fft_t1, label='t=0.002s')
plt.plot(x_arr[:N], psi_fft_t2, label='t=0.004s')
plt.plot(x_arr[:N], psi_fft_t3, label='t=0.006s')
plt.plot(x_arr[:N], psi_fft_t4, label='t=0.012s')
plt.plot(x_arr[:N], psi_fft_t5, label='t=0.1s')

# Plot
plt.xlabel('x')
plt.ylabel('phi')
plt.title('Amplitude at each position at different t by Fourier transformation')
plt.savefig('Amplitude at each position at different t by Fourier transformation.pdf')
plt.legend()
plt.show()

