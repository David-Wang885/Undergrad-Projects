import matplotlib.pyplot as plt
import numpy as np

g = 9.80
l = g / (4 * (np.pi ** 2))

w = (2*np.pi)


def f(r, t):
    x1 = r[0]
    x2 = r[1]
    dx1 = x2
    dx2 = - w ** 2 * x1
    return np.array([dx1, dx2], float)


def Question1_a():
    a = 0.0
    b = 60.0
    N = 10000
    h = (b - a) / N

    time_data = np.arange(a, b, h)
    r = np.array([1, 0], float)
    x = []
    y = []
    for t in time_data:
        x.append(r[0])
        y.append(r[1])
        k1 = h * f(r, t)
        k2 = h * f(r + 0.5 * k1, t + 0.5 * h)
        k3 = h * f(r + 0.5 * k2, t + 0.5 * h)
        k4 = h * f(r + k3, t + h)
        r += (k1 + 2 * k2 + 2 * k3 + k4) / 6

    plt.plot(y, x)
    plt.xlabel("Velocity unit: m/s")
    plt.ylabel("Displacement: cm")
    plt.title("Posistion time graph")
    plt.savefig("1a_phase.png")
    plt.show()


def van_der_pol(r, t, u):
    x1 = r[0]
    x2 = r[1]
    dx1 = x2
    dx2 = u * (1 - x1 ** 2) * dx1 - (w ** 2) * x1
    return np.array([dx1, dx2], float)


def Question1_b(u):
    a = 0.0
    b = 60.0
    N = 10000
    h = (b - a) / N

    time_data = np.arange(a, b, h)
    r = np.array([1, 0], float)
    x = []
    y = []
    for t in time_data:
        x.append(r[0])
        y.append(r[1])
        k1 = h * van_der_pol(r, t, u)
        k2 = h * van_der_pol(r + 0.5 * k1, t + 0.5 * h, u)
        k3 = h * van_der_pol(r + 0.5 * k2, t + 0.5 * h, u)
        k4 = h * van_der_pol(r + k3, t + h, u)
        r += (k1 + 2 * k2 + 2 * k3 + k4) / 6
    plt.plot(y, x)
    plt.xlabel("Velocity unit: m/s")
    plt.ylabel("Displacement unit: m")
    plt.title("Van der pol equation phase graph with µ = 0.1")
    plt.savefig("2b_1e-1_phase.png")
    plt.show()


def van_der_pol_driven(r, t, u, w2, A):
    x1 = r[0]
    x2 = r[1]
    dx1 = x2
    dx2 = u * (1 - x1 ** 2) * dx1 - (w ** 2) * x1 + A * np.sin(w2 * t)
    return np.array([dx1, dx2], float)


def Question1_c(u, w2):
    a = 0.0
    b = 60.0
    N = 10000
    h = (b - a) / N

    time_data = np.arange(a, b, h)
    r = np.array([1, 0], float)
    x = []
    y = []
    for t in time_data:
        x.append(r[0])
        y.append(r[1])
        k1 = h * van_der_pol_driven(r, t, u, w2, 8.53)
        k2 = h * van_der_pol_driven(r + 0.5 * k1, t + 0.5 * h, u, w2, 8.53)
        k3 = h * van_der_pol_driven(r + 0.5 * k2, t + 0.5 * h, u, w2, 8.53)
        k4 = h * van_der_pol_driven(r + k3, t + h, u, w2, 8.53)
        r += (k1 + 2 * k2 + 2 * k3 + k4) / 6
    plt.plot(y, x)
    plt.ylabel("Displacement unit: m")
    plt.xlabel("Velocity unit: m/s")
    plt.title("Van der pol equation with driven frequency ω2 = 0.5ω")
    plt.savefig("1c_driven_2e-1_phase.png")
    plt.show()


if __name__ == "__main__":
    Question1_c(1, 2*w)
