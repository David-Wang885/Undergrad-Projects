import math as m
import matplotlib.pyplot as plt
import numpy as np


def f(x, y):
    distance = (x ** 2) + (y ** 2)
    return -4 * ((1 / (distance ** 6)) - (1 / (distance ** 3))) / m.sqrt(distance)


def Question1(p1_inital, p2_inital):
    step = 0.01

    partical_1_x = []
    partical_1_y = []

    partical_2_x = []
    partical_2_y = []

    p1_x = p1_inital[0]
    p1_y = p1_inital[1]

    p2_x = p2_inital[0]
    p2_y = p2_inital[1]

    distance = np.sqrt((p1_x - p2_x) ** 2 + (p1_y - p2_y) ** 2)

    p1_vx = 0 + 0.5 * step * f(p2_x - p1_x, p2_y - p1_y) * ((p2_x - p1_x) / distance)
    p1_vy = 0 + 0.5 * step * f(p2_x - p1_x, p2_y - p1_y) * ((p2_y - p1_y) / distance)

    p2_vx = 0 + 0.5 * step * f(p2_x - p1_x, p2_y - p1_y) * ((p1_x - p2_x) / distance)
    p2_vy = 0 + 0.5 * step * f(p2_x - p1_x, p2_y - p1_y) * ((p1_y - p2_y) / distance)

    for x in range(0, 100, 1):
        p1_x += step * p1_vx
        p1_y += step * p1_vy

        p2_x += step * p2_vx
        p2_y += step * p2_vy

        distance = np.sqrt((p1_x - p2_x) ** 2 + (p1_y - p2_y) ** 2)

        p1_kx = step * f(p2_x - p1_x, p2_y - p1_y) * ((p2_x - p1_x) / distance)
        p1_ky = step * f(p2_x - p1_x, p2_y - p1_y) * ((p2_y - p1_y) / distance)

        p2_kx = step * f(p2_x - p1_x, p2_y - p1_y) * ((p1_x - p2_x) / distance)
        p2_ky = step * f(p2_x - p1_x, p2_y - p1_y) * ((p1_y - p2_y) / distance)

        p1_vx += 0.5 * p1_kx
        p1_vy += 0.5 * p1_ky

        p2_vx += 0.5 * p2_kx
        p2_vy += 0.5 * p2_ky

        p1_vx += p1_kx
        p1_vy += p1_ky

        p2_vx += p2_kx
        p2_vy += p2_ky

        partical_1_x.append(p1_x)
        partical_1_y.append(p1_y)

        partical_2_x.append(p2_x)
        partical_2_y.append(p2_y)

    plt.plot(partical_1_x, partical_1_y, ".")
    plt.plot(partical_2_x, partical_2_y, ".")
    plt.title("Particle trajectory")
    plt.savefig("condition3.png")
    plt.show()


def f1(v, a, b):
    x = v[0]
    y = v[1]
    dx = 1 - (b + 1) * x + a * (x ** 2) * y
    dy = b * x - a * (x ** 2) * y
    return np.array([dx, dy], float)


def Question3(t1, t2, a, b):
    delta = 1e-10
    calculate_time = []

    dxPoints = []
    dyPoints = []
    calculate_time.append(t1)
    r = np.array([0, 0], float)
    dxPoints.append(0)
    dyPoints.append(0)
    result = r
    while calculate_time[-1] < t2:
        result, H = binary_search(result, 0.2, delta, a, b)
        calculate_time.append(calculate_time[-1] + H)
        dxPoints.append(result[0])
        dyPoints.append(result[1])
    # generate the graph and label the graph
    plt.plot(calculate_time, dxPoints, ".", color="green", label="X Calculation points")
    plt.plot(calculate_time, dxPoints, label="Concentration of x")
    plt.plot(calculate_time, dyPoints, ".", color="black",label="Y Calculation points")
    plt.plot(calculate_time, dyPoints, label="Concentration of y")
    plt.title("Concentration of x and y vs time")
    plt.legend()
    plt.xlabel("Time, unit: s")
    plt.ylabel("Concentration of chemicals")
    plt.savefig("Concentration.pdf")
    plt.show()


def value(r, H, a, b, delta):
    n = 1
    r1 = r + 0.5 * H * f1(r, a, b)
    r2 = r + H * f1(r1, a, b)

    R1 = np.empty([1, 2], float)
    R1[0] = 0.5 * (r1 + r2 + 0.5 * H * f1(r2, a, b))

    error = 2 * H * delta
    while error > H * delta and n < 7:

        n += 1
        h = H / n

        r1 = r + 0.5 * h * f1(r, a, b)
        r2 = r + h * f1(r1, a, b)
        for i in range(n - 1):
            r1 += h * f1(r2, a, b)
            r2 += h * f1(r1, a, b)

        R2 = R1
        R1 = np.empty([n, 2], float)
        R1[0] = 0.5 * (r1 + r2 + 0.5 * h * f1(r2, a, b))
        for k in range(1, n):
            epsilon = (R1[k - 1] - R2[k - 1]) / ((n / (n - 1)) ** (2 * k) - 1)
            R1[k] = R1[k - 1] + epsilon
        error = abs(epsilon[0])

    r = R1[n - 1]
    return r, error


def binary_search(inital_posistion, H, delta, a, b):
    r, error = value(inital_posistion, H, a, b, delta)
    # If the accuracy is nit achieved, use recursion to cut H by half
    if error > H * delta:
        return binary_search(inital_posistion, H / 2, delta, a, b)
    else:
        return r, H


if __name__ == "__main__":
    # Question1([2, 3], [7, 6])
    Question3(0, 20, 1, 4)
