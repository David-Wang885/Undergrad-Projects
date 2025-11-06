import pylab as plt
import numpy as np


# 2 (c)

d = 10 ** -4
b = 10 ** -4
beta = 0.2
alpha = 0.1
gamma = 0.1
vi = 10 ** -3


def f(v, t):
    """Return the value of f
    """
    s = v[0]
    e = v[1]
    i = v[2]
    er = v[3]
    n = s + e + i + er
    lamb = beta * i / n
    fs = b * n + vi * er - (d + lamb) * s
    fe = lamb * s - (d + alpha) * e
    fi = alpha * e - (d + gamma) * i
    fr = gamma * i - (d + vi) * er
    return np.array([fs, fe, fi, fr], float)


begin = 0.0
end = 3650
N = 36500
h = (end - begin) / N

arr_t = np.arange(begin, end, h)
arr_s = []
arr_e = []
arr_i = []
arr_r = []
arr_n = []
arr_d = []


r_init = np.array([1-1e-6, 0.0, 1e-6, 0.0], float)
for t in arr_t:
    arr_s.append(r_init[0])
    arr_e.append(r_init[1])
    arr_i.append(r_init[2])
    arr_r.append(r_init[3])
    arr_n.append(sum(r_init))
    arr_d.append(1-sum(r_init))
    k1 = h * f(r_init, t)
    k2 = h * f(r_init + 0.5 * k1, t + 0.5 * h)
    k3 = h * f(r_init + 0.5 * k2, t + 0.5 * h)
    k4 = h * f(r_init + 0.5 * k3, t + h)
    r_init += (k1 + 2 * k2 + 2 * k3 + k4) / 6


plt.plot(arr_t, arr_s, label='Susceptible')
plt.plot(arr_t, arr_e, label='Exposed')
plt.plot(arr_t, arr_i, label='Infectious')
plt.plot(arr_t, arr_r, label='Recovered')
plt.plot(arr_t, arr_n, label='Alive')
plt.plot(arr_t, arr_d, label='Died')
plt.xlabel('Time in day')
plt.ylabel('Total Population')
plt.title('Plot of different population verse time')
plt.legend()
plt.savefig('Plot of different population verse time.pdf')
plt.show()


# 2(d)

def f_high(v, t):
    """Return the value of f
    """
    s = v[0]
    e = v[1]
    i = v[2]
    er = v[3]
    n = s + e + i + er
    lamb = beta * i / n
    fs = b * n + vi * er - (d + lamb) * s
    fe = lamb * s - (d + alpha) * e
    fi = alpha * e - (d * 100 + gamma) * i
    fr = gamma * i - (d + vi) * er
    return np.array([fs, fe, fi, fr], float)


begin = 0.0
end = 15000
N = 36500
h = (end - begin) / N

arr_t = np.arange(begin, end, h)
arr_s = []
arr_e = []
arr_i = []
arr_r = []
arr_n = []
arr_d = []

day = 0
r_init = np.array([1-1e-6, 0.0, 1e-6, 0.0], float)
for t in arr_t:
    arr_s.append(r_init[0])
    arr_e.append(r_init[1])
    arr_i.append(r_init[2])
    arr_r.append(r_init[3])
    arr_n.append(sum(r_init))
    arr_d.append(1-sum(r_init))
    k1 = h * f_high(r_init, t)
    k2 = h * f_high(r_init + 0.5 * k1, t + 0.5 * h)
    k3 = h * f_high(r_init + 0.5 * k2, t + 0.5 * h)
    k4 = h * f_high(r_init + 0.5 * k3, t + h)
    r_init += (k1 + 2 * k2 + 2 * k3 + k4) / 6
    if day == 0 and sum(r_init) < 0.5:
        day = t
        print('Half of population died after day', int(day), ', year', int(day)/365)


plt.plot(arr_t, arr_s, label='Susceptible')
plt.plot(arr_t, arr_e, label='Exposed')
plt.plot(arr_t, arr_i, label='Infectious')
plt.plot(arr_t, arr_r, label='Recovered')
plt.plot(arr_t, arr_n, label='Alive')
plt.plot(arr_t, arr_d, label='Died')
plt.xlabel('Time in day')
plt.ylabel('Total Population')
plt.title('Plot of different population verse time with high death rate')
plt.legend()
plt.savefig('Plot of different population verse time with high death rate.pdf')
plt.show()


# 2(e)

def f_high(v, t):
    """Return the value of f
    """
    s = v[0]
    e = v[1]
    i = v[2]
    er = v[3]
    va = v[4]
    n = s + e + i + er + va
    lamb = beta * i / n
    if t > 5475:
        p = 1 - 2 ** ((5475 - t) / 365)
    else:
        p = 0
    fs = b * n * (1-p) + vi * er - (d + lamb) * s
    fe = lamb * s - (d + alpha) * e
    fi = alpha * e - (d * 100 + gamma) * i
    fr = gamma * i - (d + vi) * er
    fv = b * n * p - d * va
    return np.array([fs, fe, fi, fr, fv], float)


begin = 0.0
end = 3650 * 5
N = 36500
h = (end - begin) / N

arr_t = np.arange(begin, end, h)
arr_s = []
arr_e = []
arr_i = []
arr_r = []
arr_n = []
arr_d = []
arr_v = []

day = 0
r_init = np.array([1-1e-6, 0.0, 1e-6, 0.0, 0.0], float)
for t in arr_t:
    arr_s.append(r_init[0])
    arr_e.append(r_init[1])
    arr_i.append(r_init[2])
    arr_r.append(r_init[3])
    arr_v.append(r_init[4])
    arr_n.append(sum(r_init))
    arr_d.append(1-sum(r_init))
    k1 = h * f_high(r_init, t)
    k2 = h * f_high(r_init + 0.5 * k1, t + 0.5 * h)
    k3 = h * f_high(r_init + 0.5 * k2, t + 0.5 * h)
    k4 = h * f_high(r_init + 0.5 * k3, t + h)
    r_init += (k1 + 2 * k2 + 2 * k3 + k4) / 6


plt.plot(arr_t, arr_s, label='Susceptible')
plt.plot(arr_t, arr_e, label='Exposed')
plt.plot(arr_t, arr_i, label='Infectious')
plt.plot(arr_t, arr_r, label='Recovered')
plt.plot(arr_t, arr_n, label='Alive')
plt.plot(arr_t, arr_d, label='Died')
plt.plot(arr_t, arr_v, label='Vaccinated')
plt.xlabel('Time in day')
plt.ylabel('Total Population')
plt.title('Plot of different population verse time with vaccine')
plt.legend()
plt.savefig('Plot of different population verse time with vaccine.pdf')
plt.show()
print('The population is tend to ', arr_n[-1])

