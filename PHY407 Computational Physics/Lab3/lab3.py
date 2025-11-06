import pylab as plt
import numpy as np
import math


# Question 3(a)
def f(x):
    """Define the function e^(2x), for any import x, return e^(2x)
    """
    return np.e ** (2 * x)


def central_diff(h, x):
    """Define the calculation of the first derivative by using central
    difference using h. Return the value of first derivative at x
    """
    return (f(x + h / 2) - f(x - h / 2)) / h


arr_ha = []
order = []
real = []
est = []
difference = []
for a in range(0, 17):
    order.append(a)
    arr_ha.append(10**(-1 * a))
    real.append(2)
for item in arr_ha:
    est.append(central_diff(item, 0))
    difference.append(abs(central_diff(item, 0) - 2))

print('The values of estimation produced by central difference are ', est)
plt.plot(order, est, label='Estimated value')
plt.plot(order, real, label='2')
plt.title('Comparison between the estimated value and 2')
plt.xlabel('Value of h')
plt.ylabel('Output of the first derivative')
plt.legend()
plt.savefig('Comparison between the estimated value and 2.pdf')
plt.show()
print('The minimum difference between the estimation is ', min(difference),
      ' when h=10 **', order.__getitem__(difference.index(min(difference)))*-1)
plt.plot(order, difference, label='Difference')
plt.title('Absolute difference between the estimated value and 2')
plt.xlabel('Value of h')
plt.ylabel('Difference of the first derivative')
plt.legend()
plt.savefig('Absolute difference between the estimated value and 2.pdf')
plt.show()


# Question 3(b)
M = 10
N = 1000
H = 0.038


def delta(x, m, h):
    if m > 1:
        return (delta(x + h/2, m - 1, h) - delta(x - h/2, m-1, h)) / h
    else:
        return (f(x + h/2) - f(x - h/2)) / h


def cauchy(n, m):
    i = (-1) ** (1/2)
    total = 0
    for k in range(0, n):
        zk = np.e ** ((i * 2 * np.pi * k)/n)
        total += f(zk) * zk ** (-m)
    return math.factorial(m) * total / n


print('The value of estimation produced by delta estimation is ',
      delta(0, M, H))
print('The value of estimation produced by Cauchy derivative formula is ',
      cauchy(N, M).real)

arr_hb = []
error_b = []
for b in range(0, 30):
    temp = 0.03 + b * 0.001
    arr_hb.append(temp)
    error_b.append(abs(1024 - delta(0, M, temp)))
plt.plot(arr_hb, error_b, label='Difference')
plt.title('Absolute difference between the delta estimation and the real value')
plt.xlabel('Value of h')
plt.ylabel('Difference of the delta estimation and the real value')
plt.legend()
plt.savefig('Absolute difference between the delta '
            'estimation and the real value.pdf')
plt.show()
