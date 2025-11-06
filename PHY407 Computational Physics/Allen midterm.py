import pylab as plt
import numpy as np

# # 9
# x = 0
# N = 100
# for _ in range(N):
#     r = np.random.choice([1, -1])
#     x += r
#     print(x)
#
# # 10
# N = 50000
# step = 1000000
# n = []
# for _ in range(N):
#     x = 0
#     i = 0
#     while i < step and -10 < x < 10:
#         x += np.random.choice([1, -1])
#         i += 1
#     n.append(i)
# plt.hist(n)
# plt.show()

# # 11
# N = 50000
# step = 1000000
# n = []
# for _ in range(N):
#     a = np.array([0, 0])
#     i = 0
#     while i < step and (a[0] ** 2 + a[1] ** 2) <= 100:
#         r = np.random.choice([1, 0])
#         a[r] += np.random.choice([1, -1])
#         i += 1
#     n.append(i)
# plt.hist(n)
# plt.show()


# 12
def f(x):
    return np.sin(x)**2 / (1 + (7/8)*np.cos(x))**3


a = 0
b = np.pi
N = 10000
h = (b-a)/N
integral = 0
temp = a
for _ in range(N):
    area = h * f(temp)
    integral += area
    temp += h
print(integral)
