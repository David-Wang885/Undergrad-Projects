import pylab as plt
import numpy as np
import math
import cmath

# 1(b)

v_2 = 1350  # m/s
Y = np.asarray([[0, 0, 1, 0, 0, 0],
               [0, -1/4, 0, 0, 0, 0],
               [3, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0]])


def cap_x(x, y):
    return [x**2, x*y, y**2, x, y, 1]


arr_x = np.asarray([-38.04, -35.28, -25.58, -28.80, -30.06])
arr_y = np.asarray([27.71, 17.27, 30.68, 31.50, 10.53])
arr_s = []
for i in range(5):
    arr_s.append(cap_x(arr_x[i], arr_y[i]))
s = np.matmul(np.asarray(arr_s).transpose(), np.asarray(arr_s))
s_inv = np.linalg.inv(s)
E, V = np.linalg.eig(np.matmul(s_inv, Y))


A, B, C, D, F, G = V[0][4], V[1][4], V[2][4], V[3][4], V[4][4], V[5][4]
temp_1 = np.sqrt(abs((2*(A*(F**2)+C*(D**2)+G*(B**2)-2*B*D*F-A*C*G)) /
            ((B**2-A*C)*(np.sqrt((A-C)**2+4*(B**2))-(A+C)))))
temp_2 = np.sqrt(abs((2*(A*(F**2)+C*(D**2)+G*(B**2)-2*B*D*F-A*C*G)) /
            ((B**2-A*C)*(-np.sqrt((A-C)**2+4*(B**2))-(A+C)))))
a, b = max(temp_1, temp_2), min(temp_1, temp_2)


e = np.sqrt(1-b**2/a**2)
l_1 = a*(1-e)
l_2 = a*(1+e)
T = 2*np.pi*a*b*(1.496*(10**11)) / (l_2*v_2*3600*24*365)
print(T)
