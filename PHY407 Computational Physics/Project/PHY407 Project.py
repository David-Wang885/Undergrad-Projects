import pylab as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# (a)
# Constant
sigma = 10  # the Prandtl number
r = 28  # Rayleigh number
b_1 = 8/3  # size constant


# Function
def f(s, t):  # Set up the function of ode
    x = s[0]
    y = s[1]
    z = s[2]
    fx = sigma * (y - x)
    fy = r * x - y - x * z
    fz = x * y - b_1 * z
    return np.array([fx, fy, fz], float)


# Set up array
begin = 0.0  # begin time
end = 100  # end time
h = 0.01  # steps

arr_t = np.arange(begin, end, h)
arr_x = []
arr_y = []
arr_z = []

s_init = [-10, 10, 25]  # initial position

# Main loop
for t in arr_t:  # rk4
    arr_x.append(s_init[0])
    arr_y.append(s_init[1])
    arr_z.append(s_init[2])
    k1 = h * f(s_init, t)
    k2 = h * f(s_init + 0.5 * k1, t + 0.5 * h)
    k3 = h * f(s_init + 0.5 * k2, t + 0.5 * h)
    k4 = h * f(s_init + 0.5 * k3, t + h)
    s_init += (k1 + 2 * k2 + 2 * k3 + k4) / 6

# Plotting
fig = plt.figure()  # Create graph
ax = fig.gca(projection='3d')  # set up 3d
ax.plot(arr_x, arr_y, arr_z)  # plot the graph
ax.set_xlabel('x')  # labels
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.title('Lorenz initial position at (-10, 10, 25)')
plt.savefig('Lorenz initial position at (-10, 10, 25).pdf')
plt.show()

# (b)
c = np.fft.rfft(arr_x)  # Fourier transformation
c_2 = np.zeros(len(c), complex)  # Create two arrays for keeping different
c_4 = np.zeros(len(c), complex)  # amount of Fourier coefficient
for j in range(len(c)//50):  # keep only first 2% of Fourier coefficients
    c_2[j] = c[j]
for j in range(len(c)//25):  # keep only first 4% of Fourier coefficients
    c_4[j] = c[j]

z_2 = np.fft.irfft(c_2)  # Inverse Fourier transformation
z_4 = np.fft.irfft(c_4)

# Plotting for 2%
plt.plot(arr_x)
plt.plot(z_2)
plt.title('Fourier transformation keep only first 2% of the coefficients')
plt.xlabel('t')
plt.ylabel('x')
plt.savefig('Fourier transformation keep only first 2% of the coefficients.pdf')
plt.show()
# Plotting for 4%
plt.plot(arr_x)
plt.plot(z_4)
plt.title('Fourier transformation keep only first 4% of the coefficients')
plt.xlabel('t')
plt.ylabel('x')
plt.savefig('Fourier transformation keep only first 4% of the coefficients.pdf')
plt.show()


# (c)
# Set up array
arr_x_1 = []
arr_y_1 = []
arr_z_1 = []

s_init_1 = [-10, 10, 25.001]  # initial position

# Main loop
for t in arr_t:
    arr_x_1.append(s_init_1[0])
    arr_y_1.append(s_init_1[1])
    arr_z_1.append(s_init_1[2])
    k1 = h * f(s_init_1, t)
    k2 = h * f(s_init_1 + 0.5 * k1, t + 0.5 * h)
    k3 = h * f(s_init_1 + 0.5 * k2, t + 0.5 * h)
    k4 = h * f(s_init_1 + 0.5 * k3, t + h)
    s_init_1 += (k1 + 2 * k2 + 2 * k3 + k4) / 6

# Plotting
fig_2 = plt.figure()
ax_2 = fig_2.gca(projection='3d')
ax_2.plot(arr_x_1, arr_y_1, arr_z_1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.title('Lorenz initial position at (-10, 10, 25.001)')
plt.savefig('Lorenz initial position at (-10, 10, 25.001).pdf')
plt.show()

# (d)
# Set up array
arr_x_2 = []
arr_y_2 = []
arr_z_2 = []
ratio = []  # array storing the ratio of points inside the certain area
count = 0  # count stores the number of points inside the certain area

s_init_2 = [10, -10, 25]
gap = 1/99  # set up gap for 100 initial positions between 25 to 26


# Main loop
for i in range(100):  # loop for 100 initial positions
    for t in arr_t:  # loop for time
        if -5 <= s_init_2[0] <= 5 and -5 <= s_init_2[1] <= 5:
            count += 1  # for points in certain area, we store it in count
        arr_x_2.append(s_init_2[0])
        arr_y_2.append(s_init_2[1])
        arr_z_2.append(s_init_2[2])
        k1 = h * f(s_init_2, t)
        k2 = h * f(s_init_2 + 0.5 * k1, t + 0.5 * h)
        k3 = h * f(s_init_2 + 0.5 * k2, t + 0.5 * h)
        k4 = h * f(s_init_2 + 0.5 * k3, t + h)
        s_init_2 += (k1 + 2 * k2 + 2 * k3 + k4) / 6

    ratio.append(count / len(arr_t))  # calculate the ratio
    arr_x_2 = []  # set each array back to zero to prevent the stack up when
    arr_y_2 = []  # looping over 100 initial positions
    arr_z_2 = []
    count = 0  # set count back to 0
    s_init_2 = [10, -10, 25 + i * gap]  # set the initial to the new positions

print(ratio)  # print out the result

# (f)
# Constant
a = 0.1
b = 0.1
c = 14


# Function
def g(u, t):
    x = u[0]
    y = u[1]
    z = u[2]
    gx = -y - z
    gy = x + a * y
    gz = b + (x - c) * z
    return np.array([gx, gy, gz], float)


# Set up array
begin = 0.0  # begin time
end = 200  # end time
h = 0.01  # steps

arr_t_r = np.arange(begin, end, h)
arr_x_3 = []
arr_y_3 = []
arr_z_3 = []

u_init = [-10, 10, 10]  # initial position

# Main loop
for t in arr_t_r:
    arr_x_3.append(u_init[0])
    arr_y_3.append(u_init[1])
    arr_z_3.append(u_init[2])
    k1 = h * g(u_init, t)
    k2 = h * g(u_init + 0.5 * k1, t + 0.5 * h)
    k3 = h * g(u_init + 0.5 * k2, t + 0.5 * h)
    k4 = h * g(u_init + 0.5 * k3, t + h)
    u_init += (k1 + 2 * k2 + 2 * k3 + k4) / 6

# Plotting
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(arr_x_3, arr_y_3, arr_z_3)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.title('Rossler initial position at (-10, 10, 10)')
plt.savefig('Rossler initial position at (-10, 10, 10).pdf')
plt.show()


# Set up array
arr_x_4 = []
arr_y_4 = []
arr_z_4 = []

u_init_1 = [-10, 10.01, 10]

# Main loop
for t in arr_t_r:
    arr_x_4.append(u_init_1[0])
    arr_y_4.append(u_init_1[1])
    arr_z_4.append(u_init_1[2])
    k1 = h * g(u_init_1, t)
    k2 = h * g(u_init_1 + 0.5 * k1, t + 0.5 * h)
    k3 = h * g(u_init_1 + 0.5 * k2, t + 0.5 * h)
    k4 = h * g(u_init_1 + 0.5 * k3, t + h)
    u_init_1 += (k1 + 2 * k2 + 2 * k3 + k4) / 6

# Plotting
fig_2 = plt.figure()
ax_2 = fig_2.gca(projection='3d')
ax_2.plot(arr_x_4, arr_y_4, arr_z_4)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.title('Rossler initial position at (-10, 10.01, 10)')
plt.savefig('Rossler initial position at (-10, 10.01, 10).pdf')
plt.show()


# Set up array
arr_x_5 = []
arr_y_5 = []
arr_z_5 = []
ratio_1 = []
count_1 = 0

u_init_2 = [-10, 10, 10]
gap = 1/99


# Main loop
for i in range(100):
    for t in arr_t_r:
        if 10 <= u_init_2[0] <= 15 and 5 <= u_init_2[1] <= 10:
            count_1 += 1
        arr_x_5.append(u_init_2[0])
        arr_y_5.append(u_init_2[1])
        arr_z_5.append(u_init_2[2])
        k1 = h * g(u_init_2, t)
        k2 = h * g(u_init_2 + 0.5 * k1, t + 0.5 * h)
        k3 = h * g(u_init_2 + 0.5 * k2, t + 0.5 * h)
        k4 = h * g(u_init_2 + 0.5 * k3, t + h)
        u_init_2 += (k1 + 2 * k2 + 2 * k3 + k4) / 6

    ratio_1.append(count_1 / len(arr_t_r))
    arr_x_5 = []
    arr_y_5 = []
    arr_z_5 = []
    count_1 = 0
    u_init_2 = [-10, 10 + i * gap, 10]


print(ratio_1)


