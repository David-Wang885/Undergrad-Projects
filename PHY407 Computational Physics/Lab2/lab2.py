
import pylab as plt
import math as m
import numpy as np
import scipy.constants as sp
import scipy.special


def method_1_1b(data_set):
    """
    Implementation for quuation (5)
    """

    x_mean = sum(data_set) / len(data_set)                  # find the mean of the data

    result = 0
    size = len(data_set)

    for i in range(0, size):
        result += (data_set[i] - x_mean) ** 2               # Calculate the sum of the data

    std = m.sqrt(result / (size - 1))                       # Calculate the standard deviation

    numpy_result = np.std(data_set, ddof=1)                 # Numpy's standard deviation

    return abs((std - numpy_result)) / numpy_result         # Calculate relative error


def method_2_1b(data_set):
    """
    Implementation of equation (6)
    """
    x_mean = sum(data_set) / len(data_set)
    size = len(data_set)
    y = 0

    for i in range(0, size):
        y += (data_set[i] ** 2) - size * (x_mean ** 2)      # Calculate x_i^2 + n * x_mean^2
    if y < 0:                                               # Condition check to see if the sum is less than zero
        print("Warning, square rooting a negative "
              "number，taking absolute value insted")       # Print a error message if y is less than zero
        y = -y
    std = m.sqrt(y / (size - 1))                            # Take the square root of the sum

    numpy_result = np.std(data_set, ddof=1)                 # Calculate the std using numpy.std

    return abs(std - numpy_result) / numpy_result           # Calculate the realative error and return it

def fixed_one_pass(data_set, shift_amount):
    x_mean = sum(data_set) / len(data_set)
    size = len(data_set)
    y = 0
    for i in range(0, size):
        y += ((data_set[i]-shift_amount) ** 2) - size * ((x_mean - shift_amount) ** 2) # Calculate x_i^2 + n * x_mean^2
    if y < 0:  # Condition check to see if the sum is less than zero
        print("Warning, square rooting a negative "
              "number，taking absolute value insted")  # Print a error message if y is less than zero
        y = -y
    std = m.sqrt(y / (size - 1))  # Take the square root of the sum

    numpy_result = np.std(data_set, ddof=1)  # Calculate the std using numpy.std

    return abs(std - numpy_result) / numpy_result


def p(u):
    """ Helper function for p(u)"""
    return (1-u)**8

def q(u):
    """Helper function for q(u)"""
    return 1 - (8 * u) + (28 * (u ** 2)) - (56 * (u ** 3)) + (70 * (u ** 4)) - (56 * (u ** 5)) +\
           (28 * (u ** 6)) - (8 * (u ** 7)) + (u ** 8)
def q_mean(u):
    """Helper function for q(u)"""
    return ((-(8 * u)**2) + ((28 * (u ** 2))**2) + ((-56 * (u ** 3))**2) + ((70 * (u ** 4))**2) + ((-56 * (u ** 5))**2) +\
           ((28 * (u ** 6))**2) + ((-8 * (u ** 7))**2) + ((u ** 8)**2))/8

def Question_2a():
    N = []                                              # Initialize the parameters
    p_u = []
    q_u = []
    data_point = 0.98
    for r in range(0, 500):                             # When the loop ends, data_point will have a value of 1.02
        data_point += 0.00008
        N.append(data_point)                            # Calculat p and q and save the data to corresponding array
        p_u.append(p(data_point))
        q_u.append(q(data_point))
    plt.plot(N, p_u, "bo", label="Plot for p(u)")       # Plot p(u) using blue dot and q(u) using blue dot
    plt.plot(N, q_u, "ro", label="Plot for q(u)")
    plt.xlabel("x value")                               # Label the axis , save the figure and show it
    plt.ylabel("y value")
    plt.title("p(u) vs q(u)")
    plt.legend()
    plt.savefig("Question_2a.png")
    plt.show()

def Question_2b():
    result = []                                       # Initialize the array
    mean = []
    data_point = 0.98
    for r in range(0, 500):                           # When the loop ends when data point is 1.02
        data_point += 0.00008
        result.append(p(data_point) - q(data_point))  # Save the data into a array
        mean.append(q_mean(data_point))
    plt.hist(result, 30)
    plt.title("Histogram for p(u) - q(u)")            # Plot the histogram

    np.array(result)
    total = 0
    for i in range(0, len(result)):                   # Calculate the estimated standard deviation
        total += (result[i] ** 2)/44

    estimate_std = (10**(-16)) * m.sqrt(44) * m.sqrt(sum(mean)/len(result))
    numpy_std = np.std(result)                        # Calculate the standard deviation using numpy
    print(sum(mean)/500)
    print(estimate_std)
    print(numpy_std)
    plt.savefig("2b_distribution.png")                # Save the histogram and shoe it
    plt.show()

def Question_2c():
    N = []                                      # Initalize all the varables
    result = []
    data_point = 0.98
    for x in range(0, 500):                     # When the loop end, data_point will have a value of 0.999
        data_point += 0.000038
        if abs(q(data_point)) != 0:             # Check if q(data_point) is zero, if it is zero, we will have a divide by 0 error
            N.append(data_point)
            result.append(abs(p(data_point) - q(data_point))/abs(q(data_point)))  # Calcluate relitive error
    plt.plot(N, result, "bo")                   # Plot the relitive error and save it
    plt.ylim((0, 1))
    plt.title("Error of p - q")
    plt.xlabel("Value of u")
    plt.ylabel("Error")
    plt.savefig("Plot for Question_2c.png")
    plt.show()

def f(u):
    """
    Helper function for f in question 2d
    """
    return u**8/((u**4)*(u**4))

def Question_2d():
    N = []                                  # Initialize all the parameters
    result = []
    data_point = 0.98
    for r in range(0, 500):                 # Calculate 500 data point from 0.98 to 1.02 and save them to array
        data_point += 0.00008
        N.append(data_point)
        result.append(f(data_point) - 1 )

    plt.plot(N, result, "bo")               # Plot the result using blue circule and label the axis
    plt.title("Error of p - q")
    plt.xlabel("Value of u")
    plt.ylabel("value of f(u)")
    plt.savefig("question_4d.png")
    plt.show()


# 3
w = 0.01
c = sp.c
h = sp.h
k = sp.k


def f(x: float):
    """Return the result of the function that simplified from the blackbody
    function
    """
    return x**3 / ((1-x)**5 * (np.e**(x/(1-x))-1))


def black_body_t(wide: float):
    """Return the integral of the black body function using trapezoidal rule
    """
    n = int(1 // wide)
    total = 0
    for j in range(1, n):
        total += wide * f(j * wide)
    return total


def black_body_s(wide: float):
    """Return the integral of the black body function using Simpson rule
    """
    n = int(1 // wide)
    total = 0
    for j in range(1, n, 2):
        total += wide * 4 * f(j * wide) / 3
    for j in range(2, n, 2):
        total += wide * 2 * f(j * wide) / 3
    return total


def sigma(integral: float):
    """Return the value of sigma from the estimation of black body function
    """
    return 2 * np.pi * k**4 * integral / (h**3 * c**2)


print(sigma(black_body_s(w)))
print(sp.sigma)


# 4
r = 0.003
# radius in m
z = 0.0
# height in m
Q = 10 ** (-13)
# charge in Coulomb
length = 0.001
# length in m


def v(u: float, height_z: float, dis_r: float):
    """return the value of potential for a certain radius, a function used in
    Simpson rule
    """
    cons = (Q / (4 * np.pi * sp.epsilon_0))
    func = np.e**((-1)*(np.tan(u))**2)/\
           (np.cos(u)**2 * np.sqrt((height_z-length*np.tan(u))**2+dis_r**2))
    return cons * func


def v_s(wide: float):
    """return the value of potential calculated by Simpson rule
    """
    n = int(np.pi // wide)
    total = wide * (v(-np.pi/2, z, r)+v(np.pi/2, z, r))/3
    for j in range(1, n, 2):
        total += wide * 4 * v(-np.pi/2 + j * wide, z, r) / 3
    for j in range(2, n, 2):
        total += wide * 2 * v(-np.pi/2 + j * wide, z, r) / 3
    return total * Q / (4 * np.pi * sp.epsilon_0 * length)


def v_e():
    """return the value of potential calculated by equation (4)
    """
    ex = np.e**(r**2/(2*length**2))
    cons = Q / (4 * np.pi * sp.epsilon_0 * length)
    return cons * ex * scipy.special.kn(0, r**2/(2*length**2))


# plot and show the effect of accuracy on adding more interval, N
wid = []
diff_v = []
for b in range(1, 100):
    a = np.pi/b
    wid.append(b)
    diff_v.append(abs(v_s(a)-v_e()))

plt.plot(wid, diff_v)
plt.show()


# plot a field showing the potential in vector around the line of charge
r_x = []
z_y = []
dir_x = []
dir_y = []
for rx in range(25, 500, 25):
    for zy in range(-50, 50, 5):
        x = rx * 10 ** -5
        y = zy * 10 ** -4
        r_x.append(x)
        z_y.append(y)
        theta = np.arctan(y/x)
        vc = v(np.pi/8, y, x)
        dir_x.append(np.cos(theta)*vc)
        dir_y.append(np.sin(theta)*vc)


plt.quiver(r_x, z_y, dir_x, dir_y, norm=True)
plt.show()
