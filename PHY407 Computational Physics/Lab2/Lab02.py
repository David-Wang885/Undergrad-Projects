import numpy as np
import math as m
import pylab as plt

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



if __name__ =="__main__":
    #print(method_1_1b(np.loadtxt("cdata.txt")))
    #print(method_2_1b(np.loadtxt("cdata.txt")))
    x = np.random.normal(0., 1., 2000)
    y = np.random.normal(1.e7, 1., 2000)
    #print(method_1_1b(x))
    #print(method_2_1b(np.loadtxt("cdata.txt")))
    #print(fixed_one_pass(np.loadtxt("cdata.txt"), 299))
    #print(method_2_1b(y))
    Question_2b()
    #Question_2d()
