
import pylab as plt
import math
import time
import numpy as np
import scipy.optimize
import random
from typing import Tuple, List
#
#
# def solution_to_1c(x_initial, y_initial, x_v_inital, y_v_initial, time_step, end_time):
#     """
#     This programs plots the solution to 1c.
#     """
#     x_coordinate = [0]                                                                                                  # Add the position of the sun
#     y_coordinate = [0]                                                                                                  # size of x_coordinate and y_coordinate should be = to size of time
#     x_new_initial = x_initial
#     x_new_v_initial = x_v_inital
#     y_new_initial = y_initial
#     y_new_v_initial = y_v_initial
#
#     iteration = int(end_time / time_step)
#
#     for x in range(0, iteration):
#
#         r = math.sqrt((x_new_initial ** 2) + (y_new_initial ** 2))                                                      # Calculate r
#
#         x = x_new_initial + x_new_v_initial * time_step - ((39.5 * 1 * x_new_initial) / (r ** 3)) * (time_step ** 2)    # Calculate the new step value for x
#         y = y_new_initial + y_new_v_initial * time_step - ((39.5 * 1 * y_new_initial) / (r ** 3)) * (time_step ** 2)    # Calculate the new step value for y
#
#         x_v = x_new_v_initial - ((39.5 * 1 * x_new_initial) / (r ** 3)) * time_step                                     # Calculate the new value for x velocity
#         y_v = y_new_v_initial - ((39.5 * 1 * y_new_initial) / (r ** 3)) * time_step                                     # Calculate the new value for y velocity
#
#         y_coordinate.append(y)                                                                                          # Add new coordinates to list
#         x_coordinate.append(x)
#
#         x_new_initial = x                                                                                               # Update starting position values
#         y_new_initial = y
#
#         x_new_v_initial = x_v                                                                                           # Update starting velocity values
#         y_new_v_initial = y_v
#
#     #plt.plot(x_coordinate, y_coordinate, "bo")                                                                          # Plot them to the graph
#     plt.xlabel("X position, Unit: AU")
#     plt.ylabel("Y position, Unit: AU")
#     plt.savefig("1c_ouptput.png")
#     plt.show()
#
#
# def solution_to_1d(x_initial, y_initial, x_v_inital, y_v_initial, time_step, end_time):
#     """
#     This programs plots the solution to 1d, where the gravitational forces factor predict by general relativity is added
#     to the equation.
#     """
#     x_coordinate = [0]                                                                                                  # Add the position of the sun
#     y_coordinate = [0]                                                                                                  # size of x_coordinate and y_coordinate should be = to size of time
#     x_new_initial = x_initial
#     x_new_v_initial = x_v_inital
#     y_new_initial = y_initial
#     y_new_v_initial = y_v_initial
#
#     iteration = int(end_time / time_step)
#
#     for x in range(0, iteration):
#
#         r = math.sqrt((x_new_initial ** 2) + (y_new_initial ** 2))                                                      # Calculate r
#         rc = (1 + (0.01 / (r**2)))                                                                                      # Calculate general relativity coefficent
#
#         x = x_new_initial + x_new_v_initial * time_step - ((39.5 * 1 * x_new_initial) * rc / (r ** 3)) *(time_step ** 2)# Calculate the new step value for x
#         y = y_new_initial + y_new_v_initial * time_step - ((39.5 * 1 * y_new_initial) * rc / (r ** 3)) *(time_step ** 2)# Calculate the new step value for y
#
#         x_v = x_new_v_initial - ((39.5 * 1 * x_new_initial) * rc / (r ** 3)) * time_step                                # Calculate the new value for x velocity
#         y_v = y_new_v_initial - ((39.5 * 1 * y_new_initial) * rc / (r ** 3)) * time_step                                # Calculate the new value for y velocity
#
#         y_coordinate.append(y)                                                                                          # Add new coordinates to list
#         x_coordinate.append(x)
#
#         x_new_initial = x                                                                                               # Update starting position values
#         y_new_initial = y
#
#         x_new_v_initial = x_v                                                                                           # Update starting velocity values
#         y_new_v_initial = y_v
#
#     #plt.plot(x_coordinate, y_coordinate, "bo")                                                                          # Plot them to the graph
#     plt.xlabel("X position, Unit: AU")
#     plt.ylabel("Y position, Unit: AU")
#     plt.savefig("1d_ouptput.png")
#     plt.show()
#
# def solution_to_problem_3_N():
#     """
#     This function measures the time for matrix muiltiplication on matrix of size N, the function will produce a graph
#     """
#     # Below is the code is for ploting time as function of N
#     N = []
#     time_spend = []
#     for x in range(2, 512, 1):
#         time_average = []
#         for y in range(0, 1000):
#
#             A = np.ones([x, x], float) * 5
#             B = np.ones([x, x], float) * 8
#
#             inital_time = time.time() * 1000
#
#             C = A * B
#
#             final_time = time.time() * 1000
#
#             time_average.append(final_time - inital_time)
#
#         N.append(x)
#         time_spend.append(sum(time_average)/len(time_average))
#
#     x1, y1 = scipy.optimize.curve_fit(helper_Q3, N, time_spend, 0.01)
#     #plt.plot(x1, y1)
#     plt.xlabel("Matrix size")
#     plt.ylabel("Time spent")
#     plt.title("Time spent for matrix mutilplication as a function of N")
#     plt.show()
#
# def helper_Q3(x, c,):
#
#     return c * np.power(x, 3)
#
#
#
# if __name__ == "__main__":
#     # solution_to_1c(0.47, 0.0, 0.0, 8.17, 0.0001, 1)
#     # solution_to_1d(0.47, 0.0, 0.0, 8.17, 0.0001, 1)
#      solution_to_problem_3_N()


N = 1000000
# Set up total number of particles that need to generate


def generate_point(n: int):
    """This function take in total number of particles and generate random
    height, z, and calculate angle, theta. Then store them into two separated
    arrays and return those two arrays.
    """
    arr_z = []
    arr_theta = []
    for _ in range(n):
        z = random.uniform(-1.0, 1.0)
        arr_z.append(z)
        if z >= 0:
            angle = 90 + np.arctan(np.sqrt(1 - z ** 2) / z) * 180 / np.pi
        else:
            angle = 270 + np.arctan(np.sqrt(1 - z ** 2) / z) * 180 / np.pi
        arr_theta.append(angle)
    return arr_z, arr_theta


def relative_prob(a: Tuple[int, int], b: Tuple[int, int], arr: List):
    """This function take in two range in the form of tuple and an array of
    number. Return the relative probability of getting this two range of angle
    """
    arr_a = []
    arr_b = []
    for angle in arr:
        if a[0] < angle < a[1]:
            arr_a.append(theta)
        if b[0] < angle < b[1]:
            arr_b.append(theta)
    return len(arr_a) / len(arr_b)


z, theta = generate_point(N)
# Create the array for z and theta
plt.plot(z, theta, "bo")
plt.title('Input z verse angle of theta')
plt.xlabel('Value of z')
plt.ylabel('Angle of theta in degree')
plt.legend()
# Plot it
# plt.show()
# Show the graph

print(relative_prob((170, 190), (90, 110), theta))
# Calculate the relative probability
