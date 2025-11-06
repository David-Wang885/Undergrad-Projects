import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


data = np.unpackbits(np.load("Earth.npy")).reshape(2160, 4320)



def Uniform_draw_Q1():
    """Random number generator"""
    theta = np.random.randint(0, 2159)
    phi = np.random.randint(0, 4319)
    return theta, phi


def calculate_area_percentage_Q1_d():
    """Calculate the percetage of the land by iterating through all data"""
    sea_area = 0
    for x in range(0, 2160):
        for y in range(0, 4320):
            if data[x][y] == 0:
                sea_area += 1
    print(1 - sea_area / (2160 * 4320))
    return 1 - sea_area/(2160*4320)


def calculate_samples_c(sample_num):
    """Calculate the land percentage by randomly select points on the earth"""
    count_ocean = 0
    count_land = 0
    points = []
    x = []
    y = []
    z = []
    plt.figure()
    ax = plt.axes(projection='3d')
    for i in range(0, sample_num):
        theta, phi = Uniform_draw_Q1()                      # pick a random point on earth
        # convert latitude and longtitude to 3D cartesian space
        points.append([theta/12, phi/12])
        x.append(np.sin(theta/12) * np.cos(phi/12))
        y.append(np.sin(theta/12)*np.sin(phi/12))
        z.append(np.cos(theta/12))

        if data[theta][phi] == 0:
            count_ocean += 1
        else:
            count_land += 1
    # Plot the randomly chosen points in 3D space
    ax.scatter(x, y, z)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.title("Data point chosen")
    plt.savefig("point Selection.png")
    plt.show()
    return count_land / (count_land + count_ocean)


if __name__ == "__main__":
    print(calculate_samples_c(100000)/ 0.26880369084362143)
