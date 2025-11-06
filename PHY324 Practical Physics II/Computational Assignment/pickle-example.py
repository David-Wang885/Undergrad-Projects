# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 15:35:35 2017

@author: Brian
"""

import pickle

with open('noisy_sine_wave','rb') as file:
    data_from_file=pickle.load(file)
"""
the above few lines makes an array called data_from_file which contains
a noisy sine wave as long as you downloaded the file "noisy_sine_wave" 
and put it in the same directory as this python file

pickle is a Python package which nicely saves data to files. it can be
a little tricky when you save lots of data, but this file only has one
object (an array) saved so it is pretty easy
"""

import matplotlib.pyplot as plt

plt.plot(data_from_file)
xmax=300
plt.xlim(0,xmax)
plt.show()

number=len(data_from_file)
message="There are " + \
        str(number) + \
        " data points in total, only drawing the first " + \
        str(xmax)
print(message)
