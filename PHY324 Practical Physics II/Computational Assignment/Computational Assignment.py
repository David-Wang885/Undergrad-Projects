import matplotlib.pyplot as plt
import numpy as np
from random import gauss
import pickle


# # Exercise 1
#
N = 200  # Number of data points
t = np.arange(N)
A1 = 3  # Amplitude of the first wave
A2 = 6  # Amplitude of the second wave
T1 = 9  # Period of the first wave
T2 = 4  # Period of the second wave
y = A1*np.sin(2*np.pi*t/T1) + A2*np.sin(2*np.pi*t/T2)


plt.plot(t/N, y)
plt.title("Position-Time Graph of Composition of Two Sine Waves")
plt.xlabel("Time")
plt.ylabel("Position")
plt.savefig('Exercise 1.pdf')
plt.show()

z = np.fft.fft(y)
plt.plot(np.abs(z))
plt.title("Frequency-Amplitude Graph of Composition of Two Sine Waves")
plt.xlabel("Frequency")
plt.ylabel("Amplitude")
plt.savefig('FT of Exercise 1.pdf')
plt.show()
#
#
# print(np.abs(z))
#
#
# # Exercise 2
# A3 = 5.   # wave amplitude
# T3 = 17.  # wave period
# noise_amp = A3/2
# noise=[gauss(0,noise_amp) for _usused_variable in range(len(y))]
# x_original = A3*np.sin(2.*np.pi*t/T3)
# x = x_original + noise
# z2 = np.fft.fft(x)
#
# M = len(z2)
# freq = np.arange(M)  # frequency values, like time is the time values
# width = 32  # width=2*sigma**2 where sigma is the standard deviation
# peak = 12    # ideal value is approximately N/T1
#
# filter_function = (np.exp(-(freq-peak)**2/width)+np.exp(-(freq+peak-M)**2/width))
# z_filtered = z2*filter_function
#
# fig, (ax1,ax2,ax3)=plt.subplots(3,1,sharex='col')
# # this gives us an array of 3 graphs, vertically aligned
# ax1.plot(np.abs(z2))
# ax2.plot(np.abs(filter_function))
# ax3.plot(np.abs(z_filtered))
#
# fig.subplots_adjust(hspace=0)
# ax1.set_ylim(0,480)
# ax2.set_ylim(0,1.2)
# ax3.set_ylim(0,480)
# ax1.set_ylabel('Noisy FFT')
# ax2.set_ylabel('Filter Function')
# ax3.set_ylabel('Filtered FFT')
# ax3.set_xlabel('Absolute value of FFT of Position-Time\n(Amplitude-Frequency)')
#
# plt.tight_layout()
# plt.savefig('Exercise 2 Filtered FT.pdf')
# plt.show()
#
# plt.plot(np.abs(np.fft.fft(x_original)), label="FT of the original sine wave")
# plt.plot(np.abs(z_filtered), label='Filtered FT')
# plt.title('Comparison of FT of the Original Wave and Filtered')
# plt.xlabel('Period')
# plt.ylabel('Amplitude')
# plt.legend()
# plt.savefig('Comparison of FT of the Original Wave and Filtered.pdf')
# plt.show()
#
# cleaned=np.fft.ifft(z_filtered)
# fig, (ax1,ax2,ax3)=plt.subplots(3,1,sharex='col',sharey='col')
# ax1.plot(t/N,x)
# ax2.plot(t/N,np.real(cleaned))
# ax3.plot(t/N,x_original)
# fig.subplots_adjust(hspace=0)
# ax1.set_ylim(-13,13)
# ax1.set_ylabel('Original Data')
# ax2.set_ylabel('Filtered Data')
# ax3.set_ylabel('Ideal Result')
# ax3.set_xlabel('Position-Time')
# plt.savefig('SingleWaveAndNoiseFFT.pdf')
# plt.show()
#
#
# # Exercise 3
#
#
# with open('noisy_sine_wave','rb') as file:
#     data_from_file=pickle.load(file)
# """
# the above few lines makes an array called data_from_file which contains
# a noisy sine wave as long as you downloaded the file "noisy_sine_wave"
# and put it in the same directory as this python file
#
# pickle is a Python package which nicely saves data to files. it can be
# a little tricky when you save lots of data, but this file only has one
# object (an array) saved so it is pretty easy
# """
#
# plt.plot(data_from_file)
# xmax = 300
# plt.xlim(0, xmax)
# plt.xlabel('Time')
# plt.ylabel('Position')
# plt.title('Exercise 3')
# plt.savefig('Exercise 3.pdf')
# plt.show()
#
# number = len(data_from_file)
# message = "There are " + \
#         str(number) + \
#         " data points in total, only drawing the first " + \
#         str(xmax)
# print(message)
#
# z3 = np.fft.fft(data_from_file)
# plt.plot(np.abs(z3))
# plt.xlabel('Frequency')
# plt.ylabel('Amplitude')
# plt.title('FT of Exercise 3')
# plt.savefig('FT of Exercise 3.pdf')
# plt.show()
#
#
# first_i = np.where(np.abs(z3) == max(np.abs(z3)))[0][0]
# print(first_i)
# second_i = np.where(np.abs(z3)[:first_i] == max(np.abs(z3)[:first_i]))[0][0]
# print(second_i)
# third_i = np.where(np.abs(z3)[:second_i] == max(np.abs(z3)[:second_i]))[0][0]
# print(third_i)
#
# print(np.abs(z3)[first_i])
# print(np.abs(z3)[second_i])
# print(np.abs(z3)[third_i])
#
# point = np.arange(2000)
# A4 = 9.485
# A5 = 5.008
# A6 = 2.334
# T4 = 7
# T5 = 13
# T6 = 17
# est = A4*np.sin(2*np.pi*point/T4) + A5*np.sin(2*np.pi*point/T5) + A6*np.sin(2*np.pi*point/T6)
# plt.plot(data_from_file, label='Original')
# plt.plot(est, label='Calculated')
# plt.xlim(0, xmax)
# plt.title('Comparison between Original Graph and Calculated Result')
# plt.xlabel('Time')
# plt.ylabel('Position')
# plt.legend()
# plt.savefig('Comparison between Original Graph and Calculated Result.pdf')
# plt.show()
#
#
# freq = point  # frequency values, like time is the time values
# width = 1  # width=2*sigma**2 where sigma is the standard deviation
# peak1 = 286    # ideal value is approximately N/T1
# peak2 = 154
# peak3 = 118
#
# filter_function = (np.exp(-(freq-peak1)**2/width)+np.exp(-(freq+peak1-2000)**2/width))\
#                   +(np.exp(-(freq-peak2)**2/width)+np.exp(-(freq+peak2-2000)**2/width))\
#                   +(np.exp(-(freq-peak3)**2/width)+np.exp(-(freq+peak3-2000)**2/width))
#
# z_filtered = z3*filter_function
# fig, (ax1,ax2,ax3)=plt.subplots(3,1,sharex='col')
# # this gives us an array of 3 graphs, vertically aligned
# ax1.plot(np.abs(z3))
# ax2.plot(np.abs(filter_function))
# ax3.plot(np.abs(z_filtered))
#
# fig.subplots_adjust(hspace=0)
# ax1.set_ylabel('Noisy FFT')
# ax2.set_ylabel('Filter Function')
# ax3.set_ylabel('Filtered FFT')
# ax3.set_xlabel('Absolute value of FFT of Position-Time\n(Amplitude-Frequency)')
# plt.tight_layout()
# plt.savefig('Pickle Filtered.pdf')
# plt.show()
#
# z_est = np.fft.fft(est)
# plt.plot(np.abs(z_filtered))
# plt.plot(np.abs(z_est))
# plt.show()
#
# c = np.fft.ifft(z_filtered)
# plt.plot(est)
# plt.plot(np.real(c))
# plt.xlim(0, xmax)
# plt.show()
#
# cleaned=np.fft.ifft(z_filtered)
# fig, (ax1,ax2,ax3)=plt.subplots(3,1,sharex='col',sharey='col')
# ax1.plot(point/2000,data_from_file)
# ax2.plot(point/2000,np.real(cleaned))
# ax3.plot(point/2000,est)
# fig.subplots_adjust(hspace=0)
# ax1.set_ylim(-13,13)
# plt.xlim(0, xmax/2000)
# ax1.set_ylabel('Original Data')
# ax2.set_ylabel('Filtered Data')
# ax3.set_ylabel('Ideal Result')
# ax3.set_xlabel('Position-Time')
# plt.savefig('SingleWaveAndNoiseFFT.pdf')
# plt.show()
#
#
#
# # Exercise 4
# A = 5
# a = 0.125
# N = 200
# t = np.arange(N)
#
#
# def f(x):
#     return a * x
#
#
# y = f(t)
# plt.plot(y)
# plt.title('y = 0.125t^2')
# plt.xlabel('Time')
# plt.ylabel('Position')
# plt.savefig('w = 0.125t.pdf')
# plt.show()
#
# z = np.fft.fft(y)
# plt.plot(np.abs(z))
# plt.title('Fourier Transformation of y = 0.125t^2')
# plt.xlabel('Frequency')
# plt.ylabel('Amplitude')
# plt.savefig('Fourier Transformation of y = 0.125t^2.pdf')
# plt.show()
