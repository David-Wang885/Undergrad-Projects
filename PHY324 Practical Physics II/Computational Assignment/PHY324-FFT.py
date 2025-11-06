# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 13:53:25 2017

@author: Brian
"""

save=True # if True then we save images as files

from random import gauss
import matplotlib.pyplot as plt
import numpy as np

N=200   # N is how many data points we will have in our sine wave

time=np.arange(N)

A1=5.   # wave amplitude
T1=17.  # wave period
y=A1*np.sin(2.*np.pi*time/T1)


noise_amp=A1/2. 
# set the amplitude of the noise relative to sine's amp

"""
i=0
noise=[]
while i < N:
    noise.append(gauss(0,noise_amp))
    i+=1
"""
noise=[gauss(0,noise_amp) for _usused_variable in range(len(y))]
# this line, and the commented block above, do exactly the same thing

x=y+noise
# y is our pure sine wave, x is y with noise added

z1=np.fft.fft(y)
z2=np.fft.fft(x)
# take the Fast Fourier Transforms of both x and y

fig, ( (ax1,ax2), (ax3,ax4) ) = plt.subplots(2,2,sharex='col',sharey='col')
""" 
this setups up a 2x2 array of graphs, based on the first two arguments
of plt.subplots()

the sharex and sharey force the x- and y-axes to be the same for each 
column
"""

ax1.plot(time/N,y)
ax2.plot(np.abs(z1))
ax3.plot(time/N,x)
ax4.plot(np.abs(z2))
""" 
our graphs are now plotted

(ax1,ax2) is a list of figures which are the top row of figures

therefore ax1 is top-left and ax2 is top-right

we plot the position-time graphs rescaled by a factor of N so that
the FFT x-axis agrees with the frequency we could measure from the
position-time graph. by default, both graphs use "data-point number"
on their x-axes, so would go 0 to 200 since N=200.
"""

fig.subplots_adjust(hspace=0)
# remove the horizontal space between the top and bottom row
ax3.set_xlabel('Position-Time')
ax4.set_xlabel('Absolute value of FFT of Position-Time\n(Amplitude-Frequency)')
ax3.set_ylim(-13,13)
ax4.set_ylim(0,480)
ax1.set_ylabel('Pure Sine Wave')
ax3.set_ylabel('Same Wave With Noise')

mydpi=300
plt.tight_layout()

if (save): plt.savefig('SingleWaveAndNoiseWithFFT.png',dpi=mydpi)
plt.show()
"""
plt.show() displays the graph on your computer

plt.savefig will save the graph as a .png file, useful for including
in your report so you don't have to cut-and-paste
"""


M=len(z2)
freq=np.arange(M)  # frequency values, like time is the time values
width=128  # width=2*sigma**2 where sigma is the standard deviation
peak=14    # ideal value is approximately N/T1

filter_function=(np.exp(-(freq-peak)**2/width)+np.exp(-(freq+peak-M)**2/width))
z_filtered=z2*filter_function
"""
we choose Gaussian filter functions, fairly wide, with
one peak per spike in our FFT graph

we eyeballed the FFT graph to figure out decent values of 
peak and width for our filter function

a larger width value is more forgiving if your peak value
is slightly off

making width a smaller value, and fixing the value of peak,
will give us a better final result
"""

fig, (ax1,ax2,ax3)=plt.subplots(3,1,sharex='col')
# this gives us an array of 3 graphs, vertically aligned
ax1.plot(np.abs(z2))  
ax2.plot(np.abs(filter_function))
ax3.plot(np.abs(z_filtered))
"""
note that in general, the fft is a complex function, hence we plot
the absolute value of it. in our case, the fft is real, but the
result is both positive and negative, and the absolute value is still
easier to understand

if we plotted (abs(fft))**2, that would be called the power spectra
"""

fig.subplots_adjust(hspace=0)
ax1.set_ylim(0,480)
ax2.set_ylim(0,1.2)
ax3.set_ylim(0,480)
ax1.set_ylabel('Noisy FFT')
ax2.set_ylabel('Filter Function')
ax3.set_ylabel('Filtered FFT')
ax3.set_xlabel('Absolute value of FFT of Position-Time\n(Amplitude-Frequency)')

plt.tight_layout() 
""" 
the \n in our xlabel does not save to file well without the
tight_layout() command
"""

if(save): plt.savefig('FilteringProcess.png',dpi=mydpi)
plt.show()

cleaned=np.fft.ifft(z_filtered)
"""
ifft is the inverse FFT algorithm

it converts an fft graph back into a sinusoidal graph

we took the data, took the fft, used a filter function 
to eliminate most of the noise, then took the inverse fft
to get our "cleaned" version of the original data
"""

fig, (ax1,ax2,ax3)=plt.subplots(3,1,sharex='col',sharey='col')
ax1.plot(time/N,x)
ax2.plot(time/N,np.real(cleaned))
ax3.plot(time/N,y)
"""
we plot the real part of our cleaned data - but since the 
original data was real, the result of our tinkering should 
be real so we don't lose anything by doing this

if you don't explicitly plot the real part, python will 
do it anyway and give you a warning message about only
plotting the real part of a complex number. so really, 
it's just getting rid of a pesky warning message
"""

fig.subplots_adjust(hspace=0)
ax1.set_ylim(-13,13)
ax1.set_ylabel('Original Data')
ax2.set_ylabel('Filtered Data')
ax3.set_ylabel('Ideal Result')
ax3.set_xlabel('Position-Time')

if(save): plt.savefig('SingleWaveAndNoiseFFT.png',dpi=mydpi)
plt.show()
