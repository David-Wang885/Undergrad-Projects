import numpy as np
from scipy import optimize as op
import matplotlib.pyplot as plt

def get_tau_step():
    """calculate how far a photon travels before it gets scattered .
    Input : tau - optical depth of the atmosphere Output: optical depth traveled"""
    delta_tau = -np.log(np.random.random())
    return delta_tau

def emit_photon( tau_max ) :
    """Emit a photon from the stellar core. Input : tau max - max optical depth Output :
    tau: optical depth at which the photon is created mu: directional cosine of the photon emitted """
    tau = tau_max
    delta_tau = get_tau_step()
    mu = np.random.random()
    return tau-delta_tau*mu, mu

def scatter_photon(tau):
    """Scatter a photon . Input : tau âĹŠ optical depth of the atmosphere Output : tau: new optical depth
    mu: directional cosine of the photon scattered"""
    delta_tau = get_tau_step()
    # sample mu uniformly from -1 to 1
    mu = 2*np.random.random()-1
    tau = tau + delta_tau*mu
    return tau, mu


def Question_2a(tau_max):
    """Calculate the photon's final scatter angle mu"""
    tau_value, mu = emit_photon(tau_max)
    iteration = 1
    while tau_value >= 0:
        tau_value, mu = scatter_photon(tau_value)
        iteration += 1

        if tau_value >= tau_max:                    # If the photon got bounced back to the core
            tau_value, mu = emit_photon(tau_max)    # Emit a new photon
            iteration = 1
    return mu

def Question_2b_histogram(tau_max):
    """
    Calculate the final mu for 10**5 photons and plot histogram and curve fit the intesnsity vs mu plot
    """
    protons = 10**5
    final_mu = np.empty(protons, dtype=float)
    count = 0
    num_ones = 0
    while count < protons:
        final_mu[count] = abs(Question_2a(tau_max))     # Absolute value for better readibility
        if final_mu[count] >= 0.95:               # Count the number of photon that is in the interval of 0.95 ~ 1 (mu)
            num_ones += 1
        count += 1
    plt.figure()
    hist, bins = np.histogram(final_mu, bins=20)
    hist = hist/num_ones
    width = np.diff(bins)
    center = (bins[:-1] + bins[1:]) / 2
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(center, hist, align='center', width=width)
    ax.set_xticks(bins)
    plt.xlabel("µ", fontsize=20)
    plt.ylabel("N(µ)/N(1)", fontsize=20)
    plt.title("Histogram of N(µ)/N(1) vs µ", fontsize=20)
    plt.savefig("Histogram.png")
    Question_2b_curve(final_mu)

def Question_2b_curve(data):
    point = []
    value = []
    start = 0
    end = 1/20
    for i in range(0, 20):
        value.append(count_intensity(data, start, end) / count_intensity(data, 1 - 1/20, 1))
        point.append(start + (end - start)/2)
        start = end
        end += 1/20
    plt.figure()
    point = np.asarray(point)
    result, covar = op.curve_fit(f, point, value, [1, 0.1])
    print(result)
    fit_result = f(point, result[0], result[1])
    plt.plot(point, fit_result, label="Curve fit result")
    plt.plot(point, value,"ro", label="Intensity")
    plt.legend()
    plt.xlabel("µ")
    plt.ylabel("I(µ)")
    plt.title("I(µ) vs µ")
    plt.savefig("Curve2b.png")
    plt.show()

def f(x,a,b):
    return a/x + b

def count_intensity(raw_data, start, end):
    count = 0
    for x in raw_data:
        if x >=start and x < end:
            count += 1
    avg = start + (end - start)/2
    return (count / avg)

if __name__ == "__main__":
    #Question_2b(10)
    Question_2b_histogram(0.0001)
