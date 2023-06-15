import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft

# Plotting settings
plt.rcParams["font.size"] = 11

# Data parameters
filename = "Book1 2.csv"
t_s = 20 # Time step [fs]
t_start = 150 # Start of temporal window [fs]
t_end = 1000 # End of temporal window [fs]


if __name__ == "__main__":
    df = pd.read_csv(filename)

    time = df["Time"]
    # If you want to select a different column, change "Amplitude" to "Amplitude.1" or "Amplitude.2"
    amplitude = df["Amplitude"]

    # Plot raw data
    plt.figure()
    plt.plot(time, amplitude, color="k", linewidth=0.75)
    plt.title("Raw data")
    plt.xlabel("Time (fs)")
    plt.ylabel("Amplitude")
    plt.show()

    # Zoom in on temporal window
    amplitude = amplitude[np.logical_and(time>t_start, time<t_end)]
    time = time[np.logical_and(time>t_start, time<t_end)]

    # Center signal at amplitude = 0
    # amplitude = amplitude - np.mean(amplitude)

    # Zero pad
    # amplitude = np.pad(amplitude, 100)
    # time = np.arange(0, len(amplitude) * t_s, t_s)

    # Plot processed data
    plt.figure()
    plt.plot(time, amplitude, color="k", linewidth=0.75, marker="d")
    plt.title("Filtered data")
    plt.xlabel("Time (fs)")
    plt.ylabel("Amplitude")
    plt.show()

    # Take the Fourier transform
    t_s = t_s * 1e-15
    s_r = 1 / t_s
    X = fft(amplitude)
    N = len(X)
    n = np.arange(N)
    T = N / s_r
    freq = n/T

    fq = freq * 1e-12 # [THz]
    fq_amplitude = np.abs(X)

    # Normalize
    fq_amplitude = fq_amplitude / np.max(fq_amplitude)

    # Plot Fourier transform
    plt.plot(fq, fq_amplitude, color="k", linewidth=0.75, marker="d")
    plt.title("FFT")
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Amplitude")
    plt.xlim(5, np.max(fq)/2)
    plt.ylim(0, 0.005)

    # Save figure
    plt.savefig("fft_figure.jpg", dpi=500)
    
    plt.show()

    # Save data
    np.savetxt("fft_data.csv", np.c_[fq, fq_amplitude], delimiter=",")