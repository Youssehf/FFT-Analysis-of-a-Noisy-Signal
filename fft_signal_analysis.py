import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Sampling parameters
Fs = 1000  # Sampling frequency (Hz)
T = 1 / Fs  # Sampling interval
L = 1000  # Number of samples
t = np.linspace(0, L*T, L, endpoint=False)  # Time vector

# Create the signal: mix of 50Hz and 120Hz sine waves
signal = 0.7 * np.sin(2 * np.pi * 50 * t) + 1.0 * np.sin(2 * np.pi * 120 * t)

# Add Gaussian noise
noise = 0.5 * np.random.randn(L)
signal_noisy = signal + noise

# Apply FFT
yf = fft(signal_noisy)
xf = fftfreq(L, T)[:L//2]

# Plot results
plt.figure(figsize=(12, 6))

# Time domain
plt.subplot(2, 1, 1)
plt.plot(t, signal_noisy, label='Noisy Signal')
plt.plot(t, signal, label='Clean Signal', alpha=0.7)
plt.title('Time Domain Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()

# Frequency domain
plt.subplot(2, 1, 2)
plt.plot(xf, 2.0/L * np.abs(yf[:L//2]))
plt.title('Frequency Domain (FFT)')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()
