import numpy as np
import matplotlib.pyplot as plt

# Create a sample signal with three frequencies
t = np.linspace(0, 1, 500, False)  # 1 second
sig = 1*np.cos(2*np.pi*10*t) +2* np.sin(2*np.pi*30*t) +10* np.sin(2*np.pi*50*t)

# Compute the Fast Fourier Transform (FFT)
fft = np.fft.fft(sig)

# Compute the magnitudes of the FFT
mag_before = np.abs(fft)

# Compute the frequencies associated with the FFT
freqs = np.fft.fftfreq(len(fft))

# Define the gains for the four sliders
# (In a real application, these would be controlled by the user)
gains = [1,6,1]  # Change these values to see the effect

# Divide the positive frequency range into four parts
freq_ranges = np.linspace(0, max(freqs), 4)
for i in range(3):
    # Find the indices of the FFT coefficients that correspond to this frequency range
    idx = np.where((freqs >= freq_ranges[i]) & (freqs < freq_ranges[i+1]))

    # Apply the gain to these coefficients
    fft[idx] *= gains[i]

    # Also apply the gain to the corresponding negative frequencies
    idx_neg = np.where((freqs >= -freq_ranges[i+1]) & (freqs < -freq_ranges[i]))
    fft[idx_neg] *= gains[i]

# Now we can reconstruct the signal with the equalizer applied
equalized_sig = np.fft.ifft(fft)

# Compute the magnitudes of the equalized FFT
mag_after = np.abs(fft)

# Let's plot the original and equalized signal in separate subplots
fig, axs = plt.subplots(4, 1, figsize=(10, 15))

# Original signal in time domain
axs[0].plot(t, sig)
axs[0].set(xlabel='Time [s]', ylabel='Amplitude')
axs[0].set_title('Original Signal')

# Original signal FFT magnitude
axs[1].plot(freqs, mag_before)
axs[1].set(xlabel='Frequency [Hz]', ylabel='Magnitude')
axs[1].set_title('Original FFT Magnitude')

# Equalized signal in time domain
axs[2].plot(t, np.real(equalized_sig))
axs[2].set(xlabel='Time [s]', ylabel='Amplitude')
axs[2].set_title('Equalized Signal')

# Equalized signal FFT magnitude
axs[3].plot(freqs, mag_after)
axs[3].set(xlabel='Frequency [Hz]', ylabel='Magnitude')
axs[3].set_title('Equalized FFT Magnitude')

plt.tight_layout()
plt.show()

