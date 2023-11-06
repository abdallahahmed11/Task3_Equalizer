import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Step 1: Generate a sample signal
fs = 500  # Sample rate (Hz)
t = np.arange(0, 1, 1/fs)  # Time array
freq1, freq2 = 5, 25  # Frequencies of sinusoids (Hz)
sig = np.sin(2*np.pi*freq1*t) + np.sin(2*np.pi*freq2*t)

# Step 2: Take the FFT
fft = np.fft.fft(sig)

# Create a figure with a plot for the input signal and sliders
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.3)

# Step 3: Create Slider objects for each frequency band
slider_ax1 = plt.axes([0.1, 0.15, 0.8, 0.05])
slider_ax2 = plt.axes([0.1, 0.05, 0.8, 0.05])
slider1 = Slider(slider_ax1, 'Low Freqs', 0.1, 2.0, valinit=1.0)
slider2 = Slider(slider_ax2, 'High Freqs', 0.1, 2.0, valinit=1.0)

# Function to be called whenever a slider value changes
def update(val):
    # Get the slider values
    val1 = slider1.val
    val2 = slider2.val

    # Segment the FFT results into low and high frequencies
    low_freqs = fft[:len(fft)//2]  # First half of the array
    high_freqs = fft[len(fft)//2:]  # Second half of the array

    # Modify the FFT results using the slider values
    modified_fft = np.concatenate([low_freqs*1, high_freqs*0.4])

    # Step 4: Take the inverse FFT
    reconstructed_sig = np.fft.ifft(modified_fft).real

    # Update the plot
    ax.clear()
    ax.plot(t, sig, label='Original')
    ax.plot(t, reconstructed_sig, label='Reconstructed')
    ax.legend()
    plt.show()

# Call the update function whenever a slider value changes
slider1.on_changed(update)
slider2.on_changed(update)


update(3)

plt.show()
