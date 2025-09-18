import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.signal import butter, lfilter

# Audio settings
RATE = 44100  # Sampling rate
DURATION = 5  # Duration of generated noise in seconds

# Frequency bands for EQ
FREQ_BANDS = [100, 300, 1000, 3000, 10000]  # Hz
GAIN_LEVELS = [1.0] * len(FREQ_BANDS)  # Initial gains

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq + .00001
    high = highcut / nyq + .00001
    print(low, high)
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_eq(noise, gains, fs):
    filtered_signal = np.zeros_like(noise)
    band_limits = [0] + FREQ_BANDS + [fs // 2]

    for i in range(len(FREQ_BANDS)):
        lowcut, highcut = band_limits[i], band_limits[i + 2]
        b, a = butter_bandpass(lowcut, highcut, fs)
        filtered_signal += gains[i] * lfilter(b, a, noise)

    return filtered_signal

def audio_callback(outdata, frames, time, status):
    global noise, GAIN_LEVELS
    processed_signal = apply_eq(noise[:frames], GAIN_LEVELS, RATE)
    outdata[:] = processed_signal.reshape(-1, 1)


# Generate white noise
np.random.seed(0)
noise = np.random.normal(0, 0.1, RATE * DURATION)  # 5 seconds of noise

# Start audio stream
stream = sd.OutputStream(samplerate=RATE, channels=1, callback=audio_callback)
stream.start()

# Create GUI
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
sliders = []
ax.set_title("Equalizer")
ax.set_yticks([])
ax.set_xticks([])

for i, freq in enumerate(FREQ_BANDS):
    ax_slider = plt.axes([0.25, 0.05 + i * 0.1, 0.65, 0.03])
    sliders.append(Slider(ax_slider, f"{freq} Hz", 0.1, 3.0, valinit=1.0))

def update(val):
    global GAIN_LEVELS
    GAIN_LEVELS = [slider.val for slider in sliders]


for slider in sliders:
    slider.on_changed(update)

plt.show()

# Cleanup
stream.stop()
stream.close()
