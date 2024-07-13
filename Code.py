import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
import scipy.io.wavfile as wav
from scipy.fft import fft, ifft, fftshift, fftfreq


# Function to record audio
def record_audio(filename, duration, samplerate):
    print("Recording...")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float64')
    sd.wait()
    wav.write(filename, samplerate, recording)
    print(f"Recording saved to {filename}")


# Function to plot time signal
def plot_signal(signal, samplerate, title):
    t = np.arange(signal.shape[0]) / samplerate
    plt.figure()
    plt.plot(t, signal)
    plt.title(title)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.show()


# Function to scale and shift time signal
def scale_and_shift(signal, scale, shift, samplerate):
    t = np.arange(signal.shape[0]) / samplerate
    yt = np.interp(t, (t - shift) / scale, signal, left=0, right=0)
    return yt


# Function to add two signals
def add_signals(signal1, signal2):
    return signal1 + signal2


# Function to compute Fourier transform
def compute_fourier(signal):
    return fft(signal)


# Function to inverse Fourier transform
def inverse_fourier(signal_freq):
    return ifft(signal_freq).real


def shift_frequency(fourier_transform, ws):
    shifted = np.roll(fourier_transform, int(ws))
    return shifted


# Function to apply low pass filter
def low_pass_filter(signal_freq, cutoff, samplerate):
    freq = np.fft.fftfreq(signal_freq.size, 1 / samplerate)
    filter_mask = np.abs(freq) <= cutoff
    return signal_freq * filter_mask


# Function to apply high pass filter
def high_pass_filter(signal_freq, cutoff, samplerate):
    freqs = np.fft.fftfreq(signal_freq.size, 1 / samplerate)
    filter_mask = np.abs(freqs) > cutoff
    return signal_freq * filter_mask


# Function to play a WAV file
def play_wav(data, sample_rate):
    sd.play(data, sample_rate)
    sd.wait()


def triangular_filter(filepath, wc):
    samplerate, data = wav.read(filepath)
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    N = len(data)
    yf = fft(data)
    xf = fftfreq(N, 1 / samplerate)

    response = np.zeros_like(xf)

    for i in range(len(xf)):
        if -wc <= xf[i] <= -wc / 2:
            response[i] = (xf[i] + wc) / (wc / 2)
        elif -wc / 2 < xf[i] <= wc / 2:
            response[i] = 1 - 2 * abs(xf[i]) / wc
        elif wc / 2 < xf[i] < wc:
            response[i] = (wc - xf[i]) / (wc / 2)

    yf_filtered = yf * response
    filtered_signal = ifft(yf_filtered)
    play_wav(filtered_signal.real, samplerate)
    plt.figure(figsize=(12, 6))
    plt.plot(xf, response)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.title('Triangular filter response')
    plt.show()




# Load recorded audio
def mainMethod(filename, samplerate):
    data, samplerate = sf.read(filename)
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
        print("play the signal")
        play_wav(data, samplerate)
        
        # Plot original time signal
        
        plot_signal(data, samplerate, 'Original Time Signal ' )
        # Scale and shift the signal
        scale = 2  # Scaling factor
        shifted = 2  # Time shift
        scaled_shifted_signal = scale_and_shift(data, scale, shifted, samplerate)
        # Plot scaled and shifted signal
        plot_signal(scaled_shifted_signal, samplerate, 'Scaled and Shifted Time Signal')
        print("play the signal")
        play_wav(scaled_shifted_signal, samplerate)
        added_signal = add_signals(data, scaled_shifted_signal)
        plot_signal(added_signal, samplerate, 'Added Signal')
        print("play the signal")
        play_wav(added_signal, samplerate)
        fourier_transform = compute_fourier(data)
        # Plot Fourier transform
        plt.figure()
        plt.plot(np.fft.fftfreq(data.size, 1 / samplerate), np.abs(fourier_transform))
        plt.title('Fourier Transform')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude')
        plt.show()
        ws = 1000
        shiftedfreq = shift_frequency(fourier_transform, ws)
        plt.figure()
        plt.plot(np.fft.fftfreq(data.size, 1 / samplerate), np.abs(shiftedfreq))
        plt.title('Fourier Transform shifted')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude')
        plt.show()
        shiftedIntime = inverse_fourier(shiftedfreq)
        plot_signal(shiftedIntime, samplerate, 'Shifted Time Signal')
        print("play the signal")
        play_wav(shiftedIntime, samplerate)

    cutoff = 1000  # Cutoff frequency
    low_passed_freq = low_pass_filter(fourier_transform, cutoff, samplerate)
    low_passed_signal = inverse_fourier(low_passed_freq)
    plot_signal(low_passed_signal, samplerate, 'Low Passed Signal')
    print("play the signal")
    play_wav(low_passed_signal, samplerate)
    
    high_passed_freq = high_pass_filter(fourier_transform, cutoff, samplerate)
    high_passed_signal = inverse_fourier(high_passed_freq)
    plot_signal(high_passed_signal, samplerate, 'High Passed Signal')
    print("play the signal")
    play_wav(high_passed_signal, samplerate)
    
    triangular_filter(filename, cutoff)
    
    print("end")

# Kirat Voice
samplerate = 44100  # Sampling frequency
kiratsFile = "C:/Users/Ahmed.Kirat/Desktop/Kirat.wav"
mainMethod(kiratsFile, samplerate)
nourhansFile="C:/Users/Ahmed.Kirat/Desktop/Nourhan.wav"
mainMethod(nourhansFile, samplerate)