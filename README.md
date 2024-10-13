# Sound Recording and Analysis

## Project Description
This project enables recording audio, performing time and frequency domain analyses, and applying different signal transformations. It includes Fourier Transform operations, filtering, and time scaling. The project also features signal visualization and allows users to apply various filters such as low-pass, high-pass, and triangular filters to audio files.

## Features
- Audio recording and playback
- Time-domain signal manipulation (scaling, shifting)
- Fourier Transform and frequency-domain analysis
- Low-pass, high-pass, and triangular filters
- Visualization of signals and frequency spectra

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/n-o-u-r-h-a-n/Sound-Recording-and-Analysis.git
    ```
2. Install the required Python libraries:
    ```bash
    pip install numpy matplotlib sounddevice soundfile scipy
    ```

## Usage
1. To record audio:
    ```python
    record_audio('filename.wav', duration, samplerate)
    ```
2. To load an existing audio file for analysis:
    ```python
    mainMethod('path_to_audio_file.wav', samplerate)
    ```
3. The following actions can be performed on the audio signal:
   - Time-domain signal plotting
   - Time scaling and shifting
   - Fourier Transform and inverse Fourier Transform
   - Frequency shift
   - Low-pass and high-pass filtering
   - Triangular filtering

## Functions
- `record_audio(filename, duration, samplerate)`: Records and saves an audio file.
- `plot_signal(signal, samplerate, title)`: Plots the time-domain signal.
- `scale_and_shift(signal, scale, shift, samplerate)`: Scales and shifts the time-domain signal.
- `compute_fourier(signal)`: Computes the Fourier Transform.
- `low_pass_filter(signal_freq, cutoff, samplerate)`: Applies a low-pass filter to the signal.
- `high_pass_filter(signal_freq, cutoff, samplerate)`: Applies a high-pass filter to the signal.
- `triangular_filter(filepath, wc)`: Applies a triangular filter and plots the response.

## Example
To run the example provided in the code:
1. Set the correct file paths for `kiratsFile` and `nourhansFile` variables.
2. Call `mainMethod` to load and analyze the file:
    ```python
    mainMethod(kiratsFile, 44100)
    ```

## License
This project is licensed under the MIT License.
