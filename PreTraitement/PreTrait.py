import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wav
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import os

def high_pass_filter(data, sample_rate, cutoff_frequency):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_frequency / nyquist
    b, a = signal.butter(1, normal_cutoff, btype='high', analog=False)
    filtered_data = signal.lfilter(b, a, data)
    return filtered_data

def undersample(data, factor):
    return data[::factor]

def process_wav_file(uploaded_file, apply_high_pass, cutoff_frequency, apply_undersample, undersample_factor, output_file):
    sample_rate, data = wav.read(uploaded_file)
    
    # Keep a copy of the original data for comparison
    original_data = data.copy()
    
    # Apply high-pass filter if selected
    if apply_high_pass:
        data = high_pass_filter(data, sample_rate, cutoff_frequency)
    
    # Undersample the data if selected
    if apply_undersample:
        data = undersample(data, undersample_factor)
        output_sample_rate = sample_rate // undersample_factor
    else:
        output_sample_rate = sample_rate
    
    # Save the processed data to a new .wav file
    wav.write(output_file, output_sample_rate, data.astype(np.int16))
    return sample_rate, original_data, output_sample_rate, data, output_file

def plot_waveforms(sample_rate, original_data, output_sample_rate, processed_data):
    # Normalize the data
    original_data = original_data / np.max(np.abs(original_data))
    processed_data = processed_data / np.max(np.abs(processed_data))
    
    # Apply rolling mean for visualization
    window_size = 1000
    original_data_smooth = pd.Series(original_data).rolling(window=window_size).mean().fillna(0)
    processed_data_smooth = pd.Series(processed_data).rolling(window=window_size).mean().fillna(0)
    
    # Plot original waveform
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    time_original = np.linspace(0, len(original_data) / sample_rate, num=len(original_data))
    ax[0].plot(time_original, original_data_smooth)
    ax[0].set_title('Original Waveform')
    ax[0].set_xlabel('Time [s]')
    ax[0].set_ylabel('Amplitude')
    ax[0].set_ylim([-1, 1])  # Set y-axis limits

    # Plot processed waveform
    time_processed = np.linspace(0, len(processed_data) / output_sample_rate, num=len(processed_data))
    ax[1].plot(time_processed, processed_data_smooth)
    ax[1].set_title('Processed Waveform')
    ax[1].set_xlabel('Time [s]')
    ax[1].set_ylabel('Amplitude')
    ax[1].set_ylim([-1, 1])  # Set y-axis limits

    st.pyplot(fig)

# Streamlit app
st.title("WAV File High-Pass Filter and Undersampler")

uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file is not None:
    apply_high_pass = st.checkbox("Apply High-Pass Filter", value=True)
    cutoff_frequency = st.number_input("Cutoff Frequency (Hz)", min_value=1, max_value=20000, value=1000)
    
    apply_undersample = st.checkbox("Apply Undersampling", value=True)
    undersample_factor = st.number_input("Undersample Factor", min_value=1, max_value=10, value=2)
    
    input_file_name = uploaded_file.name
    default_output_file_name = f"{os.path.splitext(input_file_name)[0]}_pre-processing.wav"
    output_file_name = st.text_input("Output file name", value=default_output_file_name)
    
    if st.button("Process"):
        sample_rate, original_data, output_sample_rate, processed_data, output_file = process_wav_file(
            uploaded_file, apply_high_pass, cutoff_frequency, apply_undersample, undersample_factor, output_file_name)
        
        # Calculate file sizes
        input_file_size = uploaded_file.size
        output_file_size = os.path.getsize(output_file)
        
        st.success(f"File processed and saved as {output_file_name}")
        st.audio(output_file_name)
        
        plot_waveforms(sample_rate, original_data, output_sample_rate, processed_data)
        
        # Display file sizes
        st.write(f"Original file size: {input_file_size / (1024 * 1024):.2f} MB")
        st.write(f"Processed file size: {output_file_size / (1024 * 1024):.2f} MB")
