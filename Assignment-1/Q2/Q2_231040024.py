import cv2
import numpy as np
import librosa

# Function to extract spectrogram
def extractSpectrogram(audioPath, nfft=2048, hopLength=512):
    """Function to exctract frequencies using stft"""
    y, sr = librosa.load(audioPath)
    return np.abs(librosa.stft(y, n_fft=nfft, hop_length=hopLength))

def solution(audio_path):
    ############################################
    ############################################
    # Author: CIPH3R <aravindswami135@gmail.com>
    ############################################

    # Getting the mean of spectrographic values
    meanAmplitude = np.mean(extractSpectrogram(audio_path).flatten())
    threshold = 0.5 # Tweakable according to needs
    
    if meanAmplitude > threshold:
        class_name = "metal"  # High quality
    else:
        class_name =  "cardboard"  # Low quality

    # Display result
    print(audio_path, end=": ")
    print(class_name)

    ############################
    ############################
    return class_name
