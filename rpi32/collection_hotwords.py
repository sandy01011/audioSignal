import os
import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf
import sys

# Define the length of audio chunks (in seconds)
chunk_duration = 1.0

# Define the sample rate for audio recordings
sample_rate = 16000

# Define the labels for hotword and non-hotword audio samples
hotword_label = 1
non_hotword_label = 0

# Define the output directory for saving audio samples and metadata
output_dir = 'data/'

# Define the number of samples to collect for each label
num_hotword_samples = 10
num_non_hotword_samples = 10

# Define a callback function for recording audio in chunks
def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    global audio_buffer
    audio_buffer.append(indata.copy())

# Collect hotword audio samples
print('Recording hotword audio samples...')
for i in range(num_hotword_samples):
    # Start recording audio stream
    print(f'Sample {i + 1}/{num_hotword_samples} - Say the hotword now!')
    audio_buffer = []
    with sd.InputStream(channels=1, blocksize=int(chunk_duration * sample_rate),
                        samplerate=sample_rate, callback=callback):
        sd.sleep(int(chunk_duration * 1000))

    # Concatenate audio chunks into one long audio sample
    audio = np.concatenate(audio_buffer, axis=0)

    # Extract features from audio
    features = librosa.feature.mfcc(audio[:, 0], sr=sample_rate, n_mfcc=13)

    # Save audio sample and metadata to disk
    sample_path = os.path.join(output_dir, f'hotword_sample_{i}.npy')
    metadata_path = os.path.join(output_dir, f'hotword_sample_{i}.txt')
    np.save(sample_path, features)
    with open(metadata_path, 'w') as f:
        f.write(f'label: {hotword_label}')

# Collect non-hotword audio samples
print('Recording non-hotword audio samples...')
for i in range(num_non_hotword_samples):
    # Start recording audio stream
    print(f'Sample {i + 1}/{num_non_hotword_samples} - Say something that is not the hotword.')
    audio_buffer = []
    with sd.InputStream(channels=1, blocksize=int(chunk_duration * sample_rate),
                        samplerate=sample_rate, callback=callback):
        sd.sleep(int(chunk_duration * 1000))

    # Concatenate audio chunks into one long audio sample
    audio = np.concatenate(audio_buffer, axis=0)

    # Extract features from audio
    features = librosa.feature.mfcc(audio[:, 0], sr=sample_rate, n_mfcc=13)

    # Save audio sample and metadata to disk
    sample_path = os.path.join(output_dir, f'non_hotword_sample_{i}.npy')
    metadata_path = os.path.join(output_dir, f'non_hotword_sample_{i}.txt')
    np.save(sample_path, features)
    with open(metadata_path, 'w') as f:
        f.write(f'label: {non_hotword_label}')

# Load audio samples and metadata from disk
#samples = []
#labels = []
#for label, label_name in enumerate(['non_hotword', 'hotword']):
#    for i in range(num_non_hotword_samples if label == 0 else num_hotword_samples):
#        sample_path = os.path.join(output_dir, f'{label_name}_sample_{i}.npy')
#        metadata_path = os.path.join(output_dir, f'{label_name}_sample
