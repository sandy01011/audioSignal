import os
import numpy as np
import tensorflow as tf
import librosa

# Define the length of audio chunks (in seconds)
chunk_duration = 1.0

# Define the sample rate for audio recordings
sample_rate = 16000

# Define the labels for hotword and non-hotword audio samples
hotword_label = 1
non_hotword_label = 0

# Define the output directory for saving the trained model
model_dir = 'hotword_model'

# Define the number of samples to collect for each label
num_hotword_samples = 10
num_non_hotword_samples = 10

# Define the neural network architecture
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(13, 32)),
    tf.keras.layers.Reshape((13, 32, 1)),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Loading audio samples
print('loading audio samples...')
samples = []
labels = []
for label, label_name in enumerate(['non_hotword', 'hotword']):
    for i in range(num_non_hotword_samples if label == 0 else num_hotword_samples):
        sample_path = os.path.join('data', f'{label_name}_sample_{i}.npy')
        #print('sample_path id{}, label {}, type {}, shape {}, size {}, ndim {}'.format(i, label_name, type(sample_path),sample_path.shape,sample_path.size,sample_path.ndim))
        metadata_path = os.path.join('data', f'{label_name}_sample_{i}.txt')
        print(sample_path)

        # Load audio sample and metadata from disk
        sample = np.load(sample_path)
        #sample = tf.ragged.constant(sample)
        with open(metadata_path, 'r') as f:
            metadata = f.read()
            print('---metadata----',metadata)
            sample_label = int(metadata.split(':')[1])

        # Add sample and label to list
        samples.append(sample)
        labels.append(sample_label)
        print(type(sample_label), type(sample))

for i in range(len(samples)):
    print('id {}, len {}, type {}, shape {}, size {}, ndim {}'.format(i,len(samples[i]),type(samples[i]), samples[i].size, samples[i].size, samples[i].ndim))


# Pad audio samples to have consistent shape
#max_length = max(sample.shape[0] for sample in samples)
max_length = 63
print('max length>>>',max_length)
print('samples length>>>',len(samples))
padded_samples = np.zeros((len(samples), 13, max_length))
print("##############")
for i, sample in enumerate(samples):
    print('**************')
    print('id {}, type {}, shape {}, size {}, ndim {}'.format(i,type(sample), sample.size, sample.size, sample.ndim))
    padded_samples[i, :len(sample)] = sample

for i in range(len(padded_samples)):
    print('id {}, type {}, shape {}, size {}, ndim {}'.format(i,type(padded_samples[i]), padded_samples[i].size, padded_samples[i].size, padded_samples[i].ndim))

# Convert labels to numpy array
labels = np.array(labels)
print('np labels {}, len {}, size {}, shape {} dim {}'.format(labels, len(labels),labels.size, labels.shape, labels.ndim))
#samples =  tf.ragged.constant(samples)

# Split data into training and test sets
dataset = tf.data.Dataset.from_tensor_slices((samples, labels))
batch_size = 4
train_size = int(len(labels) * 0.8)
train_dataset = dataset.take(train_size).shuffle(train_size).batch(batch_size)
test_dataset = dataset.skip(train_size).batch(batch_size)
print(train_dataset)

# # Train the model
# print('Training model...')
# epochs = 10
# model.fit(train_dataset, epochs=epochs)

# # Evaluate the model on the test set
# print('Evaluating model...')
# test_loss, test_accuracy = model.evaluate(test_dataset)
# print(f'Test accuracy: {test_accuracy}')

# # Save the model to disk
# print('Saving model...')
# if not os.path.exists(model_dir):
#     os.makedirs(model_dir)
# model.save(os.path.join(model_dir, 'hotword_model'))
