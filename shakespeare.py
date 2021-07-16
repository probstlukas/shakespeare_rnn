import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

import numpy as np
import os
import time

# Fetch text
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

# Preprocess text by reading and then decoding
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
vocab = sorted(set(text))

ids_from_chars = preprocessing.StringLookup(vocabulary=list(vocab), mask_token=None)

# Invert the ID-representation to recover human-readable strings from it
chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)

# Divide the text into example sequences
all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))

# Convert the text vector into a stream of character indices
ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

seq_length = 100

# Convert these individual characters to sequences of the desired size
sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)

### Create Training Examples
# Take a sequence as input, duplicate, and shift it to align the input and label for each timestep
def split_input_target(sequence):
    input_text = sequence[:-1] # Every char except the last one
    target_text = sequence[1:] # Every char except the first one
    return input_text, target_text

dataset = sequences.map(split_input_target)

# Batch size
BATCH_SIZE = 64

# Buffer size to shuffle the dataset
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)

vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024

### Build Model
class BuildModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        # An LSTM layer can also be used
        self.gru = tf.keras.layers.GRU(rnn_units, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x

model = BuildModel(
    # Make sure that the vocabulary size matches the 'StringLookup' layers
    vocab_size=len(ids_from_chars.get_vocabulary()),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units)

### Train Model
loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

# Attach an optimizer
model.compile(optimizer='adam', loss=loss)

# Configure checkpoints
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)

# Number of epochs (iterations on a dataset)
EPOCHS = 30

history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

class OneStep(tf.keras.Model):
    def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.model = model
        self.chars_from_ids = chars_from_ids
        self.ids_from_chars = ids_from_chars

        # Create a mask to prevent unknown words from being generated
        skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
        sparse_mask = tf.SparseTensor(
            # Put a -inf at each bad index.
            values=[-float('inf')]*len(skip_ids),
            indices=skip_ids,
            # Match the shape to the vocabulary
            dense_shape=[len(ids_from_chars.get_vocabulary())])
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)

    @tf.function
    def generate_one_step(self, inputs, states=None):
        # Convert strings to token IDs
        input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
        input_ids = self.ids_from_chars(input_chars).to_tensor()

        # Run the model
        predicted_logits, states = self.model(inputs=input_ids, states=states, return_state=True)
        # Only use the last prediction
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits/self.temperature
        # Apply the prediction mask
        predicted_logits = predicted_logits + self.prediction_mask

        # Sample the output logits to generate token IDs
        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)

        # Convert from token IDs to characters
        predicted_chars = self.chars_from_ids(predicted_ids)

        # Return the characters and model state
        return predicted_chars, states

one_step_model = OneStep(model, chars_from_ids, ids_from_chars)

### Generate Text
start = time.time()
states = None
next_char = tf.constant(['ROMEO:'])
result = [next_char]

for n in range(1000):
  next_char, states = one_step_model.generate_one_step(next_char, states=states)
  result.append(next_char)

result = tf.strings.join(result)
end = time.time()
generated_text = result[0].numpy().decode('utf-8') 
print(generated_text, '\n\n' + '_'*80)
print('\nRun time:', end - start)

# Write generated text in .txt file
output_path = 'output/out.txt'
out_text = open(output_path, "w")
out_text.write(generated_text)
out_text.close()
