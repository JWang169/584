import re
import string
from unicodedata import normalize
import numpy as np
import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Attention
from tensorflow.keras.models import Model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot, plot_model
from sklearn.model_selection import train_test_split

# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, mode='rt', encoding='utf-8')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

# split a loaded document into sentences
def to_pairs(doc):
    lines = doc.strip().split('\n')
    pairs = [line.split('\t') for line in  lines]
    return pairs

def clean_data(lines):
    cleaned = list()
    # prepare regex for char filtering
    re_print = re.compile('[^%s]' % re.escape(string.printable))
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)
    for pair in lines:
        clean_pair = list()
        for line in pair:
            # normalize unicode characters
            line = normalize('NFD', line).encode('ascii', 'ignore')
            line = line.decode('UTF-8')
            # tokenize on white space
            line = line.split()
            # convert to lowercase
            line = [word.lower() for word in line]
            # remove punctuation from each token
            line = [word.translate(table) for word in line]
            # remove non-printable chars form each token
            line = [re_print.sub('', w) for w in line]
            # remove tokens with numbers in them
            line = [word for word in line if word.isalpha()]
            # store as string
            clean_pair.append(' '.join(line))
        cleaned.append(clean_pair)
    return np.array(cleaned)


filename = "fra.txt"
doc = load_doc(filename)
pairs = to_pairs(doc)
# choose sample size
n_train = 20000
clean_pairs = clean_data(pairs)[0:n_train, :]
# clean_pairs = clean_data(pairs)
input_texts = clean_pairs[:, 0]
target_texts = ['\t' + text + '\n' for text in clean_pairs[:, 1]]
max_encoder_seq_length = max(len(line) for line in input_texts)
max_decoder_seq_length = max(len(line) for line in target_texts)


# split data into training and testing 
X_train, X_test, y_train, y_test = train_test_split(input_texts, target_texts, test_size=0.2, random_state=42)

def text2sequences(max_len, lines):
    tokenizer = Tokenizer(char_level=True, filters='')
    tokenizer.fit_on_texts(lines)
    seqs = tokenizer.texts_to_sequences(lines)
    seqs_pad = pad_sequences(seqs, maxlen=max_len, padding='post')
    return seqs_pad, tokenizer.word_index

encoder_input_seq, input_token_index = text2sequences(max_encoder_seq_length, X_train)
decoder_input_seq, target_token_index = text2sequences(max_decoder_seq_length, y_train)

num_encoder_tokens = len(input_token_index) + 1
num_decoder_tokens = len(target_token_index) + 1

print('num_encoder_tokens: ' + str(num_encoder_tokens))
print('num_decoder_tokens: ' + str(num_decoder_tokens))

def onehot_encode(sequences, max_len, vocab_size):
    n = len(sequences)
    data = np.zeros((n, max_len, vocab_size))
    for i in range(n):
        data[i, :, :] = to_categorical(sequences[i], num_classes=vocab_size)
    return data

# encode and decode data
encoder_input_data = onehot_encode(encoder_input_seq, max_encoder_seq_length, num_encoder_tokens)
decoder_input_data = onehot_encode(decoder_input_seq, max_decoder_seq_length, num_decoder_tokens)
decoder_target_seq = np.zeros(decoder_input_seq.shape)
decoder_target_seq[:, 0:-1] = decoder_input_seq[:, 1:]
decoder_target_data = onehot_encode(decoder_target_seq,
                                    max_decoder_seq_length,
                                    num_decoder_tokens)


# encoder model
latent_dim = 256

# inputs of the encoder network
encoder_inputs = Input(shape=(None, num_encoder_tokens), 
                       name='encoder_inputs')

# set the LSTM layer
encoder_lstm = LSTM(latent_dim, return_state=True, 
                    dropout=0.5, name='encoder_lstm')
_, state_h, state_c = encoder_lstm(encoder_inputs)

# build the encoder network model
encoder_model = Model(inputs=encoder_inputs, 
                      outputs=[state_h, state_c],
                      name='encoder')

# decoder model

# inputs of the decoder network
decoder_input_h = Input(shape=(latent_dim,), name='decoder_input_h')
decoder_input_c = Input(shape=(latent_dim,), name='decoder_input_c')
decoder_input_x = Input(shape=(None, num_decoder_tokens), name='decoder_input_x')

# set the LSTM layer
decoder_lstm = LSTM(latent_dim, return_sequences=True, 
                    return_state=True, dropout=0.5, name='decoder_lstm')
decoder_lstm_outputs, state_h, state_c = decoder_lstm(decoder_input_x, 
                                                  initial_state=[decoder_input_h, decoder_input_c])

# set the attention layer
attention_layer = Attention()([state_h, decoder_lstm_outputs])

# attention_layer = Attention()([state_h, decoder_input_h])
attention_decoder_concat = Concatenate()([attention_layer, decoder_lstm_outputs])

# set the dense layer
decoder_dense = Dense(num_decoder_tokens, activation='softmax', name='decoder_dense')
decoder_outputs = decoder_dense(decoder_lstm_outputs)

# build the decoder network model
decoder_model = Model(inputs=[decoder_input_x, decoder_input_h, decoder_input_c],
                      outputs=[decoder_outputs, state_h, state_c],
                      name='decoder')


# connect encoder and decoder

# input layers
encoder_input_x = Input(shape=(None, num_encoder_tokens), name='encoder_input_x')
decoder_input_x = Input(shape=(None, num_decoder_tokens), name='decoder_input_x')

# connect encoder to decoder
encoder_final_states = encoder_model([encoder_input_x])
decoder_lstm_output, _, _ = decoder_lstm(decoder_input_x, initial_state=encoder_final_states)
decoder_pred = decoder_dense(decoder_lstm_output)

model = Model(inputs=[encoder_input_x, decoder_input_x], 
              outputs=decoder_pred, 
              name='model_training')


import tensorflow
class LossHistory(tensorflow.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
#         self.x = []
#         self.losses = []
#         self.val_losses = []
#         self.logs = []

    def on_batch_end(self, batch, logs={}):
#         self.logs.append(logs)
#         self.x.append(self.i)
#         self.losses.append(logs.get('loss'))
#         self.val_losses.append(logs.get('val_loss'))
#         # print every 200 iteration. batch_size*200
#         if self.i % (200*64) == 0:
        
        print('The current Loss is',logs)
#         self.i += 1

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])





