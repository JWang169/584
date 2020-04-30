import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import re
import string
from unicodedata import normalize
from pickle import dump
from keras.utils import to_categorical
import numpy as np
from keras.layers import Input, GRU, Flatten, Bidirectional, Permute, Dense, concatenate, Embedding, RepeatVector, Activation, Lambda, Concatenate, Dot, LSTM, Multiply, Reshape
from keras.models import Model
import keras
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras.backend as K

def text2sequences(max_len, lines):
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(lines)
    seqs = tokenizer.texts_to_sequences(lines)
    seqs_pad = pad_sequences(seqs, maxlen=max_len, padding='post')
    return seqs_pad, tokenizer.word_index

# customize softmax function
def softmax(x, axis=1):
    """
    Softmax activation function.
    """
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim > 2:
        e = K.exp(x - K.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D')

def one_step_attention(a, s_prev):
    """
    Attention mechanism, return weighted Context Vector
   
    """
    
    # repeat s_prev Tx times, one for each word
    s_prev = repeator(s_prev)
    # connect BiRNN hidden state to s_prev
    concat = concatenator([a, s_prev])
    # compute energies
    e = densor_tanh(concat)
    energies = densor_relu(e)
    # compute weights
    alphas = activator(energies)
    # get weighted Context Vector
    context = dotor([alphas, a])
    
    return context

def pretrained_embedding_layer(word_to_vec_map, source_vocab_to_int):
    """
    build Embedding layer and pretrain word embedding
	"""
    
    vocab_len = len(source_vocab_to_int) + 1        # Keras Embedding API +1
    emb_dim = word_to_vec_map["the"].shape[0]
    
    # initialize embedding matrix
    emb_matrix = np.zeros((vocab_len, emb_dim))
    
    # fit wordvec to embedding layers
    for word, index in source_vocab_to_int.items():
        word_vector = word_to_vec_map.get(word, np.zeros(emb_dim))
        emb_matrix[index, :] = word_vector

    # build embedding layer, cannot be trained
    embedding_layer = Embedding(vocab_len, emb_dim, trainable=False)

    # build embedding layer
    embedding_layer.build((None,))
    
    # set weights
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer

# load pretrained word embedding from glove
with open("glove.6B/glove.6B.50d.txt", 'r') as f:
    words = set()
    word_to_vec_map = {}
    for line in f:
        line = line.strip().split()
        curr_word = line[0]
        words.add(curr_word)
        word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

# Embedding layer
embedding_layer = pretrained_embedding_layer(word_to_vec_map, source_vocab_to_int)


df = pd.read_csv('5k_para.csv')
all_artists = df['Artists']
all_titles = df['Titles']
all_lyrics = df['Lyrics']

# to category
cat_list = list(set(all_artists))
cat_dict = {k: v for v, k in enumerate(cat_list)}


input_cats = list() 
input_texts = list()
target_texts = list()
for i in range(len(all_titles)):
    if isinstance(all_lyrics[i], str):
        input_cats.append(all_artists[i])
        input_texts.append(all_titles[i])
        target_texts.append(all_lyrics[i])


max_encoder_seq_length = max(len(line) for line in input_texts)
max_decoder_seq_length = max(len(line) for line in target_texts)
print('max length of input  sentences: %d' % (max_encoder_seq_length))
print('max length of target sentences: %d' % (max_decoder_seq_length))

source_text_to_int, source_vocab_to_int = text2sequences(max_encoder_seq_length, 
                                                      input_texts)
target_text_to_int, target_vocab_to_int = text2sequences(max_decoder_seq_length, 
                                                       target_texts)

source_int_to_vocab = {word: idx for idx, word in source_vocab_to_int.items()}
target_int_to_vocab = {word: idx for idx, word in target_vocab_to_int.items()}

X = source_text_to_int
Y = target_text_to_int

num_encoder_tokens = len(source_vocab_to_int)
num_decoder_tokens = len(target_vocab_to_int)


# onehot encoding
Xoh = np.array(list(map(lambda x: to_categorical(x, num_classes=len(source_vocab_to_int)), X)))
Yoh = np.array(list(map(lambda x: to_categorical(x, num_classes=len(target_vocab_to_int)), Y)))

Tx = max_encoder_seq_length 
Ty = max_decoder_seq_length

# one hot encode category
input_cat_list = list(set(source_cats))
cat_dict = {k: v for v, k in enumerate(input_cat_list)}
cat_n = len(source_cats)
cat_data = np.zeros((cat_n, 1, 50))
for i in range(cat_n):
    cat_data[i, :, :] = to_categorical(cat_dict[source_cats[i]], num_classes=50)

repeator = RepeatVector(Tx)
concatenator = Concatenate(axis=-1)
densor_tanh = Dense(32, activation = "tanh")
densor_relu = Dense(1, activation = "relu")
activator = Activation(softmax, name='attention_weights')
dotor = Dot(axes = 1)

n_a = 32 # Hidden size of Bi-LSTM
n_s = 128 # Decoder LSTM hidden size 
decoder_LSTM_cell = LSTM(n_s, return_state=True)
output_layer = Dense(len(target_vocab_to_int), activation=softmax)

# define model layers
reshapor = Reshape((1, len(target_vocab_to_int)))
concator = Concatenate(axis=-1)

# input text 
X0 = Input(shape=(Tx,))
# input artist label
X1 = Input(shape=(1,50))

# word Embedding layer
embed = embedding_layer(X0)

# concatenate text and artist
concat = K.concatenate([embed, X1], axis=1)

# initialize Decoder LSTM
s0 = Input(shape=(n_s,), name='s0')
c0 = Input(shape=(n_s,), name='c0')

# Decoder input for LSTM layer
out0 = Input(shape=(num_decoder_tokens, ), name='out0')
out = reshapor(out0)

s = s0
c = c0
    
# save outputs
outputs = []
    
# define Bi-LSTM
a = Bidirectional(LSTM(n_a, return_sequences=True))(embed)
    
# Decoder, iterate max_decoder_seq_length rounds, each iteration generates one result
for t in range(max_decoder_seq_length):
    
    # get Context Vector
    context = one_step_attention(a, s)
        
    # concat Context Vector and the previous translated result
    context = concator([context, reshapor(out)])
    s, _, c = decoder_LSTM_cell(context, initial_state=[s, c])
        
    # connect lstm output and dense layer
    out = output_layer(s)
        
    # save output result
    outputs.append(out)

model = Model([X0, X1, s0, c0, out0], outputs)

model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.001),
              metrics=['accuracy'],
              loss='categorical_crossentropy')

X1 = cat_data
X0 = source_text_to_int
m = source_text_to_int.shape[0] # num of training sample
s0 = np.zeros((m, n_s))
c0 = np.zeros((m, n_s))
out0 = np.zeros((m, len(target_vocab_to_int)))
outputs = list(Yoh.swapaxes(0,1))

history = model.fit([X0, X1, s0, c0, out0], outputs,
          epochs=20, 
          batch_size=64,
          verbose=0)

model.save_weights('word_level.h')






