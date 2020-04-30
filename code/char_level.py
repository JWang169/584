import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import re
import string
from unicodedata import normalize
from pickle import dump
import numpy as np
from keras.utils import to_categorical
from keras.layers import Input, GRU, Dense, concatenate
from keras.models import Model
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras.backend as K
import matplotlib.pyplot as plt

def read_file(file):
	"""
	read in csv file,
	return 3 lists, each one matchees: input categories, input titles, target text 
	"""
    df = pd.read_csv('5k_para.csv')
    all_artists = df['Artists']
    all_titles = df['Titles']
    all_lyrics = df['Lyrics']
    cat_list = list(set(all_artists))
    cat_dict = {k: v for v, k in enumerate(cat_list)}
    input_cats = list()  # save artists
    input_texts = list()  # save titles
    target_texts = list()  # save lyrics
    for i in range(len(all_lyrics)):
        if isinstance(all_lyrics[i], str):
            input_cats.append(cat_dict[all_artists[i]])
            input_texts.append(all_titles[i])
            target_texts.append('\t' + all_lyrics[i] + '\n')
    return input_cats, input_texts, target_texts


def onehot_encode(sequences, max_len, vocab_size):
    n = len(sequences)
    data = np.zeros((n, max_len, vocab_size))
    for i in range(n):
        data[i, :, :] = to_categorical(sequences[i], num_classes=vocab_size)
    return data


def text2sequences(max_len, lines):
	# encode and pad sequences
    tokenizer = Tokenizer(char_level=True, filters='')
    tokenizer.fit_on_texts(lines)
    seqs = tokenizer.texts_to_sequences(lines)
    seqs_pad = pad_sequences(seqs, maxlen=max_len, padding='post')
    return seqs_pad, tokenizer.word_index


# make predictions:
def beam_decode(candidates):
    cur_hypo = list() # cumulative log for all character, hidden_state as an array
    for candi in candidates:
        prev_log, prev_c, prev_state = candi
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, target_token_index[prev_c[-1]]] = 1. # first decoder input is start sign '\t'
        output_tokens, h = decoder_model.predict([target_seq,prev_state])
        sorted_arr = output_tokens[0][0].argsort() # sorted hypothesis
        for i in range(-1, -beam_size-1, -1): # get top k hypothesis
            idx = sorted_arr[i]
            cur_log = np.log(output_tokens[0][0][idx]) # log for current hypothesis
            cur_c = reverse_target_char_index[idx]
            if cur_c == '<pad>':
                print("sentence completed", prev_c + cur_c)
                continue
            new_c = prev_c + cur_c
            all_hypo.append(new_c)
            avg_log = (prev_log * len(prev_c) + cur_log)/len(new_c)
            cur_hypo.append((avg_log, new_c, h))
    return cur_hypo


# compute rouge score, k=1
def compute_rouge(res, target):
    length = len(res)
    target = target[:length]
    res_set = set(res.split())
    target_set = set(target.split())
    count_match = 0
    for w in res_set:
        if w in target_set:
            count_match += 1
    score = count_match / len(res_set)
    return score


max_encoder_seq_length = max(len(line) for line in input_texts)
max_decoder_seq_length = max(len(line) for line in target_texts)
print('max length of input  sentences: %d' % (max_encoder_seq_length))
print('max length of target sentences: %d' % (max_decoder_seq_length))

# manually set max length for encoder inputs and decoder outputs
max_encoder_seq_length = 30
max_decoder_seq_length = 2000

# encoding texts
encoder_input_seq, input_token_index = text2sequences(max_encoder_seq_length, input_texts)
decoder_input_seq, target_token_index = text2sequences(max_decoder_seq_length, target_texts)

num_encoder_tokens = len(input_token_index) + 1
num_decoder_tokens = len(target_token_index) + 1

print('num_encoder_tokens: ' + str(num_encoder_tokens))
print('num_decoder_tokens: ' + str(num_decoder_tokens))

reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())
reverse_target_char_index[0] = '<pad>'

encoder_input_data = onehot_encode(encoder_input_seq, max_encoder_seq_length, num_encoder_tokens)
decoder_input_data = onehot_encode(decoder_input_seq, max_decoder_seq_length, num_decoder_tokens)

decoder_target_seq = np.zeros(decoder_input_seq.shape)
decoder_target_seq[:, 0:-1] = decoder_input_seq[:, 1:]
decoder_target_data = onehot_encode(decoder_target_seq, max_decoder_seq_length, num_decoder_tokens)

print(encoder_input_data.shape)
print(decoder_input_data.shape)

# one hot encode category
cat_n = len(input_cats)
cat_data = np.zeros((cat_n, 1, num_encoder_tokens))
for i in range(cat_n):
    cat_data[i, :, :] = to_categorical(input_cats[i], num_classes=num_encoder_tokens)

"""
build encoder and decoder. structure
"""
latent_dim = 256
encoder_inputs = Input(shape=(None, num_encoder_tokens),
                      name='encoder_inputs')
encoder_cats = Input(shape=(None, num_encoder_tokens),
                   name='encoder_cats')
encoder_concat = concatenate([encoder_inputs, encoder_cats], axis=1)
encoder = GRU(latent_dim, return_state=True,
             name='encoder_gru')
encoder_outputs, state_h = encoder(encoder_concat)
encoder_model = Model(inputs=[encoder_inputs,encoder_cats], 
                     outputs=[state_h],
                     name='encoder')


# inputs of the decoder network
decoder_input_x = Input(shape=(None, num_decoder_tokens), name='decoder_input_x') # one-hot encode of target sequence
decoder_input_h = Input(shape=(latent_dim, ), name='decoder_input_h') # the initial hidden state h

# set the GRU layer
decoder_gru = GRU(latent_dim, return_sequences=True,
                 return_state=True, name='decoder_gru')

decoder_gru_outputs, state_h = decoder_gru(decoder_input_x, initial_state=[decoder_input_h])

# set the dense layer
decoder_dense = Dense(num_decoder_tokens, activation='softmax', name='decoder_dense')
decoder_outputs = decoder_dense(decoder_gru_outputs)

# build the decoder network model
decoder_model = Model(inputs=[decoder_input_x, decoder_input_h], 
                      outputs=[decoder_outputs, state_h],
                     name='decoder')


# input layers
encoder_input_x = Input(shape=(None, num_encoder_tokens), name='encoder_input_x')
encoder_cats = Input(shape=(None, num_encoder_tokens), name='encoder_cats')
decoder_input_x = Input(shape=(None, num_decoder_tokens), name='decoder_input_x')

encoder_final_state = encoder_model([encoder_input_x, encoder_cats])
decoder_gru_output, _ = decoder_gru(decoder_input_x, initial_state=encoder_final_state)
decoder_pred = decoder_dense(decoder_gru_output)

model = Model(inputs=[encoder_input_x, encoder_cats, decoder_input_x],
             outputs=decoder_pred,
             name='model_training')


model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
             metrics=['accuracy'])
history = model.fit([encoder_input_data, cat_data, decoder_input_data], decoder_target_data,
          batch_size=64,
          epochs=30,
          validation_split=0.2)






