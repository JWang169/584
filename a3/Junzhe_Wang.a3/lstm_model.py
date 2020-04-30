import re
import string
import keras
import tensorflow as tf
from unicodedata import normalize
import numpy as np
import keras.utils as ku 
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, BatchNormalization
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def load_doc(filename):
    # open and read file as read only
    file = open(filename, mode='rt', encoding='utf-8')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

def to_sentence(doc):
    # @para: doc is the whole context
    # return: a list of sentences.
    lines = doc.strip().split('\n')
    sentences = [line.split('\t') for line in lines]
    return sentences

def create_model(dropout=0.2):
    print("-----------------Creationg the Language model--------------")
    model = Sequential()
    model.add(Embedding(total_words, 15, input_length=20)) # input_length: x_train size
    model.add(LSTM(output_dim=256, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(output_dim=128))
    model.add(Dropout(0.2))
    model.add(Dense(total_words, activation='softmax')) # outputlayer with vocab size
    adam = keras.optimizers.Adam(lr=0.001,decay=1e-6) # set learning rate to 0.001(which is default)
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['mse','acc'])
    return model

f1 = load_doc('a3-data/train.txt')
lines = to_sentence(f1) # convert how passage to sentences
corpus = []
for c in lines:
    corpus.append(c[0])

tokenizer = Tokenizer(oov_token='unk') # unknown words as unk
tokenizer.fit_on_texts(corpus)
word_index = tokenizer.word_index

total_words = len(tokenizer.word_index) + 2 # +1 for zero padding, word_index starts from 1, +2 for unk

index_word = {}
for k, v in word_index.items():
    index_word[v] = k
sequences = tokenizer.texts_to_sequences(corpus)

# create training dataset
# if sentence length < 21: padding
# else sliding window size = 21 
 
training = False # weights of trained model already saved.
seq_train = [] 
for line in sequences:
    if len(line) < 21:
        seq_train.append(line)
    else:
        for i in range(20, len(line)):
            seq_train.append(line[i-20:i+1])
padded = pad_sequences(seq_train, maxlen=21)
# create predictors and labels for each training dataset
x_train, y_train = padded[:,:-1], padded[:,-1]
# onehot encoding labels
y_train = ku.to_categorical(y_train, num_classes=total_words) 


# train model
if (training == True):
    filepath = "saved-best-model-train.hdf5"
    model = create_model()
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='auto', period=1)
    model.fit(x_train, y_train, epochs=30, batch_size=50, verbose=1, callbacks=[checkpoint])
    print("-----------------Training the model--------------")

# evaluate model
fval = load_doc('a3-data/valid.txt')
lines_val = to_sentence(fval)
corpus_val = []
for c in lines_val:
    corpus_val.append(c[0])
sequences_val = tokenizer.texts_to_sequences(corpus_val)
padded_val = np.array(pad_sequences(sequences_val, maxlen=21, padding='pre')) # window 20

x_val, y_val = padded_val[:,:-1], padded_val[:,-1]
y_val = ku.to_categorical(y_val, num_classes=total_words) 
model = create_model()
model.load_weights("saved-best-model-colab.hdf5")
print("-----------------Loading weights for the model--------------")
model_loss, model_mse, model_acc = model.evaluate(x_val, y_val)
print("-----------------Predicting---------------------------------")
pred = model.predict(x_val)

# calculate perplexity:
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=tf.argmax(y_val, 1))
print("------Calculating the cross_entrpy loss of the testing dataset-------")
cost = tf.reduce_mean(cross_entropy)
sess = tf.InteractiveSession()
cost_value = cost.eval()
perplexity = 2 ** cost_value

print('The perplexity is ', perplexity)