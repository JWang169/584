import pandas as pd
import numpy as np
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from matplotlib import pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras import regularizers, layers
from keras.layers import Dropout, Flatten, BatchNormalization, Conv1D, GlobalMaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
from keras.models import load_model


def get_data(path):
    f = open(path)
    context = f.read().strip()
    f.close()
    raw = context.split('\n')
    sentences = []
    for sentence in raw:
        cur = sentence.split()
        if len(cur) > 4:
             sentences.append(cur)
    return sentences


def split_data(s,l,X_train, X_test, Y_train, Y_test):
    x_train, x_test, y_train, y_test = train_test_split(s, l, test_size=0.2, random_state=42)
    X_train.extend(x_train)
    X_test.extend(x_test)
    Y_train.extend(y_train)
    Y_test.extend(y_test)
    return X_train, X_test, Y_train, Y_test


def preprocess():
    authors =  ['Fyodor Dostoyevsky', 'Conan Doyle', 'Jane Austen']
    p1, p2, p3 = ['data/28054-0.txt', 'data/pg1661.txt', 'data/pg31100.txt']

    # manully remove headers 
    s1 = get_data(p1)[91:]
    s2 = get_data(p2)[23:]
    s3 = get_data(p3)[13:]
    l1 = [0]* len(s1)
    l2 = [1]* len(s2)
    l3 = [2]* len(s3)

    # split training and testing data set accordingly 
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    X_train, X_test, Y_train, Y_test = split_data(s1,l1,X_train, X_test, Y_train, Y_test)
    X_train, X_test, Y_train, Y_test = split_data(s2,l2,X_train, X_test, Y_train, Y_test)
    X_train, X_test, Y_train, Y_test = split_data(s3,l3,X_train, X_test, Y_train, Y_test)

    tokenizer = Tokenizer(oov_token='unk')
    tokenizer.fit_on_texts(X_train)

    word_index = tokenizer.word_index
    vocab_size = len(tokenizer.word_index) + 1

    # text tokenize
    seq_train = tokenizer.texts_to_sequences(X_train)
    seq_test = tokenizer.texts_to_sequences(X_test)
    max_sentence_len = 0
    for x in seq_train:
        length = len(x)
        max_sentence_len = length if length > max_sentence_len else max_sentence_len
    
    training_data = pad_sequences(seq_train, padding='post', maxlen=max_sentence_len)
    test_data = pad_sequences(seq_test, padding='post', maxlen=max_sentence_len)
    
    training_labels = to_categorical(Y_train)
    test_labels = to_categorical(Y_test)

    return training_data, test_data, training_labels, test_labels, vocab_size, max_sentence_len


def cnn_2_layer_model(vocab_size, max_sentence_len):
    # build model
    model = Sequential()
    model.add(layers.Embedding(vocab_size, 100, input_length=max_sentence_len))
    model.add(layers.Conv1D(128,5))
    model.add(layers.MaxPooling1D(3,3, padding='same'))
    model.add(layers.Conv1D(64,5))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(layers.Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.5))
    model.add(layers.Dense(3, activation='sigmoid'))
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    model.summary()
    return model


def cnn_1_layer_model(vocab_size, max_sentence_len):
    model = Sequential()
    model.add(layers.Embedding(vocab_size, 100, input_length=max_sentence_len))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(layers.Dense(32, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.5))
    model.add(layers.Dense(3, activation='sigmoid'))
    adam = keras.optimizers.Adam(lr=0.001,decay=1e-6) # set learning rate to 0.001(which is default)

    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    model.summary()
    return model



training_data, test_data, training_labels, test_labels,vocab_size, max_sentence_len = preprocess()
model = cnn_1_layer_model(vocab_size, max_sentence_len)
filepath = "saved-1cnn-model.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='auto', period=1)
history = model.fit(training_data, 
                    training_labels,
                    validation_split=0.3,
                    batch_size=50, 
                    epochs=5, 
                    callbacks=[checkpoint])

scores = model.evaluate(test_data, test_labels) # two layers

# The result for cnn_1_layer_model:
# test loss:0.5594399030183734, 
# test accuracy: 0.8428394142496701,

# The result for cnn_2_layer_model: 
# test loss: 0.578757287553387
# test accuracy: 0.850881111941405
# [0.578757287553387, 0.850881111941405]

