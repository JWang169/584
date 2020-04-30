import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from string import punctuation
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras import regularizers, layers
from keras.layers import Embedding, Dense, Dropout, Flatten, BatchNormalization, Conv1D, GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

f = open('datasetSplit.txt')
text = f.read()
f.close()
text = text.split()
idx_label = {} # key is train/val/test, value is the sentence number
idx_label['1'] = []
idx_label['2'] = []
idx_label['3'] = []
for i in text[1:]:
    try:
        x = i.split(',')
        label = x[1]
        idx = x[0]
        idx_label[label].append(idx)
    except ValueError:
        print(i)


f = open('datasetSentences.txt')
text_sentences = f.read()
f.close()
text_sentences = text_sentences.split('\n')
idx_sentence = {} # key is the number of sentence, value is the sentence text
for s in text_sentences[1:]:
    try:
        ss = s.split('\t')
        idx = ss[0]
        sentence = ss[1]
        idx_sentence[idx] = sentence
    except IndexError:
        print(s)


f = open('sentiment_labels.txt')
text = f.readlines()
f.close()
idx_sentiment = {} # key is the number of sentence, value is the sentiment
for s in text[1:]:
    try:
        ss = s.split('|')
        idx = ss[0]
        sentiment = ss[1].split('\n')
        idx_sentiment[idx] = float(sentiment[0])
    except IndexError:
        print(s)


f = open('dictionary.txt')
text = f.read()
f.close()
lines = text.split('\n')
phrase_sentiment = {}
for line in lines:
    try:
        ss = line.split('|')
        phrase = ss[0]
        idx = ss[1]
        phrase_sentiment[phrase] = idx
    except IndexError:
        print(line)


Xtrain = []
Xval = []
Xtest = []
for idx in idx_label['1']: # training set
    Xtrain.append(idx_sentence[idx])
for idx in idx_label['2']: # training set
    Xval.append(idx_sentence[idx])
for idx in idx_label['3']: # training set
    Xtest.append(idx_sentence[idx])


ytrain = {}
yval = {}
ytest = {}
notin = []
for i in Xtrain:
    if i not in phrase_sentiment:
        notin.append(i)
        continue
    senti_idx = phrase_sentiment[i]
    sentiment = idx_sentiment[senti_idx]
    ytrain[i] = sentiment
    
for i in Xval:
    if i not in phrase_sentiment:
        notin.append(i)
        continue
    senti_idx = phrase_sentiment[i]
    sentiment = idx_sentiment[senti_idx]
    yval[i] = sentiment

for i in Xtest:
    if i not in phrase_sentiment:
        notin.append(i)
        continue
    senti_idx = phrase_sentiment[i]
    sentiment = idx_sentiment[senti_idx]
    ytest[i] = sentiment


x_train = []
y_train = []
for k in ytrain:
    y = ytrain[k]
    x_train.append(k)
    y_train.append(y)

x_val = []
y_val = []
for k in yval:
    y = yval[k]
    x_val.append(k)
    y_val.append(y)

x_test = []
y_test = []
for k in ytest:
    y = ytest[k]
    x_test.append(k)
    y_test.append(y)

max_sentence_len = 0
for x in x_train:
    length = len(x)
    max_sentence_len = length if length > max_sentence_len else max_sentence_len

tokenizer = Tokenizer(oov_token='unk')
tokenizer.fit_on_texts(x_train)

word_index = tokenizer.word_index
vocab_size = len(tokenizer.word_index) + 1
maxlen = max_sentence_len
# maxlen = 30

seq_train = tokenizer.texts_to_sequences(x_train)
seq_val = tokenizer.texts_to_sequences(x_val)
seq_test = tokenizer.texts_to_sequences(x_test)

training_data = pad_sequences(seq_train, padding='post', maxlen=maxlen)
val_data = pad_sequences(seq_val, padding='post', maxlen=maxlen)
test_data = pad_sequences(seq_test, padding='post', maxlen=maxlen)

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
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='mse',
              metrics=['accuracy'])
model.summary()

filepath = "saved-a42-cnn.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='auto', period=1)

history = model.fit(training_data, 
                    y_train,
                    validation_data=(val_data, y_val),
                    batch_size=30, 
                    epochs=100,
                    callbacks=[checkpoint])


pred = model.predict(test_data)
model.evaluate(test_data, y_test)
MSE_scaled = mean_squared_error(y_test, pred)

# the output of evaluate:
# [0.06994478800267066, 0.007670182166826462]
# training loss: 0.0094
# training acc: 0.0042
# validation loss: 0.0503
# validation acc: 0.0033
# root mean square error: 0.046