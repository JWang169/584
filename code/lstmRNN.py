import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import pandas as pd

def predict(patterm):
	in_title = []
	# pattern = 'death of the bachlor do i look lonely i see the shadows on my face people have told me i dont look the same maybe i lost weight'
	# pattern = pattern[:100]
	for p in pattern:
    	num = char_to_int[p]
    	in_title.append(char_to_int[p])
    res = ''
    # generate characters
	for i in range(500):
    	x = np.reshape(pattern, (1, len(pattern), 1))
    	x = x / float(n_vocab)
    	prediction = model.predict(x, verbose=0)
    	index = np.argmax(prediction)
    	result = int_to_char[index]
    	res = res + result
    	seq_in = [int_to_char[value] for value in pattern]
    	pattern.append(index)
    	pattern = pattern[1:len(pattern)]
    return res


df = pd.read_csv('5k_para.csv')
all_artists = df['Artists']
all_titles = df['Titles']
all_lyrics = df['Lyrics']

text = ''
for l in all_lyrics:
    text = text + str(l)

text = text[:100000]
chars = sorted(list(set(text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
n_chars = len(text)
n_vocab = len(chars)

seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = text[i:i + seq_length]
    seq_out = text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print ("Total Patterns: ", n_patterns)

# reshape X to be [samples, time steps, features]
X = np.reshape(dataX, (n_patterns, seq_length, 1))

X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)

model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

history = model.fit(X, y, nb_epoch=20, batch_size=128)

res = predict(pattern)




