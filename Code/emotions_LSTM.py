# !pip install contractions
import re
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, Activation, Flatten, Input, concatenate, Conv1D, GlobalMaxPooling1D, MaxPooling1D

# Text preprocessing function

str_punc = string.punctuation.replace(',', '').replace("'",'')

def clean(text):
    global str_punc
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = text.lower()
    return text

# Read dataset & Preprocess text

OR_PATH = os.getcwd()
# os.chdir("..") # Change to the parent directory
PATH = os.getcwd()
DATA_DIR = OR_PATH + os.path.sep + 'NLP_Final_project' + os.path.sep +'Data' + os.path.sep
sep = os.path.sep


os.chdir(OR_PATH)

df_train = pd.read_csv(DATA_DIR+'train.txt', names=['Text', 'Emotion'], sep=';')
df_val = pd.read_csv(DATA_DIR+'val.txt', names=['Text', 'Emotion'], sep=';')
df_test = pd.read_csv(DATA_DIR+'test.txt', names=['Text', 'Emotion'], sep=';')

df_test = df_test[df_test['Emotion'].isin(['sadness','anger','joy','fear'])]
df_val = df_val[df_val['Emotion'].isin(['sadness','anger','joy','fear'])]
df_train = df_train[df_train['Emotion'].isin(['sadness','anger','joy','fear'])]

X_train = df_train['Text'].apply(clean)
y_train = df_train['Emotion']

X_test = df_test['Text'].apply(clean)
y_test = df_test['Emotion']

X_val = df_val['Text'].apply(clean)
y_val = df_val['Emotion']

# Visualize classes counts
# convert the Series to a DataFrame
y_train = pd.DataFrame({'data': y_train})
y_test = pd.DataFrame({'data': y_test})
y_val = pd.DataFrame({'data': y_val})


# plot the countplot using the DataFrame
sns.countplot(x='data', data=y_train)
# sns.countplot(y_train['Emotion'])
plt.title("Training data - classes counts")
plt.show()

sns.countplot(x='data', data=y_test)
plt.title("Testing data - classes counts")
plt.show()

sns.countplot(x='data', data=y_val)
plt.title("Validation data - classes counts")
plt.show()

# Encode labels

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)
y_val = le.transform(y_val)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)

# Tokenize words
tokenizer = Tokenizer()
tokenizer.fit_on_texts(pd.concat([X_train, X_test], axis=0))


sequences_train = tokenizer.texts_to_sequences(X_train)
sequences_test = tokenizer.texts_to_sequences(X_test)
sequences_val = tokenizer.texts_to_sequences(X_val)

X_train = pad_sequences(sequences_train, maxlen=256, truncating='pre')
X_test = pad_sequences(sequences_test, maxlen=256, truncating='pre')
X_val = pad_sequences(sequences_val, maxlen=256, truncating='pre')

vocabSize = len(tokenizer.index_word) + 1
print(f"Vocabulary size = {vocabSize}")

# Embedding
import gensim
from gensim.models import Word2Vec

# train Word2Vec on your text corpus
embedding_dim=200
corpus = [text.split() for text in df_train['Text']]
model_emb = Word2Vec(corpus, vector_size=embedding_dim, window=5, min_count=1, workers=4)

# create an embedding matrix
embedding_matrix = np.zeros((vocabSize, embedding_dim))
for word, i in tokenizer.word_index.items():
    if word in model_emb.wv.key_to_index:
        embedding_matrix[i] = model_emb.wv[word]

max_features = 20000
maxlen = 100
embedding_size = 128

# Convolution
kernel_size = 5
filters = 64
pool_size = 4

# LSTM
lstm_output_size = 70

# Training
#batch_size = 30
#epochs = 2

# Embedding
max_features = vocabSize
maxlen = X_train.shape[1]
embedding_size = 200

# Convolution
kernel_size = 5
filters = 128
pool_size = 4

# LSTM
lstm_output_size = 128

# Training
#batch_size = 30
#epochs = 2

print('Build model...')

model = Sequential()
# model.add(Embedding(vocabSize, embedding_size, input_length=X_train.shape[1]))
model.add(Embedding(vocabSize, embedding_dim, input_length=maxlen, weights=[embedding_matrix], trainable=False))
model.add(Dropout(0.25))
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
model.add(MaxPooling1D(pool_size=pool_size))
model.add(LSTM(lstm_output_size))
model.add(Dense(4))
model.add(Activation('softmax'))
# model.add(LSTM(128))
# model.add(Dense(4, activation='softmax'))
model.summary()


adam = Adam(learning_rate=0.005)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

callback = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
)

# Fit model
history = model.fit(X_train,
                    y_train,
                    validation_data=(X_val, y_val),
                    verbose=1,
                    batch_size=256,
                    epochs=30,
                    callbacks=[callback]
                   )

model.evaluate(X_test, y_test, verbose=1)

plt.figure(figsize=(15,7))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'Validation'], loc='upper left')
plt.show()

plt.figure(figsize=(15,7))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss Function')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(['train', 'Validation'], loc='upper left')
plt.show()

# Classify custom sample
def predict(text):
    print(sentence)
    sentence = clean(sentence)
    sentence = tokenizer.texts_to_sequences([sentence])
    sentence = pad_sequences(sentence, maxlen=256, truncating='pre')
    result = le.inverse_transform(np.argmax(model.predict(sentence), axis=-1))[0]
    proba =  np.max(model.predict(sentence))
    print(f"{result} : {proba}\n\n")


import pickle

with open('tokenizer.pickle', 'wb') as f:
    pickle.dump(tokenizer, f)

with open('labelEncoder.pickle', 'wb') as f:
    pickle.dump(le, f)

model.save('Emotion Detection.h5')