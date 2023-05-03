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

from tensorflow.keras.optimizers import Adam, SGD, Adagrad
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, Activation, Flatten, Input, concatenate, Conv1D, GlobalMaxPooling1D, MaxPooling1D
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
from nltk.corpus import stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Text preprocessing function

str_punc = string.punctuation.replace(',', '').replace("'",'')

def clean(text):
    global str_punc
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove mentions
    text = re.sub(r'@[A-Za-z0-9]+', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove punctuation
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize
    words = text.split()
    # Remove stop words
    words = [word for word in words if word not in stop_words]
    # # Stem words
    # words = [stemmer.stem(word) for word in words]
    # Lemmatize words
    words = [lemmatizer.lemmatize(word) for word in words]
    # Join words back into a string
    text = ' '.join(words)
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

df_test = df_test[df_test['Emotion'].isin(['sadness','anger','joy','fear','surprise','love'])]
df_val = df_val[df_val['Emotion'].isin(['sadness','anger','joy','fear','surprise','love'])]
df_train = df_train[df_train['Emotion'].isin(['sadness','anger','joy','fear','surprise','love'])]

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
plt.title("Training data - classes counts")
plt.show()

sns.countplot(x='data', data=y_test)
plt.title("Testing data - classes counts")
plt.show()

sns.countplot(x='data', data=y_val)
plt.title("Validation data - classes counts")
plt.show()

# from imblearn.over_sampling import RandomOverSampler
#
# # Instantiate RandomOverSampler
# oversampler = RandomOverSampler(random_state=42)
#
# # Fit and resample data
# X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

df_train['length'] = df_train['Text'].apply(len) # number of characters
plt.figure(figsize=(10,7))
sns.kdeplot(x=df_train["length"], hue=df_train["Emotion"])
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

# # create an embedding matrix
# embedding_matrix = np.zeros((vocabSize, embedding_dim))
# for word, i in tokenizer.word_index.items():
#     if word in model_emb.wv.key_to_index:
#         embedding_matrix[i] = model_emb.wv[word]


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


print('Build model...')

model = Sequential()
model.add(Embedding(vocabSize, embedding_size, input_length=X_train.shape[1]))
#model.add(Embedding(vocabSize, embedding_dim, input_length=maxlen, weights=[embedding_matrix], trainable=False))
model.add(Dropout(0.25))
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
model.add(MaxPooling1D(pool_size=pool_size))
model.add(LSTM(lstm_output_size))
model.add(Dense(6))
model.add(Activation('softmax'))
model.summary()


# adam = Adam(learning_rate=0.005)
# adam = Adam(learning_rate=0.001)
adam = Adam(learning_rate=0.001)
# sgd = SGD(learning_rate=0.0001)
# Ada = Adagrad(learning_rate=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
# model.compile(loss='categorical_crossentropy', optimizer=Ada, metrics=['accuracy'])

callback = EarlyStopping(
    monitor="val_loss",
    patience=10,
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

from sklearn.metrics import f1_score

# Predict on the test set
y_pred = model.predict(X_test)

# Convert predictions to binary labels
y_pred_binary = np.argmax(y_pred, axis=1)
y_test_binary = np.argmax(y_test, axis=1)

# Calculate F1 macro and F1 micro scores
f1_macro = f1_score(y_test_binary, y_pred_binary, average='macro')
f1_micro = f1_score(y_test_binary, y_pred_binary, average='micro')

print("F1 macro score:", f1_macro)
print("F1 micro score:", f1_micro)

# Classify custom sample
def predict(text):
    print(text)
    sentence = clean(text)
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

# # !pip install contractions
# import re
# import string
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, Activation, Flatten, Input, concatenate, Conv1D, GlobalMaxPooling1D, MaxPooling1D
# import re
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer, WordNetLemmatizer
# from tensorflow.keras.preprocessing.text import one_hot
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from gensim.models import Word2Vec
# import torch
# import torch.nn as nn
# from collections import Counter
# from torch.utils.data import TensorDataset, DataLoader
#
# nltk.download('stopwords')
# nltk.download('wordnet')
# stop_words = set(stopwords.words('english'))
# stemmer = PorterStemmer()
# lemmatizer = WordNetLemmatizer()
#
#
# is_cuda = torch.cuda.is_available()
# if is_cuda:
#     device = torch.device("cuda")
#     print("GPU is available")
# else:
#     device = torch.device("cpu")
#     print("GPU not available, CPU used")
#
# # Read dataset & Preprocess text
# OR_PATH = os.getcwd()
# # os.chdir("..") # Change to the parent directory
# PATH = os.getcwd()
# DATA_DIR = OR_PATH + os.path.sep + 'NLP_Final_project' + os.path.sep +'Data' + os.path.sep
# sep = os.path.sep
# os.chdir(OR_PATH)
#
# df_train = pd.read_csv(DATA_DIR+'train.txt', names=['Text', 'Emotion'], sep=';')
# df_val = pd.read_csv(DATA_DIR+'val.txt', names=['Text', 'Emotion'], sep=';')
# df_test = pd.read_csv(DATA_DIR+'test.txt', names=['Text', 'Emotion'], sep=';')
#
#
# df_test = df_test[df_test['Emotion'].isin(['sadness','anger','joy','fear','surprise','love'])]
# df_val = df_val[df_val['Emotion'].isin(['sadness','anger','joy','fear','surprise','love'])]
# df_train = df_train[df_train['Emotion'].isin(['sadness','anger','joy','fear','surprise','love'])]
#
# # x_train = df_train['Text']
# # y_train = df_train['Emotion']
# #
# # x_test = df_test['Text']
# # y_test = df_test['Emotion']
# #
# # x_val = df_val['Text']
# # y_val = df_val['Emotion']
#
# X,y = df_train['Text'].values,df_train['Emotion'].values
# x_train,x_test,y_train,y_test = train_test_split(X,y,stratify=y)
# print(f'shape of train data is {x_train.shape}')
# print(f'shape of test data is {x_test.shape}')
#
# df_train['length'] = df_train['Text'].apply(len) # number of characters
# plt.figure(figsize=(10,7))
# sns.kdeplot(x=df_train["length"], hue=df_train["Emotion"])
# plt.show()
# maxlen:350

# x_train = df_train['Text'].apply(clean)
# y_train = df_train['Emotion']
#
# x_test = df_test['Text'].apply(clean)
# y_test = df_test['Emotion']
#
# x_val = df_val['Text'].apply(clean)
# y_val = df_val['Emotion']

# def clean(s):
#     s = re.sub(r"[^\w\s]", '', s)
#     s = re.sub(r"\s+", '', s)
#     s = re.sub(r"\d", '', s)
#     return s
#
# def tockenize(x_train, y_train, x_val, y_val):
#     word_list = []
#     stop_words = set(stopwords.words('english'))
#     max_length = 0
#     for sent in x_train:
#         tokenized_sent = []
#         for word in sent.lower().split():
#             word = clean(word)
#             if word not in stop_words and word != '':
#                 tokenized_sent.append(word)
#         word_list.extend(tokenized_sent)
#         if len(tokenized_sent) > max_length:
#             max_length = len(tokenized_sent)
#
#     corpus = Counter(word_list)
#     corpus_ = sorted(corpus, key=corpus.get, reverse=True)[:1000]
#     onehot_dict = {w: i + 1 for i, w in enumerate(corpus_)}
#
#     final_list_train, final_list_test = [], []
#     for sent in x_train:
#         tokenized_sent = [onehot_dict[clean(word)] for word in sent.lower().split() if clean(word) in onehot_dict.keys()]
#         padded_sent = np.pad(tokenized_sent, (0, max_length - len(tokenized_sent)), 'constant')
#         final_list_train.append(padded_sent)
#     for sent in x_val:
#         tokenized_sent = [onehot_dict[clean(word)] for word in sent.lower().split() if clean(word) in onehot_dict.keys()]
#         padded_sent = np.pad(tokenized_sent, (0, max_length - len(tokenized_sent)), 'constant')
#         final_list_test.append(padded_sent)
#
#     encoded_train = [
#         0 if label == 'anger' else 1 if label == 'love' else 2 if label == 'joy' else 3 if label == 'fear' else 4 if label == 'sadness' else 5 if label == 'surprise' else -1
#         for label in y_train]
#     encoded_test = [
#         0 if label == 'anger' else 1 if label == 'love' else 2 if label == 'joy' else 3 if label == 'fear' else 4 if label == 'sadness' else 5 if label == 'surprise' else -1
#         for label in y_val]
#
#     return np.array(final_list_train), np.array(encoded_train), np.array(final_list_test), np.array(encoded_test), onehot_dict
#
# x_train,y_train,x_test,y_test,vocab = tockenize(x_train,y_train,x_test,y_test)
#
#
# def padding_(sentences, seq_len):
#     features = np.zeros((len(sentences), seq_len),dtype=int)
#     for ii, review in enumerate(sentences):
#         if len(review) != 0:
#             features[ii, -len(review):] = np.array(review)[:seq_len]
#     return features
#
# # x_train_pad = padding_(x_train,200)
# # x_test_pad = padding_(x_test,200)
# x_train_pad = padding_(x_train,300)
# x_test_pad = padding_(x_test,300)
#
#
# train_data = TensorDataset(torch.from_numpy(x_train_pad), torch.from_numpy(y_train))
# valid_data = TensorDataset(torch.from_numpy(x_test_pad), torch.from_numpy(y_test))
#
#
# # batch_size = 10
# batch_size = 100
#
# train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last = True)
# valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size, drop_last = True)
#
# class SentimentRNN(nn.Module):
#     def __init__(self, no_layers, vocab_size, hidden_dim, embedding_dim, drop_prob=0.5):
#         super(SentimentRNN, self).__init__()
#
#         self.output_dim = output_dim
#         self.hidden_dim = hidden_dim
#
#         self.no_layers = no_layers
#         self.vocab_size = vocab_size
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=self.hidden_dim, num_layers=no_layers, batch_first=True)
#         self.dropout = nn.Dropout(0.3)
#         self.fc = nn.Linear(self.hidden_dim, output_dim)
#         self.sig = nn.Sigmoid()
#
#     def forward(self, x, hidden):
#         batch_size = x.size(0)
#         embeds = self.embedding(x)
#         lstm_out, hidden = self.lstm(embeds, hidden)
#         lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
#         out = self.dropout(lstm_out)
#         out = self.fc(out)
#         sig_out = self.sig(out)
#         sig_out = sig_out.view(batch_size, -1)
#         sig_out = sig_out[:, -1]
#         return sig_out, hidden
#
#     def init_hidden(self, batch_size):
#         h0 = torch.zeros((self.no_layers, batch_size, self.hidden_dim)).to(device)
#         c0 = torch.zeros((self.no_layers, batch_size, self.hidden_dim)).to(device)
#         hidden = (h0, c0)
#         return hidden
#
# no_layers = 2
# vocab_size = len(vocab) +1
# embedding_dim = 64
# output_dim = 1
# hidden_dim = 256
#
#
# model = SentimentRNN(no_layers,vocab_size,hidden_dim,embedding_dim,drop_prob=0.5)
# model.to(device)
# lr=0.001
# criterion = nn.BCELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#
# def acc(pred,label):
#     pred = torch.round(pred.squeeze())
#     return torch.sum(pred == label.squeeze()).item()
#
# clip = 5
# epochs = 30
# valid_loss_min = np.Inf
# epoch_tr_loss, epoch_vl_loss = [], []
# epoch_tr_acc, epoch_vl_acc = [], []
#
# for epoch in range(epochs):
#     train_losses = []
#     train_acc = 0.0
#     model.train()
#     h = model.init_hidden(batch_size)
#     for inputs, labels in train_loader:
#         inputs, labels = inputs.to(device), labels.to(device)
#         h = tuple([each.data for each in h])
#         model.zero_grad()
#         output, h = model(inputs, h)
#         loss = criterion(output.squeeze(), labels.float())
#         loss.backward()
#         train_losses.append(loss.item())
#         accuracy = acc(output, labels)
#         train_acc += accuracy
#         nn.utils.clip_grad_norm_(model.parameters(), clip)
#         optimizer.step()
#
#     val_h = model.init_hidden(batch_size)
#     val_losses = []
#     val_acc = 0.0
#     model.eval()
#     for inputs, labels in valid_loader:
#         val_h = tuple([each.data for each in val_h])
#         inputs, labels = inputs.to(device), labels.to(device)
#         output, val_h = model(inputs, val_h)
#         val_loss = criterion(output.squeeze(), labels.float())
#         val_losses.append(val_loss.item())
#         accuracy = acc(output, labels)
#         val_acc += accuracy
#
#     epoch_train_loss = np.mean(train_losses)
#     epoch_val_loss = np.mean(val_losses)
#     epoch_train_acc = train_acc / len(train_loader.dataset)
#     epoch_val_acc = val_acc / len(valid_loader.dataset)
#     epoch_tr_loss.append(epoch_train_loss)
#     epoch_vl_loss.append(epoch_val_loss)
#     epoch_tr_acc.append(epoch_train_acc)
#     epoch_vl_acc.append(epoch_val_acc)
#     print(f'Epoch {epoch + 1}')
#     print(f'train_loss : {epoch_train_loss} val_loss : {epoch_val_loss}')
#     print(f'train_accuracy : {epoch_train_acc * 100:.2f}% val_accuracy : {epoch_val_acc * 100:.2f}%')
#
#




# # Text preprocessing function
# str_punc = string.punctuation.replace(',', '').replace("'",'')
#
# # def clean(text):
# #     global str_punc
# #     text = re.sub(r'[^a-zA-Z ]', '', text)
# #     text = text.lower()
# #     return text
# def clean(text):
#     global str_punc
#     # Remove URLs
#     text = re.sub(r'http\S+', '', text)
#     # Remove mentions
#     text = re.sub(r'@[A-Za-z0-9]+', '', text)
#     # Remove numbers
#     text = re.sub(r'\d+', '', text)
#     # Remove punctuation
#     text = re.sub(r'[^a-zA-Z]', ' ', text)
#     # Convert to lowercase
#     text = text.lower()
#     # Tokenize
#     words = text.split()
#     # Remove stop words
#     words = [word for word in words if word not in stop_words]
#     # # Stem words
#     # words = [stemmer.stem(word) for word in words]
#     # Lemmatize words
#     words = [lemmatizer.lemmatize(word) for word in words]
#     # Join words back into a string
#     text = ' '.join(words)
#     return text
#
# # Read dataset & Preprocess text
# OR_PATH = os.getcwd()
# # os.chdir("..") # Change to the parent directory
# PATH = os.getcwd()
# DATA_DIR = OR_PATH + os.path.sep + 'NLP_Final_project' + os.path.sep +'Data' + os.path.sep
# sep = os.path.sep
# os.chdir(OR_PATH)
#
# df_train = pd.read_csv(DATA_DIR+'train.txt', names=['Text', 'Emotion'], sep=';')
# df_val = pd.read_csv(DATA_DIR+'val.txt', names=['Text', 'Emotion'], sep=';')
# df_test = pd.read_csv(DATA_DIR+'test.txt', names=['Text', 'Emotion'], sep=';')
#
# df_train['length'] = df_train['Text'].apply(len) # number of characters
# plt.figure(figsize=(10,7))
# sns.kdeplot(x=df_train["length"], hue=df_train["Emotion"])
# plt.show()
# # maxlen:350
#
# df_test = df_test[df_test['Emotion'].isin(['sadness','anger','joy','fear','surprise','love'])]
# df_val = df_val[df_val['Emotion'].isin(['sadness','anger','joy','fear','surprise','love'])]
# df_train = df_train[df_train['Emotion'].isin(['sadness','anger','joy','fear','surprise','love'])]
#
# X_train = df_train['Text'].apply(clean)
# y_train = df_train['Emotion']
#
# X_test = df_test['Text'].apply(clean)
# y_test = df_test['Emotion']
#
# X_val = df_val['Text'].apply(clean)
# y_val = df_val['Emotion']
#
# # Visualize classes counts
# # convert the Series to a DataFrame
# y_train = pd.DataFrame({'Emotion': y_train})
# y_test = pd.DataFrame({'Emotion': y_test})
# y_val = pd.DataFrame({'Emotion': y_val})
#
#
# # plot the countplot using the DataFrame
# sns.countplot(x='Emotion', data=y_train)
# plt.title("Training data - classes counts")
# plt.show()
#
# sns.countplot(x='Emotion', data=y_test)
# plt.title("Testing data - classes counts")
# plt.show()
#
# sns.countplot(x='Emotion', data=y_val)
# plt.title("Validation data - classes counts")
# plt.show()
# vocabsize= 18000
# maxlen = 300
# def encode(text):
#     one_hot_words = [one_hot(input_text=word, n=vocabsize) for word in text]
#     padded = pad_sequences(sequences = one_hot_words, maxlen =maxlen, padding ="pre") # padding is used to provide uniformity in the sentences.
#     print(text.shape)
#     return padded
#
# x_train = encode(df_train['Text'])
# x_val = encode(df_val['Text'])
# x_test = encode(df_test['Text'])
#
# # Encode labels
#
# le = LabelEncoder()
# y_train = le.fit_transform(y_train)
# y_test = le.transform(y_test)
# y_val = le.transform(y_val)
#
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
# y_val = to_categorical(y_val)
#
# # Tokenize words
# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(pd.concat([X_train, X_test], axis=0))
#
# sequences_train = tokenizer.texts_to_sequences(X_train)
# sequences_test = tokenizer.texts_to_sequences(X_test)
# sequences_val = tokenizer.texts_to_sequences(X_val)
#
# X_train = pad_sequences(sequences_train, maxlen=256, truncating='pre')
# X_test = pad_sequences(sequences_test, maxlen=256, truncating='pre')
# X_val = pad_sequences(sequences_val, maxlen=256, truncating='pre')
#
# vocabSize = len(tokenizer.index_word) + 1
# print(f"Vocabulary size = {vocabSize}")
# # Vocabulary size = 14324
#
# Embedding
# import gensim
# from gensim.models import Word2Vec
# # train Word2Vec on your text corpus
# embedding_dim=200
# corpus = [text.split() for text in df_train['Text']]
# model_emb = Word2Vec(corpus, vector_size=embedding_dim, window=5, min_count=1, workers=4)
#
# # create an embedding matrix
# embedding_matrix = np.zeros((vocabSize, embedding_dim))
# for word, i in tokenizer.word_index.items():
#     if word in model_emb.wv.key_to_index:
#         embedding_matrix[i] = model_emb.wv[word]
#
# # Embedding
# max_features = vocabsize
# maxlen = 300
# embedding_size = 128
#
# # Convolution
# kernel_size = 5
# filters = 64
# pool_size = 4
#
# # LSTM
# lstm_output_size = 70
#
#
# print('Build model...')
# model = Sequential()
# model.add(Embedding(vocabsize, embedding_size, input_length=256))
# # model.add(Embedding(vocabSize, embedding_dim, input_length=maxlen, weights=[embedding_matrix], trainable=False))
# model.add(Dropout(0.25))
# model.add(Conv1D(filters,
#                  kernel_size,
#                  padding='valid',
#                  activation='relu',
#                  strides=1))
# model.add(MaxPooling1D(pool_size=pool_size))
# model.add(LSTM(lstm_output_size))
# model.add(Dense(6))
# model.add(Activation('softmax'))
# model.summary()
#
# # adam = Adam(learning_rate=0.005)
# adam = Adam(learning_rate=0.001)
# # adam = Adam(learning_rate=0.01)
# model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
#
# callback = EarlyStopping(
#     monitor="val_loss",
#     # patience=3,
#     patience=10,
#     restore_best_weights=True,
# )
#
# # Fit model
# history = model.fit(X_train,
#                     y_train,
#                     validation_data=(X_val, y_val),
#                     verbose=1,
#                     batch_size=256,
#                     epochs=30,
#                     callbacks=[callback]
#                    )
#
# model.evaluate(X_test, y_test, verbose=1)

# plt.figure(figsize=(15,7))
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'Validation'], loc='upper left')
# plt.show()
#
# plt.figure(figsize=(15,7))
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Loss Function')
# plt.ylabel('Loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'Validation'], loc='upper left')
# plt.show()
#
# # Classify custom sample
# def predict(text):
#     print(text)
#     sentence = clean(text)
#     sentence = tokenizer.texts_to_sequences([sentence])
#     sentence = pad_sequences(sentence, maxlen=256, truncating='pre')
#     result = le.inverse_transform(np.argmax(model.predict(sentence), axis=-1))[0]
#     proba =  np.max(model.predict(sentence))
#     print(f"{result} : {proba}\n\n")
#
# import pickle
#
# with open('tokenizer.pickle', 'wb') as f:
#     pickle.dump(tokenizer, f)
#
# with open('labelEncoder.pickle', 'wb') as f:
#     pickle.dump(le, f)
#
# model.save('Emotion Detection.h5')




from sklearn.preprocessing import OneHotEncoder
# onehot_encoder = OneHotEncoder()
# y_train = np.array(y_train)
# y_train = onehot_encoder.fit_transform(y_train.reshape(-1,1)).toarray()
# print(y_train)
#
# y_val = np.array(y_val)
# y_val = onehot_encoder.fit_transform(y_val.reshape(-1,1)).toarray()
# print(y_val)
#
# y_test = np.array(y_test)
# y_test = onehot_encoder.fit_transform(y_test.reshape(-1,1)).toarray()
# print(y_test)

# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
#
# def tokenize_text(row):
#     return tokenizer.encode_plus(row['Text'], padding='max_length', max_length=256, truncation=True)
#
# df_train['tokenized_text'] = df_train.apply(tokenize_text, axis=1)
# df_test['tokenized_text'] = df_test.apply(tokenize_text, axis=1)
# df_val['tokenized_text'] = df_val.apply(tokenize_text, axis=1)






























#
# #========================================================================================
