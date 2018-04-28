import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping


def get_coefs(word1, *arr):
    return word1, np.asarray(arr, dtype='float32')


# size of each word vector
embed_size = 300
# number of rows in the embedding vector (unique words)
max_features = 100000
# max words in a comment; pad if not this length
maxlen = 150
batch_size = 128
epochs = 2

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
tokenizer = Tokenizer(num_words=max_features)

list_sentences_train = train["comment_text"].fillna("_na_").values
list_sentences_test = test["comment_text"].fillna("_na_").values

tokenizer.fit_on_texts(list(list_sentences_train))

list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)

X_tr = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)

# load embedding file
embeddings_index = dict(get_coefs(*o.strip().split()) for o in open('glove.840B.300d.txt'))
print ("Loaded GloVe Embedding, embed size = 300")

all_embedding = np.stack(embeddings_index.values())
emb_mean, emb_std = all_embedding.mean(), all_embedding.std()
print "Mean and SD of embedding %s, %s" % (emb_mean, emb_std)

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
x = Bidirectional(LSTM(batch_size, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
x = GlobalMaxPool1D()(x)
x = Dense(batch_size, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(6, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
early = EarlyStopping(monitor="val_acc", mode="max", patience=5)
callbacks_list = [early]
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-3), metrics=['accuracy'])
model.summary()
model.fit(X_tr, y, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=callbacks_list)

y_predict = model.predict([X_te], batch_size=1024, verbose=1)

sample_submission = pd.read_csv('sample_submission.csv')
sample_submission[list_classes] = y_predict
sample_submission.to_csv('submission_lstm_keras.csv', index=False)
