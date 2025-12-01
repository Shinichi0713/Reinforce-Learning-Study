import csv

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# パラメータ
vocab_size = 10
max_length = 4


# 学習・テストデータ
def load_movie_comment_data():
    docs = csv.reader(open("movie_comment_sample.csv"))
    next(docs, None)
    docs = list(docs)
    texts = [d[0] for d in docs]
    labels = [int(d[1]) for d in docs]
    return texts, labels


# 前処理
def preprocessing(texts, labels):
    tokenizer = Tokenizer(vocab_size)
    tokenizer.fit_on_texts(texts)
    encoded_docs = tokenizer.texts_to_sequences(texts)
    padded_docs = pad_sequences(encoded_docs,
                                maxlen=max_length,
                                padding='post')

    train_padded_docs = padded_docs[:6]
    test_padded_docs = padded_docs[6:7]

    train_labels = labels[:6]
    test_labels = labels[6:7]
    return train_padded_docs, test_padded_docs, train_labels, test_labels


# モデルの構築
def build_lstm_model():
    model = Sequential()
    model.add(
        Embedding(vocab_size, 2, input_length=max_length, mask_zero=True))
    model.add(LSTM(3, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(
        optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    print(model.summary())
    return model


# モデルの学習
def train(model,
          train_padded_docs,
          train_labels,
          epochs=1000):
    return model.fit(train_padded_docs,
                     train_labels,
                     epochs=epochs,
                     verbose=0)


# モデルの評価
def evaluate(model, train_padded_docs, train_labels):
    loss, accuracy = model.evaluate(train_padded_docs,
                                    train_labels,
                                    verbose=0)
    print("-" * 10)
    print('loss: {}'.format((loss)))
    print('accuracy: {}'.format((accuracy * 100)))


if __name__ == "__main__":
    texts, labels = load_movie_comment_data()
    train_padded_docs, test_padded_docs, train_labels, test_labels = \
        preprocessing(texts, labels)
    model = build_lstm_model()
    hist = train(model, train_padded_docs, train_labels)
    model.save("simple_lstm_weight.h5")
    evaluate(model, train_padded_docs, train_labels)
