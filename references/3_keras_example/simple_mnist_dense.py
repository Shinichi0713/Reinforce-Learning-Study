from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# パラメータ
batch_size = 128
num_classes = 10
epochs = 12


# 学習とテスト、データの読み込み、前処理
def load_data_for_dense():
    (x_train, y_train), (x_test,
                         y_test) = mnist.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    return x_train, x_test, y_train, y_test


# モデルの構築
def build_dense_model():
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(),
                  metrics=['accuracy'])

    print(model.summary())
    return model


# モデルの学習
def train(model, x_train, y_train, epochs=10):
    return model.fit(x_train, y_train, epochs=epochs)


if __name__ == "__main__":
    x_train, x_test, y_train, y_test = load_data_for_dense(
    )
    model = build_dense_model()
    hist = train(model, x_train, y_train)
    model.save("simple_mnist_dense_weight.h5")
