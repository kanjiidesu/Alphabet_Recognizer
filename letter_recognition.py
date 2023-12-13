""" Copied from assignment, modified by Kevin, with support and fixes from Jonas. """

from __future__ import annotations

import imageio as imageio
# import os
# import pandas as pd
# import seaborn as sns
# from PIL import Image
import matplotlib.pyplot as plt
# from keras.datasets import mnist
# from sklearn.metrics import confusion_matrix
import numpy as np
from PIL import Image
from numpy import ndarray
from datasets import load_dataset
from keras.models import load_model
from keras import Sequential, Model
from keras.src.utils import to_categorical
from keras.src.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
# from sklearn.metrics import confusion_matrix

MODEL_FILENAME: str = "letter_recognizer.h5"
CHARACTER_COUNT = 26
mapping = {str(i): chr(i+65) for i in range(CHARACTER_COUNT)}

dataset = load_dataset("pittawat/letter_recognition")

# save input image dimensions
img_rows, img_cols = 28, 28


def _load_data() -> tuple:

    # Y is true value, X is test
    x_train: ndarray
    y_train: ndarray
    x_test: ndarray
    y_test: ndarray

    x_train = np.stack(tuple(~np.array(img.convert('L')) for img in dataset['train']['image']))
    y_train = np.array(dataset['train']['label'])
    assert x_train.shape == (len(dataset['train']), img_rows, img_cols)  # (26000, 28, 28)
    assert len(y_train) == len(x_train)

    x_test = np.stack(tuple(~np.array(img.convert('L')) for img in dataset['test']['image']))
    y_test = np.array(dataset['test']['label'])
    assert x_test.shape == (len(dataset['test']), img_rows, img_cols)  # (26000, 28, 28)
    assert len(y_test) == len(x_test)

    return ((x_train, y_train), (x_test, y_test))


def train_model(filename: str) -> None:

    # Y is true value, X is test
    x_train: ndarray
    y_train: ndarray
    x_test: ndarray
    y_test: ndarray

    (x_train, y_train), (x_test, y_test) = _load_data()

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    x_train = np.float_(x_train)
    x_test = np.float_(x_test)

    x_train /= 255.0
    x_test /= 255.0

    y_train = to_categorical(y_train, CHARACTER_COUNT)
    y_test = to_categorical(y_test, CHARACTER_COUNT)

    # Convert images to NumPy array and normalize
    # images = np.array(images) / 255.0

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(img_rows, img_cols, 1)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(CHARACTER_COUNT, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    batch_size = 128
    epochs = 10

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    model.save(filename)


def predict_letter(image: str, model: Model | str = None) -> str:

    # (x_train, y_train), (x_test, y_test) = _load_data()

    im = imageio.imread(image)

    # Convert the image to grayscale
    gray = np.dot(im[..., :3], [0.299, 0.587, 0.114])

    # Resize the image to match the training data dimensions
    gray = np.array(Image.fromarray(gray).resize((img_rows, img_cols)))

    # reshape the image
    gray = gray.reshape(1, img_rows, img_cols, 1)

    # normalize image
    gray = gray / 255

    if isinstance(model, str):
        model: Model = load_model(model)

    # predict digit
    prediction = model.predict(gray)

    return prediction.argmax()


# def confusing_matrix() -> None:
#     """ Generated by ChatGPT """
#
#     # Load MNIST data
#     x_train: ndarray
#     y_train: ndarray
#     x_test: ndarray
#     y_test: ndarray
#     (x_train, y_train), (x_test, y_test) = _load_data()
#
#     # Assuming img_rows and img_cols are the dimensions expected by your model
#     img_rows, img_cols = 28, 28
#
#     # Preprocess test data
#     x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
#     x_test = x_test.astype('float32') / 255
#
#     # Load the model
#     model: Sequential = load_model(MODEL_FILENAME)
#
#     # Predict on test data
#     y_pred = model.predict(x_test)
#     y_true = to_categorical(y_test, num_classes=10)
#
#     # Get predicted labels
#     y_pred_labels = np.argmax(y_pred, axis=1)
#     y_true_labels = np.argmax(y_true, axis=1)
#
#     # Create confusion matrix
#     # TODO Kevin: Maybe fix plz IDK
#     cm = confusion_matrix(y_true_labels, y_pred_labels)
#
#     # Plot confusion matrix
#     f, ax = plt.subplots(figsize=(8, 8))
#     sns.heatmap(cm, annot=True, linewidths=0.5, linecolor="red", fmt=".0f", ax=ax)
#     plt.xlabel("Predicted Label")
#     plt.ylabel("True Label")
#     plt.title("Confusion Matrix")
#     plt.show()


if __name__ == '__main__':
    train_model(MODEL_FILENAME)
    # confusing_matrix()