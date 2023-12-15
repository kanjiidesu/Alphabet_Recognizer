#!env/bin/python3
""" Copied from assignment, modified by Kevin, with support and fixes from Jonas. """

from __future__ import annotations

import os
from os import path
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
from datasets import load_dataset, DatasetDict, Dataset, IterableDatasetDict, IterableDataset
from keras.models import load_model
from keras import Sequential, Model
from keras.src.utils import to_categorical
from keras.src.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

MODEL_FILENAME: str = "letter_recognizer.h5"
CHARACTER_COUNT = 26
mapping = {str(i): chr(i+65) for i in range(CHARACTER_COUNT)}

dataset: DatasetDict | Dataset | IterableDatasetDict | IterableDataset = None

# save input image dimensions
IMG_ROWS, IMG_COLS = 28, 28

SAVE_DIR: str = 'images_folder'


def _load_data(with_user_images=True) -> tuple:

    global dataset
    if dataset is None:
        dataset = load_dataset("pittawat/letter_recognition")

    # Y is true value, X is the image
    x_train: ndarray
    y_train: ndarray
    x_test: ndarray
    y_test: ndarray

    x_train = np.stack(tuple(np.array(img.convert('L')) for img in dataset['train']['image']))
    y_train = np.array(dataset['train']['label'])
    assert x_train.shape == (len(dataset['train']), IMG_ROWS, IMG_COLS)  # (26000, 28, 28)
    assert len(y_train) == len(x_train)

    x_test = np.stack(tuple(np.array(img.convert('L')) for img in dataset['test']['image']))
    y_test = np.array(dataset['test']['label'])
    assert x_test.shape == (len(dataset['test']), IMG_ROWS, IMG_COLS)  # (26000, 28, 28)
    assert len(y_test) == len(x_test)

    if with_user_images:
        sorted_files = sorted(os.listdir(SAVE_DIR))

        # Set the ratio for train/test split
        split_ratio = 0.8

        # Determine the split index
        split_index = int(len(sorted_files) * split_ratio)

        # Split files into training and test sets
        train_files = sorted_files[:split_index]
        test_files = sorted_files[split_index:]

        # Y is true value, X is the image
        x_train2 = []
        y_train2 = []
        x_test2 = []
        y_test2 = []

        # Iterate through training files
        for file_name in train_files:
            image_path = os.path.join(SAVE_DIR, file_name)
            # Extract the letter from the filename (assuming it's the first character)
            letter = file_name.split('-')[0]
            #letter = path.split(file_name)[-1].split('-')[0]
            # Read and preprocess the image
            image = np.array(Image.open(image_path).resize((28, 28)).convert('L')) / 255.0
            # Append to arrays
            x_train2.append(image)
            y_train2.append(ord(letter) - ord('A'))  # Assuming letters are uppercase A-Z

        # Iterate through test files
        for file_name in test_files:
            image_path = os.path.join(SAVE_DIR, file_name)
            # Extract the letter from the filename (assuming it's the first character)
            letter = file_name.split('-')[0]
            #letter = path.split(file_name)[-1].split('-')[0]
            # Read and preprocess the image
            image = np.array(Image.open(image_path).resize((28, 28)).convert('L')) / 255.0
            # Append to arrays
            x_test2.append(image)
            y_test2.append(ord(letter) - ord('A'))  # Assuming letters are uppercase A-Z

        x_train = np.concatenate((np.array(x_train2), x_train))
        y_train = np.concatenate((np.array(y_train2), y_train))
        x_test = np.concatenate((np.array(x_test2), x_test))
        y_test = np.concatenate((np.array(y_test2), y_test))

    # user_images = {letter: _imagefile_to_ndarray(imagefile) }

    return ((x_train, y_train), (x_test, y_test))


def train_model(filename: str) -> None:

    # Y is true value, X is test
    x_train: ndarray
    y_train: ndarray
    x_test: ndarray
    y_test: ndarray

    (x_train, y_train), (x_test, y_test) = _load_data()

    x_train = x_train.reshape(x_train.shape[0], IMG_ROWS, IMG_COLS, 1)
    x_test = x_test.reshape(x_test.shape[0], IMG_ROWS, IMG_COLS, 1)

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
                     input_shape=(IMG_ROWS, IMG_COLS, 1)))

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


def format_for_prediction(seq) -> ndarray:
    # Convert the image to grayscale
    gray = np.dot(seq[..., :3], [0.299, 0.587, 0.114])

    # Resize the image to match the training data dimensions
    gray = np.array(Image.fromarray(gray).resize((IMG_ROWS, IMG_COLS)))

    if gray.shape != (1, IMG_ROWS, IMG_COLS, 1):
        # reshape the image
        gray = gray.reshape(1, IMG_ROWS, IMG_COLS, 1)

    # normalize image
    gray = gray / 255

    return gray


def _imagefile_to_ndarray(image: str) -> ndarray:
    im = imageio.imread(image)
    return format_for_prediction(im)


def predict_letter(image: str | ndarray, model: Model | str = MODEL_FILENAME, show_converted_letter: bool = False) -> str:

    # (x_train, y_train), (x_test, y_test) = _load_data()

    if isinstance(image, str):
        image = _imagefile_to_ndarray(image)
    assert isinstance(image, ndarray)

    if isinstance(model, str):
        model: Model = load_model(model)

    assert model is not None

    # predict digit
    prediction = model.predict(image)

    # Get the predicted letter
    predicted_letter = chr(65 + prediction.argmax())

    if show_converted_letter:
        # Display the image with the predicted letter, shows image plot
        plt.imshow(image.reshape(IMG_ROWS, IMG_COLS), cmap='Greys')
        plt.title(f"Predicted Letter: {predicted_letter}")
        plt.show()

    return predicted_letter


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


def letter_from_dataset(letter: str, num_show: int) -> None:
    (x_train, y_train), (x_test, y_test) = _load_data()

    assert len(letter) == 1
    assert ord(letter) > 64 and ord(letter) < 65+CHARACTER_COUNT

    for img, num in zip(x_train, y_train):
        if chr(65+num) == letter:
            plt.imshow(img, cmap='Greys')
            plt.show()
            num_show -= 1
        if num_show <= 0:
            break


if __name__ == '__main__':
    # train_model(MODEL_FILENAME)
    letter_from_dataset('B', 1)
    # confusing_matrix()
