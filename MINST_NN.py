from time import time

import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense

# view the data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print("X_train shape: ", X_train.shape)
print("y_train shape: ", y_train.shape)
print("X_test shape: ", X_test.shape)
print("y_test shape: ", y_test.shape)

# Preprocess
X_test = X_test / 255.0
X_train = X_train / 255.0

print(y_train[0])

# get my data
X_test_my = np.zeros_like(X_test[0:10])
y_test_my = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

for i in range(11):
    im = cv2.imread("Digit_" + str(i) + ".png")
    X_test_my[i] = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

X_test_my = X_test_my / 255.0

plt.figure(figsize=(5, 5))
for i in range(11):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_test_my[i], cmap=plt.cm.binary)
    plt.xlabel(y_test_my[i])
plt.show()

# train_accuracies = []
# test_accuracies = []
# time_to_train = []
# epochs = []

# set network architecture
model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)),
                             tf.keras.layers.Dense(128, activation='relu'),
                             tf.keras.layers.Dense(10)])

# set settings
model.compile(optimizer='adam',
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
metrics=['accuracy'])
start = time()

# train
model.fit(X_train, y_train, epochs=10)

# time_to_train.append(time() - start)

# _, test_accuracy = model.evaluate(X_test, y_test)
# print('Test Accuracy: %.2f' % (test_accuracy * 100))
#
# _, train_accuracy = model.evaluate(X_train, y_train)
# print('Train Accuracy: %.2f' % (train_accuracy * 100))

# test_accuracies.append(test_accuracy)
# train_accuracies.append(train_accuracy)
# epochs.append(i)

# plt.plot(epochs, test_accuracies, label='testing accuracy')
# plt.plot(epochs, train_accuracies, label='training accuracy')
# plt.legend()
# plt.xlabel('Epochs')
# plt.ylabel('accuracy')
# plt.show()

# plt.figure(1)
# plt.plot(epochs, time_to_train)
# plt.xlabel('epochs')
# plt.ylabel('time')
# plt.show()

# evaluate the keras model
_, accuracy = model.evaluate(X_test_my, y_test_my)
print('Accuracy: %.2f' % (accuracy*100))

# model.predict_classes(X)
