
import tensorflow as tf
from matplotlib import pyplot
from keras.datasets import cifar10
# from emnist import extract_training_samples
from tensorflow.keras import *
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG19

#airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
num_classes = 10
x_train = x_train/255.
y_train = tf.keras.utils.to_categorical(y_train, num_classes)

x_test = x_test/255.
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

conv_base = VGG19(weights=None,
                  include_top=False,
                  input_shape=(32, 32, 3))

model3 = models.Sequential()
model3.add(conv_base)
model3.add(layers.Flatten())
# model3.add(layers.Dense(256, activation='relu'))
# model3.add(layers.Dense(256, activation='relu'))
# model3.add(layers.Dropout(rate=0.2))
# model3.add(layers.Dense(256, activation='relu'))
# model3.add(layers.Dense(256, activation='relu'))

model3.add(layers.Dense(num_classes, activation='softmax'))

model3.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

history3 = model3.fit(x_train, y_train, batch_size=128, epochs=200,validation_data=(x_test, y_test))


model3.save('my_model.h5')
