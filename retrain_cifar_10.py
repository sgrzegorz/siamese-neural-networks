import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
from utils import *
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

train_images.astype('float32') 
test_images.astype('float32') 
train_images = train_images/255.
test_images = test_images/255.

train_labels_cat = to_categorical(train_labels)
test_labels_cat = to_categorical(test_labels)

a= int(sys.argv[1])
b = a+100

X1 = np.concatenate([train_images,train_images])
X2 = np.concatenate([np.concatenate([train_images[a:], train_images[:a]]),
                    np.concatenate([train_images[b:], train_images[:b]])])
y1 = (train_labels == np.concatenate([train_labels[a:], train_labels[:a]])).reshape(-1)
y2 = (train_labels == np.concatenate([train_labels[b:], train_labels[:b]])).reshape(-1)
y = np.concatenate([y1, y2])
y_tst = (test_labels == np.concatenate([test_labels[1:], test_labels[:1]])).reshape(-1)

model = tf.keras.models.load_model("siamese_vanilla_cifar10_diff.h5")

history = model.fit([X1, X2], y, epochs=10, batch_size=512, shuffle=True, class_weight={0:1, 1:2},
         validation_data=([test_images, np.concatenate([test_images[1:], test_images[:1]])], y_tst))

model.save('siamese_vanilla_cifar10_diff.h5')