import os
import glob
import numpy as np
import cv2


def load_data(path, image_shape):
    images = []
    labels = []
    labels_p = []
    for full_path in glob.glob(f"{path}/**/*.png", recursive=True):
        head, tail = os.path.split(full_path)
        labels_p.append((tail.split("_")[0]))
        img = cv2.imread(full_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, image_shape)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        images.append(np.asarray(img))
    classes = list(np.unique(labels_p))

    print("[INFO] Label indexes:")
    for index, class_name in enumerate(classes):
        print(f"Class: {class_name} --- Index: {index}")

    for elem in labels_p:
        labels.append(classes.index(elem))
    return np.array(images), np.array(labels)


def load_test_data(path, image_shape):
    images = []
    labels = []
    labels_p = []
    for full_path in glob.glob(f"{path}/**/query/*.png", recursive=True):
        head, tail = os.path.split(full_path)
        labels_p.append((tail.split("_")[0]))
        img = cv2.imread(full_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, image_shape)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        images.append(np.asarray(img))
    classes = list(np.unique(labels_p))

    print("[INFO] Label indexes:")
    for index, class_name in enumerate(classes):
        print(f"Class: {class_name} --- Index: {index}")

    for elem in labels_p:
        labels.append(classes.index(elem))
    return np.array(images), np.array(labels)