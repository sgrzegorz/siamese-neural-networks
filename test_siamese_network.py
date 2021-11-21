# USAGE
# python test_siamese_network.py --input examples

# import the necessary packages
from pyimagesearch import config
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, average_precision_score, roc_auc_score
import numpy as np
from prepare_data import load_data, load_test_data
from pyimagesearch.utils import prepare_im
import copy

# grab the test dataset image paths and then randomly generate a
# total of 10 image pairs
print("[INFO] loading test dataset...")
(testX, testY) = load_data(config.TEST_DIR, config.IMG_SHAPE[:2])
(placeX, placeY) = load_test_data(config.LABELS_DIR, config.IMG_SHAPE[:2])
placeX = np.array([prepare_im(x) for x in placeX])

# load the model from disk
print("[INFO] loading siamese model...")
model = load_model(config.MODEL_PATH)

# calculate similarities
print("[INFO] model is evaluating data...")
labels = []
probs = []
if config.SUBSAMPLE:
    idx = [i * 10 for i in range(len(testY)//10)]
    testY = testY[idx]
    for x in idx:
        x = prepare_im(testX[x])
        lst = []
        for q in placeX:
            preds = model.predict([x, q])
            lst.append(preds[0][0])
        labels.append(max(list(zip(lst, placeY)), key=lambda x: x[0])[1])
        probs.append(copy.copy(lst))
else:
    for x in testX:
        x = prepare_im(x)
        lst = []
        for q in placeX:
            preds = model.predict([x, q])
            lst.append(preds[0][0])
        labels.append(max(list(zip(lst, placeY)), key=lambda x: x[0])[1])
        probs.append(copy.copy(lst))

# calculate metrics
print("[INFO] Results:")
print("[INFO] Confusion matrix:")
print(confusion_matrix(y_true=testY, y_pred=labels))
print("[INFO] Accuracy:")
print(accuracy_score(y_true=testY, y_pred=labels))
print("[INFO] mAP:")
aps = []
for cls in set(testY):
    a = [max([y[0] for y in list(zip(probs[i], placeY)) if y[1] == cls]) for i in range(len(probs))]
    labels_cls = [1 if x == cls else 0 for x in labels]
    aps.append(average_precision_score(labels_cls, a))
print(sum(aps)/len(aps))
