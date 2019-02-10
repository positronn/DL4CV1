#!/anaconda3/envs/vision/bin/python
# -*- coding: utf-8 -*-
# keras_cifar10.py


import argparse
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.optimizers import SGD
from keras.layers.core import Dense
from keras.models import Sequential
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split


# construct argument parse
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", default = "plot/keras_cifar10.png",
                help = "path to the output loss/accuracy plot")
args = vars(ap.parse_args())


# load the training and testing data
print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()

# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# scaling data into the range [0, 1]
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

# splitting into training and validation sets
(trainX, valX, trainY, valY) = train_test_split(trainX, trainY, test_size = 0.25)

# reshaping the data to fit it in the input layer of neural network
trainX = trainX.reshape((trainX.shape[0], 3072))
valX = valX.reshape((valX.shape[0], 3072))
testX = testX.reshape((testX.shape[0], 3072))

# initialize the label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog",
                "frog", "horse", "ship", "truck"]


# define the 3072-1024-512-10 atchitecture using keras
model = Sequential()
model.add(Dense(1024, input_shape = (3072,), activation = "relu"))
model.add(Dense(512, activation = "relu"))
model.add(Dense(10, activation = "softmax"))


# train the model using SGD
print("[INFO] training network...")
sgd = SGD(lr = 0.01)
model.compile(loss = "categorical_crossentropy", optimizer = sgd, metrics = ["accuracy"])
H = model.fit(trainX, trainY, validation_data = (valX, valY), epochs = 100, batch_size = 16)


# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size = 16)
print(classification_report(testY.argmax(axis = 1),
                            predictions.argmax(axis = 1),
                            target_names = labelNames))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label = "train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label = "val_loss")
plt.plot(np.arange(0, 100), H.history["acc"], label = "train_acc")
plt.plot(np.arange(0, 100), H.history["val_acc"], label = "val_acc")
plt.title("Training Loss and accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss / Accuracy")
plt.legend()
plt.savefig(args["output"])