#!/anaconda3/envs/vision/bin/python
# -*- coding: utf-8 -*-
# keras_mnist.py


import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.layers.core import Dense
from keras.models import Sequential
from keras.optimizers import SGD


# construct argument parse
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", default = "plot/keras_mnist.png",
                help = "path to the output loss/accuracy plot")
args = vars(ap.parse_args())


# grab the MNIST dataset
print("[INFO] loading MNIST (full) dataset...")
dataset = datasets.fetch_mldata("MNIST Original")


# scale the raw pixel intensities to the range [0, 1.0],
# then contruct the training and testing splits
data = dataset.data.astype("float") / 255.0
(trainX, testX, trainY, testY) = train_test_split(data, dataset.target, test_size = 0.5)

# convert labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)


# define the 784-256-128-10 architecture using Keras
model = Sequential()
model.add(Dense(256, input_shape = (784,), activation = "sigmoid"))
model.add(Dense(128, activation = "sigmoid"))
model.add(Dense(10, activation = "softmax"))


# train the model using SGD
print("[INFO] training network...")
sgd = SGD(lr = 0.01)
model.compile(loss = "categorical_crossentropy", optimizer = sgd, metrics = ["accuracy"])
H = model.fit(trainX, trainY, validation_data = (testX, testY), epochs = 100, batch_size = 128)


# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size = 128)
print(classification_report(testY.argmax(axis = 1), predictions.argmax(axis = 1),
                            target_names = [str(x) for x in lb.classes_]))


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