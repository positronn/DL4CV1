# -*- coding: utf-8 -*-
# cifar10_monitor.py

# import matplotlib and change backend
import matplotlib
matplotlib.use("Agg")

import os
import argparse
from keras.datasets import cifar10
from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.callbacks import TrainingMonitor
from pyimagesearch.nn.conv import MiniVGGNet


# construct argument parse
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required = True, help = "path to the output directory")
ap.add_argument("-e", "--epochs", default = 20, help = "int: number of epochs to train the model")
args = vars(ap.parse_args())


# show information on the process ID
print("[INFO] process ID: {}".format(os.getpid()))


# loading the training and testing data, then scale it into 
# the ragne [0, 1]
print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0


# convert labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)


# initialize the label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer",
                "dog", "frog", "horse", "ship", "truck"]


# initialize the SGD optimizer, but without any learning rate decay
print("[INFO] compiling model...")
opt = SGD(lr = 0.01, momentum = 0.9, nesterov = True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics =["accuracy"])


# construct the set of callbacks
figPath = os.path.sep.join([args["output"], "{}.png".format(os.getpid())])
jsonPath = os.path.sep.join([args["output"], "{}.json".format(os.getpid())])
callbacks = [TrainingMonitor(figPath, jsonPath = jsonPath)]


# train the networks
print("[INFO] training network...")
epochs = int(args["epochs"])
model.fit(trainX, trainY, validation_data = (testX, testY),
            batch_size = 64, epochs = epochs, callbacks = callbacks, verbose = 1)