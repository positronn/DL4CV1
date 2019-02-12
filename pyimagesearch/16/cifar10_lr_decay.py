# -*- coding: utf-8 -*-
# cifar10_lr_decay.py


# import matplotlib and change backend
import matplotlib
matplotlib.use("Agg")

import argparse
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.nn.conv import MiniVGGNet




def step_decay(epoch:int):
    '''
    A learning rate scheduler.
    Calculates next step decay value for the current epoch
    as:
        alpha = initAlpha * (factor ** np.floor((1 + epoch) / dropEvery))

    parameters
    ----------
        epoch: current epoch to update the learning rate

    returns
    -------
        alpha: the next learning rate value
    '''
    # initiaize the base initial learning rate, drop factor
    # and epochs to drop every
    initAlpha = 0.01
    factor = 0.25
    dropEvery = 5


    # compute learning rate for the current epoch
    alpha = initAlpha * (factor ** np.floor((1 + epoch) / dropEvery))

    return float(alpha)


# construct the argument parse
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required = True, help = "path to the putput loss/accuracy plot")
ap.add_argument("-e", "--epochs", default = 20, help = "int: number of epochs to train the model")
args = vars(ap.parse_args())


# load the training and testing data, then scale it
# into the range [0, 1]
print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0


# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)


# initialize the label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer",
                "dog", "frog", "horse", "ship", "truck"]


# define the set of callbacks to be passed to the
# model during training
callbacks = [LearningRateScheduler(step_decay)]


# initialize the optimizer and model
opt = SGD(lr = 0.01, momentum = 0.9, nesterov = True)
model = MiniVGGNet.build(width = 32, height = 32, depth = 3, classes = 10)
model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics = ["accuracy"])


# train the network
epochs = int(args["epochs"])
H = model.fit(trainX, trainY, validation_data = (testX, testY),
                batch_size = 64, epochs = epochs, callbacks = callbacks, verbose = True)


# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size = 64)
print(classification_report(testY.argmax(axis = 1), predictions.argmax(axis = 1),
                            target_names = labelNames))


# plot the traiing loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs), H.history["loss"], label = "train_loss")
plt.plot(np.arange(0, epochs), H.history["val_loss"], label = "val_loss")
plt.plot(np.arange(0, epochs), H.history["acc"], label = "train_acc")
plt.plot(np.arange(0, epochs), H.history["val_acc"], label = "val_acc")
plt.title("Training Loss and Accuracy on CIFAR-10")
plt.xlabel("Epoch #")
plt.ylabel("Loss / Accuracy")
plt.legend()
plt.savefig(args["output"])