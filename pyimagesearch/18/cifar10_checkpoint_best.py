# -*- coding: utf-8 -*-
# cifar10_checkpoint_best.py


import argparse
from keras.datasets import cifar10
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.nn.conv import MiniVGGNet


# construct the argument parse
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", required = True,
                help = "path to weights directory")
ap.add_argument("-e", "--epochs", default = 20,
                help = "int: number of epochs to train the model")
args = vars(ap.parse_args())


# load the training and testing data, then
# scale it into the range [0, 1]
print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0


# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)


# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr = 0.01, decay = 0.01 / 40, momentum = 0.9, nesterov = True)
model = MiniVGGNet.build(width = 32, height = 32, depth = 3, classes = 10)
model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics = ["accuracy"])



checkpoint = ModelCheckpoint(args["weights"], monitor = "val_loss", save_best_only = True,
                                verbose = 1)
callbacks = [checkpoint]

# train the nwtwork
print("[INFO] training network...")
epochs = int(args["epochs"])
H = model.fit(trainX, trainY, validation_data = (testX, testY),
                batch_size = 64, epochs = epochs, callbacks = callbacks, verbose = 1)