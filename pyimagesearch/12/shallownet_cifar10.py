# -*- coding: utf-8 -*-
# shallownet_cifar10.py


import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from pyimagesearch.nn.conv import ShallowNet

# loading the training and testing data
print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()


# scaling data into range [0, 1]
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0


# convert the albels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)


# initialize the lavel names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer",
                "dog", "frog", "horse", "ship", "truck"]


# initialize the optimizer model
print("[INFO] compiling model...")
opt = SGD(lr = 0.01)
model = ShallowNet.build(width = 32, height = 32, depth = 3, classes = 10)
model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics = ["accuracy"])


# train the network
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_split = 0.1, batch_size = 32, epochs = 40, verbose = 1)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size = 32)
print(classification_report(testY.argmax(axis = 1), predictions.argmax(axis = 1),
      target_names = labelNames))


# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 40), H.history["loss"], label = "train_loss")
plt.plot(np.arange(0, 40), H.history["val_loss"], label = "val_loss")
plt.plot(np.arange(0, 40), H.history["acc"], label = "train_acc")
plt.plot(np.arange(0, 40), H.history["val_acc"], label = "val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss / Accuracy")
plt.legend()
plt.show()