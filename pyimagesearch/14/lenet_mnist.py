# -*- coding: utf-8 -*-
# lenet_mnist.py


import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from keras.optimizers import SGD
from sklearn import datasets
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.nn.conv import LeNet



# grab the MNIST dataset
print("[INFO] accesing MNIST...")
dataset = datasets.fetch_mldata("MNIST Original")
data = dataset.data


# if we're using "channels first" ordering, then reshape the
# design matrix such that the design matrix is:
# num_samples * depth * rows * columns
if K.image_data_format() == "channels_first":
    data = data.reshape(data.shape[0], 1, 28, 28)


# other wise, we are using "channels last" ordering, so the design
# matrix shape should be: num_samples * rows * columns * depth
else:
    data = data.reshape(data.shape[0], 28, 28, 1)


# scale the input data into the range [0, 1]
(trainX, testX, trainY, testY) = train_test_split(data / 255.0,
                                        dataset.target.astype("int"),
                                        test_size = 0.25,
                                        random_state = 42)


# convert the labels from integers to vectors
le = LabelBinarizer()
trainY = le.fit_transform(trainY)
testY = le.transform(testY)


# initialize the optimizer model
print("[INFO] compiling model...")
opt = SGD(lr = 0.01)
model = LeNet.build(width = 28, height = 28, depth = 1, classes = 10)
model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics = ["accuracy"])


# train the network
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data = (testX, testY), batch_size = 128,
              epochs = 20, verbose = 1)


# evaluating network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size = 128)
print(classification_report(testY.argmax(axis = 1), predictions.argmax(axis = 1),
                            target_names = [str(x) for x in le.classes_]))


# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 20), H.history["loss"], label = "train_loss")
plt.plot(np.arange(0, 20), H.history["loss"], label = "train_loss")
plt.plot(np.arange(0, 20), H.history["loss"], label = "train_loss")
plt.plot(np.arange(0, 20), H.history["loss"], label = "train_loss")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss / Accuracy")
plt.legend()
plt.savefig("lenet_mnist.png")