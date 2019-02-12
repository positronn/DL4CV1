# minivggnet_cifar10.py

# import matplotlib and change backend
import matplotlib
matplotlib.use("Agg")


# import the necessary packages
import argparse
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.optimizers import SGD
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.nn.conv import MiniVGGNet


# construt the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required = True, help = "path to the output loss/accuract plot")
args = vars(ap.parse_args())


# load the training and testing data, then scale it into the range [0, 1]
print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0


# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)


# initialize the label names from the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse",
                "ship", "truck"]


# initialie the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr = 0.01, decay = 0.01 / 40, momentum = 0.9, nesterov = True)
model = MiniVGGNet.build(width = 32, height = 32, depth = 3, classes = 10, batchNorm = False)
model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics = ["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data = (testX, testY), batch_size = 64, epochs = 40,
              verbose = 1)


# evaluate tje network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size = 64)
print(classification_report(testY.argmax(axis = 1), predictions.argmax(axis = 1),
                            target_names = labelNames))


# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 40), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 40), H.history["loss"], label = "train_loss")
plt.plot(np.arange(0, 40), H.history["val_loss"], label = "val_loss")
plt.plot(np.arange(0, 40), H.history["acc"], label = "train_acc")
plt.title()
plt.title("Training loss and accuracy on CIFAR-10")
plt.xlabel("Epoch #")
plt.ylabel("Loss/ Accuracy")
plt.legend()
plt.savefig(args["output"])