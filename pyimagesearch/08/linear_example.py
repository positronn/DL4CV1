# -*- coding: utf-8 -*-
# linear_example.py

import cv2
import numpy as np



# initialize class labels and set the seed of the
# pseudorandom generator so we can reproduce our results
labels = ["dog", "cat", "panda"]


# randomly initialize our weight matrix and bias vector
# in a *real* training and classification task, these parameters
# would be *learned* by our model, but for the sake of the example
# let's use random values
W = np.random.randn(3, 3072)
b = np.random.randn(3)


# load our example image, resize it, and the flatten it into our
# "feature vector" representation
orig = cv2.imread("beagle.jpg")
image = cv2.resize(orig, (32, 32)).flatten()


# compute the output scores by taking the dot product between the
# weight matrix and image pixels, followed by adding in the bias
scores = W.dot(image) + b


# loop over the scores + labels and display them
for (label, score) in zip(labels, scores):
    print("[INFO] {}: {:.2f}".format(label, score))


# draw the label with the highest score on the image as our
# prediction
cv2.putText(orig, "Label: {}".format(labels[np.argmax(scores)]),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# display out input image
cv2.imshow("image", orig)
cv2.waitKey(0)