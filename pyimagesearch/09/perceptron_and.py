# -*- coding: utf-8 -*-
# perceptron_and.py

import numpy as np
from pyimagesearch.nn import Perceptron


# construct the AND dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [0], [0], [1]])


# define our perceptron and train it
print("[INFO] training perceptron...")
p = Perceptron(X.shape[1], alpha = 0.1)
p.fit(X, y, epochs = 20)


# evaluate perceptron
print("[INFO] testing perceptron...")

# now that out network is trained, loop over the data points
for (x, target) in zip(X, y):
    # make a prediction on the data point and display the
    # result to our console
    pred = p.predict(x)
    print("[INFO] data = {}, ground-truth = {}, pred = {}".format(x, target[0], pred))