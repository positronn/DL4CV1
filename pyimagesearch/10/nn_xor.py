# -*- coding: utf-8 -*-
# nn_xor.py


import numpy as np
from pyimagesearch.nn import NeuralNetwork


# construct the XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])



# define neural network architecture and train it
nn = NeuralNetwork([2, 2, 1], alpha = 0.5)
nn.fit(X, y, epochs = 5000)


# loop over XOR dataset
for (x, target) in zip(X, y):
    # make prediciton on the data point and display the result
    # to out console
    pred = nn.predict(x)[0][0]
    step = 1 if pred > 0.5 else 0
    print(f"[INFO] data: {x}, ground-truth: {target[0]}, pred: {pred}, step: {step}")