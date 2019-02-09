# -*- coding: utf-8 -*-
# gradientdescent.py

import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs



def sigmoidActivation(x:float) -> float:
    '''
        Compute the sigmoid activation value for a given input.

            sigma(x) = 1.0 / (1 + exp(-x))

        parameters
        ----------
            x: A float number to calculate its sigmoid activation according
                to the sigmoid function.

        returns
        -------
            activation_value: float
    '''

    return 1.0 / (1 + np.exp(-x))



def predict(X:np.ndarray, W:np.ndarray) -> np.ndarray:
    '''
        Threshold the predictions obtained by sigmoidActivation: 
        any prediction with a value <= 0.5 is set to 0 while any prediction
        with a value > 0.5 is set to 1.

        parameters
        ----------
            X: Input matrix representing a a set of data points
            W: The weight matrix representing our model as in:
                    s = f(x_i, W)
                or
                    y = f(X, W)

        returns
        -------
            preds: an array (list)
    '''
    
    # take the dot product between our features and weight matrix
    preds = sigmoidActivation(X.dot(W))

    # apply a step function to threshold the outputs to binary
    # class labels
    preds[preds <= 0.5] = 0
    preds[preds > 0] = 1

    # return the predictions
    return preds



def main():
    '''
    '''
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--epochs", type = float, default = 100, help = "# of epochs")
    ap.add_argument("-a", "--alpha", type = float, default = 0.01, help = "learning rate")
    args = vars(ap.parse_args())


    # generate a 2-class classification problem with
    # 1,000 data points where each data point is a 2D feature vector
    print("[INFO] creating dataset...")
    (X, y) = make_blobs(n_samples = 1000, n_features = 2, centers = 2, cluster_std = 1.5,
                            random_state = 1)
    y = y.reshape((y.shape[0], 1))


    # insert a column of 1's as the last entry in the feature
    # matrix -- this little trick allows us to treat the bias
    # as a trainable parameter within the weight matrix
    X = np.c_[X, np.ones((X.shape[0]))]


    # partition the data into training and testing splits using 50%
    # of rhe data for training and the remaining 50% for testing
    print("[INFO] splitting dataset...")
    (trainX, testX, trainY, testY) = train_test_split(X, y, test_size = 0.5, random_state = 42)


    # ================================= Training stage =================================
    # 
    # initialize our weight matrix and list of losses
    print("[INFO] training...")
    W = np.random.randn(X.shape[1], 1)
    losses = []


    # loop over the desired number of epochs
    for epoch in np.arange(0, args["epochs"]):
        # take the dot product between our features 'X' and the wight
        # matrix 'W', then pass this value through our sigmoid activation
        # function, thereby giving us our predictions on the dataset
        preds = sigmoidActivation(trainX.dot(W))

        # now that we have our predictions, we need to determine the 
        # 'error', which is the difference between our predictions and
        # the true values
        error = preds - trainY
        loss = np.sum(error ** 2)
        losses.append(loss)

        # the gradient descent update is the dot product between our
        # features and the error of the predictions
        gradient = trainX.T.dot(error)

        # in the update stage, all we need to do is "nudge" the weight
        # matrix in the negative direction of the gradient (hence the term
        # gradient descent") by taking a small step towards a set of
        # "more optimal" oarameters
        W += -args["alpha"] * gradient


    # ================================= Evaluation stage =================================
    # 
    # evaluate our model
    print("[INFO] evaluating...\n")
    preds = predict(testX, W)
    print(classification_report(testY, preds))


    # ================================= Potting stage ====================================
    # 
    # plotting the testing classification data
    print("[INFO] plotting...")
    plt.style.use("ggplot")
    plt.figure()
    plt.title("Data")
    plt.scatter(testX[:, 0], testX[:, 1], c = testY[:, 0], marker = 'o', s = 20, cmap = 'jet')


    # construct a figure that plots the loss over time
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, args["epochs"]), losses)
    plt.title("Training loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.show()


    print("[INFO] closing program...")


if __name__ == '__main__':
    main()
