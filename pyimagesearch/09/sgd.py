# -*- coding: utf-8 -*-
# sgd.py

'''
Stochastic Gradient Descent (SGD), a simple modification to the standard
gradient descent algorithm that computes the gradient and updates the weight
matrix W on small batches of training data, rather than the entire training set.
While this modification leads to “more noisy” updates, it also allows us to
take more steps along the gradient (one step per each batch versus one
step per epoch), ultimately leading to faster convergence and no negative
affects to loss and classification accuracy.


SGD is arguably the most important algorithm when it comes to training
deep neural networks. Even though the original incarnation of SGD was introduced
over 57 years ago, it is still the engine that enables us to train large
networks to learn patterns from data points. Above all other algorithms
covered in this book, take the time to understand SGD.
'''

import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
from gradientdescent import sigmoidActivation, predict



def nextBatch(X, y, batchSize):
    '''
        Loops over the dataset 'X' in mini-batches, yielding
        a tuple of the current batched data and labels

        parameters
        ----------
            X: Input matrix representing a a set of data points
            y: output vector representing the (true) labels of their corresponding
                inputs
            batchSize: size of the mini-batch

        returns
        -------
            yields subsets of both X and y as mini-batches
    '''

    for i in np.arange(0, X.shape[0], batchSize):
        yield(X[i:i + batchSize], y[i:i + batchSize])



def main():
    '''
    '''
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--epochs", type = float, default = 100, help = "# of epochs")
    ap.add_argument("-a", "--alpha", type = float, default = 0.01, help = "learning rate")
    ap.add_argument("-b", "--batch-size", type = int, default = 32, help = "size of SGD mini-batches")
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


    # loop over desired number of epochs
    for epoch in np.arange(0, args["epochs"]):
        # initialize the total loss for the epoch
        epochLoss = []

        # loop over our data in batches
        for (batchX, batchY) in nextBatch(X, y, args["batch_size"]):
            # take the dot product between our current batch of features
            # and the weight matrix, the pass this value though our activation
            # function
            preds = sigmoidActivation(batchX.dot(W))

            # now that we have our predictions, we need to determine the
            # 'error', which is the difference between our predictions
            # and the true values
            error = preds - batchY
            epochLoss.append(np.sum(error ** 2))

            # the gradient descent update is the dot product between our
            # current batch and the error on the batch
            gradient = batchX.T.dot(error)

            # in the update stage, all we need to do is "nudge" the
            # weight matrix in the negative direction of the gradient
            # (hence the term "gradient descent") by takking small step
            # towards a set of 'more optimal' parameters
            W += -args["alpha"] * gradient

            # update our loss history by taking the average loss across
            # all batches
        loss = np.average(epochLoss)
        losses.append(loss)

            # check to see if an update should be displayed
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print("[INFO] epoch: {0:4},\tloss: {1:.7f}".format(int(epoch + 1), loss))


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