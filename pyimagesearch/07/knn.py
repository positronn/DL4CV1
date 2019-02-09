# -*- coding: utf-8 -*-
# knn.py


import argparse
from imutils import paths
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader


'''
    LabelEncoder:
        A helper utility to convert labels represented as strings to integers
        where there is one unique integer per class label.
'''


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True, help = "path to input dataset")
ap.add_argument("-k", "--neighbors", type = int, default = 1, help = "number of nearest neighbors")
ap.add_argument("-j", "--jobs", type = int, default = -1, help = "cores to use in process")
args = vars(ap.parse_args())


# grab the list of images that we'll be describing
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))


# initialize the image preprocessor, load the dataset from disk
# and reshape the data matrix
sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors = [sp])
(data, labels) = sdl.load(imagePaths, verbose = 500)


'''
    In order to apply the k-NN algorithm, we need to “flatten” our images
    from a 3D representation to a single list of pixel intensities.

    We flatten the 32 × 32 × 3 images into an array with shape (3000, 3072).
    The actual image data hasn’t changed at all – the images are simply represented as
    a list of 3,000 entries, each of 3,072-dim (32 × 32 × 3 = 3, 072).
'''
data = data.reshape((data.shape[0], 3072))


# show information on memory consumption of the images
print("[INFO] features matrix: {:.1f} MB".format(data.nbytes / (1024 * 1000.0)))


# encode labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)


# partitioning data into training and test sets
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size = 0.25,
                                                        random_state = 42)


# train and evaluate a k-NN classifier on the raw pixel intensities
print("[INFO] evaluating k-NN classifier...")

model = KNeighborsClassifier(n_neighbors = args["neighbors"], n_jobs = args["jobs"])
model.fit(trainX, trainY)

print(classification_report(testY, model.predict(testX), target_names = le.classes_))
