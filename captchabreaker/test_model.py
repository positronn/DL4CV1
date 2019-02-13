# test_model.py

import cv2
import imutils
import argparse
import numpy as np
from imutils import paths
from imutils import contours
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from pyimagesearch.utils.captchaHelper import preprocess



# construct the argument parse and aprse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required = True, help = "path to input directory of images")
ap.add_argument("-m", "--model", required = True, help  = "path to input model")
args = vars(ap.parse_args())


# load the pre-trained network
print("[INFO] loading pretrained network...")
model = load_model(args["model"])

# randomly sample a few of the input images
imagePaths = list(paths.list_images(args["input"]))
imagePaths = np.random.choice(imagePaths, size = (10,), replace = False)