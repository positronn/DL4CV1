# download_images.py

import os
import time
import requests
import argparse


# constrcut argument parse
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required = True, help = "path to output directory of images")
ap.add_argument("-n", "--num-images", type = int, default = 500, help = "# of images to download")
args = vars(ap.parse_args())


# initialize the URL that contains the captcha images that we will
# be downloading along with the total number of images downloaded
# this far
url = "https://www.e-zpassny.com/vector/jcaptcha.do"
total = 0


# loop over the number of images to download
for i in range(0, args["num_images"]):

    try:
        # try to grab a new captcha image
        r = requests.get(url, timeout = 60)

        # save the image to disk
        p = os.path.sep([args["output"], "{}.jpg".frormat(str(total).zfill(5))])
        f = open(p, 'wb')
        f.write(r.content)
        f.close()

        # update the counter
        print(f"[INFO] downloaded: {p}")
        total += 1

    except:
        print("[INFO] error downloading image...")

    # insert a small sleep to be courteous to the server
    time.sleep(0.1)

