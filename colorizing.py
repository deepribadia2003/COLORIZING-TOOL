import numpy as np
import argparse
import cv2
import os

##model load karne ka path
DIR = r"C:\Users\deepr\MiniProject2BTE"
PROTOTXT = os.path.join(DIR, r"model/colorization_deploy_v2.prototxt")
POINTS = os.path.join(DIR, r"model/pts_in_hull.npy")
MODEL = os.path.join(DIR, r"model/colorization_release_v2.caffemodel")

##argparser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True, help="path to input black and white image")
args = vars(ap.parse_args())

##model loading
print("Load model")
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
pts = np.load(POINTS)

##load centers for ab channel
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

##loading input image
image = cv2.imread(args["image"])
scaled = image.astype("float32") / 255.0
lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

resized = cv2.resize(lab, (224, 224))                                                 
L = cv2.split(resized)[0]
L -= 50

print("Colorizing the image")
net.setInput(cv2.dnn.blobFromImage(L))
ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

L = cv2.split(lab)[0]
colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
colorized = np.clip(colorized, 0, 1)

colorized = (255 * colorized).astype("uint8")

# Save the colorized image with unique filename
output_dir = os.path.join(DIR, r"colorized-images")
os.makedirs(output_dir, exist_ok=True)
colorized_output_path = os.path.join(output_dir, "colorized_image_1.jpg")
cv2.imwrite(colorized_output_path, colorized)
print(f"Colorized image saved at: {colorized_output_path}")

# Resize the colorized image back to original dimensions
colorized_resized = cv2.resize(colorized, (image.shape[1], image.shape[0]))

# Downscale the colorized image for display
downscale_factor = 0.5  # Adjust this factor as needed
colorized_downscaled = cv2.resize(colorized_resized, None, fx=downscale_factor, fy=downscale_factor)

# Downscale the input image for display
input_downscaled = cv2.resize(image, None, fx=downscale_factor, fy=downscale_factor)

## Displaying both original and colorized images
cv2.imshow("Original", input_downscaled)
cv2.imshow("Colorized", colorized_downscaled)
cv2.waitKey(0)
