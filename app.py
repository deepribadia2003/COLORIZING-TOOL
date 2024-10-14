from flask import Flask, render_template, request, send_file
import cv2
import numpy as np
import os
import io
from datetime import datetime

app = Flask(__name__)

# Model paths
DIR = r"C:\Users\deepr\MiniProject2BTE"
PROTOTXT = os.path.join(DIR, r"model/colorization_deploy_v2.prototxt")
POINTS = os.path.join(DIR, r"model/pts_in_hull.npy")
MODEL = os.path.join(DIR, r"model/colorization_release_v2.caffemodel")

# Colorize image function
def colorize_image(image):
    # Colorization logic
    # Load colorization model
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
    pts = np.load(POINTS)

    # Load the centers for ab channel
    class8 = net.getLayerId("class8_ab") 
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    # Convert the image to LAB color space
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    # Resize image to model input size
    resized = cv2.resize(lab, (224, 224))                                                 
    L = cv2.split(resized)[0]
    L -= 50

    # Colorize the image
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))
    L = cv2.split(lab)[0]
    colorized_lab = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

    # Convert colorized LAB image back to BGR
    colorized_bgr = cv2.cvtColor(colorized_lab, cv2.COLOR_LAB2BGR)
    colorized_bgr = np.clip(colorized_bgr, 0, 1) * 255
    colorized_bgr = colorized_bgr.astype(np.uint8)    
    return colorized_bgr

def save_image(image):
    # Generate a unique filename using current timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"colorized_image_{timestamp}.jpg"
    save_path = os.path.join("colorized_images", filename)
    
    # Ensure the directory exists, create it if not
    os.makedirs("colorized_images", exist_ok=True)
    
    # Save the image
    cv2.imwrite(save_path, image)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/colorize', methods=['POST'])
def colorize():
    file = request.files['file']
    image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Colorize the image
    colorized_image = colorize_image(image)

    # Save the colorized image
    save_image(colorized_image)

    # Convert image to bytes for serving
    img_bytes = cv2.imencode('.jpg', colorized_image)[1].tobytes()
    return send_file(io.BytesIO(img_bytes), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
