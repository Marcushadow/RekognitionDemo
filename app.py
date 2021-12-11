from flask import Flask, render_template, Response, redirect
import cv2, boto3
from VideoCamera import VideoCamera
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
videoStream = VideoCamera()


curr_detection_mode = 0
# Defaults to faces


@app.route('/')

def reroute():
    return redirect("/main")


@app.route("/main")

def mainPage():
    return render_template("Frontend/home.html")

@app.route("/camera")

def cameraPage():
    return render_template("Frontend/camera.html")

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/png\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
# Sent directly to video camera / webcam element
def video_feed():
    return Response(gen(videoStream), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect_faces')

def detect_faces():
    global curr_detection_mode

    curr_detection_mode = 0

    return render_template("Frontend/output.html")

@app.route('/detect_labels')

def detect_labels():
    global curr_detection_mode

    curr_detection_mode = 1

    return render_template("Frontend/output.html")

@app.route('/detect_text')

def detect_text():
    global curr_detection_mode

    curr_detection_mode = 2

    return render_template("Frontend/output.html")

@app.route('/source_image')
# Send directly to the image element
def source_image():
    global curr_detection_mode
    if(curr_detection_mode == 0):
        picture = videoStream.face_detections()

    elif(curr_detection_mode == 1):
        picture = videoStream.label_detections()

    else:
        picture = videoStream.text_detections()

    return picture