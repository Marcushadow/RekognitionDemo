import cv2 , boto3, os
import numpy as np
from flask import make_response
from dotenv import load_dotenv
import base64

client = boto3.client(
    'rekognition', 
    region_name="us-east-1", 
    aws_access_key_id= os.getenv("aws_access_key_id") ,
    aws_secret_access_key= os.getenv("aws_secret_access_key"),
    aws_session_token =  os.getenv("aws_session_token")
)

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()        

    def get_frame(self):
        ret, frame = self.video.read()
        # Frame is a 3d array
       
        # For real time frame processing you can do it here i.e detecting objects
        # But this is going to be slow as you need to wait for a response from rekognition 
        # Maybe look at YOLO for possible real time 

        ret, png = cv2.imencode('.png', frame)
        return png.tobytes()

    def face_detections(self):
        imageByteString = self.get_frame()

        # Obtain the shape for bounding boxes
        img_as_np = np.frombuffer(imageByteString, dtype=np.uint8)
        img = cv2.imdecode(img_as_np, flags=-1)
        imgh, imgw, channels = img.shape 
    
        # Preparing the binary encoding needed by rekognition
        base64_img =  base64.b64encode(imageByteString)
        base_64_binary = base64.decodebytes(base64_img)

        try:    
            response = client.detect_faces(Image={'Bytes': base_64_binary})

            for face in response['FaceDetails']:
                if(face['Confidence'] > 75):
                    # Setting a confidence threshold
                    boundingBoxDetails = face['BoundingBox']
                    width = int(boundingBoxDetails['Width'] * imgw)
                    height = int(boundingBoxDetails['Height'] * imgh)

                    topLeft_x = int(boundingBoxDetails['Left'] * imgw)
                    topLeft_y = int(boundingBoxDetails['Top'] * imgh)
                    topLeft_coords = (topLeft_x, topLeft_y)

                    bottomRight_x = int(topLeft_x + width)
                    bottomRight_y = int(topLeft_y + height)
                    bottomRight_coords =(bottomRight_x, bottomRight_y)


                    img = cv2.rectangle(img, topLeft_coords, bottomRight_coords, (255,255,255), 2)


            ret, resImg = cv2.imencode('.png', img)
        
        except:
            print("Call to rekognition failed")

        
        return make_response(resImg.tobytes())


    def label_detections(self):
        imageByteString = self.get_frame()

        # Obtain the shape for bounding boxes
        img_as_np = np.frombuffer(imageByteString, dtype=np.uint8)
        img = cv2.imdecode(img_as_np, flags=-1)
        imgh, imgw, channels = img.shape 
    
        # Preparing the binary encoding needed by rekognition
        base64_img =  base64.b64encode(imageByteString)
        base_64_binary = base64.decodebytes(base64_img)
   
        response = client.detect_labels(Image={'Bytes': base_64_binary})


        for item in response['Labels']:
            if(item['Confidence'] > 75):
                # Creating a confidence threshold
                for instance in item['Instances']:

                    if(instance['Confidence'] > 75):
                        boundingBoxDetails = instance['BoundingBox']
                        width = int(boundingBoxDetails['Width'] * imgw)
                        height = int(boundingBoxDetails['Height'] * imgh)

                        topLeft_x = int(boundingBoxDetails['Left'] * imgw)
                        topLeft_y = int(boundingBoxDetails['Top'] * imgh)
                        topLeft_coords = (topLeft_x, topLeft_y)

                        bottomRight_x = int(topLeft_x + width)
                        bottomRight_y = int(topLeft_y + height)
                        bottomRight_coords =(bottomRight_x, bottomRight_y)

                        img = cv2.rectangle(img, topLeft_coords, bottomRight_coords, (255,255,255), 2)
                        label_coords = (topLeft_x, topLeft_y - 10)
                        img = cv2.putText(img, item['Name'], label_coords, cv2.FONT_HERSHEY_DUPLEX, 0.85, (100,255,0), 2)

        ret, resImg = cv2.imencode('.png', img)
        
        return make_response(resImg.tobytes())

    def text_detections(self):
        imageByteString = self.get_frame()

        # Obtain the shape for bounding boxes
        img_as_np = np.frombuffer(imageByteString, dtype=np.uint8)
        img = cv2.imdecode(img_as_np, flags=-1)
        imgh, imgw, channels = img.shape 
    
        # Preparing the binary encoding needed by rekognition
        base64_img =  base64.b64encode(imageByteString)
        base_64_binary = base64.decodebytes(base64_img)

        response = client.detect_text(Image={'Bytes': base_64_binary})

        for textDetection in response['TextDetections']:
            if(textDetection['Confidence'] > 75):
                # Creating a confidence threshold
                
                boundingBoxDetails = textDetection['Geometry']['BoundingBox']
                width = int(boundingBoxDetails['Width'] * imgw)
                height = int(boundingBoxDetails['Height'] * imgh)

                topLeft_x = int(boundingBoxDetails['Left'] * imgw)
                topLeft_y = int(boundingBoxDetails['Top'] * imgh)
                topLeft_coords = (topLeft_x, topLeft_y)

                bottomRight_x = int(topLeft_x + width)
                bottomRight_y = int(topLeft_y + height)
                bottomRight_coords =(bottomRight_x, bottomRight_y)

                img = cv2.rectangle(img, topLeft_coords, bottomRight_coords, (255,255,255), 2)
                label_coords = (topLeft_x, topLeft_y - 10)
                img = cv2.putText(img, textDetection['DetectedText'], label_coords, cv2.FONT_HERSHEY_DUPLEX, 0.85, (100,255,0), 2)

        ret, resImg = cv2.imencode('.png', img)
        


        
        return make_response(resImg.tobytes())




