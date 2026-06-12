import cv2 
import os

# Load the file for face detection
haar_file = "haarcascade_frontalface_default.xml"

datasets = "datasets"
subdata = "Gabriel"

# Create the directory for storing images if they don't exist.
path = os.path.join(datasets,subdata)
if not os.path.isdir(path):
    os.makedirs(path)
width,height = 130,100
face_cascade = cv2.CascadeClassifier(haar_file)
webcam = cv2.VideoCapture(0)


count = 1 
while count <= 5:
    res,frames = webcam.read()
    if not res:
        print("Failed To Capture Video / Image")
        break
    gray = cv2.cvtColor(frames,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,4) # 1.3 is scale factor, 4 is the min number of neighbours
    