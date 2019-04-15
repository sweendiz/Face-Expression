#Imports
from __future__ import print_function
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import math
import argparse
import imutils
import cv2
import label_image
import os
#disable error. Maybe visit later
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#value for resizing image: Adjust and test stuff
size = 2
# testing speed stuff
cv2.setUseOptimized(False)
cv2.useOptimized()

# We load the xml file
classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
vs = WebcamVideoStream(src=0).start() #Using default WebCam connected to the PC.
#if vs = null then break gently...
while True:
    im = vs.read()
    # Flip camera
    im=cv2.flip(im,1,0)
    # Resize the image to speed up detection
    mini = cv2.resize(im, (int(im.shape[1]/size), int(im.shape[0]/size)))
    # detect MultiScale / faces 
    faces = classifier.detectMultiScale(mini)

    # Draw rectangles around each face
    for f in faces:
        (x, y, w, h) = [v * size for v in f] #Scale the shapesize backup
        cv2.rectangle(im, (x,y), (x+w,y+h), (255,255,255), 3)
        
        #Save just the rectangle faces in SubRecFaces
        sub_face = im[y:y+h, x:x+w]

        FaceFileName = "test.jpg" #Saving the current image to overlay images
        cv2.imwrite(FaceFileName, sub_face)
        
        text = label_image.main(FaceFileName)# Getting the Result from the label_image file, i.e., Classification Result.
        text = text.title()
        font = cv2.FONT_HERSHEY_SIMPLEX
        #check value and make s_img = path of image file. :) IE(s_img = cv2.imread("smaller_image.png"))
        #x_offset=y_offset=50
        #l_img[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1]] = s_img
        xw = math.ceil((x+ w)*.75)
        yh = (y -20)
        #place text over this face
        cv2.putText(im, text,(xw,yh), font, 1, (255,192,203), 2)

    # Show the image
    cv2.imshow('Capture',   im)
    key = cv2.waitKey(1)
    # if Esc key is press then break out of the loop 
    if key == 27: #The Esc key
        break

