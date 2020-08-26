# python3 videotest.py --encodings encodings.pickle --face face_detector

# import the necessary packages
from imutils.video import VideoStream
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2
import os
from keras.models import model_from_json
from keras.preprocessing import image
import numpy as np

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
    help="path to serialized db of facial encodings")
ap.add_argument("-o", "--output", type=str,
    help="path to output video")
ap.add_argument("-y", "--display", type=int, default=1,
    help="whether or not to display output frame to screen")
ap.add_argument("-d", "--detection-method", type=str, default="hog",
    help="face detection model to use: either `hog` or `cnn`")
ap.add_argument("-f", "--face", required=True,
	help="path to face detector model directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

data = pickle.loads(open(args["encodings"], "rb").read())
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
model_emotion = model_from_json(open("emotion.json", "r").read())
model_emotion.load_weights('emotion.h5')
model= model_from_json(open("API.json", "r").read())
model.load_weights('API.h5')

vs = VideoStream(src=0).start()
index = 0
def saveimg(index, frame):
    filename = 'h1/{:03d}.jpg'.format(index)
    cv2.imwrite(filename, frame)
    print(filename)
writer = None
time.sleep(1.0)

while True:
        frame = vs.read()
        frame = cv2.resize(frame, (1000, 700))
        r = frame.shape[1] / float(frame.shape[1])
        boxes = face_recognition.face_locations(frame,
        model=args["detection_method"])
        encodings = face_recognition.face_encodings(frame, boxes)
        names = []
    # loop over the facial embeddings
        for encoding in encodings:
            matches = face_recognition.compare_faces(data["encodings"],
               encoding)
            global name 
            name = "Unknown"
            if True in matches:
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1 
                name= max(counts, key=counts.get)
        # update the list of names
            names.append(name)     
        for ((top, right, bottom, left), name) in zip(boxes, names):
            top = int(top * r)
            right = int(right * r)
            bottom = int(bottom * r)
            left = int(left * r)
            cv2.rectangle(frame, (left, top), (right, bottom),
                (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, f"Person:{name}", (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 255, 0), 2)
    # if the video writer is None *AND* we are supposed to write
    # the output video to disk initialize the writer
        if writer is None and args["output"] is not None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args["output"], fourcc, 20,
                (frame.shape[1], frame.shape[0]), True)
    # if the writer is not None, write the frame with recognized
    # faces t odisk
        if writer is not None:
            writer.write(frame)
        if args["display"] > 0:
            cv2.imshow("getting ID", frame)
            key = cv2.waitKey(1) & 0xFF     
        if key & 0xFF == ord('p'):
            saveimg(index, frame)
            index +=1    
            continue 
        if key == ord("q"):
                break
cv2.destroyWindow("getting ID")

while True:
    frame2=vs.read()
    #rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    RGB_img= cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    faces_detected = face_haar_cascade.detectMultiScale(RGB_img, 1.08, 6)

    for (x,y,w,h) in faces_detected:
        cv2.rectangle(frame2,(x,y),(x+w,y+h),(255,0,0),thickness=7)
        roi_gray=RGB_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image
        roi_gray=cv2.resize(roi_gray,(198,198))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)
        
        predicted_age = int(np.sum(predictions[0])*100)
        
        races_index = np.argmax(predictions[1])
        races = ('white', 'black', 'asian', 'idian','others')
        predicted_races = races[races_index] 
                
        predicted_genders=np.argmax(predictions[2])
        if  np.argmax(predictions[2])==0:
            predicted_genders = "Male"
        else:
            predicted_genders = "Female"
        cv2.putText(frame2, f'{predicted_age},{predicted_races},{predicted_genders}', (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    resized_img2 = cv2.resize(frame2, (800, 800))
    cv2.imshow('Facial analysis',resized_img2)
    key = cv2.waitKey(1) & 0xFF 
    if key & 0xFF == ord('p'):
        saveimg(index, resized_img2)
        index +=1    
        continue 
    if key == ord("q"):
            break    
    #if cv2.waitKey(100) == ord('q'):#wait until 'q' key is pressed
    #    break
cv2.destroyWindow("Facial analysis")

while True:
    test_img = vs.read()
    gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.05, 5)
    
    for (x,y,w,h) in faces_detected :
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,255,0),thickness=3)
        roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image
        roi_gray=cv2.resize(roi_gray,(48,48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 255
        predictions = model_emotion.predict(img_pixels)
        #find max indexed array
        max_index = np.argmax(predictions[0])
        emotions = ('angry', 'disgust', 'neutral', 'happy', 'sad', 'surprise', 'fear')
        predicted_emotion = emotions[max_index]
                   
        cv2.putText(test_img, f"{name},{predicted_age},{predicted_genders},{predicted_races} is ({predicted_emotion})", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,0), 2)

    resized_img = cv2.resize(test_img, (700, 700))
    cv2.imshow("emotions!", resized_img)
    key = cv2.waitKey(1) & 0xFF     
    if key & 0xFF == ord('p'):
        saveimg(index, resized_img)
        index +=1    
        continue 
    if key == ord("q"):
            break

cv2.destroyAllWindows()
vs.stop()

if writer is not None:
   writer.release()
