import tensorflow
from tensorflow import keras
from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import os
import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier(r'E:\DL_Project\DL Project\haarcascade_frontalface_default.xml')
classifier =load_model(r'E:\DL_Project\DL Project\model.h5')

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

cap = cv2.VideoCapture(0)



while True:
    _, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)



        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            prediction = classifier.predict(roi)[0]
            label=emotion_labels[prediction.argmax()]
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        else:
            cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.imshow('Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



####################################################################################
####################################################################################



# import cv2
# import numpy as np
# from keras.models import load_model
# from keras.preprocessing.image import img_to_array
# import os


# face_classifier = cv2.CascadeClassifier(r'E:\DL_Project\DL Project\haarcascade_frontalface_default.xml')
# classifier = load_model(r'E:\DL_Project\DL Project\model.h5')

# emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

# image_path = r'E:\DL_Project\DL Project\images\test2.jpg'

# if not os.path.exists(image_path):
#     print(f"Image not found at path: {image_path}")
#     exit()

# image = cv2.imread(image_path)
# if image is None:
#     print("Image is invalid or corrupted.")
#     exit()

# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

# for (x, y, w, h) in faces:
#     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)
#     roi_gray = gray[y:y + h, x:x + w]
#     roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

#     if np.sum([roi_gray]) != 0:
#         roi = roi_gray.astype("float") / 255.0
#         roi = img_to_array(roi)
#         roi = np.expand_dims(roi, axis=0)

#         # Predict emotion
#         prediction = classifier.predict(roi)[0]
#         label = emotion_labels[prediction.argmax()]
#         label_position = (x, y - 10)
#         cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# cv2.imshow("Emotion Detector - Image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

