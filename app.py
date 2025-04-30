from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import threading

app = Flask(__name__)

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classifier = load_model('model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

camera = None
output_frame = None
lock = threading.Lock()
running = False


def detect_emotion():
    global camera, output_frame, running
    while running:
        success, frame = camera.read()
        if not success:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray)
        emotion_text = ""

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                label = emotion_labels[prediction.argmax()]
                emotion_text = label
                cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        frame = cv2.resize(frame, (480, 360))
        with lock:
            output_frame = frame.copy()
            if emotion_text:
                app.emotion_label = emotion_text


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    def generate():
        global output_frame
        while True:
            with lock:
                if output_frame is None:
                    continue
                ret, buffer = cv2.imencode('.jpg', output_frame)
                frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start')
def start():
    global camera, running
    if not running:
        camera = cv2.VideoCapture(0)
        running = True
        threading.Thread(target=detect_emotion).start()
    return jsonify({'status': 'started'})


@app.route('/stop')
def stop():
    global camera, running
    running = False
    if camera is not None:
        camera.release()
        camera = None
    return jsonify({'status': 'stopped'})


@app.route('/get_emotion')
def get_emotion():
    label = getattr(app, 'emotion_label', '')
    return jsonify({'emotion': label})


if __name__ == '__main__':
    app.run(debug=True)