import socketio
import eventlet
import numpy as np
from flask import Flask
from keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2


sio = socketio.Server()
app = Flask(__name__)

speed_limit = 20

@sio.on('telemetry')
def telemetry(sid, data):
    print('merge2')
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = img_preprocess(image)
    image = np.array([image])
    steering_angle = float(model.predict(image))
    throttle = 1.0 - speed/speed_limit
    print('{}{}{}'.format(steering_angle, throttle, speed))
    send_control(steering_angle, throttle)

def img_preprocess(img):
    img = img[60:135,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img

@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    send_control(0, 0)
    
def send_control(steering_angle, throttle):
    print('merge3')
    sio.emit(
             'steer', 
             data = {
                     'steering_angle': steering_angle.__str__(),
                      'throttle': throttle.__str__()
                     },
             skip_sid=True
             )
    

    
if __name__ == '__main__':
    model = load_model('model4.h5')
    print('merge1')
    app = socketio.Middleware(sio, app)
    print('merge4')
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
    print('merge5')