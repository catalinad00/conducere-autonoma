import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense, Conv2D
from keras.callbacks import ModelCheckpoint
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa
import cv2
import pandas as pd
import ntpath
import random


datadir = 'track'
columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
data = pd.read_csv(os.path.join(datadir, 'data', 'driving_log.csv'), names = columns)
#pd.set_option('display.max_colwidth', -1)
data.head()

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail
data['center'] = data['center'].apply(path_leaf)
data['left'] = data['left'].apply(path_leaf)
data['right'] = data['right'].apply(path_leaf)
data.head()

num_bins = 25
samples_per_bin = 400
hist, bins = np.histogram(data['steering'], num_bins)
center = (bins[:-1]+ bins[1:]) * 0.5
#plt.bar(center, hist, width=0.05)
#plt.plot((np.min(data['steering']), np.max(data['steering'])), (samples_per_bin, samples_per_bin))
#plt.show()

#print('total data:', len(data))
#4303
#print(data.shape)
#(4303, 7)

remove_list = []
for j in range(num_bins):
    list_ = []
    for i in range(len(data['steering'])):
        if data['steering'][i] >= bins[j] and data['steering'][i] <= bins[j+1]:
            list_.append(i)
    list_ = shuffle(list_)
    list_ = list_[samples_per_bin:]
    remove_list.extend(list_)
    
#print('removed:', len(remove_list))
#2733
data.drop(data.index[remove_list], inplace = True)
#print('remaining: ', len(data))
#1570

hist, _ = np.histogram(data['steering'], (num_bins))
#plt.bar(center, hist, width=0.05)
#plt.plot((np.min(data['steering']), np.max(data['steering'])),(samples_per_bin, samples_per_bin))
#plt.show()

def load_img_steering(datadir, df):
    image_path = []
    steering = []
    for i in range(len(data)):
        indexed_data = data.iloc[i]
        center, left, right = indexed_data[0], indexed_data[1], indexed_data[2]
        image_path.append(os.path.join(datadir, center.strip()))
        steering.append(float(indexed_data[3]))
        #left image append
        image_path.append(os.path.join(datadir, left.strip()))
        steering.append(float(indexed_data[3])+0.15)
        #right image append
        image_path.append(os.path.join(datadir, right.strip()))
        steering.append(float(indexed_data[3])-0.15)
    image_paths = np.asarray(image_path)
    steerings = np.asarray(steering)
    return image_paths, steerings

image_paths, steerings = load_img_steering('track/data/IMG', data)
#print('image_paths: ', len(image_paths))
#print('steerings: ', len(steerings))

X_train, X_valid, y_train, y_valid = train_test_split(image_paths, steerings, test_size=0.2, random_state=6)
#print('Training Samples: {}\nValid Samples {}' .format(len(X_train), len(X_valid)))
#3768, 942

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(y_train, bins=num_bins, width=0.05, color='blue')
axes[0].set_title('Training set')
axes[1].hist(y_valid, bins=num_bins, width=0.05, color='red')
axes[1].set_title('Validation set')


def zoom(image):
    zoom = iaa.Affine(scale=(1, 1.3))
    image = zoom.augment_image(image)
    return image

#image = image_paths[random.randint(0,1000)]
#original_image = mpimg.imread(image)
#zoomed_image = zoom(original_image)

#fig, axis = plt.subplots(1, 2, figsize=(15, 10))
#fig.tight_layout()

#axis[0].imshow(original_image)
#axis[0].set_title('Original Image')

#axis[1].imshow(zoomed_image)
#axis[1].set_title('Zoomed Image')
#plt.show()

def pan(image):
    pan = iaa.Affine(translate_percent= {"x" : (-0.1, 0.1), "y" : (-0.1, 0.1)})
    image = pan.augment_image(image)
    return image

#image = image_paths[random.randint(0,1000)]
#original_image = mpimg.imread(image)
#panned_image = pan(original_image)

#axis[0].imshow(original_image)
#axis[0].set_title("Original Image")

#axis[1].imshow(panned_image)
#axis[1].set_title("Panned Image")
#plt.show()

def img_random_brightness(image):
    brightness = iaa.Multiply(0.7, 1)
    image = brightness.augment_image(image)
    return image

#image = image_paths[random.randint(0,1000)]
#original_image = mpimg.imread(image)
#brightness_altered_image = img_random_brightness(original_image)

#axis[0].imshow(original_image)
#axis[0].set_title('Original Image')

#axis[1].imshow(brightness_altered_image)
#axis[1].set_title('Brightness altere image')
#plt.show()

def img_random_flip(image, steering_angle):
    image = cv2.flip(image, 1)
    steering_angle = -steering_angle
    return image, steering_angle

#random_index = random.randint(0,1000)
#image = image_paths[random_index]
#steering_angle = steerings[random_index]

#original_image = mpimg.imread(image)
#flipped_image, flipped_steering_angle = img_random_flip(original_image, steering_angle)


#axis[0].imshow(original_image)
#axis[0].set_title('Original Image - ' + 'Steering Angle:' + str(steering_angle))

#axis[1].imshow(flipped_image)
#axis[1].set_title('Flipped Image - ' + 'Steering Angle: ' + str(flipped_steering_angle))
#plt.show()

def random_augment(image, steering_angle):
    image = mpimg.imread(image)
    if np.random.rand() < 0.5:
        image = pan(image)
    if np.random.rand() > 0.5:
        image = zoom(image)
    if np.random.rand() < 0.5:
        image = img_random_brightness(image)
    if np.random.rand() == 0.5:
        image, steering_angle = img_random_flip(image, steering_angle)
    
    return image, steering_angle

def img_preproceess(img):
    img = img[60:135,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img

image = image_paths[100]
original_image = mpimg.imread(image)
preprocessed_image = img_preproceess(original_image)

fig, ax = plt.subplots(1, 2, figsize=(15,10))
fig.tight_layout()
ax[0].imshow(original_image)
ax[0].set_title('Original Image')
ax[1].imshow(preprocessed_image)
ax[1].set_title('Preprocessed Image')
#plt.show()

def batch_generator(image_paths, steering_ang, batch_size, istraining):
    while True:
        batch_img = []
        batch_steering = []
        
        for i in range(batch_size):
            random_index = random.randint(0, len(image_paths) - 1)
            
            if istraining:
                im, steering = random_augment(image_paths[random_index], steering_ang[random_index])
            
            else:
                im= mpimg.imread(image_paths[random_index])
                steering = steering_ang[random_index]
            
            im = img_preproceess(im)
            batch_img.append(im)
            batch_steering.append(steering)
        yield(np.asarray(batch_img), np.asarray(batch_steering))
        
x_train_gen, y_train_gen = next(batch_generator(X_train, y_train, 1, 1))
x_valid_gen, y_valid_gen = next(batch_generator(X_valid, y_valid, 1, 0))

ax[0].imshow(x_train_gen[0])
ax[0].set_title('Training Image')

ax[1].imshow(x_valid_gen[0])
ax[1].set_title('Validation Image')
plt.show()

def nvidia_model():
    model = Sequential()
    model.add(Conv2D(24,(5,5), strides=(2,2), input_shape=(66,200,3), activation='elu'))
    model.add(Conv2D(36,(5,5), strides=(2,2), activation='elu'))
    model.add(Conv2D(48,(5,5), strides=(2,2), activation='elu'))
    model.add(Conv2D(64,(3,3), activation='elu'))
    model.add(Conv2D(64,(3,3), activation='elu'))
    
    model.add(Flatten())
    
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(loss='mse', optimizer=optimizer)
    return model

model = nvidia_model()
print(model.summary())



checkpoint = ModelCheckpoint('training/model-{epoch}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=False,
                                 save_freq=50)

history = model.fit_generator(batch_generator(X_train, y_train, 68, 1),
                              steps_per_epoch=300,
                              epochs=25,
                              validation_data=batch_generator(X_valid, y_valid, 68, 0),
                              validation_steps=300,
                              verbose=1,
                              shuffle=1)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()

model.save('model4.h5')
