import os
import keras
import pickle
import numpy as np
from keras.applications import vgg16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
from keras.layers import Dense, Flatten
from keras.applications.vgg16 import preprocess_input
from sklearn.metrics import mean_squared_error

# Open answers file
with open('../data/meta_data/annotation_training.pkl', 'rb') as f:
    annotation_training = pickle.load(f, encoding='latin1')

# Get all IDs for videos for the training set
vid_ids = os.listdir('../data/image_data/training_data')[0:500]
y_train = [annotation_training['interview'][i + '.mp4'] for i in vid_ids]

# Create empty array to store image data
X_train = np.empty(shape=(0, 224, 224, 3))

for video in vid_ids:
    # Load the image
    filename = '../data/image_data/training_data/{}/frame50.jpg'.format(video)
    original = load_img(filename, target_size=(224, 224))

    # Convert to numpy array
    numpy_image = img_to_array(original)

    # Resize and store in one big array
    image_temp = np.expand_dims(numpy_image, axis=0)
    image_temp = vgg16.preprocess_input(image_temp)
    X_train = np.concatenate((X_train, image_temp), axis=0)

# Open answers file
with open('../data/meta_data/annotation_test.pkl', 'rb') as f:
    annotation_test = pickle.load(f, encoding='latin1')

# Get all IDs for videos for the test set
vid_ids = os.listdir('../data/image_data/test_data')[0:200]
y_test = [annotation_test['interview'][i + '.mp4'] for i in vid_ids]

# Create empty array to store image data
X_test = np.empty(shape=(0, 224, 224, 3))

for video in vid_ids:
    # Load the image
    filename = '../data/image_data/test_data/{}/frame50.jpg'.format(video)
    original = load_img(filename, target_size=(224, 224))

    # Convert to numpy array
    numpy_image = img_to_array(original)

    # Resize and store in one big array
    image_temp = np.expand_dims(numpy_image, axis=0)
    image_temp = vgg16.preprocess_input(image_temp)
    X_test = np.concatenate((X_test, image_temp), axis=0)

# Load the VGG16 model
base_model = vgg16.VGG16(weights="imagenet", include_top=False, input_shape =(224,224,3))
x = base_model.output

# Create the last two additional layers
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
preds = Dense(1, activation='linear')(x)

# Freeze all original layers
for layer in base_model.layers:
    layer.trainable = False

# Create the model
model = Model(base_model.input, preds)
model.compile(optimizer='Adam', loss='mean_squared_error')
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

