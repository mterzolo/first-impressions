import os
import keras
import pickle
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications import vgg16
from keras.models import Model
from keras.layers import Dense, Input
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Nadam

# Open answers file
with open('data/annotation_training.pkl', 'rb') as f:
    annotation_training = pickle.load(f, encoding='latin1')

# Get all IDs for videos for the training set
vid_ids = os.listdir('imageData/trainingData')[0:10]
y_train = [annotation_training['interview'][i + '.mp4'] for i in vid_ids]

# Create empty array to store image data
X_train = np.empty(shape=(0, 100, 224, 224, 3))

for video in vid_ids:

    images = os.listdir('ImageData/trainingData/{}'.format(video))
    X_temp = np.empty(shape=(0, 224, 224, 3))

    for image in images:
        # Load the image
        filename = 'ImageData/trainingData/{}/frame50.jpg'.format(video)
        original = load_img(filename, target_size=(224, 224))

        # Convert to numpy array
        numpy_image = img_to_array(original)

        # Resize and store in one big array
        image_temp = np.expand_dims(numpy_image, axis=0)
        image_temp = vgg16.preprocess_input(image_temp)
        X_temp = np.concatenate((X_temp, image_temp), axis=0)

    X_temp = np.expand_dims(X_temp, axis=0)
    X_train = np.concatenate((X_train, X_temp), axis=0)

video = Input(shape=(100, 224, 224, 3))

cnn_base = vgg16.VGG16(input_shape=(224, 224, 3), weights="imagenet", include_top=False)

cnn_out = GlobalAveragePooling2D()(cnn_base.output)
cnn = Model(input=cnn_base.input, output=cnn_out)
cnn.trainable = False

encoded_frames = TimeDistributed(cnn)(video)
encoded_sequence = LSTM(256)(encoded_frames)

hidden_layer = Dense(units=1024, activation="relu")(encoded_sequence)
outputs = Dense(units=1, activation="linear")(hidden_layer)
model = Model([video], outputs)

optimizer = Nadam(lr=0.002,
                  beta_1=0.9,
                  beta_2=0.999,
                  epsilon=1e-08,
                  schedule_decay=0.004)

model.compile(loss="mean_squared_error",
              optimizer=optimizer,
              metrics=["mse"])

# Create the model
model.compile(optimizer='Adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=10, batch_size=32)

