import pickle
import keras
from keras.applications import vgg16
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout
from keras.callbacks import ModelCheckpoint

import lib

# Transform images into numpy arrays
X_train, y_train = lib.img2array(data_split='training', num_samples=6000, frame_num=10)
X_test, y_test = lib.img2array(data_split='test', num_samples=2000, frame_num=10)

# Load the VGG16 model
base_model = vgg16.VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
x = base_model.output

# Create additional layers
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(.4)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(.4)(x)
preds = Dense(1, activation='linear')(x)

# Freeze all original layers
for layer in base_model.layers:
    layer.trainable = False

# Optimizer
optimizer = keras.optimizers.Adam(lr=0.001, decay=0.0006)

# fit model
filename = '../output/image_model.h5'
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# Create the model
model = Model(base_model.input, preds)
model.compile(optimizer=optimizer, loss='mean_squared_error')
model.fit(X_train, y_train,
          validation_data=(X_test, y_test),
          epochs=150,
          batch_size=32,
          callbacks=[checkpoint])
