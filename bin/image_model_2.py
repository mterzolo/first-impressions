
import pickle
from keras.applications import vgg16
from keras.models import Model
from keras.layers import Dense, Input
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Nadam

import lib

X_train, y_train = lib.img2array3D(data_split='training', num_samples=500, num_frames=100)
X_test, y_test = lib.img2array3D(data_split='test', num_samples=100, num_frames=100)

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

# Save model
pkl_filename = '../output/image_model_2.pkl'
with open(pkl_filename, 'wb') as file:
    pickle.dump(model, file)
