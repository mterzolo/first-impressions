from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential

import lib

X_train, y_train = lib.audio2melspec(data_split='training', num_samples=5)
X_test, y_test = lib.audio2melspec(data_split='test', num_samples=5)

X_train = X_train.reshape(X_train.shape[0], 128, 662, 1)
X_test = X_test.reshape(X_test.shape[0], 128, 662, 1)

input_shape = (128, 662, 1)

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='Adam', loss='mean_squared_error')
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32)

# Save model
pkl_filename = '../output/audio_model.pkl'
with open(pkl_filename, 'wb') as file:
    pickle.dump(model, file)