from keras.applications import vgg16
from keras.models import Model
from keras.layers import Dense, Flatten
import lib

# Transform images into numpy arrays
X_train, y_train = lib.img2array(data_split='training', num_samples=500, frame_num=50)
X_test, y_test = lib.img2array(data_split='test', num_samples=100, frame_num=50)


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

