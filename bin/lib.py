import os
import pickle
import numpy as np
from keras.applications import vgg16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


def img2array(data_split, num_samples, frame_num):

    # Open answers file
    with open('../data/meta_data/annotation_{}.pkl'.format(data_split), 'rb') as f:
        label_file = pickle.load(f, encoding='latin1')

    # Get all IDs for videos for the training set
    vid_ids = os.listdir('../data/image_data/{}_data'.format(data_split))[0:num_samples]
    y = [label_file['interview'][i + '.mp4'] for i in vid_ids]

    # Create empty array to store image data
    X = np.empty(shape=(num_samples, 224, 224, 3))
    counter = 0

    for video in vid_ids:

        # Load the image
        filename = '../data/image_data/{}_data/{}/frame{}.jpg'.format(data_split, video, frame_num)
        original = load_img(filename, target_size=(224, 224))

        # Convert to numpy array
        numpy_image = img_to_array(original)

        # Resize and store in one big array
        image_temp = np.expand_dims(numpy_image, axis=0)
        image_temp = vgg16.preprocess_input(image_temp)
        X[counter] = image_temp

        counter += 1

    return X, y
