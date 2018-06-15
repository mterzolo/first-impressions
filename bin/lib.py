import os
import pickle
import librosa
import cv2
import numpy as np
from keras.applications import vgg16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import datetime
import logging
import os
import moviepy.editor as mp
import pandas as pd
import yaml
from keras.preprocessing.sequence import pad_sequences

# Global variables
CONFS = None
BATCH_NAME = None


def load_confs(confs_path='../conf/confs.yaml'):
    """
    Load configurations from file.
     - If configuration file is available, load it
     - If configuraiton file is not available attempt to load configuration template
    Configurations are never explicitly validated.
    :param confs_path: Path to a configuration file, appropriately formatted for this application
    :type confs_path: str
    :return: Python native object, containing configuration names and values
    :rtype: dict
    """
    global CONFS

    if CONFS is None:

        try:
            logging.info('Attempting to load confs from path: {}'.format(confs_path))

            # Attempt to load confs from confPath
            CONFS = yaml.load(open(confs_path))

        except IOError:
            logging.warn('Unable to open user conf file. Attempting to run with default values from confs template')

            # Attempt to load confs from template path
            template_path = confs_path + '.template'
            CONFS = yaml.load(open(template_path))

    return CONFS


def get_conf(conf_name):
    """
    Get a configuration parameter by its name
    :param conf_name: Name of a configuration parameter
    :type conf_name: str
    :return: Value for that conf (no specific type information available)
    """
    return load_confs()[conf_name]

def get_batch_name():
    """
    Get the name of the current run. This is a unique identifier for each run of this application
    :return: The name of the current run. This is a unique identifier for each run of this application
    :rtype: str
    """
    global BATCH_NAME

    if BATCH_NAME is None:
        logging.info('Batch name not yet set. Setting batch name.')
        BATCH_NAME = str(datetime.datetime.utcnow()).replace(' ', '_').replace('/', '_').replace(':', '_')
        logging.info('Batch name: {}'.format(BATCH_NAME))
    return BATCH_NAME


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


def img2array3D(data_split, num_samples, num_frames):

    # Open answers file
    with open('../data/meta_data/annotation_{}.pkl'.format(data_split), 'rb') as f:
        label_file = pickle.load(f, encoding='latin1')

    # Get all IDs for videos for the training set
    vid_ids = os.listdir('../data/image_data/{}_data'.format(data_split))[0:num_samples]
    y = [label_file['interview'][i + '.mp4'] for i in vid_ids]

    # Create empty array to store image data
    X = np.empty(shape=(num_samples, num_frames, 224, 224, 3))
    out_counter = 0

    for video in vid_ids:

        images = os.listdir('../data/image_data/{}_data/{}'.format(data_split, video))
        X_temp = np.zeros(shape=(num_frames, 224, 224, 3))
        in_counter = 0

        for image in images:

            # Load the image
            original = load_img('../data/image_data/{}_data/{}/{}'.format(data_split, video, image), target_size=(224, 224))

            # Convert to numpy array
            numpy_image = img_to_array(original)

            # Resize and store in one big array
            image_temp = np.expand_dims(numpy_image, axis=0)
            image_temp = vgg16.preprocess_input(image_temp)
            X_temp[in_counter] = image_temp

            # Increment counter for number of images in observation
            in_counter += 1

        X_temp = np.expand_dims(X_temp, axis=0)
        X[out_counter] = X_temp

        # Increment counter for observations in dataset
        out_counter += 1

    return X, y


def audio2melspec(data_split, num_samples):

    # Open answers file
    with open('../data/meta_data/annotation_{}.pkl'.format(data_split), 'rb') as f:
        label_file = pickle.load(f, encoding='latin1')

    # Get all IDs for videos for the training set
    audio_files = os.listdir('../data/audio_data/{}_data'.format(data_split))[0:num_samples]
    audio_files = [i.split('.mp3')[0] for i in audio_files]
    y = [label_file['interview'][i + '.mp4'] for i in audio_files]

    # Create empty array to store image data
    X = np.zeros(shape=(num_samples, 128, 662))
    counter = 0

    for audio in audio_files:

        aud, sr = librosa.load('../data/audio_data/{}_data/{}.mp3'.format(data_split, audio))
        mel_spec = librosa.feature.melspectrogram(y=aud)
        mel_spec = cv2.resize(mel_spec, dsize=(662, 128), interpolation=cv2.INTER_CUBIC)
        mel_spec = mel_spec / np.max(mel_spec)
        X[counter] = mel_spec

    return X, y


def extract_images(partition, num_frames):

    logging.info('Begin image extraction on {} partition'.format(partition))

    file_chunks = os.listdir('../data/video_data')
    file_chunks = [i for i in file_chunks if partition in i]

    for chunk in file_chunks:

        files = os.listdir('../data/video_data/{}'.format(chunk))

        for file_name in files:

            # Create video object
            cap = cv2.VideoCapture('../data/video_data/{}/{}'.format(chunk, file_name))

            # Get file name
            file_name = (file_name.split('.mp4'))[0]

            # Create new folder for images
            try:
                if not os.path.exists('../data/image_data/{}_data/{}'.format(partition, file_name)):
                    os.makedirs('../data/image_data/{}_data/{}'.format(partition, file_name))

            except OSError:
                logging.warning('Error: Creating directory of data')

            # Set number of frames to grab
            cap.set(cv2.CAP_PROP_FRAME_COUNT, num_frames + 1)
            length = num_frames + 1
            count = 0

            while (cap.isOpened()):
                count += 1

                # Exit if at the end
                if length == count:
                    break

                # Create the image
                ret, frame = cap.read()

                # Skip if there is no frame
                if frame is None:
                    continue

                # Resize image
                frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_CUBIC)

                # Save image to jpg
                name = '../data/image_data/{}_data/{}/frame{}.jpg'.format(partition, file_name, count)
                cv2.imwrite(name, frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                logging.info('{}: {}: frame{}'.format(chunk, file_name, count))


def extract_audio(partition):

    logging.info('Begin Audio Extraction on {} partition'.format(partition))

    file_chunks = os.listdir('../data/video_data')
    file_chunks = [i for i in file_chunks if partition in i]

    # Create new folder for images
    if not os.path.exists('../data/audio_data/{}_data/'.format(partition)):
        os.makedirs('../data/audio_data/{}_data/'.format(partition))

    for chunk in file_chunks:

        files = os.listdir('../data/video_data/{}'.format(chunk))

        for file_name in files:
            file_name = file_name.split('.mp4')[0]

            # Create video object
            clip = mp.VideoFileClip('../data/video_data/{}/{}.mp4'.format(chunk, file_name))
            clip.audio.write_audiofile("../data/audio_data/{}_data/{}.mp3".format(partition, file_name))

def extract_text(partition):
    """

    Takes transcripts and saves them as dataframes
    :param partition: Training set, test set, or validation set
    :return:
    """

    logging.info('Begin Text Extraction')

    # Open transcript
    with open('../data/meta_data/transcription_{}.pkl'.format(partition), 'rb') as f:
        transcript = pickle.load(f, encoding='latin1')

    # Transform into a dataframe
    pd.DataFrame({'file': list(transcript.keys()),
                  'transcript': list(transcript.values())})

    # Save into text data directory
    pd.to_csv('../data/text_data/{}_data.csv'.format(partition), index=False)
