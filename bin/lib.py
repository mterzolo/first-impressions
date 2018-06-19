import subprocess
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
import pandas as pd
import yaml
from keras.preprocessing.sequence import pad_sequences
from gensim.utils import simple_preprocess
from collections import defaultdict

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
            logging.warning('Unable to open user conf file. Attempting to run with default values from confs template')

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


def img2array(partition, frame_num):

    logging.info('Begin image transformations for {} partition'.format(partition))

    # Open answers file
    with open('../data/meta_data/annotation_{}.pkl'.format(partition), 'rb') as f:
        label_file = pickle.load(f, encoding='latin1')

    # Get all IDs for videos for the training set
    vid_ids = os.listdir('../data/image_data/{}_data'.format(partition))
    y = [label_file['interview'][i + '.mp4'] for i in vid_ids]

    # Create empty array to store image data
    X = np.empty(shape=(len(y), 224, 224, 3))
    counter = 0

    for video in vid_ids:

        # Load the image
        filename = '../data/image_data/{}_data/{}/frame{}.jpg'.format(partition, video, frame_num)
        original = load_img(filename, target_size=(224, 224))

        # Convert to numpy array
        numpy_image = img_to_array(original)

        # Resize and store in one big array
        image_temp = np.expand_dims(numpy_image, axis=0)
        image_temp = vgg16.preprocess_input(image_temp)
        X[counter] = image_temp
        counter += 1

    # Save arrays as pickled files
    with open('../data/image_data/pickle_files/X_{}.pkl'.format(partition), 'wb') as output:
        pickle.dump(X, output, protocol=4)
    with open('../data/image_data/pickle_files/y_{}.pkl'.format(partition), 'wb') as output:
        pickle.dump(y, output, protocol=4)

    pass


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
            original = load_img('../data/image_data/{}_data/{}/{}'.format(data_split, video, image),
                                target_size=(224, 224))

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


def audio2melspec(partition):
    """
    Reads an audio file and outputs a Mel-spectrogram.
    """

    logging.info('Begin audio transformations for {} partition'.format(partition))

    # Open answers file
    with open('../data/meta_data/annotation_{}.pkl'.format(partition), 'rb') as f:
        label_file = pickle.load(f, encoding='latin1')

    # Get all IDs for videos for the training set
    audio_files = os.listdir('../data/audio_data/{}_data'.format(partition))
    audio_files = [i.split('.mp3')[0] for i in audio_files]
    y = [label_file['interview'][i + '.mp4'] for i in audio_files]
    y = np.array(y)

    # Create empty array to store image data
    X = np.zeros(shape=(len(y), 96, 704, 1))
    counter = 0

    for audio in audio_files:

        logging.debug('Transforming partition: {} file: {}'.format(partition, audio))

        # Mel-spectrogram parameters
        SR = 12000
        N_FFT = 512
        N_MELS = 96
        HOP_LEN = 256
        DURA = 15

        # Load audio file
        src, sr = librosa.load('../data/audio_data/{}_data/{}.mp3'.format(partition, audio), sr=SR)
        n_sample = src.shape[0]
        n_sample_wanted = int(DURA * SR)

        # Trim the signal at the center
        if n_sample < n_sample_wanted:  # if too short
            src = np.hstack((src, np.zeros((int(DURA * SR) - n_sample,))))
        elif n_sample > n_sample_wanted:  # if too long
            src = src[(n_sample - n_sample_wanted) // 2:
                      (n_sample + n_sample_wanted) // 2]

        # Convert to log scaled mel-spec
        logam = librosa.core.power_to_db
        melgram = librosa.feature.melspectrogram
        logmelspec = logam(melgram(y=src, sr=SR, hop_length=HOP_LEN,
                                   n_fft=N_FFT, n_mels=N_MELS) ** 2, ref=1.0)
        logmelspec = np.expand_dims(logmelspec, axis=3)
        X[counter] = logmelspec

    logging.info('{} audio transformation complete, saving to file'.format(partition))

    # Save arrays as pickled files
    with open('../data/audio_data/pickle_files/X_{}.pkl'.format(partition), 'wb') as output:
        pickle.dump(X, output, protocol=4)
    with open('../data/audio_data/pickle_files/y_{}.pkl'.format(partition), 'wb') as output:
        pickle.dump(y, output, protocol=4)

    pass


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

            while cap.isOpened():
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
    pass


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

            subprocess.call(['ffmpeg',
                             '-y',
                             '-acodec',
                             'libmp3lame',
                             '-i',
                             '../data/video_data/{}/{}.mp4'.format(chunk, file_name),
                             '../data/audio_data/{}_data/{}.mp3'.format(partition, file_name)])
    pass


def extract_text(partition):
    """

    Takes transcripts and saves them as dataframes
    :param partition: Training set, test set, or validation set
    :return:
    """

    logging.info('Begin text extraction for {} partition'.format(partition))

    # Open transcript and annotations
    with open('../data/meta_data/transcription_{}.pkl'.format(partition), 'rb') as f:
        transcript = pickle.load(f, encoding='latin1')
    with open('../data/meta_data/annotation_{}.pkl'.format(partition), 'rb') as f:
        annotation = pickle.load(f, encoding='latin1')

    # Transform into a data frame
    text_df = pd.DataFrame({'file': list(transcript.keys()),
                            'transcript': list(transcript.values())})

    # Map in annotations
    text_df['interview_score'] = text_df['file'].map(annotation['interview'])

    # Create directory if it doesnt exist
    if not os.path.exists('../data/text_data/{}_data/'.format(partition)):
        os.makedirs('../data/text_data/{}_data/'.format(partition))

    # Save into text data directory
    text_df.to_csv('../data/text_data/{}_data/{}_transcripts.csv'.format(partition, partition), index=False)


def transform_text(partition):

    logging.info('Begin text transformation on {}'.format(partition))

    # Load in embedding resources and transcripts
    with open('../resources/embedding_matrix.pkl', 'rb') as f:
        embedding_matrix = pickle.load(f, encoding='latin1')
    with open('../resources/word_to_index.pkl', 'rb') as f:
        word_to_index = pickle.load(f, encoding='latin1')
    observations = pd.read_csv('../data/text_data/{}_data/{}_transcripts.csv'.format(partition, partition))

    # Transform embedding resources
    default_dict_instance = defaultdict(lambda: word_to_index['UNK'])
    default_dict_instance.update(word_to_index)
    word_to_index = default_dict_instance

    # TODO remove words that were cutoff at the end of the video

    # Convert text to normalized tokens. Unknown tokens will map to 'UNK'.
    observations['tokens'] = observations['transcript'].apply(simple_preprocess)

    # Convert tokens to indices
    observations['indices'] = observations['tokens'].apply(lambda token_list: map(lambda token: word_to_index[token],
                                                                                  token_list))
    observations['indices'] = observations['indices'].apply(lambda x: np.array(x))

    # Pad indices list with zeros, so that every article's list of indices is the same length
    observations['padded_indices'] = pad_sequences(observations['indices'], 85)

    # Create data sets for model
    X = observations['padded_indices'].values
    y = observations['interview_score'].values

    # Save as pickled files
    with open('../data/text_data/pickle_files/X_{}.pkl'.format(partition), 'wb') as output:
        pickle.dump(X, output, protocol=4)
    with open('../data/text_data/pickle_files/y_{}.pkl'.format(partition), 'wb') as output:
        pickle.dump(y, output, protocol=4)

    pass
