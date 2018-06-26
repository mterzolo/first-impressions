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
import resources

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
                             '-i',
                             '../data/video_data/{}/{}.mp4'.format(chunk, file_name),
                             '../data/audio_data/{}_data/{}.wav'.format(partition, file_name)])
    pass


def extract_text(partition, training=True):
    """

    Takes transcripts and saves them as dataframes
    :param partition: Training set, test set, or validation set
    :return:
    """

    logging.info('Begin text extraction for {} partition'.format(partition))

    # Open transcript and annotations
    with open('../data/meta_data/transcription_{}.pkl'.format(partition), 'rb') as f:
        transcript = pickle.load(f, encoding='latin1')

    if training:
        with open('../data/meta_data/annotation_{}.pkl'.format(partition), 'rb') as f:
            annotation = pickle.load(f, encoding='latin1')

    # Transform into a data frame
    text_df = pd.DataFrame({'video_id': list(transcript.keys()),
                            'transcript': list(transcript.values())})

    text_df['transcript'] = text_df['transcript'].fillna('UNK')
    text_df['token'] = text_df['transcript'].str.replace(r'\[.*\]', '')

    if training:
        # Map in annotations
        text_df['interview_score'] = text_df['video_id'].map(annotation['interview'])

    # Create directory if it doesnt exist
    if not os.path.exists('../data/text_data/{}_data/'.format(partition)):
        os.makedirs('../data/text_data/{}_data/'.format(partition))

    with open('../data/text_data/{}_data/{}_text_df.pkl'.format(partition, partition), 'wb') as output:
        pickle.dump(text_df, output, protocol=4)

    pass


def transform_images_5d_chunks(partition, num_frames, training=True):

    logging.info('Begin transform images 5d for the {} partition'.format(partition))

    if not os.path.exists('../data/image_data/npy_files/{}_data/'.format(partition)):
        os.makedirs('../data/image_data/npy_files/{}_data/'.format(partition))

    if training:
        # Open answers file
        with open('../data/meta_data/annotation_{}.pkl'.format(partition), 'rb') as f:
            label_file = pickle.load(f, encoding='latin1')

    # Get all IDs for videos for the training set
    vid_ids = os.listdir('../data/image_data/{}_data'.format(partition))
    file_ids = [i + '.mp4' for i in vid_ids]

    if training:
        y = [label_file['interview'][i + '.mp4'] for i in vid_ids]

    out_counter = 0

    for video in vid_ids:

        images = os.listdir('../data/image_data/{}_data/{}'.format(partition, video))
        X_temp = np.zeros(shape=(num_frames, 224, 224, 3))
        in_counter = 0

        for image in images:

            # Load the image
            original = load_img('../data/image_data/{}_data/{}/{}'.format(partition, video, image),
                                target_size=(224, 224))

            # Convert to numpy array
            numpy_image = img_to_array(original)

            # Resize and store in one big array
            image_temp = np.expand_dims(numpy_image, axis=0)
            image_temp = vgg16.preprocess_input(image_temp)
            X_temp[in_counter] = image_temp

            # Increment counter for number of images in observation
            in_counter += 1

        # Append to numpy array
        X_temp = np.expand_dims(X_temp, axis=0)

        # Save the images numpy array
        np.save('../data/image_data/npy_files/{}_data/{}.npy'.format(partition, out_counter), X_temp)

        # Increment counter for observations in dataset
        out_counter += 1

    if training:
        with open('../data/image_data/pickle_files/y_5d_{}.pkl'.format(partition), 'wb') as output:
            pickle.dump(y, output, protocol=4)

    with open('../data/image_data/pickle_files/vid_ids_5d_{}.pkl'.format(partition), 'wb') as output:
        pickle.dump(file_ids, output, protocol=4)

    pass


def transform_audio(partition, n_mfcc, training=True):
    """

    compute features
    :return:
    """

    logging.info('Begin audio transformations for {} partition'.format(partition))

    if training:
        # Open answers file
        with open('../data/meta_data/annotation_{}.pkl'.format(partition), 'rb') as f:
            label_file = pickle.load(f, encoding='latin1')

    # Get all IDs for videos for the training set
    audio_files = os.listdir('../data/audio_data/{}_data'.format(partition))
    audio_files = [i.split('.wav')[0] for i in audio_files]
    id_array = [i + '.mp4' for i in audio_files]

    if training:
        score = [label_file['interview'][i + '.mp4'] for i in audio_files]
        score = np.array(score)

    # Set column names
    mfcc_mean_cols = ['mfcc_mean_' + str(i) for i in range(n_mfcc)]
    mfcc_std_cols = ['mfcc_mean_' + str(i) for i in range(n_mfcc)]
    other_cols = [

        'energey_mean',
        'energy_std',
        'zero_cross_mean',
        'zero_cross_std',
        'tempo_mean',
        'tempo_std',
        'flatness_mean',
        'flatness_std',
        'bandwidth_mean',
        'bandwidth_std',
        'rolloff_mean',
        'rolloff_std',
        'contrast_mean',
        'contrast_std',
        'tonnetz_mean',
        'tonnetz_std'
    ]
    cols = mfcc_mean_cols + mfcc_std_cols + other_cols

    # Create empty 2d array with place holders for all features
    audio_matrix = np.empty((len(audio_files), n_mfcc * 2 + 16))
    counter = 0

    for aud in audio_files:

        logging.debug('Begin feature extraction for {}.wav'.format(aud))

        # Convert wav to librosa object
        y, sr = librosa.load('../data/audio_data/{}_data/{}.wav'.format(partition, aud))

        # Create array to store values (Will become a row in the final df)
        values = np.zeros((len(cols)))

        mfcc = librosa.feature.mfcc(y, n_mfcc=n_mfcc)
        energy = librosa.feature.rmse(y)
        zero_cross = librosa.feature.zero_crossing_rate(y)
        tempo = librosa.feature.tempogram(y)
        flatness = librosa.feature.spectral_flatness(y)
        bandwidth = librosa.feature.spectral_bandwidth(y)
        rolloff = librosa.feature.spectral_rolloff(y)
        contrast = librosa.feature.spectral_contrast(y)
        tonnetz = librosa.feature.tonnetz(y)

        values[0:n_mfcc] = mfcc.mean(axis=1)
        values[n_mfcc:n_mfcc * 2] = mfcc.std(axis=1)
        values[n_mfcc * 2] = np.mean(energy)
        values[n_mfcc * 2 + 1] = np.std(energy)
        values[n_mfcc * 2 + 2] = np.mean(zero_cross)
        values[n_mfcc * 2 + 3] = np.std(zero_cross)
        values[n_mfcc * 2 + 4] = np.mean(tempo)
        values[n_mfcc * 2 + 5] = np.std(tempo)
        values[n_mfcc * 2 + 6] = np.mean(flatness)
        values[n_mfcc * 2 + 7] = np.std(flatness)
        values[n_mfcc * 2 + 8] = np.mean(bandwidth)
        values[n_mfcc * 2 + 9] = np.std(bandwidth)
        values[n_mfcc * 2 + 10] = np.mean(rolloff)
        values[n_mfcc * 2 + 11] = np.std(rolloff)
        values[n_mfcc * 2 + 12] = np.mean(contrast)
        values[n_mfcc * 2 + 13] = np.std(contrast)
        values[n_mfcc * 2 + 14] = np.mean(tonnetz)
        values[n_mfcc * 2 + 15] = np.std(tonnetz)

        # Append values to matrix
        audio_matrix[counter] = values
        counter += 1

    # Create final dataframe
    audio_df = pd.DataFrame(audio_matrix, columns=cols)

    if training:
        audio_df['interview_score'] = score

    audio_df['video_id'] = id_array

    audio_df.to_csv('../data/audio_data/pickle_files/{}_df.csv'.format(partition, partition), index=False)

    pass


def transform_text(partition, word_to_index, training=True):

    logging.info('Begin text transformation on {}'.format(partition))

    # Load transcripts
    with open('../data/text_data/{}_data/{}_text_df.pkl'.format(partition, partition), 'rb') as f:
        observations = pickle.load(f, encoding='latin1')

    # Transform embedding resources
    default_dict_instance = defaultdict(lambda: word_to_index['UNK'])
    default_dict_instance.update(word_to_index)
    word_to_index = default_dict_instance

    # Convert text to normalized tokens. Unknown tokens will map to 'UNK'.
    observations['tokens'] = observations['transcript'].apply(simple_preprocess)

    # Convert tokens to indices
    observations['indices'] = observations['tokens'].apply(lambda token_list: list(map(lambda token: word_to_index[token],
                                                                                       token_list)))
    observations['indices'] = observations['indices'].apply(lambda x: np.array(x))

    # Pad indices list with zeros, so that every article's list of indices is the same length
    X = pad_sequences(observations['indices'], 80)

    # Create data sets for model
    if training:
        y = observations['interview_score'].values

    vid_id = observations['video_id'].values

    # Save as pickled files
    with open('../data/text_data/pickle_files/X_{}.pkl'.format(partition), 'wb') as output:
        pickle.dump(X, output, protocol=4)
    with open('../data/text_data/pickle_files/vid_ids_{}.pkl'.format(partition), 'wb') as output:
        pickle.dump(vid_id, output, protocol=4)

    if training:
        with open('../data/text_data/pickle_files/y_{}.pkl'.format(partition), 'wb') as output:
            pickle.dump(y, output, protocol=4)

    pass
