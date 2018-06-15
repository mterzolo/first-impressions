import gzip
import logging
import os
import gensim
import requests
import pickle
import subprocess

from lib import get_conf


def download_first_impressions():
    """
    Download raw video and annotations data sets. Extracts the zipped folders and organizes them appropriately
    :return:
    """

    logging.info('Attempting to either validate or download first impressions data set.')

    # Set location of the server to download files from
    server = 'http://158.109.8.102/FirstImpressionsV2/'

    # Download dictionaries that map relationships between downloaded files
    with open('../resources/file_tree.pkl', 'rb') as input_file:
        file_tree = pickle.load(input_file)
    with open('../resources/meta_tree.pkl', 'rb') as input_file:
        meta_tree = pickle.load(input_file)

    # Set the encryption keys to unzip password protected files
    encryption_key = 'zeAzLQN7DnSIexQukc9W'
    alt_encryption_key = '.chalearnLAPFirstImpressionsSECONDRoundICPRWorkshop2016.'

    # Check if the meta files are in the correct location
    for file in meta_tree.keys():
        if not os.path.exists('../data/meta_data/{}'.format(file)):

            # Extract and save to correct location
            subprocess.call(['unzip',
                             '-n',
                             '-P',
                             encryption_key,
                             '../resources/compressed/{}'.format(meta_tree[file]),
                             '-d',
                             '../data/meta_data/'])
    download_links = [

        'train-1.zip',
        'train-2.zip',
        'train-3.zip',
        'train-4.zip',
        'train-5.zip',
        'train-6.zip',
        'test-1e.zip',
        'test-2e.zip',
        'val-1.zip',
        'val-2.zip',
    ]

    for file in download_links:

        # Define path to compressed files
        fi_downloaded_path = '../resources/compressed/{}'.format(file)

        # Download embeddings, if necessary
        if not os.path.exists(fi_downloaded_path):
            logging.warning('{} does not exist. Downloading {}.'.format(file, file))
            logging.info(
                'Downloading embedding data from: {} to: {}'.format(server + file, fi_downloaded_path))

            # Download the file
            download_file(server + file, fi_downloaded_path, auth=True)

    for file_chunk in file_tree.keys():

        if not os.path.exists('../data/video_data/{}'.format(file_chunk)):

            auth = encryption_key

            logging.debug('Extracting {}'.format(file_tree[file_chunk]))

            # Unzip file chunks from main download blocks
            subprocess.call(['unzip',
                             '-n',
                             '-P',
                             encryption_key,
                             '../resources/compressed/{}'.format(file_tree[file_chunk]),
                             '-d',
                             '../data/video_data/'])

            # Handle funky zipping of the test sets
            if file_tree[file_chunk] == 'test-1e.zip':
                all_files = os.listdir('../data/video_data/test-1/')
                for file in all_files:
                    subprocess.call(['mv', '../data/video_data/test-1/{}'.format(file), '../data/video_data/'])
                subprocess.call(['rm', '-r', '../data/video_data/test-1/'])
                auth = alt_encryption_key

            elif file_tree[file_chunk] == 'test-2e.zip':
                all_files = os.listdir('../data/video_data/test-2/')
                for file in all_files:
                    subprocess.call(['mv', '../data/video_data/test-2/{}'.format(file), '../data/video_data/'])
                subprocess.call(['rm', '-r', '../data/video_data/test-2/'])
                auth = alt_encryption_key

            # Unzip contents into newly created directory
            zipped_chunks = [i for i in os.listdir('../data/video_data/') if '.zip' in i]
            for to_extract in zipped_chunks:
                subprocess.call(['unzip',
                                 '-P',
                                 auth,
                                 '../data/video_data/{}'.format(to_extract),
                                 '-d',
                                 '../data/video_data/{}/'.format(to_extract.split('.zip')[0])])
            for to_extract in zipped_chunks:

                # Remove empty folder that is created in the process
                subprocess.call(['rm', '../data/video_data/{}'.format(to_extract)])

def download_embedding():
    """
    Prepare GoogleNews pre-trained word embeddings.
     - Check if compressed embeddings are available
     - If compressed embeddings are not available, download them
     - Check if uncompressed embeddings are available
     - If compressed embeddings are not available, uncompress embeddings
    :return: None
    :rtype: None
    """

    logging.info('Attempting to either validate or download and extract embeddings.')

    # Reference variables
    embedding_download_link = 'https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz'
    embedding_downloaded_path = '../resources/compressed/GoogleNews-vectors-negative300.bin.gz'

    # Download embeddings, if necessary
    if not os.path.exists(embedding_downloaded_path):
        logging.warning('embedding_downloaded_path does not exist. Downloading embedding.')
        logging.info(
            'Downloading embedding data from: {} to: {}'.format(embedding_download_link, embedding_downloaded_path))

        download_file(embedding_download_link, embedding_downloaded_path, auth=False)

    # Extract embeddings, if necessary
    if not os.path.exists(get_conf('embedding_path')):
        logging.warning('embedding_path does not exist. Extracting embedding.')
        logging.info(
            'Extracting embedding data from: {} to: {}'.format(embedding_downloaded_path, get_conf('embedding_path')))

    with gzip.open(embedding_downloaded_path, 'rb') as zipped, \
            open(get_conf('embedding_path'), 'wb') as unzipped:
        for line in zipped:
            unzipped.write(line)

    logging.info('Embeddings available at: {}'.format(get_conf('embedding_path')))


def download_file(url, local_file_path, auth=False):
    """
    Download the file at `url` in chunks, to the location at `local_file_path`
    :param url: URL to a file to be downloaded
    :type url: str
    :param local_file_path: Path to download the file to
    :type local_file_path: str
    :param auth: is authentication required to download file
    :type auth: Boolean
    :return: The path to the file on the local machine (same as input `local_file_path`)
    :rtype: str
    """

    # Get user name and password
    username = get_conf('user_name')
    password = get_conf('password')

    # Reference variables
    chunk_count = 0

    if auth:

        # Create connection to the stream
        r = requests.get(url, auth=(username, password), stream=True)
    else:

        # Create connection without password
        r = requests.get(url, stream=True)

    # Open output file
    with open(local_file_path, 'wb') as f:

        # Iterate through chunks of file
        for chunk in r.iter_content(chunk_size=64*1024):

            logging.debug('Downloading chunk: {} for file: {}'.format(chunk_count, local_file_path))

            # Write chunk to file
            f.write(chunk)

            # Increase chunk counter
            chunk_count = chunk_count + 1

    return local_file_path


def create_embedding_matrix():
    """
    Load embedding assets from file.
     - Load embedding binaries w/ gsensim
     - Extract embedding matrix from gensim model
     - Extract word to index lookup from gensim model
    :return: embedding_matrix, word_to_index
    :rtype: (numpy.array, {str:int})
    """

    logging.info('Reading embedding matrix and word to index dictionary from file')

    # Get word weights from file via gensim
    model = gensim.models.KeyedVectors.load_word2vec_format(get_conf('embedding_path'), binary=True)
    embedding_matrix = model.syn0

    # Filter out words with index not in w2v range
    word_to_index = dict([(k, v.index) for k, v in model.vocab.items()])

    logging.info('Created embedding matrix, of shape: {}'.format(embedding_matrix.shape))
    logging.info('Created word to index lookup, with min index: {}, max index: {}'.format(min(word_to_index.values()),
                                                                                          max(word_to_index.values())))

    return embedding_matrix, word_to_index
