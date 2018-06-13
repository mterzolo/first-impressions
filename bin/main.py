import cv2
import os
import logging
import moviepy.editor as mp

import lib
import resources


def main():
    """

    main entry point for code
    :return:
    """

    logging.getLogger().setLevel(level=logging.DEBUG)

    extract()
    transform()
    model()
    load()

    pass

def extract():
    """

    Downloads raw data needed and extracts image, audio, and text from video files
    :return:
    """

    # Download resources
    resources.download_first_impressions()
    resources.download_embedding()

    # Extract images, audio files, and text transcripts for each partition
    for partition in ['training', 'test', 'validation']:

        lib.extract_images(partition, num_frames=50)
        lib.extract_audio(partition)
        lib.extract_text()


def transform():

    embedding_matrix, word_to_index = resources.create_embedding_matrix()


def model():










    # Main section
    if __name__ == '__main__':
        main()
