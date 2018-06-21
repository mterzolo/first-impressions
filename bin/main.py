import pickle
import logging
import models
import lib
import resources
from keras.callbacks import ModelCheckpoint


def main():
    """

    main entry point for code
    :return:
    """

    logging.getLogger().setLevel(level=logging.DEBUG)

    #extract()
    transform()
    #model()

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

        # Chop video up into images and save into separate directory
        lib.extract_images(partition, num_frames=20)

        # Strip audio from mp4 and save in separate directory
        lib.extract_audio(partition)

        # Take text from transcripts
        lib.extract_text(partition)

    pass


def transform():

    embedding_matrix, word_to_index = resources.create_embedding_matrix()

    for partition in ['training', 'test', 'validation']:

        # Transform raw audio to melspectrograms
        #lib.audio2melspec(partition=partition)

        # Transform raw audio to feature matrix
        lib.transform_audio(partition=partition, n_mfcc=13)

        # Transform raw jpegs into numpy arrays
        #lib.img2array(partition=partition, frame_num=4)

        # Transform text to tokens
        lib.transform_text(partition=partition, word_to_index=word_to_index)

    pass


def model(image=False, audio=True, text=False):

    embedding_matrix, word_to_index = resources.create_embedding_matrix()

    if image:

        # Load data
        with open('../data/image_data/pickle_files/X_training.pkl', 'rb') as file:
            X_train = pickle.load(file)
        with open('../data/image_data/pickle_files/y_training.pkl', 'rb') as file:
            y_train = pickle.load(file)
        with open('../data/image_data/pickle_files/X_test.pkl', 'rb') as file:
            X_test = pickle.load(file)
        with open('../data/image_data/pickle_files/y_test.pkl', 'rb') as file:
            y_test = pickle.load(file)

        # Create model and fit
        image_model = models.image_cnn_model()
        filename = '../output/image_model.h5'
        checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        image_model.fit(X_train, y_train,
                        validation_data=(X_test, y_test),
                        epochs=150, batch_size=32,
                        callbacks=[checkpoint])

    if audio:

        # Load data
        with open('../data/audio_data/pickle_files/X_training.pkl', 'rb') as file:
            X_train = pickle.load(file)
        with open('../data/audio_data/pickle_files/y_training.pkl', 'rb') as file:
            y_train = pickle.load(file)
        with open('../data/audio_data/pickle_files/X_test.pkl', 'rb') as file:
            X_test = pickle.load(file)
        with open('../data/audio_data/pickle_files/y_test.pkl', 'rb') as file:
            y_test = pickle.load(file)

        # Create model object and fit
        audio_model = models.audio_cnn_model()
        filename = '../output/audio_model.h5'
        checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        audio_model.fit(X_train, y_train,
                        batch_size=32, epochs=100,
                        validation_data=(X_test, y_test),
                        callbacks=[checkpoint])
    if text:

        # Load data
        with open('../data/text_data/pickle_files/X_training.pkl', 'rb') as file:
            X_train = pickle.load(file)
        with open('../data/text_data/pickle_files/y_training.pkl', 'rb') as file:
            y_train = pickle.load(file)
        with open('../data/text_data/pickle_files/X_test.pkl', 'rb') as file:
            X_test = pickle.load(file)
        with open('../data/text_data/pickle_files/y_test.pkl', 'rb') as file:
            y_test = pickle.load(file)

        # Create model object and fit
        text_model = models.text_cnn_model(embedding_matrix=embedding_matrix)
        filename = '../output/audio_model.h5'
        checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        text_model.fit(X_train, y_train,
                       batch_size=32, epochs=100,
                       validation_data=(X_test, y_test),
                       callbacks=[checkpoint])

        pass



    pass


# Main section
if __name__ == '__main__':
    main()
