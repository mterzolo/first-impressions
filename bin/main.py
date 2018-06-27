import pickle
import logging
import models
import lib
import resources
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from my_classes import DataGenerator


def main():
    """
    Main entry point for code
    :return:
    """

    logging.getLogger().setLevel(level=logging.INFO)

    extract()
    transform()
    model()
    ensemble()
    score_new_vid()

    pass


def extract():
    """
    Downloads raw data needed and extracts data for all 3 models.
    Image - Convert mp4 files into a series of jpeg images
    Audio - Extract mp3 files from each mp4 file
    Text - Extract text from annotation files
    :return:
    """

    # Download resources
    resources.download_first_impressions()
    resources.download_embedding()

    # Extract images, audio files, and text transcripts for each partition
    for partition in ['training', 'test', 'validation']:

        # Chop video up into images and save into separate directory
        lib.extract_images(partition, num_frames=10)

        # Strip audio from mp4 and save in separate directory
        lib.extract_audio(partition)

        # Take text from transcripts
        lib.extract_text(partition)

    pass


def transform():
    """
    Transforms all features for the 3 models.
    Image - Convert jpegs to numpy arrays and preprocess for the vgg16 model
    Audio - Use librosa to extract features and save dataframe with all features for each video
    Text - Tokenize, and convert to indices based on the google news 20 word embeddings
    :return:
    """

    embedding_matrix, word_to_index = resources.create_embedding_matrix()

    for partition in ['training', 'test', 'validation']:

        # Transform raw jpegs into numpy arrays
        lib.transform_images(partition=partition, num_frames=10)

        # Transform raw audio to feature matrix
        lib.transform_audio(partition=partition, n_mfcc=13)

        # Transform text to tokens
        lib.transform_text(partition=partition, word_to_index=word_to_index)

    pass


def model(image=False, audio=False, text=False):
    """
    Train all 3 models
    :param image: Whether or not to train the image model on this run
    :param audio: Whether or not to train the audio model on this run
    :param text: Whether or not to train the text model on this run
    :return:
    """

    if image:

        # Parameters
        params = {'dim': (10, 224, 224),
                  'batch_size': 16,
                  'n_channels': 3,
                  'shuffle': True}

        # Load labels set
        with open('../data/image_data/pickle_files/y_5d_training.pkl', 'rb') as file:
            training_labels = pickle.load(file)
        with open('../data/image_data/pickle_files/y_5d_test.pkl', 'rb') as file:
            test_labels = pickle.load(file)

        # Generators
        training_generator = DataGenerator(partition='training',
                                           list_IDs=range(6000),
                                           labels=training_labels, **params)
        validation_generator = DataGenerator(partition='test',
                                             list_IDs=range(2000),
                                             labels=test_labels, **params)

        # Create model
        model = models.image_lrcn()

        # Train model on data set
        model.fit_generator(generator=training_generator,
                            validation_data=validation_generator,
                            use_multiprocessing=True,
                            workers=6,
                            epochs=5)

        model.save_weights('../output/image_model.h5')

    if audio:

        # Read in aduio data
        training_set = pd.read_csv('../data/audio_data/pickle_files/training_df.csv')
        test_set = pd.read_csv('../data/audio_data/pickle_files/test_df.csv')

        # Concat data sets in order to use all data for CV
        all_data = pd.concat((training_set, test_set), axis=0)
        X_all = all_data.drop(['interview_score', 'video_id'], axis=1)
        y_all = all_data['interview_score']

        logging.info('Start training audio model')

        # Create model and fit to data
        audio_model = models.audio_rand_forest()
        audio_model.fit(X_all, y_all)

        logging.info(audio_model.best_params_)
        logging.info('Train score with best estimator: {}'.format(max(audio_model.cv_results_['mean_train_score'])))
        logging.info('Test score with best estimator: {}'.format(max(audio_model.cv_results_['mean_test_score'])))

        # Save to disk
        with open('../output/audio_model.pkl', 'wb') as fid:
            pickle.dump(audio_model, fid)

    if text:

        # Load in word embeddings
        embedding_matrix, word_to_index = resources.create_embedding_matrix()

        # Load text data
        with open('../data/text_data/pickle_files/X_training.pkl', 'rb') as file:
            X_train = pickle.load(file)
        with open('../data/text_data/pickle_files/y_training.pkl', 'rb') as file:
            y_train = pickle.load(file)
        with open('../data/text_data/pickle_files/X_test.pkl', 'rb') as file:
            X_test = pickle.load(file)
        with open('../data/text_data/pickle_files/y_test.pkl', 'rb') as file:
            y_test = pickle.load(file)

        # Create model object and fit
        text_model = models.text_lstm_model(embedding_matrix=embedding_matrix)
        filename = '../output/text_model.h5'
        checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        text_model.fit(X_train, y_train,
                       batch_size=32, epochs=55,
                       validation_data=(X_test, y_test),
                       callbacks=[checkpoint],
                       shuffle=True)

    pass


def ensemble():

    logging.info('Begin Ensemble model building, loading models')

    # Load models
    image_model = models.image_lrcn()
    image_model.load_weights('../output/image_model.h5')
    audio_model = pickle.load(open('../output/audio_model.pkl', 'rb'))
    text_model = load_model('../output/text_model.h5')

    # Load labels set
    with open('../data/image_data/pickle_files/y_5d_training.pkl', 'rb') as file:
        training_labels = pickle.load(file)
    with open('../data/image_data/pickle_files/y_5d_test.pkl', 'rb') as file:
        test_labels = pickle.load(file)

    # Load generators
    training_generator = DataGenerator(partition='training', list_IDs=range(6000),
                                       labels=training_labels, batch_size=16,
                                       n_channels=3, dim=(10, 224, 224),
                                       shuffle=False)
    validation_generator = DataGenerator(partition='test', list_IDs=range(2000),
                                         labels=test_labels, batch_size=16,
                                         n_channels=3, dim=(10, 224, 224),
                                         shuffle=False)
    holdout_generator = DataGenerator(partition='validation', list_IDs=range(2000),
                                      labels=test_labels, batch_size=16,
                                      n_channels=3, dim=(10, 224, 224),
                                      shuffle=False)

    logging.info('Load data files')

    # Load image data
    with open('../data/image_data/pickle_files/y_training.pkl', 'rb') as file:
        y_img_train = pickle.load(file)
    with open('../data/image_data/pickle_files/y_test.pkl', 'rb') as file:
        y_img_test = pickle.load(file)
    with open('../data/image_data/pickle_files/y_validation.pkl', 'rb') as file:
        y_img_val = pickle.load(file)
    with open('../data/image_data/pickle_files/vid_ids_training.pkl', 'rb') as file:
        id_img_train = pickle.load(file)
    with open('../data/image_data/pickle_files/vid_ids_test.pkl', 'rb') as file:
        id_img_test = pickle.load(file)
    with open('../data/image_data/pickle_files/vid_ids_validation.pkl', 'rb') as file:
        id_img_val = pickle.load(file)

    # Load audio data
    aud_train = pd.read_csv('../data/audio_data/pickle_files/training_df.csv')
    aud_test = pd.read_csv('../data/audio_data/pickle_files/test_df.csv')
    aud_val = pd.read_csv('../data/audio_data/pickle_files/validation_df.csv')
    X_aud_train = aud_train.drop(['interview_score', 'video_id'], axis=1)
    id_aud_train = aud_train['video_id']
    X_aud_test = aud_test.drop(['interview_score', 'video_id'], axis=1)
    id_aud_test = aud_test['video_id']
    X_aud_val = aud_val.drop(['interview_score', 'video_id'], axis=1)
    id_aud_val = aud_val['video_id']

    # Load text data
    with open('../data/text_data/pickle_files/X_training.pkl', 'rb') as file:
        X_text_train = pickle.load(file)
    with open('../data/text_data/pickle_files/X_test.pkl', 'rb') as file:
        X_text_test = pickle.load(file)
    with open('../data/text_data/pickle_files/X_validation.pkl', 'rb') as file:
        X_text_val = pickle.load(file)
    with open('../data/text_data/pickle_files/vid_ids_training.pkl', 'rb') as file:
        id_text_train = pickle.load(file)
    with open('../data/text_data/pickle_files/vid_ids_test.pkl', 'rb') as file:
        id_text_test = pickle.load(file)
    with open('../data/text_data/pickle_files/vid_ids_validation.pkl', 'rb') as file:
        id_text_val = pickle.load(file)

    logging.info('Getting predictions for all 3 models')

    # Get predictions
    img_train_df = pd.DataFrame({'img_preds': [i[0] for i in image_model.predict_generator(training_generator)],
                                 'video_ids': id_img_train,
                                 'interview_score': y_img_train})
    img_test_df = pd.DataFrame({'img_preds': [i[0] for i in image_model.predict_generator(validation_generator)],
                                'video_ids': id_img_test,
                                'interview_score':y_img_test})
    img_val_df = pd.DataFrame({'img_preds': [i[0] for i in image_model.predict_generator(holdout_generator)],
                               'video_ids': id_img_val,
                               'interview_score': y_img_val})
    aud_train_df = pd.DataFrame({'aud_preds': audio_model.predict(X_aud_train),
                                 'video_ids': id_aud_train})
    aud_test_df = pd.DataFrame({'aud_preds': audio_model.predict(X_aud_test),
                                'video_ids': id_aud_test})
    aud_val_df = pd.DataFrame({'aud_preds': audio_model.predict(X_aud_val),
                               'video_ids': id_aud_val})
    text_train_df = pd.DataFrame({'text_preds': [i[0] for i in text_model.predict(X_text_train)],
                                  'video_ids': id_text_train})
    text_test_df = pd.DataFrame({'text_preds': [i[0] for i in text_model.predict(X_text_test)],
                                 'video_ids': id_text_test})
    text_val_df = pd.DataFrame({'text_preds': [i[0] for i in text_model.predict(X_text_val)],
                                'video_ids': id_text_val})

    logging.info('Merge predictions together into single data frame')

    # Merge predictions
    train_preds = img_train_df.merge(aud_train_df, on='video_ids')
    train_preds = train_preds.merge(text_train_df, on='video_ids')
    test_preds = img_test_df.merge(aud_test_df, on='video_ids')
    test_preds = test_preds.merge(text_test_df, on='video_ids')
    val_preds = img_val_df.merge(aud_val_df, on='video_ids')
    val_preds = val_preds.merge(text_val_df, on='video_ids')

    # Score models
    img_train_score = np.sqrt(mean_squared_error(train_preds['interview_score'], train_preds['img_preds']))
    img_test_score = np.sqrt(mean_squared_error(test_preds['interview_score'], test_preds['img_preds']))
    img_val_score = np.sqrt(mean_squared_error(val_preds['interview_score'], val_preds['img_preds']))
    aud_train_score = np.sqrt(mean_squared_error(train_preds['interview_score'], train_preds['aud_preds']))
    aud_test_score = np.sqrt(mean_squared_error(test_preds['interview_score'], test_preds['aud_preds']))
    aud_val_score = np.sqrt(mean_squared_error(val_preds['interview_score'], val_preds['aud_preds']))
    text_train_score = np.sqrt(mean_squared_error(train_preds['interview_score'], train_preds['text_preds']))
    text_test_score = np.sqrt(mean_squared_error(test_preds['interview_score'], test_preds['text_preds']))
    text_val_score = np.sqrt(mean_squared_error(val_preds['interview_score'], val_preds['text_preds']))

    # Print scores to screen
    logging.info('Image score on the training set: {}'.format(img_train_score))
    logging.info('Image score on the test set: {}'.format(img_test_score))
    logging.info('Image score on the val set: {}'.format(img_val_score))
    logging.info('Audio score on the training set: {}'.format(aud_train_score))
    logging.info('Audio score on the test set: {}'.format(aud_test_score))
    logging.info('Audio score on the val set: {}'.format(aud_val_score))
    logging.info('Text score on the training set: {}'.format(text_train_score))
    logging.info('Text score on the test set: {}'.format(text_test_score))
    logging.info('Text score on the val set: {}'.format(text_val_score))

    # Split target variable and features
    X_train = train_preds[['img_preds', 'aud_preds', 'text_preds']]
    y_train = train_preds[['interview_score']]
    X_test = test_preds[['img_preds', 'aud_preds', 'text_preds']]
    y_test = test_preds[['interview_score']]
    X_val = val_preds[['img_preds', 'aud_preds', 'text_preds']]
    y_val = val_preds[['interview_score']]

    logging.info('Build OLS model to combine model outputs')

    # Build OLS model
    ols_model = LinearRegression()
    ols_model.fit(X_train, y_train)

    # Score model
    train_score = np.sqrt(mean_squared_error(y_train, ols_model.predict(X_train)))
    test_score = np.sqrt(mean_squared_error(y_test, ols_model.predict(X_test)))
    val_score = np.sqrt(mean_squared_error(y_val, ols_model.predict(X_val)))

    logging.info('OLS Score on training set: {}'.format(train_score))
    logging.info('OLS Score on test set: {}'.format(test_score))
    logging.info('OLS Score on val set: {}'.format(val_score))

    # Save model
    with open('../output/ensemble_model.pkl', 'wb') as fid:
        pickle.dump(ols_model, fid)

    logging.info('Ensemble model saved')

    return


def score_new_vid():

    logging.info('Begin extraction for scoring partition')

    # Extract features from vids
    lib.extract_images(partition='score', num_frames=10)
    lib.extract_audio(partition='score')
    lib.extract_text(partition='score', training=False)

    logging.info('Begin transformation for scoring partition')

    # Transform features
    embedding_matrix, word_to_index = resources.create_embedding_matrix()
    lib.transform_images(partition='score', num_frames=10, training=False)
    lib.transform_audio(partition='score', n_mfcc=13, training=False)
    lib.transform_text(partition='score', word_to_index=word_to_index, training=False)

    logging.info('Load models for evaluation of the scoring partition')

    # Load models
    image_model = models.image_lrcn()
    image_model.load_weights('../output/image_model.h5')
    audio_model = pickle.load(open('../output/audio_model.pkl', 'rb'))
    text_model = load_model('../output/text_model.h5')
    ensemble_model = pickle.load(open('../output/ensemble_model.pkl', 'rb'))

    logging.info('Load transformed data')

    # Load image data
    with open('../data/image_data/pickle_files/vid_ids_5d_score.pkl', 'rb') as file:
        id_img_score = pickle.load(file)

    # Load audio data
    aud_to_score = pd.read_csv('../data/audio_data/pickle_files/score_df.csv')
    X_aud_score = aud_to_score.drop(['video_id'], axis=1)
    id_aud_score = aud_to_score['video_id']

    # Load text data
    with open('../data/text_data/pickle_files/X_score.pkl', 'rb') as file:
        X_text_score = pickle.load(file)
    with open('../data/text_data/pickle_files/vid_ids_score.pkl', 'rb') as file:
        id_text_score = pickle.load(file)

    # Load generator
    score_generator = DataGenerator(partition='training', list_IDs=range(len(id_aud_score)),
                                    labels=[0 for i in range(len(id_aud_score))], batch_size=len(id_aud_score),
                                    n_channels=3, dim=(10, 224, 224),
                                    shuffle=False)

    logging.info('Predict values with image, text and audio models')

    # Predict values
    img_score_df = pd.DataFrame({'img_preds': [i[0] for i in image_model.predict_generator(score_generator)],
                                 'video_ids': id_img_score})
    aud_score_df = pd.DataFrame({'aud_preds': audio_model.predict(X_aud_score),
                                 'video_ids': id_aud_score})
    text_score_df = pd.DataFrame({'text_preds': [i[0] for i in text_model.predict(X_text_score)],
                                  'video_ids': id_text_score})

    logging.info('Make final predictions')

    # Merge predictions
    score_preds = img_score_df.merge(aud_score_df, on='video_ids')
    score_preds = score_preds.merge(text_score_df, on='video_ids')

    # Make final prediction
    X_score = score_preds[['img_preds', 'aud_preds', 'text_preds']]
    score_preds['final_prediction'] = ensemble_model.predict(X_score)

    # Save predictions to disk
    score_preds.to_csv('../output/predictions.csv', index=False)

    pass


# Main section
if __name__ == '__main__':
    main()
