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
from keras import backend as K
from my_classes import DataGenerator


def main():
    """

    main entry point for code
    :return:
    """

    logging.getLogger().setLevel(level=logging.DEBUG)

    #extract()
    transform()
    #model()
    #ensemble()
    score_new_vid()

    pass


def extract():
    """

    Downloads raw data needed and extracts image, audio, and text from video files
    :return:
    """

    # Download resources
    #resources.download_first_impressions()
    #resources.download_embedding()

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

    embedding_matrix, word_to_index = resources.create_embedding_matrix()

    for partition in ['training', 'test', 'validation']:

        # Transform raw jpegs into numpy arrays
        lib.transform_images_5d_chunks(partition=partition, num_frames=10)

        # Transform raw jpegs into numpy arrays
        #lib.transform_images(partition=partition, frame_num=4)

        # Transform raw audio to feature matrix
        #lib.transform_audio(partition=partition, n_mfcc=13)

        # Transform text to tokens
        #lib.transform_text(partition=partition, word_to_index=word_to_index)

    pass


def model(image=False, audio=False, text=False, image_5d_chunks=True):

    if image_5d_chunks:

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

        model = models.image_lrcn()

        # Train model on data set
        model.fit_generator(generator=training_generator,
                            validation_data=validation_generator,
                            use_multiprocessing=True,
                            workers=6,
                            epochs=5)

        model.save_weights('../output/image_model.h5')

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
                        batch_size=64, epochs=100,
                        callbacks=[checkpoint],
                        shuffle=True)

    if audio:

        training_set = pd.read_csv('../data/audio_data/pickle_files/training_df.csv')
        test_set = pd.read_csv('../data/audio_data/pickle_files/test_df.csv')

        all_data = pd.concat((training_set, test_set), axis=0)
        X_all = all_data.drop(['interview_score', 'video_id'], axis=1)
        y_all = all_data['interview_score']

        logging.info('Start training audio model')

        audio_model = models.audio_rand_forest()
        audio_model.fit(X_all, y_all)

        logging.info(audio_model.best_params_)
        logging.info('Train score with best estimator: {}'.format(max(audio_model.cv_results_['mean_train_score'])))
        logging.info('Test score with best estimator: {}'.format(max(audio_model.cv_results_['mean_test_score'])))

        with open('../output/audio_model.pkl', 'wb') as fid:
            pickle.dump(audio_model, fid)

    if text:

        embedding_matrix, word_to_index = resources.create_embedding_matrix()

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
    train_score = mean_squared_error(y_train, ols_model.predict(X_train))
    test_score = mean_squared_error(y_test, ols_model.predict(X_test))
    val_score = mean_squared_error(y_val, ols_model.predict(X_val))

    logging.info('OLS Score on training set: {}'.format(train_score))
    logging.info('OLS Score on test set: {}'.format(test_score))
    logging.info('OLS Score on val set: {}'.format(val_score))



    # Simple Average
    simp_train_score = mean_squared_error(y_train, X_train.mean(axis=1))
    simp_test_score = mean_squared_error(y_test, X_test.mean(axis=1))
    simp_val_score = mean_squared_error(y_val, X_val.mean(axis=1))

    logging.info('Simple Average Score on training set: {}'.format(simp_train_score))
    logging.info('Simple Average Score on test set: {}'.format(simp_test_score))
    logging.info('Simple Average Score on val set: {}'.format(simp_val_score))

    # Save model
    with open('../output/ensemble_model.pkl', 'wb') as fid:
        pickle.dump(ols_model, fid)

    logging.info('Ensemble model saved')

    return


def score_new_vid():

    logging.info('Begin extraction for scoring partition')

    # Extract features from vids
    lib.extract_images(partition='score', num_frames=20)
    lib.extract_audio(partition='score')
    lib.extract_text(partition='score')

    logging.info('Begin transformation for scoring partition')

    # Transform features
    embedding_matrix, word_to_index = resources.create_embedding_matrix()
    lib.transform_images_5d_chunks(partition='score', num_frames=20)
    lib.transform_audio(partition='score', n_mfcc=13)
    lib.transform_text(partition='score', word_to_index=word_to_index)

    logging.info('Load models for evaluation of the scoring partition')

    # Load models
    image_model = models.image_lrcn()
    image_model.load_weights('../output/image_model.h5')
    audio_model = pickle.load(open('../output/audio_model.pkl', 'rb'))
    text_model = load_model('../output/text_model.h5')
    ensemble_model = pickle.load(open('../output/ensemble_model.pkl', 'rb'))

    logging.info('Load transformed data')

    # Load image data
    #with open('../data/image_data/pickle_files/vid_ids_score.pkl', 'rb') as file:
        #id_img_score = pickle.load(file)

    # Load audio data
    aud_to_score = pd.read_csv('../data/audio_data/pickle_files/score_df.csv')
    X_aud_score = aud_to_score.drop(['interview_score', 'video_id'], axis=1)
    id_aud_score = aud_to_score['video_id']

    # Load text data
    with open('../data/text_data/pickle_files/X_score.pkl', 'rb') as file:
        X_text_score = pickle.load(file)
    with open('../data/text_data/pickle_files/vid_ids_score.pkl', 'rb') as file:
        id_text_score = pickle.load(file)

    # Load generator
    score_generator = DataGenerator(partition='training', list_IDs=range(len(id_aud_score)),
                                    labels=[0 for i in range(len(id_aud_score))], batch_size=len(id_aud_score),
                                    n_channels=3, dim=(20, 224, 224),
                                    shuffle=False)

    logging.info('Predict values with image, text and audio models')

    # Predict values
    img_score_df = pd.DataFrame({'img_preds': [i[0] for i in image_model.predict_generator(score_generator)],
                                 'video_ids': list(range(len(id_aud_score)))})
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
