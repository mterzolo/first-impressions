import logging
import keras
from keras.applications import vgg16
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout, LSTM, Bidirectional, Embedding, Flatten, Lambda
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

import numpy as np
from keras import layers, models
from keras import backend as K
from my_classes import DataGenerator


def image_cnn_model():

    # Load the VGG16 model
    base_model = vgg16.VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    x = base_model.output

    # Create additional layers
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(.2)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(.2)(x)
    preds = Dense(1, activation='linear')(x)

    # Freeze all original layers
    for layer in base_model.layers:
        layer.trainable = False

    # Optimizer
    optimizer = keras.optimizers.Adam(lr=0.001, decay=0.0006)

    # Create the model
    model = Model(base_model.input, preds)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    return model


def image_lrcn():

    # Set learning phase to 0
    K.set_learning_phase(0)

    # Set input layer
    video = layers.Input(shape=(None, 224, 224, 3), name='video_input')

    # Load the VGG16 model
    cnn = vgg16.VGG16(weights="imagenet", include_top=False, pooling='avg')
    cnn.trainable = False

    # Wrap cnn into Lambda and pass it into TimeDistributed
    encoded_frame = layers.TimeDistributed(Lambda(lambda x: cnn(x)))(video)
    encoded_vid = layers.LSTM(64)(encoded_frame)
    encoded_vid = layers.Dropout(.05)(encoded_vid)
    adam_opt = keras.optimizers.Adam(lr=0.0005, decay=0.001)
    outputs = layers.Dense(1, activation='linear')(encoded_vid)
    model = models.Model(inputs=[video], outputs=outputs)
    model.compile(optimizer=adam_opt, loss='mean_squared_error')

    return model


def text_lstm_model(embedding_matrix):
    """
    Generate a convolutional neural network model, with an embedding layer.
    :param transcripts: A Pandas DataFrame containing the field padded_indices
    :type transcripts: pandas.DataFrame
    :param embedding_matrix: An embedding matrix, with shape (n,m), where n is the number of words, and m is the
    dimensionality of the embedding
    :type embedding_matrix: numpy.array
    :param word_to_index: A mapping from words (strings), to their index in the embedding matrix. For example
    embedding_matrix[word_to_index['pineapple']] would give the embedding vector for the word 'pineapple'
    :type: {str:int}
    :return: A keras model that can be trained on the given padded indices
    :rtype: Model
    """

    # Number of words in the word lookup index
    embedding_input_dim = embedding_matrix.shape[0]

    # Number of dimensions in the embedding
    embedding_output_dim = embedding_matrix.shape[1]

    # Maximum length of the x vectors
    embedding_input_length = 80

    logging.info('embedding_input_dim: {}, embedding_output_dim: {}, embedding_input_length: {}'
                 .format(embedding_input_dim, embedding_output_dim, embedding_input_length))

    # Create embedding layer
    embedding_layer = Embedding(input_dim=embedding_input_dim,
                                output_dim=embedding_output_dim,
                                weights=[embedding_matrix],
                                input_length=embedding_input_length,
                                trainable=False)
    sequence_input = keras.Input(shape=(embedding_input_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Dropout(.5)(embedded_sequences)
    x = Bidirectional(LSTM(64))(x)
    x = Dropout(.5)(x)
    x = Dense(units=64, activation='relu')(x)
    x = Dropout(.5)(x)
    preds = Dense(units=1, activation='linear')(x)

    # Compile architecture
    text_model = Model(sequence_input, preds)
    text_model.compile(loss='mse', optimizer='adam')

    return text_model


def audio_rand_forest():

    model = GridSearchCV(RandomForestRegressor(n_jobs=-1),
                         param_grid={'max_features': range(5, 30, 5),
                                     'max_depth': range(3, 7, 2),
                                     'n_estimators': range(60, 120, 15),
                                     },
                         scoring='neg_mean_squared_error',
                         cv=3)

    return model
