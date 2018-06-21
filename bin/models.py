import logging
import resources
import keras
import pandas as pd
from keras.layers import Embedding, Conv1D, Conv2D, MaxPooling1D
from keras.applications import vgg16
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout, Input, Reshape
from keras.callbacks import ModelCheckpoint
from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.layers.recurrent import GRU
from sklearn.ensemble import RandomForestRegressor


def image_cnn_model():

    # Load the VGG16 model
    base_model = vgg16.VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    x = base_model.output

    # Create additional layers
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(.4)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(.4)(x)
    preds = Dense(1, activation='linear')(x)

    # Freeze all original layers
    for layer in base_model.layers:
        layer.trainable = False

    # Optimizer
    optimizer = keras.optimizers.Adam(lr=0.001, decay=0.0006)

    # fit model
    filename = '../output/image_model.h5'
    checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    # Create the model
    model = Model(base_model.input, preds)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    return model


def audio_cnn_model():
    """
    Generate convulutional neural network for audio data
    :return:
    """

    # Define the dimensions for the mel spectrogram to be fed into the model
    input_shape = (96, 704, 1)
    channel_axis = 3
    time_axis = 2
    melgram_input = Input(shape=input_shape)

    # Input block
    x = ZeroPadding2D(padding=(0, 37))(melgram_input)
    x = BatchNormalization(axis=time_axis, name='bn_0_freq')(x)

    # Conv block 1
    x = Conv2D(64, 3, 3, border_mode='same', name='conv1')(x)
    x = BatchNormalization(axis=channel_axis, name='bn1')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(x)

    # Conv block 2
    x = Conv2D(128, 3, 3, border_mode='same', name='conv2')(x)
    x = BatchNormalization(axis=channel_axis, name='bn2')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(3, 3), name='pool2')(x)

    # Conv block 3
    x = Conv2D(128, 3, 3, border_mode='same', name='conv3')(x)
    x = BatchNormalization(axis=channel_axis, name='bn3')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool3')(x)

    # Conv block 4
    x = Conv2D(128, 3, 3, border_mode='same', name='conv4')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn4')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool4')(x)

    """
    # Reshape for GROs
    x = Reshape((15, 128))(x)

    # GRU block 1, 2, output
    x = GRU(32, return_sequences=True, name='gru1')(x)
    x = GRU(32, return_sequences=False, name='gru2')(x)
    """

    # Final layer for predictions
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(1, activation='linear')(x)

    # Create model
    model = Model(melgram_input, x)
    model.compile(optimizer='Adam', loss='mean_squared_error')

    return model


def text_cnn_model(embedding_matrix):
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
    embedding_input_length = 79

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
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(35)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(units=1, activation='linear')(x)

    # Compile architecture
    text_model = Model(sequence_input, preds)
    text_model.compile(loss='mse', optimizer='rmsprop')

    return text_model


def audio_rand_forest():

    model = RandomForestRegressor()

    return model