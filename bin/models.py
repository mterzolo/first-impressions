import logging

import keras
from keras.engine import Model
from keras.layers import Dense, Embedding, Conv1D, Conv2D, MaxPooling1D, Flatten
from keras.applications import vgg16
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout
from keras.callbacks import ModelCheckpoint


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

    pass


def text_cnn_model(transcripts, embedding_matrix, word_to_index):
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
    embedding_input_length = max(transcripts['padded_indices'].apply(len))

    logging.info('embedding_input_dim: {}, embedding_output_dim: {}, embedding_input_length: {}, '
                 'output_shape: {}'.format(embedding_input_dim, embedding_output_dim, embedding_input_length,
                                           output_shape))

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

    pass

def audio_cnn_model():
    """
    Generate convulutional nerual network for audio data
    :return:
    """

    # Define the dimensions for the mel spectrogram to be fed into the model
    input_shape = (128, 662, 1)

    # Define the model architecture
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1, activation='linear'))

    # Fit model
    filename = '../output/audio_model.h5'
    checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    model.compile(optimizer='Adam', loss='mean_squared_error', callbacks=[checkpoint])

    pass
