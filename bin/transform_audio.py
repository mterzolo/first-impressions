import lib
import pickle


X_train, y_train = lib.audio2melspec(data_split='training', num_samples=5)
X_test, y_test = lib.audio2melspec(data_split='test', num_samples=5)

X_train = X_train.reshape(X_train.shape[0], 128, 662, 1)
X_test = X_test.reshape(X_test.shape[0], 128, 662, 1)

# Save files as pickle objects
with open('../data/audio_data/pickle_files/X_train.pkl', 'wb') as file:
    pickle.dump(X_train, file)
with open('../data/audio_data/pickle_files/y_train.pkl', 'wb') as file:
    pickle.dump(y_train, file)
with open('../data/audio_data/pickle_files/X_test.pkl', 'wb') as file:
    pickle.dump(X_test, file)
with open('../data/audio_data/pickle_files/y_test.pkl', 'wb') as file:
    pickle.dump(y_test, file)