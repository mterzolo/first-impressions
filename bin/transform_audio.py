import librosa

y, sr = librosa.load('../data/audio_data/training_data/_0bg1TLPP-I.004.mp3')

y_rmse = librosa.feature.rmse(y=y)

y_mfcc = librosa.feature.mfcc(y=y)