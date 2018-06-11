import pickle
import numpy as np
from sklearn.metrics import mean_squared_error

with open('../data/meta_data/annotation_training.pkl', 'rb') as f:
    annotation_training = pickle.load(f, encoding='latin1')

y_train = np.fromiter(annotation_training['interview'].values(), dtype=float)
y_preds = np.full((len(y_train),), np.mean(y_train))

print(np.sqrt(mean_squared_error(y_train, y_preds)))
print(mean_squared_error(y_train, y_preds))