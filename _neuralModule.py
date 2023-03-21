import numpy as np
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import os
import csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

N_FEATURES = 0


def create_baseline(input_size=0):
    # create model
    model = Sequential()
    model.add(
        Dense(20, input_dim=N_FEATURES, kernel_initializer="normal", activation="relu")
    )
    model.add(Dense(10, input_dim=20, kernel_initializer="normal", activation="relu"))
    model.add(Dense(5, input_dim=10, kernel_initializer="normal", activation="relu"))
    model.add(Dense(1, kernel_initializer="normal", activation="sigmoid"))
    # Compile model. We use the the logarithmic loss function, and the Adam gradient optimizer.
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


filename = "neural_set_bwv.csv"

features, outputs = [], []

row_len = 0
n_rows = 0

with open(filename) as dataset:
    reader = csv.reader(dataset)
    for line in reader:
        n_rows += 1
        numbers = [float(_) for _ in line]
        row_len = len(numbers)
        features += numbers[:-1]
        outputs.append(numbers[-1])

N_FEATURES = row_len - 1

print(f"Row len = {row_len-1} + 1")
print(f"Rows {n_rows}")

features = np.array(features)
features = features.reshape(n_rows, N_FEATURES)
outputs = np.asarray(outputs)


# print(len(features[0]))
# print(features.shape)
# exit()
model = create_baseline()

earlyStopping = EarlyStopping(monitor="loss", patience=10, verbose=0, mode="min")
mcp_save = ModelCheckpoint(
    "neuralPBRose.hdf5", save_best_only=True, monitor="loss", mode="min"
)
reduce_lr_loss = ReduceLROnPlateau(
    monitor="loss", factor=0.1, patience=7, verbose=1, min_delta=1e-4, mode="min"
)

estimator = KerasClassifier(
    model=create_baseline,
    epochs=100,
    batch_size=5,
    verbose=1,
    callbacks=[earlyStopping, mcp_save, reduce_lr_loss],
)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, features, outputs, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
