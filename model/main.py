# Required libraries
import os
import random
import pickle
import numpy as np
import tensorflow as tf
import argparse
from keras import backend as K
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import BatchNormalization, Activation, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import datetime

# DKFNet model
from dkf import DKFNet

# Set random seed for reproducibility
random.seed(0)

# Dataset paths
parser = argparse.ArgumentParser(description='Process some folders and rule names.')
parser.add_argument('--rule-name', type=str, default="rule_name", help='Name of the defense rule')
parser.add_argument('--gpu', type=str, default="5", help='Number of GPU')
parser.add_argument('--x-path', type=str, default="/path/to/x_train.pkl", help='Path to the X features file')
parser.add_argument('--y-path', type=str, default="/path/to/y_train.pkl", help='Path to the Y features file')
parser.add_argument('--model-result', type=str, default="/path/to/result.txt", help='Path to the model accuracy log')
args = parser.parse_args()

defense_rule_name = args.rule_name
X_path = args.x_path
y_path = args.y_path
model_result = args.model_result

# GPU setup (optional)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# Hyperparameters
NB_EPOCH = 30
BATCH_SIZE = 32
VERBOSE = 2
LENGTH = 8000
NB_CLASSES = 258
INPUT_SHAPE = (LENGTH, 1)
OPTIMIZER = Adamax(learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

# Function to load data
def LoadData(X_path, y_path):
    with open(X_path, 'rb') as f:
        X = pickle.load(f)
    with open(y_path, 'rb') as f:
        y = pickle.load(f)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
    
    print("train: ", X_train.shape, end='\t')
    print(y_train.shape)
    print("valid: ", X_valid.shape, end='\t')
    print(y_valid.shape)
    print("test: ", X_test.shape, end='\t')
    print(y_test.shape)

    return X_train, y_train, X_valid, y_valid, X_test, y_test

# Load and prepare data
X_train, y_train, X_valid, y_valid, X_test, y_test = LoadData(X_path, y_path)
K.set_image_data_format('channels_first')

print("train: ", X_train.shape, end='\t')
print(y_train.shape)
print("valid: ", X_valid.shape, end='\t')
print(y_valid.shape)
print("test: ", X_test.shape, end='\t')
print(y_test.shape)

# Data normalization
X_train = X_train.astype('float32')
X_valid = X_valid.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_valid = y_valid.astype('float32')
y_test = y_test.astype('float32')

# Reshape data to add channel dimension
# # we need a [Length x 1] x n shape as input to the DF CNN (Tensorflow)
X_train = X_train[:, :,np.newaxis]
X_valid = X_valid[:, :,np.newaxis]
X_test = X_test[:, :,np.newaxis]

print("train: ", X_train.shape, end='\t')
print(y_train.shape)
print("valid: ", X_valid.shape, end='\t')
print(y_valid.shape)
print("test: ", X_test.shape, end='\t')
print(y_test.shape)

# Convert class vectors to binary class matrices
y_train = to_categorical(y_train, NB_CLASSES)
y_valid = to_categorical(y_valid, NB_CLASSES)
y_test = to_categorical(y_test, NB_CLASSES)

# Model building
model = DKFNet.build(input_shape=INPUT_SHAPE, classes=NB_CLASSES)
model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER, metrics=["accuracy"])

# Start training
history = model.fit(X_train, y_train,
                batch_size=BATCH_SIZE, epochs=NB_EPOCH,
                verbose=VERBOSE, validation_data=(X_valid, y_valid))

# Model evaluation
score_test = model.evaluate(X_test, y_test, verbose=VERBOSE)
score_train = model.evaluate(X_train, y_train, verbose=VERBOSE)

print("\nEpoch: ", NB_EPOCH, ", Batch: ", BATCH_SIZE, ", feature: ", defense_rule_name)
print(model.metrics_names)
print("\n=> Train score:", score_train[0])
print("=> Train accuracy:", score_train[1])
print("\n=> Test score:", score_test[0])
print("=> Test accuracy:", score_test[1])

# Save results to a file
with open(model_result, 'w') as file:
    file.write(f"\nEpoch: {NB_EPOCH}, Batch: {BATCH_SIZE}, feature: {defense_rule_name}")
    file.write("\n" + ", ".join(model.metrics_names))
    file.write(f"\n=> Train score: {score_train[0]}")
    file.write(f"=> Train accuracy: {score_train[1]}")
    file.write(f"\n=> Test score: {score_test[0]}")
    file.write(f"=> Test accuracy: {score_test[1]}")