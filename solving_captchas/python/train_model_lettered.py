import cv2
import pickle
import os.path
import numpy as np
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense
from helpers import resize_to_fit
import itertools
import keras.callbacks


LETTER_IMAGES_FOLDER = "../../cleaning_captchas/python/extracted_letters"
MODEL_FILENAME = "captcha_model.hdf5"
MODEL_LABELS_FILENAME = "model_labels.dat"


# initialize the data and labels
data = []
labels = []
numSamplesPerLetter = 276
for directory in os.listdir(LETTER_IMAGES_FOLDER):
    letter_dir = LETTER_IMAGES_FOLDER + "/" + directory
    for image_file in os.listdir(letter_dir)[numSamplesPerLetter:]:
        # Load the image and convert it to grayscale
        image = cv2.imread(letter_dir + "/" + image_file, cv2.IMREAD_GRAYSCALE)
        # Resize the letter so it fits in a 50x50 pixel box
        image = resize_to_fit(image, 50, 50)

        # Add a third channel dimension to the image to make Keras happy
        image = np.expand_dims(image, axis=2)

        # Grab the name of the letter based on the folder it was in
        label = directory

        # Add the letter image and it's label to our training data
        data.append(image)
        labels.append(label)

# scale the raw pixel intensities to the range [0, 1] (this improves training)
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# Split the training data into separate train and test sets
(X_train, X_test, Y_train, Y_test) = train_test_split(
    data, labels, test_size=0.4, random_state=0)

# Convert the labels (letters) into one-hot encodings that Keras can work with
lb = LabelBinarizer().fit(Y_train)
Y_train = lb.transform(Y_train)
Y_test = lb.transform(Y_test)

# Save the mapping from labels to one-hot encodings.
# We'll need this later when we use the model to decode what it's predictions mean
with open(MODEL_LABELS_FILENAME, "wb") as f:
    pickle.dump(lb, f)

# Build the neural network!
model = Sequential()

# First convolutional layer with max pooling
model.add(Conv2D(20, (5, 5), padding="same",
                 input_shape=(50, 50, 1), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Second convolutional layer with max pooling
model.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

numDenseNodes = 500
# Hidden layer with 500 nodes
model.add(Flatten())
model.add(Dense(numDenseNodes, activation="relu"))

# Output layer with 20 nodes (one for each possible letter/number we predict)
model.add(Dense(20, activation="softmax"))

# Ask Keras to build the TensorFlow model behind the scenes
model.compile(loss="categorical_crossentropy",
              optimizer="adam", metrics=["accuracy"])
numEpochs = 10
tbCallBack = keras.callbacks.TensorBoard(log_dir="./logs/train/numDense" + str(numDenseNodes) +
                                         "/numEpochs"+str(numEpochs), histogram_freq=10, write_graph=True, write_images=False)

# Train the neural network
model.fit(X_train, Y_train,
          validation_data=(X_test, Y_test),
          callbacks=[tbCallBack],
          batch_size=32, epochs=numEpochs, verbose=1)

# Save the trained model to disk
model.save(MODEL_FILENAME)
