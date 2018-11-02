import cv2
import pickle
import os.path
import os
import numpy as np
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense
# from helpers import resize_to_fit


# LETTER_IMAGES_FOLDER = "extracted_letters"
TRAIN_FOLDER = "../data/solution_cleaned/"
MODEL_FILENAME = "../data/captcha_model.hdf5"
MODEL_LABELS_FILENAME = "../data/model_labels.dat"


# initialize the data and labels
data = []
labels = []

# loop over the input images
for image_file in os.listdir(TRAIN_FOLDER):
    # Load the image and convert it to grayscale
    image = cv2.imread(TRAIN_FOLDER+image_file, cv2.IMREAD_GRAYSCALE)

    # Add a third channel dimension to the image to make Keras happy
    image = np.expand_dims(image, axis=2)

    # Grab the name of the letter based on the folder it was in
    # label = image_file.split(os.path.sep)[-2]
    label = image_file.replace('_duplicate', '').replace('.jpg', '')
    # Add the letter image and it's label to our training data
    data.append(image)
    labels.append(label)

print(data)
# scale the raw pixel intensities to the range [0, 1] (this improves training)
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print(data.shape)
# Split the training data into separate train and test sets
(X_train, X_test, Y_train, Y_test) = train_test_split(
    data, labels, test_size=0.25, random_state=0)

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
                 input_shape=(44, 200, 1), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Second convolutional layer with max pooling
model.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Hidden layer with 500 nodes
model.add(Flatten())
model.add(Dense(64000000, activation="relu"))

# Output layer with 32 nodes (one for each possible letter/number we predict)
model.add(Dense(573, activation="softmax"))

# Ask Keras to build the TensorFlow model behind the scenes
model.compile(loss="categorical_crossentropy",
              optimizer="adam", metrics=["accuracy"])

# Train the neural network
model.fit(X_train, Y_train, validation_data=(
    X_test, Y_test), batch_size=32, epochs=10, verbose=1)

# Save the trained model to disk
model.save(MODEL_FILENAME)
