# Convolutional Neural Network

# Part 1 - Building the CNN

# Importing the Kera libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D # images are 2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
# (32, 3, 3) => 32 feature detectors size 3x3.
# input_shape: size of the image data. We normalise this.
# Use ReLU to remove any negative valued pixels to remove linearity
classifier.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Add second Convolutional Layer
classifier.add(Convolution2D(32, (3, 3), activation='relu'))

# Apply Max Pooling to the second Conv layer
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full Connection
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))

# Compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

# Image augmentation -> Prevents overfitting as it adds more data by rotating, zooming, shearing imgags etc.
# This adds more randomness to our training data.
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
)

# Rescale images.
test_datagen = ImageDataGenerator(rescale = 1./255)

# Create a training set
training_set = train_datagen.flow_from_directory(
    'dataset/training_set',
    target_size = (64, 64),
    batch_size = 32,
    class_mode = 'binary'
)

# Create a test set
test_set = test_datagen.flow_from_directory(
    'dataset/test_set',
    target_size = (64, 64),
    batch_size = 32,
    class_mode = 'binary'
)

classifier.fit_generator(
    training_set,
    steps_per_epoch=8000/32,
    epochs=25,
    validation_data=test_set,
    validation_steps=2000/32
)
