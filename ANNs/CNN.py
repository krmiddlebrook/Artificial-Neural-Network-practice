# Importing the Keras libraries and packages
from keras.models import Sequential  # initialise our neural network model as a sequential network
from keras.layers import Conv2D      # perform the convolution operation
from keras.layers import MaxPooling2D # used for pooling operation
from keras.layers import Flatten      # used for Flattening the 2-D arrays into one single long continous linear vector
from keras.layers import Dense        # used to perform the full connection of the neural network
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

# create an object of the Sequential class
classifier = Sequential()

# now for the Convolution step,
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# reduce the size of the images by pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# now convert the 2-D array of pooled image pixels into a 1-D single vector
classifier.add(Flatten())

# create a fully connected layer that we will later use to create a hidden layer
classifier.add(Dense(units = 128, activation = 'relu'))     # the optimal # for units is determined through trial-and-error


# create an output layer. In our case it is a binary classifier because the img either contains a dog or a cat
classifier.add(Dense(units = 1, activation = 'sigmoid'))    # final layer should only contain 1 node i.e. units=1


# now compile our CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) # Optimizer parameter is to choose the stochastic gradient descent algorithm. Loss parameter is to choose the loss function. Metrics parameter is to choose the performance metric.


# Now create synthetic data out of the same images by performing different type of operations on these images like flipping, rotating, blurring, etc.
# this will help generalize our training data so our CNN doesn't overfit our data.
train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('images/training_set', target_size = (64, 64), batch_size = 32, class_mode = 'binary')
test_set = test_datagen.flow_from_directory('images/test_set', target_size = (64, 64), batch_size = 32, class_mode = 'binary')

# Now we can fit the data to our model
classifier.fit_generator(training_set, steps_per_epoch = 8000, epochs = 25, validation_data = test_set, validation_steps = 2000)
# steps_per_epoch’ holds the number of training images. ‘epochs’, a single epoch is a single step in training a neural network
# in other words when a neural network is trained on every training samples only in one pass we say that one epoch is finished. So training process should consist more than one epochs.
#  In this case we have defined 25 epochs.



# Now let's make new predictions from our trained model!
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64)) # test_image holds the image that needs to be tested on the CNN and prepare it for our model by converting the img size to 64x64
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'