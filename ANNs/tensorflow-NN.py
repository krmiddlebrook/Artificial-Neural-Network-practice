import tensorflow as tf
import os
import numpy as np
from skimage import transform
from skimage import data
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import random


# load data into your workspace
def load_data(data_directory):
    # Get all subdirectories of data_dir. Each represents a label.
    directories = [d for d in os.listdir(data_directory)
                   if os.path.isdir(os.path.join(data_directory, d))]
    # Loop through the label directories and collect the data in
    # two lists, labels and images.
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f)
                      for f in os.listdir(label_directory)
                      if f.endswith(".ppm")]
        for f in file_names:
            images.append(data.imread(f))
            labels.append(int(d))
    return images, labels

# input the path were your test and training data is stored
ROOT_PATH = "/Users/kaimiddlebrook/Documents/codingstuff/ANNs/"

# add the training and test data to the root path
train_data_directory = os.path.join(ROOT_PATH, 'belgium-sign-data/Training')
test_data_directory =  os.path.join(ROOT_PATH, 'belgium-sign-data/Testing')

# make a list of labels and a list of images with the load_data function
images, labels = load_data(train_data_directory)


# NOW TIME FOR SOME DATA INSPECTION
# convert the images and labels lists to np arrays
images_array = np.array(images)
labels_array = np.array(labels)

# print the 'images' dims
print("'images' dimensions: " + str(images_array.ndim))

# print the number of 'images''s elements
print("number of 'images'\'s' elements: " + str(images_array.size))

# print the first instance of 'images'
print("The first instance of 'images': " + str(images_array[0]))

# Print the `labels` dimensions
print("'labels' dims: " + str(labels_array.ndim))

# Print the number of `labels`'s elements
print("Number of 'labels'\'s elements: " + str(labels_array.size))

# Count the number of labels
print("The number of labels: " + str(len(set(labels_array))))

# Make a histogram with 62 bins of the `labels` data
# plt.hist(labels, 62)

# Show the plot
# plt.show()


# Determine the (random) indexes of the images that you want to see
traffic_signs = [random.randint(1,4575), random.randint(1,4575), random.randint(1,4575), random.randint(1,4575)]

# Fill out the subplots with the random images and add shape, min and max values
for i in range(len(traffic_signs)):
    # plt.subplot(1, 4, i+1)
    # plt.axis('off')
    # plt.imshow(images[traffic_signs[i]])
    # plt.subplots_adjust(wspace=0.5)
    # plt.show()
    print("shape: {0}, min: {1}, max: {2}".format(images[traffic_signs[i]].shape,
                                                  images[traffic_signs[i]].min(),
                                                  images[traffic_signs[i]].max()))


"""-----------FEATURE EXTRACTION--------------"""
# Resize the images
images28 = [transform.resize(image, (28, 28)) for image in images]
images28 = np.array(images28)
# print(images28.ndim)

# convert images to grayscale
images28 = rgb2gray(images28)


"""-----------MODELING THE NEURAL NETWORK USING TENSORFLOW---------------"""
#----Building the NN-----#
# initialize placeholders
x = tf.placeholder(dtype = tf.float32, shape = [None, 28, 28])
y = tf.placeholder(dtype = tf.int32, shape = [None])

# flatten the input data
images_flat = tf.contrib.layers.flatten(x)

# fully connected layer
logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)

# define the loss function
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits))

# define the optimizer
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# convert logits to label indexes
correct_pred = tf.argmax(logits, 1)

# define an accuracy metric
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# check the values of some of our variables
print("images_flat: ", images_flat)
print("logits: ", logits)
print("loss: ", loss)
print("predicted_labels: ", correct_pred)

"""-------------TIME TO RUN OUR NEURAL NET!!!!-------------"""
# start a session
sess = tf.Session()

# initialize the variables
sess.run(tf.global_variables_initializer())

# train the model using 200 epcohs/training loops
for i in range(201):
        print('EPOCH', i)
        _, accuracy_val = sess.run([train_op, accuracy], feed_dict={x: images28, y: labels})
        if i % 10 == 0:
            print("Loss: ", loss)
        print('DONE WITH EPOCH')


"""--------TIME TO EVALUATE OUR NEURAL NET--------"""
# Pick 10 random images
sample_indexes = random.sample(range(len(images28)), 10)
sample_images = [images28[i] for i in sample_indexes]
sample_labels = [labels[i] for i in sample_indexes]

# Run the "predicted_labels" op.
predicted = sess.run([correct_pred], feed_dict={x: sample_images})[0]

# Print the real and predicted labels
print(sample_labels)
print(predicted)

# Display the predictions and the ground truth visually.
fig = plt.figure(figsize=(10, 10))
for i in range(len(sample_images)):
    truth = sample_labels[i]
    prediction = predicted[i]
    plt.subplot(5, 2,1+i)
    plt.axis('off')
    color='green' if truth == prediction else 'red'
    plt.text(40, 10, "Truth:        {0}\nPrediction: {1}".format(truth, prediction),
             fontsize=12, color=color)
    plt.imshow(sample_images[i])

plt.show()


# Load the test data
test_images, test_labels = load_data(test_data_directory)

# Transform the images to 28 by 28 pixels
test_images28 = [transform.resize(image, (28, 28)) for image in test_images]

# Convert to grayscale
test_images28 = rgb2gray(np.array(test_images28))

# Run predictions against the full test set.
predicted = sess.run([correct_pred], feed_dict={x: test_images28})[0]

# Calculate correct matches
match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predicted)])

# Calculate the accuracy
accuracy = match_count / len(test_labels)

# Print the accuracy
print("Accuracy: {:.3f}".format(accuracy))