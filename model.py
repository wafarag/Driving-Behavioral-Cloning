# This file to constructing an end-to-end Neural Network Model for Behavoir Coloning for Autonomous Driving
#
#       Done by:   Wael Farag
#
#==========================================================================================================

# Import Libraries
import os
import csv
import cv2
import json
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential, Model
from keras.layers import Lambda
from keras.layers import Cropping2D
from keras.layers import Convolution2D, Flatten, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation
from keras.models import model_from_json
from keras.optimizers import Adam

#===================================================================================
# Important Parameters and Initializations
BATCH_SIZE = 120
EPOCHS     = 2
LEARNING_RATE = 0.0002            # for Adam's Algorithm
LEFT_ANGLE_CORRECTION  = 0.15     # Radians
RIGHT_ANGLE_CORRECTION = -0.15    # Radians
IMAGES_PER_CSV_LINE    = 6        # No. of images (samples) produces from each drive_log CSV file
PKeep                  = 0.7      # Keep Probability - Dropout Layer

# Adjustement for proper runnig of loops within the script
ADJ_BATCH_SIZE = BATCH_SIZE / IMAGES_PER_CSV_LINE

data_file_path = "C:/work/MyFiles/Research/DeepLearning/CarND/BehavioralCloning/data/"
CSV_file = "driving_log.csv"
Model_weights_file = "model_weights.h5"
Model_file = "model.h5"

#=========================================================================================
#================================== Start of Data Visualization ==========================
# read and visualize data

CSV_lines = []
with open(data_file_path + CSV_file) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        CSV_lines.append(line)

# CSV_lines = CSV_lines[1:np.shape(CSV_lines)[0]]    # only used for supplied UDACITY Data
# CSV_lines = CSV_lines[0:8000]
# print(np.shape(CSV_lines))
train_samples, validation_samples = train_test_split(CSV_lines, test_size=0.2)

# print(np.shape(train_samples))
# print(np.shape(validation_samples))

## The following code is used to visualize indivial images in the data with the corresponding steering angle

CSV_data_line = 5000  # Selected Data Line
print('\nThe Data for CSV Line ', CSV_data_line, ' is:')
center_image = mpimg.imread(data_file_path + 'IMG/' + CSV_lines[CSV_data_line][0].split('\\')[-1] )
plt.subplot(1, 3, 1)
plt.imshow(center_image)
left_image = mpimg.imread(data_file_path + 'IMG/' + CSV_lines[CSV_data_line][1].split('\\')[-1])
plt.subplot(1, 3, 2)
plt.imshow(left_image)
right_image = mpimg.imread(data_file_path + 'IMG/' + CSV_lines[CSV_data_line][2].split('\\')[-1])
plt.subplot(1, 3, 3)
plt.imshow(left_image)
plt.show()
print('\nThe steering angle is: ', CSV_lines[CSV_data_line][3])

#Plot the Histogram of the Steering angles for the Training Data
CSV_lines = np.array(CSV_lines)
angles = CSV_lines[:,3]

hist_angles=[]
for item in angles:
    hist_angles.append(float(item))

plt.hist(hist_angles)
plt.title("Steering Angle Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()
#================================== End of Data Visualization ==========================
#=======================================================================================
#========================== The Data generator function ================================
def generator(samples, batch_size=32):
    num_samples = len(samples)
    # Loop forever so the generator never terminates
    while 1:
        shuffle(samples)
        batch_size = int(batch_size)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                #center_image_name = data_file_path +'IMG/' + batch_sample[0].split('/')[-1]
                center_image_name = data_file_path + 'IMG/' + batch_sample[0].split('\\')[-1]
                #print(center_image_name)
                center_image = mpimg.imread(center_image_name)
                center_angle = float(batch_sample[3])
                center_image_flipped = cv2.flip(center_image, 1)
                center_angle_flipped = (-1)* center_angle
                images.append(center_image)
                angles.append(center_angle)
                images.append(center_image_flipped)
                angles.append(center_angle_flipped)

                #left_image_name = data_file_path +'IMG/' + batch_sample[1].split('/')[-1]
                left_image_name = data_file_path + 'IMG/' + batch_sample[1].split('\\')[-1]
                left_image = mpimg.imread(left_image_name)
                left_angle = float(batch_sample[3]) + LEFT_ANGLE_CORRECTION
                left_image_flipped = cv2.flip(left_image, 1)
                left_angle_flipped = (-1) * left_angle
                images.append(left_image)
                angles.append(left_angle)
                images.append(left_image_flipped)
                angles.append(left_angle_flipped)

                #right_image_name = data_file_path +'IMG/' + batch_sample[2].split('/')[-1]
                right_image_name = data_file_path + 'IMG/' + batch_sample[2].split('\\')[-1]
                right_image = mpimg.imread(right_image_name)
                right_angle = float(batch_sample[3]) + RIGHT_ANGLE_CORRECTION
                right_image_flipped = cv2.flip(right_image, 1)
                right_angle_flipped = (-1) * right_angle
                images.append(right_image)
                angles.append(right_angle)
                images.append(right_image_flipped)
                angles.append(right_angle_flipped)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
#=========================================================================================

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=ADJ_BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=ADJ_BATCH_SIZE)
#

#=========================================================================================
#                 The Neural Network Model
#=========================================================================================

# The WAF-NVIDIA Model

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation
row, col, ch = 160, 320, 3  # input image format
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(row, col, ch), output_shape=(row, col, ch)))
# Cropping the images, cut 70 pixels from the top and 25 pixels from the bottom,
# The new height will be (160-70) - 25 = 65
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(row, col, ch)))
model.add(Convolution2D(nb_filter=24, nb_row=5, nb_col=5, subsample=(2,2), border_mode='valid', activation='relu'))
model.add(Convolution2D(nb_filter=36, nb_row=5, nb_col=5, subsample=(2,2), border_mode='valid', activation='relu'))
model.add(Convolution2D(nb_filter=48, nb_row=5, nb_col=5, subsample=(2,2), border_mode='valid', activation='relu'))
model.add(Convolution2D(nb_filter=64, nb_row=3, nb_col=3, border_mode='valid', activation='relu'))
model.add(Convolution2D(nb_filter=64, nb_row=3, nb_col=3, border_mode='valid', activation='relu'))
model.add(Flatten())
model.add(Dropout(PKeep))
model.add(Dense(200))
model.add(Dropout(PKeep))
model.add(Dense(100))
model.add(Dropout(PKeep))
model.add(Dense(20))
model.add(Dropout(PKeep))
model.add(Dense(1))
# load the wieghts of the previous trained model - Transfer Learning
model.load_weights(data_file_path + Model_weights_file)

# Training using ADAM's Algorithm with Adjustable learning rate
model.compile(loss='mse', optimizer=Adam(lr=LEARNING_RATE))

history_object = model.fit_generator(train_generator, samples_per_epoch= len(train_samples)*IMAGES_PER_CSV_LINE,
                                     validation_data=validation_generator,
                                     nb_val_samples=len(validation_samples)*IMAGES_PER_CSV_LINE,
                                     nb_epoch=EPOCHS, verbose=1)


################################################################
# Save the model and weights
################################################################
model_json = model.to_json()
with open(data_file_path + "model.json", "w") as json_file:
    json.dump(model_json, json_file)
model.save_weights(data_file_path + Model_weights_file)
model.save(data_file_path + Model_file)
print("Saved model to disk")


### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

#=========================================================================================


