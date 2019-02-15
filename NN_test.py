# -*- coding: utf-8 -*-

#from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt
#from iminuit import Minuit
#from probfit import Chi2Regression  # , BinnedChi2 # , BinnedLH#, , UnbinnedLH, , , Extended
#from scipy import stats
import time
import tensorflow as tf
# Import the `transform` module from `skimage`
#from skimage import transform, data
#from skimage.color import rgb2gray
import sys
from astropy.table import Table
import os
import pandas as pd

"""
def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory)
                   if os.path.isdir(os.path.join(data_directory, d))]
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

ROOT_PATH = "/home/duck/PycharmProjects/AtlasExperiment"
train_data_directory = os.path.join(ROOT_PATH, "TrafficSigns/Training/")
test_data_directory = os.path.join(ROOT_PATH, "TrafficSigns/Testing/")

print "train_data_directory:", train_data_directory, type(train_data_directory)

images, labels = load_data(train_data_directory)

# Rescale the images in the `images` array
images28 = [transform.resize(image, (28, 28)) for image in images]

images28 = np.array(images28)

# Convert `images28` to grayscale
images28 = rgb2gray(images28)

print "images28.shape:",images28.shape
# Import `tensorflow`
"""
"*************************************"
tf.reset_default_graph()

complete = 1

Ntrain_exp = 10000
Ntest_exp = 10000
slice_vars = 34
epochs = 20
global_step = tf.Variable(0, trainable=False) # Learning rate is also passed to train, which will increment it
learning_rate0 = 0.001; decay_rate = 0.96

#learning_rate0 * decay_rate ^ (global_step / decay_steps)
with tf.name_scope("Learningrate"):
    learning_rate = tf.train.exponential_decay(learning_rate0 , global_step, epochs, decay_rate)
    #tf.summary.scalar("learningrate", learning_rate)
layers = 3

df_MC05 = pd.read_msgpack("MC05_.msp")


data_electron_full = np.array(df_MC05)

if complete == True:
    #data_electron_full = np.loadtxt('/home/duck/PycharmProjects/AtlasExperiment/Data/MC_SigBkgElectrons_2000000ev.csv', delimiter=',', skiprows=1)
    data_electron_slice     = data_electron_full[:, :slice_vars]
    labels_electron_slice   = (data_electron_full[:, -4]>0.5) & (data_electron_full[:,-2]==2.0)

else:
    #data_electron_full = np.genfromtxt('Data/MC_SigBkgElectrons_500K_FixedTruth.csv', delimiter=',')
    data_electron_slice     = data_electron_full[:Ntrain_exp, :slice_vars]
    labels_electron_slice   = (data_electron_slice[:, -4]>0.5) & (data_electron_slice[:,-2]==2.0)

print "Truth sum signal:", sum(data_electron_full[:, -2]==2.0)

print "Truth freq train:", len(labels_electron_slice[labels_electron_slice>0.5])/float(len(labels_electron_slice))

#plt.hist(data_electron_full[:, -2], bins=len(np.unique(data_electron_full[:, -2])))
#sys.exit(0)

"***************************** Neurological network *****************************************"

# Initialize placeholders
x = tf.placeholder(dtype = tf.float32, shape = [None, slice_vars])
y = tf.placeholder(dtype = tf.int32, shape = [None])

# Flatten the input data
images_flat = tf.contrib.layers.flatten(x)

print "images_flat.shape", images_flat.shape

neu = 40
with tf.name_scope("layer1"):
    layer1 = tf.contrib.layers.fully_connected(images_flat, 200, tf.nn.relu)
with tf.name_scope("layer2"):
    layer2 = tf.contrib.layers.fully_connected(layer1, neu, tf.nn.sigmoid)
with tf.name_scope("layer3"):
    layer3 = tf.contrib.layers.fully_connected(layer2, neu, tf.nn.tanh)
#with tf.name_scope("layer4"):
#    layer4 = tf.contrib.layers.fully_connected(layer3, neu, tf.nn.sigmoid)
#with tf.name_scope("layer5"):
#   layer5 = tf.contrib.layers.fully_connected(layer4, neu, tf.nn.relu)
#with tf.name_scope("layer6"):
#    layer6 = tf.contrib.layers.fully_connected(layer5, neu, tf.nn.sigmoid)
#with tf.name_scope("layer7"):
#    layer7 = tf.contrib.layers.fully_connected(layer6, neu, tf.nn.tanh)
#with tf.name_scope("layer8"):
#    layer8 = tf.contrib.layers.fully_connected(layer7, neu, tf.nn.sigmoid)
#with tf.name_scope("layer9"):
#    layer9 = tf.contrib.layers.fully_connected(layer8, neu, tf.nn.relu)
#with tf.name_scope("layer10"):
#    layer10 = tf.contrib.layers.fully_connected(layer9, neu, tf.nn.sigmoid)
# Fully connected layer
with tf.name_scope("Logits"):
    logits = tf.contrib.layers.fully_connected(layer3, 2, tf.nn.relu)


# Define a loss function
with tf.name_scope("Loss"):
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits))
    tf.summary.scalar("loss", loss)

# Define an optimizer
with tf.name_scope("Optimizer"):
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)

# Convert logits to label indexes
with tf.name_scope("Correct_prediction"):
    correct_pred = tf.argmax(logits, 1)
    #tf.summary.scalar("correct_pred", correct_pred)


# Define an accuracy metric
with tf.name_scope("Accuracy"):
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar("Accuracy", accuracy)

"****************************** Hyperparams tensorboard file-string ****************************************"
# Setting up timestamp for tensoboard filename so it doesn't register files in same folder
reg_t = time.localtime()
reg_time = "{}{}{}".format(int(reg_t[3]/60./60.), int(reg_t[4]/60.), reg_t[5]%60)

# Puts hyperparams in tensorboard filename
if complete:
    hyp_param_str = "Epochs_{}|learning_rate_{}|layers_{}".format(epochs, int(str(learning_rate0)[2:]), layers)+reg_time
else:
    hyp_param_str = "Epochs_{0:d}|Ntrain_exp_{1:d}|Ntest_exp_{2:d}|learning_rate_".format(epochs, Ntrain_exp,
                                Ntest_exp) + str(learning_rate0)[2:] + "_|layers_{0:d}_|".format(layers) + reg_time


"**************************************** RUN ****************************************"
tf.set_random_seed(1234)

# Define a tensorflow session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Tensorboard setup
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("tmp/tensorboard/NN_test/" + hyp_param_str)
    writer.add_graph(sess.graph)
    
    # Calc time left
    current_time = time.localtime()[3]*60*60 + time.localtime()[4]*60 + time.localtime()[5]
    
    
    for i in range(epochs):
        # Calc time left
        time_remaining_sec = (time.localtime()[3]*60*60 + time.localtime()[4]*60 + time.localtime()[5]-current_time) * (epochs - i)
        
        curr_time = time.localtime()[3]*60*60 + time.localtime()[4]*60 + time.localtime()[5] # time after calculating weights
        epoch_time = curr_time-current_time

        # printing information out about epochs left and time left
        print'EPOCH {}/{}| Time btw epochs: {}:{}:{}| Remain: {}:{}:{}'.format( i, epochs, 
                    int(epoch_time/60./60.), int(epoch_time/60.), epoch_time%60,
                    int(time_remaining_sec/60./60.), int(time_remaining_sec/60.), time_remaining_sec%60)
        
        current_time = time.localtime()[3]*60*60 + time.localtime()[4]*60 + time.localtime()[5] # Time before calculating weights

        # Running tensorflow session
        summary,_, accuracy_val = sess.run([merged, train_op, accuracy], feed_dict={x: data_electron_slice, y: labels_electron_slice})
        writer.add_summary(summary, i) # Adds summary to tensorborad file

        print'DONE WITH EPOCH\n'


    "***************************** Testing *****************************************"
    
    data_test = pd.read_msgpack("MC05_.msp")
    if complete == True:
        data_electron_test = np.array(data_test)
        data_electron_slice_test = data_electron_test[1:, :slice_vars]
        
        labels_electron_slice_test   = (data_electron_test[:,-2]==2.0) & (data_electron_test[:,-4]>0.5)

    else:
        data_electron_slice_test     = data_electron_full[Ntrain_exp:Ntrain_exp+Ntest_exp, :slice_vars]
        labels_electron_slice_test   = (data_electron_slice_test[:,-2]==2.0) & (data_electron_slice_test[:,-4]>0.5)
                    
    # Run predictions against the full test set.
    predicted = sess.run([correct_pred], feed_dict={x: data_electron_slice_test})[0]


"****************************** Results ****************************************"

#print "zip(labels_electron_slice_test, predicted):",zip(labels_electron_slice_test, predicted)[:20]

# Calculate correct matches
match_count = sum([int(y == y_) for y, y_ in zip(labels_electron_slice_test, predicted)])

# Calculate the accuracy
accuracy = match_count / float(len(data_electron_slice_test[:,-4]))

# Print the accuracy
print "Truth freq train:", sum(labels_electron_slice)/float(len(labels_electron_slice))
print "Truth freq test :", sum(labels_electron_slice_test)/float(len(data_electron_slice_test[:,-1]))
print("Accuracy: {:.3f}".format(accuracy))


"**************************** Write hyper params to file ******************************************"

# Write hyper params to txt file
if complete == False:

    table_line = Table({'Accuracy': [accuracy], 'Ntrain_exp': [Ntrain_exp], "learning_rate0": [learning_rate0],
                        "epochs": [epochs], "Neurons": [neu]},
                        names=('Accuracy', "Ntrain_exp", "learning_rate0", "epochs", "Neurons"))

    with open('table_hyper_params.txt', mode='a') as f:
            f.seek(0, os.SEEK_END)  # Some platforms don't automatically seek to end when files opened in append mode
            table_line.write(f, format='ascii.no_header')

else:

    table_line = Table({'Accuracy': [accuracy], "epochs": [epochs], "Neurons": [neu], "learning_rate0": [learning_rate0]},
                       names=('Accuracy', "epochs", "Neurons", "learning_rate0"))

    with open('table_FullRun.txt', mode='a') as f:
        f.seek(0, os.SEEK_END)  # Some platforms don't automatically seek to end when files opened in append mode
        table_line.write(f, format='ascii.no_header')