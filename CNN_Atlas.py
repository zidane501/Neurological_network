#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May 19 08:31:41 2018

@author: duck
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py



train_dataset = h5py.File('/home/duck/PycharmProjects/AtlasExperiment/Data/images_00000000.h5', "r")["egamma"]

print train_dataset.shape
a_group_key = list(train_dataset)

# Get the data
data = np.array(train_dataset[a_group_key])

"""
for key in train_dataset.keys():
    print "Key:", key
print data.shape
print len(data[0])#,data[0]
d = np.array(data[15])
for i in range(len(data[0])):
    print i, data[0][i].shape
#plt.imshow(d)
"""