# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 19:14:52 2018

@author: maiha
"""
from __future__ import division
import pickle
import numpy as np
import time
from FJDA_Learning import Units, ModelParameters, learn, recall

np.set_printoptions(threshold=np.NaN)
"""from 1 am to 5:59 am is one period, from 6am to 9:59 am is a period, 10am to 2:59 pm,
3 pm to 6:59 pm, 7 pm to 12:59 am

Here I am trying to do training on the period from 10am to 2:59 pm EST which is 15:00 GMT to  
19:59 GMT 
"""
###############################################################################################
# First load the speed data that is saved in File "TrafficData.npy"
# The speedData array is a numpy array and each row is pointing to a road segment.
# And columns are corresponding to time of the day. The first dimension of the array is the day
# speedData dimensions is 43 X 253 X 287
# The starting day is Wed, Dec 13, 2017
###############################################################################################
numberOfLinksUsed = 100
numberOfDays = 31
numberOfTimePointsUsed = 2
numBitsPerSpeed = 1
speedThreshold = 50

speedData = np.load("TrafficData.npy")
# only use weekdays
print(speedData.shape)
speedData = speedData[[0, 1, 2, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 19, 20, 21, 22, 23, 26, 27, 28, 29, 30, 33, 34, 35,
                       36, 37, 40, 41, 42], 0 : numberOfLinksUsed, 50 : 52]
print(speedData.shape)
raw_patterns = np.zeros((numberOfLinksUsed, numberOfTimePointsUsed * numberOfDays), float)
for i in range(31):
    raw_patterns[:, 2 * i : 2 * i + 2] = speedData[i, 0:100, 0:2]
print(raw_patterns)
raw_patterns = np.append(raw_patterns.T, raw_patterns.T, axis=1)


# Prepare the test data
pickle_out = open("test_week_speed_Jan29_2.pickle", "rb")
raw_speed_test = pickle.load(pickle_out)
raw_speed_test = raw_speed_test[0:100, 50:52]
raw_speed_test = raw_speed_test.T
pickle_out.close()

if numBitsPerSpeed ==2:
    #use two bits to describe the trafic speed
    #---------------------------------
    # 0-40 km/h   |       0  0
    #--------------------------------
    # 40-50 km/h  |       0  1
    #--------------------------------
    # 50-65 km/h  |       1  0
    #--------------------------------
    # 65-99 km/h  |       1  1
    #--------------------------------

    patterns = np.zeros((raw_patterns.shape[0], raw_patterns.shape[1]*2) , bool)
    for i in range(raw_patterns.shape[0]):
        for j in range(raw_patterns.shape[1]):
            if raw_patterns[i,j] <= 40 :
                patterns[i, 2 * j]=0
                patterns[i, 2 * j + 1] = 0
            else:
                if raw_patterns[i,j] > 40 and raw_patterns[i,j]<=50 :
                    patterns[i, 2 * j] = 0
                    patterns[i, 2 * j + 1] = 1
                else:
                    if raw_patterns[i, j] > 50 and raw_patterns[i, j] <= 65:
                        patterns[i, 2 * j] = 1
                        patterns[i, 2 * j + 1] = 0
                    else:
                        patterns[i, 2 * j] = 1
                        patterns[i, 2 * j + 1] = 1

    raw_speed_test = raw_speed_test.T
    speed_test = np.zeros((raw_speed_test.shape[0], raw_speed_test.shape[1] * 2), bool)
    for i in range(raw_speed_test.shape[0]):
        for j in range(raw_speed_test.shape[1]):
            if raw_speed_test[i, j] <= 40:
                speed_test[i, 2 * j] = 0
                speed_test[i, 2 * j + 1] = 0
            else:
                if raw_speed_test[i, j] > 40 and raw_speed_test[i, j] <= 50:
                    speed_test[i, 2 * j] = 0
                    speed_test[i, 2 * j + 1] = 1
                else:
                    if raw_speed_test[i, j] > 50 and raw_speed_test[i, j] <= 65:
                        speed_test[i, 2 * j] = 1
                        speed_test[i, 2 * j + 1] = 0
                    else:
                        speed_test[i, 2 * j] = 1
                        speed_test[i, 2 * j + 1] = 1

else:
    patterns = raw_patterns > speedThreshold
    speed_test = raw_speed_test > speedThreshold

##############################################################################################
##############################################################################################
# Set the parameters

numberOfDANodes = 1024
numInput = numBitsPerSpeed * numberOfLinksUsed
numOutput = numBitsPerSpeed * numberOfLinksUsed
numHidden = 20

# Annealer Parameters
tmp_st = 30
tmp_decay = 0.2 # temp steps: 30, 24, 19, 15, 12, 10
tmp_interval = 10
DA_num_iterations = 70

# BM parameters
units = Units(numInput, numOutput, numHidden)
modelParameters = ModelParameters(units.numUnits, numberOfDANodes)
#############################################################################################
# Defining connections:
# Connections is a matrix of size numUnits x numUnits. The element "Connections[a,b]" represents the
# number of connections between nodes (a,b).
# If two nodes a and b are not connected, then Connections[a,b] = -1.
# el connections faydet-ha enaha teb2a 1 - identity matrix, such
# that the off-diagonal elements are non-zeros. There is another constraint
# however, which is there are no connections between the input and output units

connections = np.zeros((units.numUnits, units.numUnits), dtype=np.int)
for i in range(units.numInputUnits):
    for j in range(i + 1, units.numInputUnits):
        connections[i, j] = 1
    for j in range(1, units.numHiddenUnits + 1):
        connections[i, -j] = 1

for i in range(units.numOutputUnits):
    for j in range(i + 1, units.numOutputUnits):
        connections[i + units.numInputUnits, j + units.numInputUnits] = 1
    for j in range(1, units.numHiddenUnits + 1):
        connections[i + units.numInputUnits, -j] = 1

for i in range(units.numHiddenUnits, 0, -1):
    for j in range(i - 1, 0, -1):
        connections[-i, -j] = 1

connections = connections[::-1].T[::-1]

valid = np.nonzero(connections)
numConnections = np.size(valid[0])
connections_index = connections
connections_index[valid] = np.arange(1, numConnections + 1)
connections_index = connections_index + connections_index.T - 1
###########################################################################################
###########################################################################################

f_particles = np.random.choice([0, 1], size=(patterns.shape[0], numInput))
fantasy_particles = np.append(f_particles, f_particles, axis=1)
iterations = 2

#################################################################
# Learn the distribution
# learn for 'iterations' iterations and then check to see how well the
# patterns are learned by measuring the distance
# between each training pattern and the recalled pattern.
start_time = time.time()
repeat = 1
dist = np.zeros((repeat, patterns.shape[0]), float)
for i in range(repeat):
    learn(patterns, fantasy_particles, modelParameters, units, numConnections,
          connections_index, iterations, tmp_st, tmp_decay, tmp_interval, DA_num_iterations)
    #elapsed_time = time.time() - start_time
    np.savez('ModelParameters_'+str(i), fantasy_particles=fantasy_particles, modelParameters=modelParameters,
             units=units, numConnections=numConnections, connections_index=connections_index)
    for j in range(patterns.shape[0]):
        recovered = recall(patterns[j,0:numInput], units, modelParameters,
                           tmp_st, tmp_decay, tmp_interval, DA_num_iterations)
        dist[i,j] = np.linalg.norm(recovered - patterns[j,0:numInput])
    np.save('recoveredVectorsDistances', dist)
################################################################
# Test the model's prediction

recovered = recall(speed_test[1, :], units, modelParameters, tmp_st, tmp_decay, 3*tmp_interval, 3*DA_num_iterations)


print ("Recovered: ", recovered)
print ("Actual Speed: ", speed_test[1,:])
print ("not matching: " , recovered != speed_test[1,:])
#print elapsed_time, elapsed_time_testing

from tempfile import TemporaryFile
recoveredSpeed = TemporaryFile()
np.savez("recoveredSpeed_4state_2", recovered=recovered, ActualSpeed=speed_test[1,:])
