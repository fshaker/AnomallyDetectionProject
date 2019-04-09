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
"""
By: Fariba
I am trying to find the best parameters for DA. I use a small network and four vectors to learn. 
My aim is to find out if one small vector can be learned by recovering the same vector in test time.
Auto-encoder

"""



numberOfDANodes = 1024
numInput = 4
numOutput = 4
numHidden = 4
###########################################
# Annealing Parameters
tmp_st = 20
tmp_decay = 0.16 # temp steps: 20, 17, 14, 12, 10
tmp_interval = 4
DA_num_iterations = 20

############################################
patterns = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
newPattern1 = [0, 1, 0, 0]
newPattern2 = [1, 0, 0, 0]
###########################################

start_time = time.time()
units = Units(numInput, numOutput, numHidden)
modelParameters = ModelParameters(units.numUnits, numberOfDANodes)

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


f_particles = np.random.choice([0, 1], size=(patterns.shape[0], numInput))
fantasy_particles = np.append(f_particles, f_particles, axis=1)
iterations = 2000
patterns = np.append(patterns, patterns, axis=1)
print("iterations=",2000)
#############################################################################################
#Learn the distribution
# learn(patterns, fantasy_particles, modelParameters, units, numConnections, connections_index,
#      iterations, tmp_st, tmp_decay, tmp_interval, DA_num_iterations)
#
# elapsed_time = time.time() - start_time
#print("elapsed_time", elapsed_time)

############################################################################################
# following are the parameters learned after 1000 iterations with four bits training vectors and two
# hidden units
#                     [[1, 0, 0, 0],
#                      [0, 1, 0, 0],
#                      [0, 0, 1, 0],
#                      [0, 0, 0, 1]]
#
# modelParameters.weights=np.asarray( [[0,-332,  94,  96,  94, 94,106, 102, 104, 104],
#                                      [0,   0,  94,  98,  96, -2, 98, 100,  98,  -2],
#                                      [0,   0,   0,-134,-132,-96,  0,   0,   0,   0],
#                                      [0,   0,   0,   0,-130,-96,  0,   0,   0,   0],
#                                      [0,   0,   0,   0,   0,-96,  0,   0,   0,   0],
#                                      [0,   0,   0,   0,   0,  0,  0,   0,   0,   0],
#                                      [0,   0,   0,   0,   0,  0,  0,-132,-128,-106],
#                                      [0,   0,   0,   0,   0,  0,  0,   0,-132,-106],
#                                      [0,   0,   0,   0,   0,  0,  0,   0,   0,-108],
#                                      [0,   0,   0,   0,   0,  0,  0,   0,   0,   0]])
# modelParameters.weights = modelParameters.weights + modelParameters.weights.T
# ########################################################################################
# w=np.asarray([[   0, -598, -614, -584,    0,    0,    0,    0,  636,  598, -676,  614],
#               [-598,    0, -588, -582,    0,    0,    0,    0,  610,  606,  596, -642],
#               [-614, -588,    0, -560,    0,    0,    0,    0,  630, -646,  594,  606],
#               [-584, -582, -560,    0,    0,    0,    0,    0, -664,  600,  606,  608],
#               [   0,    0,    0,    0,    0, -566, -570, -570,  610,  590, -668,  608],
#               [   0,    0,    0,    0, -566,    0, -594, -584,  620,  592,  634, -694],
#               [   0,    0,    0,    0, -570, -594,    0, -586,  600, -650,  618,  596],
#               [   0,    0,    0,    0, -570, -584, -586,    0, -676,  608,  612,  610],
#               [ 636,  610,  63,  -664,  610,  620,  600, -676,    0,   26,   22,   12],
#               [ 598,  606, -646,  600,  590,  592, -650,  608,   26,    0,   22,   12],
#               [-676,  596,  594,  606, -668,  634,  618,  612,   22,   22,    0,   34],
#               [ 614, -642,  606,  608,  608, -694,  596,  610,   12,   12,   34,    0]])
# modelParameters.weights = w[::-1].T[::-1]
# #########################################################################################
modelParameters.weights = np.asarray([[   0,   20,   46,   46,  634,  646, -674,  648,  624,  618, -708,  620],
 [  20,    0,   34,   30 ,-688,  646,  626,  650, -684,  624,  642,  630],
 [  46,   34,    0,   40,  624, -712,  616,  642,  644, -674,  640,  630],
 [  46,   30,   40,    0 , 626 , 634,  610, -704,  638,  630,  652, -670],
 [ 634, -688,  624,  626,    0, -596, -594, -602,    0,    0,    0,    0],
 [ 646,  646, -712,  634, -596,    0, -598, -624,    0,    0,    0,    0],
 [-674,  626,  616,  610, -594, -598,    0, -606,    0,    0,    0,    0],
 [ 648,  650,  642, -704, -602, -624, -606,    0,    0,    0,    0,    0],
 [ 624, -684,  644,  638,    0,    0,    0,    0,    0, -606, -620, -614],
 [ 618,  624, -674,  630 ,   0 ,   0 ,   0 ,   0 ,-606 ,   0, -614, -608],
 [-708,  642 , 640 , 652,    0,    0 ,   0,    0, -620, -614,    0, -606],
 [ 620,  630,  630, -670,    0 ,   0 ,   0 ,   0, -614, -608 ,-606 ,   0]])
# Test the model's prediction
correct1 = 0
correct2 = 0
correct3 = 0
correct4 = 0
for i in range(25):
    #start_time = time.time()
    recovered = recall(patterns[0,0:4], units, modelParameters, tmp_st, tmp_decay, 2*tmp_interval, 8+2*DA_num_iterations)
    #elapsed_time_testing = time.time() - start_time
    print ("Recovered: ", recovered)
    #correct1 = correct1 + int(np.array_equal(recovered, patterns[0, 0:4]))

print ("Actual: ", patterns[0,0:4])

for i in range(25):
    #start_time = time.time()
    recovered = recall(patterns[1,0:4], units, modelParameters, tmp_st, tmp_decay, 2*tmp_interval, 8+2*DA_num_iterations)
    #elapsed_time_testing = time.time() - start_time
    print ("Recovered: ", recovered)
    #correct2 = correct2 + int(np.array_equal(recovered, patterns[1, 0:4]))

print ("Actual: ", patterns[1,0:4])

for i in range(25):
    #start_time = time.time()
    recovered = recall(patterns[2,0:4], units, modelParameters, tmp_st, tmp_decay, 2*tmp_interval, 8+2*DA_num_iterations)
    #elapsed_time_testing = time.time() - start_time
    print ("Recovered: ", recovered)
    #correct3 = correct3 + int(np.array_equal(recovered, patterns[2, 0:4]))

print ("Actual: ", patterns[2,0:4])

for i in range(25):
    #start_time = time.time()
    recovered = recall(patterns[3,0:4], units, modelParameters, tmp_st, tmp_decay, 2*tmp_interval, 8+2*DA_num_iterations)
    #elapsed_time_testing = time.time() - start_time
    print ("Recovered: ", recovered)
    #correct4 = correct4 + int(np.array_equal(recovered, patterns[3, 0:4]))

# print ("Actual: ", patterns[3,0:4])
# print("Correct1 = ", correct1)
# print("Correct2 = ", correct2)
# print("Correct3 = ", correct3)
# print("Correct4 = ", correct4)
# print("Accuracy = ", correct1 + correct2 + correct3 + correct4)

#print elapsed_time, elapsed_time_testing

print(modelParameters.weights)

with open('weights_1800.pkl','wb') as f:
    pickle.dump(modelParameters.weights, f)
# np.savez('ModelParameters_new', fantasy_particles=fantasy_particles, modelParameters=modelParameters,
#              units=units, numConnections=numConnections, connections_index=connections_index)