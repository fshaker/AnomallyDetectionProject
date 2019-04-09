# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 19:14:52 2018

@author: maiha
"""
from __future__ import division
import pickle
import numpy as np
import time
from OldCode.FJDA_Learning import Units, ModelParameters, learn, recall

np.set_printoptions(threshold=np.NaN)
"""from 1 am to 5:59 am is one period, from 6am to 9:59 am is a period, 10am to 2:59 pm,
3 pm to 6:59 pm, 7 pm to 12:59 am

Here I am trying to do training on the period from 10am to 2:59 pm EST which is 15:00 GMT to  
19:59 GMT 
"""

pickle_out = open("1_week_speed_Jan29_2.pickle","rb")
speed1 = pickle.load(pickle_out)
speed1 = speed1[0:100, 50:52]
pickle_out.close()
#speed1 = speed1[:,np.newaxis]

pickle_out = open("2_week_speed_Jan29_2.pickle","rb")
speed2 = pickle.load(pickle_out)
speed2 = speed2[0:100, 50:52]
pickle_out.close()
#speed2 = speed2[:,np.newaxis]

pickle_out = open("3_week_speed_Jan29_2.pickle","rb")
speed3 = pickle.load(pickle_out)
speed3 = speed3[0:100, 50:52]
pickle_out.close()
#speed3 = speed3[:,np.newaxis]

pickle_out = open("4_week_speed_Jan29_2.pickle","rb")
speed4 = pickle.load(pickle_out)
speed4 = speed4[0:100, 50:52]
pickle_out.close()
#speed4 = speed4[:,np.newaxis]

#________________________________________________________________________________


pickle_out = open("test_week_speed_Jan29_2.pickle","rb")
speed_test = pickle.load(pickle_out)
speed_test = speed_test[0:100, 50:52]
pickle_out.close()

# Units( numInputUnits, numOutputUnits, numHiddenUnits
numberOfDANodes = 1024
numInput = 100
numOutput = 100
numHidden = 150
speedThreshold = 50

start_time = time.time()
units = Units(numInput, numOutput, numHidden)
modelParameters = ModelParameters(units.numUnits, numberOfDANodes)

speed1 = speed1 > speedThreshold
speed1 = speed1.astype(int)
speed2 = speed2 > speedThreshold
speed2 = speed2.astype(int)
speed3 = speed3 > speedThreshold
speed3 = speed3.astype(int)
speed4 = speed4 > speedThreshold
speed4 = speed4.astype(int)
speed_test = speed_test > speedThreshold
speed_test = speed_test.astype(int)
# Defining connections:
# Connections is a matrix of size numUnits x numUnits. The element "Connections[a,b]" represents the
# number of connections between nodes (a,b).

# If two nodes a and b are not connected, then Connections[a,b] = -1. 
#el connections faydet-ha enaha teb2a 1 - identity matrix, such
# that the off-diagonal elements are non-zeros. There is another constraint
# however, which is there are no connections between the input and output units

connections = np.zeros((units.numUnits, units.numUnits), dtype=np.int)
for i in range(units.numInputUnits):
    for j in range(i+1, units.numInputUnits):
        connections[i,j] = 1
    for j in range(1, units.numHiddenUnits + 1):
        connections[i,-j] = 1   
            
for i in range(units.numOutputUnits):
    for j in range(i+1, units.numOutputUnits):
        connections[i + units.numInputUnits, j + units.numInputUnits] = 1
    for j in range(1, units.numHiddenUnits + 1):
        connections[i + units.numInputUnits, -j] = 1

for i in range(units.numHiddenUnits, 0, -1):
    for j in range(i-1, 0, -1):
        connections[-i, -j] = 1
        
connections = connections[::-1].T[::-1]

valid = np.nonzero(connections)
numConnections = np.size(valid[0])
connections_index = connections
connections_index[valid] = np.arange(1, numConnections + 1)
connections_index = connections_index + connections_index.T - 1

past_weeks_1 = np.append(speed1.T,speed1.T, axis=1) # repeat twice, accounts for input nodes and output nodes values
past_weeks_2 = np.append(speed2.T,speed2.T, axis=1)
past_weeks_3 = np.append(speed3.T,speed3.T, axis=1)
past_weeks_4 = np.append(speed4.T,speed4.T, axis=1)
past_weeks = np.append(past_weeks_1, past_weeks_2, axis=0)
past_weeks = np.append(past_weeks, past_weeks_3, axis=0)
past_weeks = np.append(past_weeks, past_weeks_4, axis=0)
patterns = past_weeks

f_particles = np.random.choice([0, 1], size=(8,100))
fantasy_particles = np.append(f_particles,f_particles, axis=1)
iterations = 100

#################################################################
#Learn the distribution
learn(patterns, fantasy_particles, modelParameters, units, numConnections, connections_index, iterations)
elapsed_time = time.time() - start_time

################################################################
#Test the model's prediction
start_time = time.time()
recovered = recall(speed_test[:,1].T, units, modelParameters)
elapsed_time_testing = time.time() - start_time

print ("Recovered: ", recovered[42:45])
print ("Actual Speed: ", speed_test[42:45,1])
print elapsed_time, elapsed_time_testing

from tempfile import TemporaryFile
recoveredSpeed = TemporaryFile()
np.savez('recoveredSpeed_fA_2', recovered=recovered, ActualSpeed=speed_test[:, 1])