# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 19:14:52 2018

@author: maiha
"""
from __future__ import division
from enum import Enum
import pickle 
import numpy as np
import time

"""from 1 am to 5:59 am is one period, from 6am to 9:59 am is a period, 10am to 2:59 pm,
3 pm to 6:59 pm, 7 pm to 12:59 am

Here I am trying to do training on the period from 10am to 2:59 pm EST which is 15:00 GMT to  
19:59 GMT 
"""
start_time = time.time()

pickle_out = open("1_week_speed_Jan29_2.pickle","rb")
speed1 = pickle.load(pickle_out)
speed1 = speed1[0:100, 50:52]
speed1_b=speed1
speed1 = speed1 > 50
speed1 = speed1.astype(int)
pickle_out.close()
#speed1 = speed1[:,np.newaxis]

pickle_out = open("2_week_speed_Jan29_2.pickle","rb")
speed2 = pickle.load(pickle_out)
speed2 = speed2[0:100, 50:52]
speed2_b=speed2
speed2 = speed2 > 50
speed2 = speed2.astype(int)
pickle_out.close()
#speed2 = speed2[:,np.newaxis]

pickle_out = open("3_week_speed_Jan29_2.pickle","rb")
speed3 = pickle.load(pickle_out)
speed3 = speed3[0:100, 50:52]
speed3_b=speed3
speed3 = speed3 > 50
speed3 = speed3.astype(int)
pickle_out.close()
#speed3 = speed3[:,np.newaxis]

pickle_out = open("4_week_speed_Jan29_2.pickle","rb")
speed4 = pickle.load(pickle_out)
speed4 = speed4[0:100, 50:52]
speed4_b=speed4
speed4 = speed4 > 50
speed4 = speed4.astype(int)
pickle_out.close()
#speed4 = speed4[:,np.newaxis]

#________________________________________________________________________________


pickle_out = open("test_week_speed_Jan29_2.pickle","rb")
speed_test = pickle.load(pickle_out)
speed_test = speed_test[0:100, 50:52]

speed_test = speed_test > 50
speed_test = speed_test.astype(int)

pickle_out.close()
#speed_test = speed_test[:,np.newaxis]

Clamp = Enum('Clamp', 'VISIBLE_UNITS NONE INPUT_UNITS')

numInputUnits = 100
numOutputUnits = 100
numHiddenUnits = 150

numVisibleUnits = numInputUnits + numOutputUnits
numUnits = numVisibleUnits+numHiddenUnits

# import DA libraries, prepare a fjda instance
from python_fjda_wrapper import fjda_wrapper
da = fjda_wrapper.fjda_wrapper()


# Defining connections:
# Connections is a matrix of size numUnits x numUnits. The element "Connections[a,b]" represents the
# number of connections between nodes (a,b).

# If two nodes a and b are not connected, then Connections[a,b] = -1. 
#el connections faydet-ha enaha teb2a 1 - identity matrix, such
# that the off-diagonal elements are non-zeros. There is another constraint
# however, which is there are no connections between the input and output units

connections = np.zeros((numUnits,numUnits), dtype=np.int)
for i in range(numInputUnits):
    for j in range(i+1,numInputUnits):
        connections[i,j] = 1
    for j in range(1,numHiddenUnits+1):
        connections[i,-j] = 1   
            
for i in range(numOutputUnits):
    for j in range(i+1,numOutputUnits):
        connections[i+numInputUnits,j+numInputUnits] = 1
    for j in range(1,numHiddenUnits+1):
        connections[i+numInputUnits,-j] = 1  

for i in range(numHiddenUnits,0,-1):
    for j in range(i-1,0,-1):
        connections[-i,-j] = 1
        
connections = connections[::-1].T[::-1]
        
valid = np.nonzero(connections)
numConnections = np.size(valid[0])
connections[valid] = np.arange(1,numConnections+1)
connections = connections + connections.T - 1

def anneal(clamp):
    global states, weights, bias
    bias = np.zeros((1024))
    bias[numUnits:] = (-2**25)*np.ones(1024-numUnits)
    weights = weights.astype(int)
    w = np.zeros((1024,1024)).astype(int)
    if clamp == Clamp.VISIBLE_UNITS:
        numUnitsToSelect = numHiddenUnits
    elif clamp == Clamp.NONE:
        numUnitsToSelect = numUnits
    else: # we want to clamp the input units only, but not the output units
        numUnitsToSelect = numHiddenUnits + numOutputUnits
    
    bias[numUnitsToSelect:numUnits]=(-1+2*states[numUnitsToSelect:numUnits])*(2**25)
    for i in range(numUnitsToSelect):
        for j in range (numUnitsToSelect,numUnits):
            bias[i] = weights[i,j]*states[j]

    w[0:numUnitsToSelect,0:numUnitsToSelect]=weights[0:numUnitsToSelect,0:numUnitsToSelect]

    bias = bias.astype(int)
    b=bias

    states = states.astype(int)
    s=states

    c = np.array(0)

    # set anneal parameters
    pa = {'offset_inc_rate': 0,
          'tmp_st': 20,
          'tmp_decay': 0.1,
          'tmp_mode': 0,
          'tmp_interval': 100,
          'noise_model': 0,
          'parallel_tempering': 0,
          'pt_interval': 1000}

    da.setAnnealParameter(pa)
    
    # do anealling
    args = {
            'eg_start': c.tolist(),
            'state_i':  ''.join([chr(ord('0')+i) for i in s.tolist()]),
            'bias':     b.tolist(),
            'weight':   w.reshape((1,1024*1024))[0].tolist(),
            #
            'num_bit':  1024,
            'num_iteration': 40,
            'num_run':  10,
            'arch':     1,
            }


    res = da.doAnneal(args)
    states = np.array([ int(x) for x in res['state_min_o_n'][0] ])

# This funtions sums the co-occurance of active states (states[i] as well as states[j] are both on)
def sumCoocurrance(clamp):                        
    sums = np.zeros(numConnections)
    for i in range(numUnits):
        if(states[i] == 1):
            for j in range(i+1,numUnits):
                if(connections[i,j]>-1 and states[j] ==1):
                    sums[connections[i,j]] += 1   
    return sums
     
def updateWeights(pplus, pminus):
    global weights
    for i in range(numUnits):
        for j in range(i+1,numUnits):            
            if connections[i,j] > -1:
                index = connections[i,j]
                weights[i,j] += 2*np.sign(pplus[index] - pminus[index])
                weights[j,i] = weights[i,j]

def recall(pattern):
    global states
        
    # Setting pattern to recall
    states[numHiddenUnits+numOutputUnits:numHiddenUnits+numOutputUnits+numInputUnits] = pattern
     
    # Assigning random values to the hidden and output states
    states[0:numHiddenUnits+numOutputUnits] = np.random.choice([0,1],numHiddenUnits+numOutputUnits)


    anneal(Clamp.INPUT_UNITS)
    
    return states[numHiddenUnits:numHiddenUnits+numOutputUnits]

def learn(patterns):
    global states, weights,bias

    numPatterns = patterns.shape[0]    
    numParticles = fantasy_particles.shape[0]
    weights = np.zeros((numUnits,numUnits))
    weights = weights.astype(int)
    states = np.zeros(1024)
    bias = np.zeros(1024)    

    for i in range(100):
        print(i)
	     # This is the Positive phase
        pplus = np.zeros(numConnections)
        for pattern in patterns:                 # This is the training data set
            
            # Setting visible units values (inputs and outputs)
            states[numHiddenUnits:numHiddenUnits+numVisibleUnits] = pattern
    
            # Assigning random values to the hidden units
            states[0:numHiddenUnits] = np.random.choice([0,1],numHiddenUnits)
    	    
            sta = time.time()
            anneal(Clamp.VISIBLE_UNITS)
            endt = time.time()-sta
            print (endt)
            pplus += sumCoocurrance(Clamp.VISIBLE_UNITS)
        pplus /= numPatterns
        
        
        # This is the Negative phase
        pminus = np.zeros(numConnections)
        ff=0
        for particle in fantasy_particles:                 # This is the fantasy particle set
            # Setting visible units values (inputs and outputs)
            states[numHiddenUnits:numHiddenUnits+numVisibleUnits] = particle
    
            # Assigning random values to the hidden units
            states[0:numHiddenUnits] = np.random.choice([0,1],numHiddenUnits)
            
            anneal(Clamp.NONE)
            fantasy_particles[ff] = states[numHiddenUnits:numHiddenUnits+numVisibleUnits]
            ff = ff+1
            pminus += sumCoocurrance(Clamp.NONE)
        pminus /= numParticles
       
        updateWeights(pplus,pminus)

past_weeks_1 = np.append(speed1.T,speed1.T, axis=1)
past_weeks_2 = np.append(speed2.T,speed2.T, axis=1)
past_weeks_3 = np.append(speed3.T,speed3.T, axis=1)
past_weeks_4 = np.append(speed4.T,speed4.T, axis=1)
past_weeks = np.append(past_weeks_1, past_weeks_2, axis=0)
past_weeks = np.append(past_weeks, past_weeks_3, axis=0)
past_weeks = np.append(past_weeks, past_weeks_4, axis=0)
patterns = past_weeks

f_particles = np.random.choice([0, 1], size=(8,100))
fantasy_particles = np.append(f_particles,f_particles, axis=1)
    
learn(patterns)

elapsed_time = time.time() - start_time

start_time = time.time()
recovered = recall(speed_test[:,1].T)
print (recovered[43:45])
elapsed_time_testing = time.time() - start_time

print elapsed_time, elapsed_time_testing
