import numpy as np
from enum import Enum
import time

DEBUG = False

# import DA libraries, prepare a fjda instance
from python_fjda_wrapper import fjda_wrapper

da = fjda_wrapper.fjda_wrapper()

Clamp = Enum('Clamp', 'VISIBLE_UNITS NONE INPUT_UNITS')

pa = {'offset_inc_rate': 0,
      'tmp_st': 20,
      'tmp_decay': 0.2,
      'tmp_mode': 0,
      'tmp_interval': 2,
      'noise_model': 0,
      'parallel_tempering': 0,
      'pt_interval': 1000}

args = {
        'eg_start': [],
        'state_i': '',
        'bias': [],
        'weight': [],
        #
        'num_bit': 1024,
        'num_iteration': 10,
        'num_run': 10, #it seems that in the first run the minimum energy is found. So no need to more run
        'arch': 1,
    }

class ModelParameters:
    def __init__(self, numberOfUnitsUsed, numberOfEntireUnits):
        # numberOfUnitsUsed: is the number of nodes in DA chip that is used for the model
        # numberOfEntireUnits: is the number of the nodes available in DA chip
        self.states = np.zeros(numberOfEntireUnits, dtype = int)
        self.weights = np.zeros((numberOfUnitsUsed, numberOfUnitsUsed), dtype = int)
        self.bias = np.zeros(numberOfEntireUnits, dtype = int)


class Units:
    def __init__(self, numInputUnits, numOutputUnits, numHiddenUnits):
        self.numInputUnits = numInputUnits
        self.numOutputUnits = numOutputUnits
        self.numHiddenUnits = numHiddenUnits
        self.numUnits = numInputUnits + numOutputUnits + numHiddenUnits
        self.numVisibleUnits = numInputUnits + numOutputUnits



def anneal(clamp,
           units,
           modelParameters,
           tmp_st,
           tmp_decay,
           tmp_interval,
           num_iterations):

    modelParameters.bias = np.zeros((1024)).astype(int)
    # Set bias for unused bits to -2**25 -1,
    modelParameters.bias[units.numUnits:] = (-2 ** 25) * np.ones(1024 - units.numUnits)
    if clamp == Clamp.VISIBLE_UNITS:
        numUnitsToSelect = units.numHiddenUnits
    elif clamp == Clamp.NONE:
        numUnitsToSelect = units.numUnits
    else:  # we want to clamp the input units only, but not the output units
        numUnitsToSelect = units.numHiddenUnits + units.numOutputUnits

    # Set bias for clamped nodes. Negative big number for units with state=0
    # and a big Positive number for units with state=1
    modelParameters.bias[numUnitsToSelect:units.numUnits] = (-1 + 2 * modelParameters.states[numUnitsToSelect:units.numUnits]) * (2 ** 25)


    for i in range(numUnitsToSelect):
        for j in range(numUnitsToSelect, units.numUnits):
            #modelParameters.bias[i] is added to the right side of the assignment
            modelParameters.bias[i] = modelParameters.bias[i] + modelParameters.weights[i, j] * modelParameters.states[j]


    w = np.zeros((1024, 1024)).astype(int)
    w[0:numUnitsToSelect, 0:numUnitsToSelect] = modelParameters.weights[0:numUnitsToSelect, 0:numUnitsToSelect]

    b = np.copy(modelParameters.bias)
    s = np.copy(modelParameters.states)
    s = s.astype(int)
    c = np.array(0)
    # set anneal parameters

    pa['tmp_st'] = tmp_st
    pa['tmp_decay'] = tmp_decay
    pa['tmp_interval']=tmp_interval

    args['eg_start'] = c.tolist()
    args['state_i'] = ''.join([chr(ord('0') + i) for i in s.tolist()])
    args['bias'] = b.tolist()
    args['weight'] = w.reshape((1,1024*1024))[0].tolist()
    args['num_iteration'] = num_iterations
    #t1=time.time()
    da.setAnnealParameter(pa)
    res = da.doAnneal(args)
    #t2=time.time()
    #debug("doAnneal timing", t2-t1)
    modelParameters.states = np.array([int(x) for x in res['state_min_o_n'][0]])
    print(res['eg_min_o_n'])


def propagate(clamp, temperature, units, modelParameters):

    if clamp == Clamp.VISIBLE_UNITS:
        numUnitsToSelect = units.numHiddenUnits
    elif clamp == Clamp.NONE:
        numUnitsToSelect = units.numUnits
    else:  # we want to clamp the input units only, but not the output units
        numUnitsToSelect = units.numHiddenUnits + units.numOutputUnits

    for i in range(numUnitsToSelect):
        # Calculating the energy of a randomly selected unit
        unit = np.random.randint(0, numUnitsToSelect)
        energy = 1.*np.dot(modelParameters.weights[unit, 0:units.numUnits], modelParameters.states[0:units.numUnits])

        p = 1. / (1. + np.exp(-energy / temperature))
        modelParameters.states[unit] = 1. if np.random.uniform() <= p else 0


# This funtions sums the co-occurance of active states (states[i] as well as states[j] are both on)
def sumCoocurrance(clamp, modelParameters, units, numConnections, connections_index):
    sums = np.zeros(numConnections)
    for epoch in range(10):
        propagate(clamp, 10, units, modelParameters)
        for i in range(units.numUnits):
            if (modelParameters.states[i] == 1):
                for j in range(i + 1, units.numUnits):
                    if (connections_index[i, j] > -1 and modelParameters.states[j] == 1):
                        sums[connections_index[i, j]] += 1
    return sums


def updateWeights(pplus,
                  pminus,
                  units,
                  connections_index,
                  modelParameters):
    #global weights
    for i in range(units.numUnits):
        for j in range(i + 1, units.numUnits):
            if connections_index[i, j] > -1:
                index = connections_index[i, j]
                #print(pplus[index] - pminus[index])
                modelParameters.weights[i, j] += 2 * np.sign(pplus[index] - pminus[index])
                modelParameters.weights[j, i] = modelParameters.weights[i, j]

def addNoise(pattern):
    probabilities = 0.8 * pattern + 0.05
    uniform = np.random.random(pattern.shape)
    return (uniform < probabilities).astype(int)

def learn(patterns,
          fantasy_particles,
          modelParameters,
          units,
          numConnections,
          connections_index,
          iterations,
          tmp_st,
          tmp_decay,
          tmp_interval,
          num_iterations
          ):
    numPatterns = patterns.shape[0]
    numParticles = fantasy_particles.shape[0]
    debug('numConnections', numConnections)

    # save the changes to weights in each  run of the 'learn' function every 10 iterations
    weightsChange = np.zeros((10, numConnections), int)
    j=0
    for i in range(iterations):
        print(i)
        ###############################################################
        ## This is the Positive phase
        ###############################################################
        pplus = np.zeros(numConnections)
        for pattern in patterns:  # This is the training data set
            # Setting visible units values (inputs and outputs)
            #add noise to pevent an infinit weight for vectors that never exist
            modelParameters.states[units.numHiddenUnits:units.numHiddenUnits + units.numVisibleUnits] = addNoise(pattern)
            # Assigning random values to the hidden units
            modelParameters.states[0:units.numHiddenUnits] = np.random.choice([0, 1], units.numHiddenUnits)
            sta = time.time()
            anneal(Clamp.VISIBLE_UNITS, units, modelParameters, tmp_st, tmp_decay, tmp_interval, num_iterations)
            endt = time.time() - sta
            debug ('Time of annealing: ', endt)
            pplus += sumCoocurrance(Clamp.VISIBLE_UNITS, modelParameters, units, numConnections, connections_index)
        pplus /= (10*numPatterns)

        ###############################################################
        ## This is the Negative phase
        ###############################################################
        pminus = np.zeros(numConnections)
        #ff = 0
        #for particle in fantasy_particles:  # This is the fantasy particle set
            # Setting visible units values (inputs and outputs)
        #modelParameters.states[units.numHiddenUnits:units.numHiddenUnits + units.numVisibleUnits] = particle
            # Assigning random values to the hidden units
        modelParameters.states[0:units.numUnits] = np.random.choice([0, 1], units.numUnits)#np.random.choice([0, 1], units.numHiddenUnits)
        anneal(Clamp.NONE, units, modelParameters, tmp_st, tmp_decay, tmp_interval, num_iterations)
            #fantasy_particles[ff] = modelParameters.states[units.numHiddenUnits:units.numHiddenUnits + units.numVisibleUnits]
            #ff = ff + 1
        pminus = sumCoocurrance(Clamp.NONE,  modelParameters, units, numConnections, connections_index)
        pminus /= 10#numParticles

        weightsChange[j,:] = np.sign(pplus - pminus)
        j += 1
        if j == 10:
            #np.save('WeightsChange_small_'+str(int(i/10)), weightsChange)
            j=0

        #################################################################
        ## Update the network weights
        ################################################################
        debug('sign(pplus-pminus):', np.sign(pplus-pminus))
        updateWeights(pplus, pminus, units, connections_index, modelParameters)
        debug('weights:', modelParameters.weights)


def recall(pattern,
           units,
           modelParameters,
           tmp_st,
           tmp_decay,
           tmp_interval,
           num_iterations):
    # Setting pattern to recall
    modelParameters.states[units.numHiddenUnits + units.numOutputUnits:units.numHiddenUnits + units.numOutputUnits + units.numInputUnits] = pattern
    # Assigning random values to the hidden and output states
    modelParameters.states[0:units.numHiddenUnits + units.numOutputUnits] = np.random.choice([0, 1], units.numHiddenUnits + units.numOutputUnits)
    anneal(Clamp.INPUT_UNITS, units, modelParameters, tmp_st, tmp_decay, tmp_interval, num_iterations)
    #return modelParameters.states[0:units.numUnits]
    return modelParameters.states[0:units.numUnits]#[units.numHiddenUnits:units.numHiddenUnits + units.numOutputUnits]


def debug(msg, arg):
    if DEBUG:
        print(msg, arg)