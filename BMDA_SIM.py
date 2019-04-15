import numpy as np
from enum import Enum
import time
import Base
from Base import Clamp
DEBUG = False

################################################
# input; number of input units
# output: number of output units
# hidden: number of hidden units
# units: number of entire units (1024 default)
################################################


class BMDA_SIM(Base.BaseBM):
    # def __init__(self, input, output, hidden, units=1024,  learning_rate=2, tmp_st=20, tmp_decay=0.16, tmp_interval=2,
    #              anneal_iterations = 10, recall_tmp_st = 20, recall_tmp_decay=0.16, recall_tmp_interval=4,
    #              recall_iterations=28, coocurrance_temp=10, coocurrance_epoch=10, save=1):
    #
    #     super(BMDA_SIM, self).__init__(input, output, hidden, units, learning_rate, tmp_st, tmp_decay, tmp_interval,
    #                                anneal_iterations, recall_tmp_st, recall_tmp_decay, recall_tmp_interval,
    #                                recall_iterations, coocurrance_temp, coocurrance_epoch, save)


    # def anneal(self, clamp, tmp_st, tmp_decay, tmp_interval, iterations):
    #     temperature = tmp_st
    #     step = np.round(iterations/tmp_interval)
    #     for i in range(np.int(step)):
    #         for j in range(tmp_interval):
    #             self.DApropagate(clamp, temperature)
    #         temperature = temperature * (1 - tmp_decay)


    def anneal(self, clamp, tmp_st, tmp_decay, tmp_interval, iterations):
        #first calculate the tmp_st based on the current network weights.
        h = np.matmul(self.weights, self.states[0:self.bmunits])  # + bias when bias is used
        delta_e = np.multiply((1 - 2 * self.states[0:self.bmunits]), h) #shows the change of overall energy when the state of each unit changes
        #we want the initial probability of changing a unit's state to be high
        p = 0.7 #initial probability
        temps = -delta_e/(np.log(1/p-1))
        tmp_st = np.amax(temps)
        if tmp_st>10000:
            print("Temperature is too big!!!!!!!")
        # print("tmp_st",tmp_st)
        #We want tmp_interval to be at least equal to the number of nodes
        #that are allowed to change state
        if clamp == Clamp.VISIBLE_UNITS:
            tmp_interval = self.hidden
        elif clamp == Clamp.NONE:
            tmp_interval = self.bmunits
        else:
            tmp_interval = self.hidden + self.output
        Tn=1 # the final temperature

        step = round(np.log(Tn/tmp_st)/np.log(1-tmp_decay))
        temperature = tmp_st
        for i in range(np.int(step)):
            for j in range(tmp_interval):
                self.DApropagate(clamp, temperature)
            temperature = temperature * (1 - tmp_decay)
    def DApropagate(self, clamp, temperature):
        if clamp == Clamp.VISIBLE_UNITS:
            numUnitsToSelect = self.hidden
        elif clamp == Clamp.NONE:
            numUnitsToSelect = self.bmunits
        else:
            numUnitsToSelect = self.hidden + self.output

            # DA: Instead of first choosing the unit at random and then decide to flip it,
            # first calculate the energy change of each node in case of flipping and
            # then choose one in random to flip

            # Calculate each node's energy change
        h = np.matmul(self.weights, self.states[0:self.bmunits])  # + bias when bias is used
        delta_e = np.multiply((1 - 2 * self.states[0:self.bmunits]), h)
        #start_index = self.bmunits - numUnitsToSelect  # we ony change unclamped units

        # p = np.exp(np.divide(-delta_e,temperature))
        p = 1. / (1. + np.exp(-delta_e / temperature))
        p[numUnitsToSelect:] = 0  # we only care about unclamped units
        uniform = np.random.random(self.bmunits)  # random numbers from uniform distribution
        acceptance = p >= uniform
        # print("before",acceptance)
        acceptance[numUnitsToSelect:] = False
        # print("after",acceptance)
        indexes = np.where(acceptance == True)[0]
        num_Candidate = indexes.shape[0]

        if (num_Candidate > 0):
            random_index = np.random.randint(0, num_Candidate)
            unit = indexes[random_index]
            self.states[unit] = 1 - self.states[unit]






