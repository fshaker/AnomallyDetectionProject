import numpy as np
import Base
from Base import Clamp
DEBUG = False

################################################
# input; number of input units
# output: number of output units
# hidden: number of hidden units
# units: number of entire units (1024 default)
################################################
# This class enforces sparse hidden unit activation by adding penalty for hidden units that are active
# most of the times during training

class BMDA_SIM_SPARSE(BMDA_SIM.BMDA_SIM):
    # def __init__(self, input, output, hidden, units=1024,  learning_rate=2, tmp_st=20, tmp_decay=0.16, tmp_interval=2,
    #              anneal_iterations = 10, recall_tmp_st = 20, recall_tmp_decay=0.16, recall_tmp_interval=4,
    #              recall_iterations=28, coocurrance_temp=10, coocurrance_epoch=10, save=1):
    #
    #     super(BMDA_SIM, self).__init__(input, output, hidden, units, learning_rate, tmp_st, tmp_decay, tmp_interval,
    #                                anneal_iterations, recall_tmp_st, recall_tmp_decay, recall_tmp_interval,
    #                                recall_iterations, coocurrance_temp, coocurrance_epoch, save)

    def train(self, input_patterns, iterations):
        numPatterns = input_patterns.shape[0]
        patterns = np.append(input_patterns, input_patterns, axis=1)
        # save the changes to weights in each  run of the 'learn' function every 10 iterations
        delta_w = []
        errors = []

        for i in range(iterations):
            print(i)
            ###############################################################
            ## This is the Positive phase
            ###############################################################
            pplus = np.zeros(self.numConnections)
            hidden_unit_tracking = np.zeros(self.hidden)
            for pattern in patterns:  # This is the training data set
                # Setting visible units values (inputs and outputs)
                # add noise to prevent an infinit weight for vectors that never exist
                self.states[self.hidden:self.bmunits] = self.addNoise(pattern)
                # Assigning random values to the hidden units
                self.states[0:self.hidden] = np.random.choice([0, 1], self.hidden)
                self.anneal(Clamp.VISIBLE_UNITS, self.tmp_st, self.tmp_decay, self.tmp_interval, self.anneal_iterations)
                # track the number of times each hidden unit is on
                hidden_unit_tracking = hidden_unit_tracking + self.states[0:self.hidden]
                pplus += self.sumCoocurrance(Clamp.VISIBLE_UNITS)
            pplus /= numPatterns
            hidden_unit_tracking /= numPatterns

            ###############################################################
            ## This is the Negative phase
            ###############################################################
            self.states[0:self.bmunits] = np.random.choice([0, 1], self.bmunits)
            self.anneal(Clamp.NONE, self.tmp_st, self.tmp_decay, self.tmp_interval, self.anneal_iterations)
            pminus = self.sumCoocurrance(Clamp.NONE)

            delta_w.append(np.linalg.norm(self.learning_rate*np.sign(pplus - pminus)))

            #################################################################
            ## Update the network weights
            ################################################################

            self.updateWeights(pplus, pminus)
            if 0 == (self.global_step % 50):
                n=patterns.shape[1]
                recovered = self.recall(patterns=patterns[:,0:n//2])
                recon_error = np.linalg.norm(patterns[:,0:n//2] - np.asarray(recovered)[:,self.hidden:self.hidden + self.output])
                # print("recovered", recovered)
                # print("patterns", patterns[:,0:n//2])
                errors.append(recon_error)
                print("Iteration ", self.global_step, "recon error is ", recon_error)
            self.global_step += 1
            if (self.global_step%self.learning_rate_cycle) == 0:
                self.learning_rate = self.learning_rate*(1.-self.learning_rate_decay)
        return delta_w, errors

    def anneal(self, clamp, tmp_st, tmp_decay, tmp_interval, iterations):
        temperature = tmp_st
        step = np.round(iterations/tmp_interval)
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






