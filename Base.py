import numpy as np
from enum import Enum
from random import randint
import time

DEBUG = False

################################################
# input; number of input units
# output: number of output units
# hidden: number of hidden units
# units: number of entire units (1024 default)
################################################

Clamp = Enum('Clamp', 'VISIBLE_UNITS NONE INPUT_UNITS')
class BaseBM:
    def __init__(self, args):
        # input, output, hidden, units=1024, learning_rate=2, tmp_st=20, tmp_decay=0.16, tmp_interval=2,
        #          anneal_iterations=10, recall_tmp_st=20, recall_tmp_decay=0.16, recall_tmp_interval=4,
        #          recall_iterations=36, coocurrance_temp=10, co_occurrence_epoch=10, save=1):
        self.input = args['input']
        self.output = args['output']
        self.hidden = args['hidden']
        self.units = args['units']  # number of entire units : 1024
        self.bmunits = self.input + self.output + self.hidden
        self.learning_rate = args['learning_rate']
        self.learning_rate_cycle = args['learning_rate_cycle']
        self.learning_rate_decay = args['learning_rate_decay']
        self.tmp_st = args['tmp_st']
        self.tmp_decay = args['tmp_decay']
        self.tmp_interval = args['tmp_interval']
        self.anneal_iterations = args['anneal_iterations']
        self.recall_tmp_st = args['recall_tmp_st']
        self.recall_tmp_decay = args['recall_tmp_decay']
        self.recall_tmp_interval = 4*args['recall_tmp_interval']
        self.recall_iterations = args['recall_iterations']
        self.co_occurrence_temp = args['co_occurrence_temp']
        self.co_occurrence_epoch = args['co_occurrence_epoch']
        self.save = args['save']
        if args['weight_type']=="int":
            self.w_type = np.int
        elif args['weight_type'] == "float":
            self.w_type = np.float64
        else:
            print("undefined type")
        self.numConnections = 0
        self.global_step = 0
        self.states = np.zeros(self.units, dtype=int)
        self.weights = np.zeros((self.input + self.output + self.hidden, self.input + self.output + self.hidden), dtype=self.w_type)
        self.bias = np.zeros(self.units, dtype=self.w_type)
        self.create_connections()
        for i in range(self.bmunits):
            for j in range(self.bmunits):
                if self.connections_index[i,j]>-1:
                    self.weights[i,j] = randint(0, 9)
                    self.weights[j,i] = self.weights[i,j]
        print("Base class: All parameters are initialized")

    def create_connections(self):
        connections = np.zeros((self.bmunits, self.bmunits), dtype=np.int)
        for i in range(0, self.hidden):
            for j in range(i + 1, self.bmunits):
                connections[i, j] = 1
        for i in range(self.hidden, self.hidden + self.output):
            for j in range(i + 1, self.hidden + self.output):
                connections[i, j] = 1
        for i in range(self.hidden + self.output, self.bmunits):
            for j in range(i + 1, self.bmunits):
                connections[i, j] = 1

        valid = np.nonzero(connections)
        numConnections = np.size(valid[0])
        connections_index = connections
        connections_index[valid] = np.arange(1, numConnections + 1)
        self.connections_index = connections_index + connections_index.T - 1
        self.numConnections = numConnections

    def train(self, input_patterns, iterations, FP):
        if FP:
            self.fantasy_particles = np.append(input_patterns, input_patterns, axis=1)
            dw , err = self.train_FP(input_patterns, iterations)
        else:
            dw, err = self.train_NoFP(input_patterns, iterations)
        return dw, err


    # Train without using fantasy particles:
    def train_NoFP(self, input_patterns, iterations):
        numPatterns = input_patterns.shape[0]
        patterns = np.append(input_patterns, input_patterns, axis=1)
        # save the changes to weights in each  run of the 'learn' function every 10 iterations
        delta_w = []
        errors = []
        # compute learning rate decay so that at the final cycle, the learning rate is one:
        self.learning_rate_decay = 1 - (1.0 / self.learning_rate) ** (self.learning_rate_cycle /iterations)
        for i in range(iterations):
            print(i)
            ###############################################################
            ## This is the Positive phase
            ###############################################################
            pplus = np.zeros(self.numConnections)
            for pattern in patterns:  # This is the training data set
                # Setting visible units values (inputs and outputs)
                # add noise to prevent an infinit weight for vectors that never exist
                self.states[self.hidden:self.bmunits] = pattern #self.addNoise(pattern)
                # Assigning random values to the hidden units
                self.states[0:self.hidden] = np.random.choice([0, 1], self.hidden)
                self.anneal(Clamp.VISIBLE_UNITS, self.tmp_st, self.tmp_decay, self.tmp_interval, self.anneal_iterations)
                pplus += self.sumCoocurrance(Clamp.VISIBLE_UNITS)
            pplus /= numPatterns

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
            if (self.global_step % 50) == 0:
                n=patterns.shape[1]
                recovered = self.recall(patterns=patterns[:,0:n//2])
                recon_error = np.linalg.norm(patterns[:,0:n//2] - np.asarray(recovered)[:,self.hidden:self.hidden + self.output])
                # print("recovered", recovered)
                # print("patterns", patterns[:,0:n//2])
                errors.append(recon_error)
                print("Iteration ", self.global_step, "recon error is ", recon_error)
                self.update_parameters()
            self.global_step += 1
            if (self.global_step%self.learning_rate_cycle) == 0:
                self.learning_rate = self.learning_rate*(1.-self.learning_rate_decay)
        return delta_w, errors

    # Train using fantasy particles
    def train_FP(self, input_patterns, iterations):
        numPatterns = input_patterns.shape[0]
        numParticles = self.fantasy_particles.shape[0]
        patterns = np.append(input_patterns, input_patterns, axis=1)
        # save the changes to weights in each  run of the 'learn' function every 10 iterations
        delta_w = []
        errors = []
        # compute learning rate decay so that at the final cycle, the learning rate is one:
        self.learning_rate_decay = 1 - (1.0 / self.learning_rate) ** (self.learning_rate_cycle / iterations)
        for i in range(iterations):
            print(i)
            ###############################################################
            ## This is the Positive phase
            ###############################################################
            pplus = np.zeros(self.numConnections)
            for pattern in patterns:  # This is the training data set
                # Setting visible units values (inputs and outputs)
                # add noise to prevent an infinit weight for vectors that never exist
                self.states[self.hidden:self.bmunits] = pattern #self.addNoise(pattern)
                # Assigning random values to the hidden units
                self.states[0:self.hidden] = np.random.choice([0, 1], self.hidden)
                self.anneal(Clamp.VISIBLE_UNITS, self.tmp_st, self.tmp_decay, self.tmp_interval, self.anneal_iterations)
                pplus += self.sumCoocurrance(Clamp.VISIBLE_UNITS)
            pplus /= numPatterns

            ###############################################################
            ## This is the Negative phase
            ###############################################################
            pminus=np.zeros(self.numConnections)
            f=0
            for particle in self.fantasy_particles:
                self.states[0:self.hidden] = np.random.choice([0, 1], self.hidden)
                self.states[self.hidden:self.bmunits] = particle
                self.anneal(Clamp.NONE, self.tmp_st, self.tmp_decay, self.tmp_interval, self.anneal_iterations)
                self.fantasy_particles[f]=self.states[self.hidden:self.bmunits]
                f+=1
                pminus += self.sumCoocurrance(Clamp.NONE)
            pminus /= numParticles

            delta_w.append(np.linalg.norm(self.learning_rate*np.sign(pplus - pminus)))

            #################################################################
            ## Update the network weights
            ################################################################
            self.updateWeights(pplus, pminus)

            if (self.global_step % 50) == 0:
                n = patterns.shape[1]
                recovered = self.recall(patterns=patterns[:, 0:n // 2])
                recon_error = np.linalg.norm(
                    patterns[:, 0:n // 2] - np.asarray(recovered)[:, self.hidden:self.hidden + self.output])
                # print("recovered", recovered)
                # print("patterns", patterns[:,0:n//2])
                errors.append(recon_error)
                print("Iteration ", self.global_step, "recon error is ", recon_error)
                self.update_parameters()
            self.global_step += 1
            if (self.global_step%self.learning_rate_cycle) == 0:
                self.learning_rate = self.learning_rate*(1.-self.learning_rate_decay)
        return delta_w, errors

    def update_parameters(self):
        #############
        # Tune tmp_st: tmp_st should be large enough compared to the maximum weight of the network
        # so we should update it after the network's weights have changed as a result of training
        # Ref: "Tips for DA parameter tuning_02.pdf"
        self.tmp_st = 100 * np.amax(self.weights)
        #compute the tmp_decay so that at the final cycle the temp if final_tmp
        final_tmp = 1
        self.tmp_decay = 1 - (final_tmp / self.tmp_st) ** (self.tmp_interval / self.anneal_iterations)


    def anneal(self, clamp, tmp_st, tmp_decay, tmp_interval, iterations):
        pass

    def propagate(self, clamp, temperature):
        if clamp == Clamp.VISIBLE_UNITS:
            numUnitsToSelect = self.hidden
        elif clamp == Clamp.NONE:
            numUnitsToSelect = self.bmunits
        else:
            numUnitsToSelect = self.hidden + self.output

        for i in range(numUnitsToSelect):
            # Calculating the energy of a randomly selected unit
            unit = np.random.randint(0, numUnitsToSelect)
            energy = 1.*np.dot(self.weights[unit, 0:self.bmunits], self.states[0:self.bmunits])

            p = 1. / (1. + np.exp(-energy / temperature))
            self.states[unit] = 1 if np.random.uniform() <= p else 0

    def sumCoocurrance(self, clamp):
        sums = np.zeros(self.numConnections)
        for epoch in range(self.co_occurrence_epoch):
            self.propagate(clamp, self.co_occurrence_temp)
            for i in range(self.bmunits):
                if (self.states[i] == 1):
                    for j in range(i + 1, self.bmunits):
                        if (self.connections_index[i, j] > -1 and self.states[j] == 1):
                            sums[self.connections_index[i, j]] += 1
        return sums / self.co_occurrence_epoch

    def updateWeights(self, pplus, pminus):
        # global weights
        for i in range(self.bmunits):
            for j in range(i + 1, self.bmunits):
                if self.connections_index[i, j] > -1:
                    index = self.connections_index[i, j]
                    self.weights[i, j] += (self.learning_rate * np.sign(pplus[index] - pminus[index]))
                    self.weights[j, i] = self.weights[i, j]

    def addNoise(self, pattern):
        probabilities = 0.8 * pattern + 0.05
        uniform = np.random.random(pattern.shape)
        return (uniform < probabilities).astype(int)

    def recall(self, patterns):
        # Setting pattern to recall
        recovered = []
        for pattern in patterns:
            self.states[self.hidden + self.output:self.bmunits] = pattern
            # Assigning random values to the hidden and output states
            self.states[0:self.hidden + self.output] = np.random.choice([0, 1], self.hidden + self.output)
            self.anneal(Clamp.INPUT_UNITS, self.recall_tmp_st, self.recall_tmp_decay, self.recall_tmp_interval,
                        self.recall_iterations)
            output_ = self.states[0:self.bmunits] #self.states[self.hidden:self.hidden + self.output]

            recovered.append(output_.tolist())
        return recovered

    def recall_hidden(self, patterns):
        # Setting pattern to recall
        recovered = []
        for pattern in patterns:
            self.states[self.hidden + self.output:self.bmunits] = pattern
            # Assigning random values to the hidden and output states
            self.states[0:self.hidden + self.output] = np.random.choice([0, 1], self.hidden + self.output)
            self.anneal(Clamp.VISIBLE_UNITS, self.recall_tmp_st, self.recall_tmp_decay, self.recall_tmp_interval,
                        self.recall_iterations)
            output_ = self.states[0:self.bmunits]
            recovered.append(output_.tolist())
        return recovered


    def print_weights(self):
        print(self.weights)

    def postTraining(self):
        pass

        #
        # Retrieve the weights and parameters
        #

    def getParameters(self):
        params = {}
        params['W'] = self.weights
        params['b'] = self.bias
        return params

        #
        # Set parameter
        #

    def setParameters(self, params):
        self.weights = params['W'].astype(int)
        self.bias = params['b'].astype(int)

