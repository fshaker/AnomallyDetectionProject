import math
import numpy as np
import time
from python_fjda import fjda

import BMDA
from Base import Clamp

da = fjda.fjda()

#c = np.array(0)
pa = {'offset_inc_rate': 0,
      'tmp_st': 20,
      'tmp_decay': 0.2,
      'tmp_mode': 0,
      'tmp_interval': 2,
      'noise_model': 0,
      'parallel_tempering': 1,
      'pt_interval': 100}

args = {
        'num_bit': 1024,
        'eg_start': 0,
        'state_i': '',
        'bias': [],
        'weight': [],
        'num_iteration': 10,
        'num_run': 10, #it seems that in the first run the minimum energy is found. So no need to more run
        'ope_mode': 1,
    }

class BMDA_DAPS(BMDA.BMDA):

    def train_FP(self, input_patterns, iterations):
        print("we are in BMDA.DAPS")
        fantasy_particles = np.append(input_patterns, input_patterns, axis=1)
        numPatterns = input_patterns.shape[0]
        numParticles = fantasy_particles.shape[0]
        patterns = np.append(input_patterns, input_patterns, axis=1)
        # save the changes to weights in each  run of the 'learn' function every 10 iterations
        delta_w = []
        errors = []
        # compute learning rate decay so that at the final cycle, the learning rate is one:
        learning_rate = self.learning_rate
        self.learning_rate_decay = 1 - (1.0 / self.learning_rate) ** (self.learning_rate_cycle / iterations)
        for i in range(iterations):
            print(i)
            # First initialize fantasy particles to modes of the model's PDF. We update
            # fantasy particles every n iterations.
            if i%2==0:
                fantasy_particles = self.initialise_particles(numParticles)
                numParticles = fantasy_particles.shape[0]
            ###############################################################
            ## This is the Positive phase
            ###############################################################
            pplus = np.zeros(self.numConnections)
            # self.tmp_st = self.update_parameters()
            for pattern in patterns:  # This is the training data set
                # Setting visible units values (inputs and outputs)
                # When using synthetic data, add noise to prevent an infinite weight for vectors that never exist.
                # when using natural data, there sre enough nose present in the dataset,
                # so there is no need to add artificial noise.
                self.states[self.hidden:self.bmunits] = pattern  # self.addNoise(pattern)
                # Assigning random values to the hidden units. For use with the DA we should set hidden units to zero.
                self.states[0:self.hidden] = np.zeros(self.hidden)
                st = time.time()
                self.anneal(Clamp.VISIBLE_UNITS, self.tmp_st, self.tmp_decay, self.tmp_interval,
                            self.anneal_iterations)
                total = time.time() - st
                self.anneal_time['total'] = self.anneal_time['total'] + total
                pplus += self.sumCoocurrance(Clamp.VISIBLE_UNITS)
            pplus /= numPatterns

            ###############################################################
            ## This is the Negative phase
            ###############################################################
            pminus = np.zeros(self.numConnections)
            f = 0
            for particle in fantasy_particles:
                self.states[0:self.bmunits] = particle
                pminus += self.sumCoocurrance(Clamp.NONE)
                fantasy_particles[f] = self.states[0:self.bmunits]
                f += 1
            pminus /= numParticles

            delta_w.append(np.linalg.norm(self.learning_rate * np.sign(pplus - pminus)))

            #################################################################
            ## Update the network weights
            ################################################################
            self.updateWeights(pplus, pminus, learning_rate)
            self.global_step += 1
            if (self.global_step % 50) == 0:
                n = patterns.shape[1]
                st = time.time()
                recovered, _ = self.recall(patterns=patterns[:, 0:n // 2])
                total = time.time() - st
                self.recall_time = self.recall_time + total
                recon_error = np.linalg.norm(
                    patterns[:, 0:n // 2] - np.asarray(recovered))
                # print("recovered", recovered)
                # print("patterns", patterns[:,0:n//2])
                errors.append(recon_error)
                print("Iteration ", self.global_step, "recon error is ", recon_error)

            if ((i + 1) % self.learning_rate_cycle) == 0:
                learning_rate = learning_rate * (1. - self.learning_rate_decay)
        return delta_w, errors

    def initialise_particles(self, n):
        # we divide the whole energy space into sub spaces by setting the state
        # of some random hidden units
        fixed_hidden_units = int(np.ceil(np.log(n)))
        num_particles = 2**fixed_hidden_units
        fantasy_particles = np.zeros((num_particles, self.bmunits),int)
        for k in range(num_particles):
            binary = np.binary_repr(k, width=fixed_hidden_units)
            index = np.random.permutation(range(self.hidden))[0:fixed_hidden_units]
            self.states = np.zeros(self.units)
            self.states[index] = np.asarray([int(x) for x in binary])
            st = time.time()
            self.anneal(Clamp.INDEXES, self.tmp_st, self.tmp_decay, self.tmp_interval, self.anneal_iterations, index)
            total = time.time() - st
            self.anneal_time['total'] = self.anneal_time['total'] + total
            # now run MCMC to make sure it is mixed well
            for _ in range(100):
                self.propagate(Clamp.NONE, 1)
            fantasy_particles[k,:] = self.states[0:self.bmunits]
        return fantasy_particles

    def anneal(self, clamp, tmp_st, tmp_decay, tmp_interval, iterations, indexes=None):
        self.bias = np.zeros((self.units)).astype(int)
        # Set bias for unused bits to -2**25 -1,
        self.bias[self.bmunits:] = (-2 ** 25) * np.ones(1024 - self.bmunits)
        if clamp == Clamp.VISIBLE_UNITS:
            FreeUnits = range(self.hidden)
        elif clamp == Clamp.NONE:
            FreeUnits = range(self.bmunits)
        elif clamp == Clamp.INDEXES:  # we want to clamp the input units only, but not the output units
            FreeUnits = [x for x in range(self.bmunits) if not x in indexes]
        else:
            FreeUnits = range(self.hidden + self.output)

        # Set bias for clamped nodes. Negative big number for units with state=0
        # and a big Positive number for units with state=1
        ClampedUnits = [x for x in range(self.bmunits) if not x in FreeUnits]
        self.bias[ClampedUnits] = (-1 + 2 * self.states[ClampedUnits]) * (2 ** 25)

        for i in FreeUnits:
            for j in ClampedUnits:
                # modelParameters.bias[i] is added to the right side of the assignment
                self.bias[i] = self.bias[i] + self.weights[i, j] * self.states[j]

        w = np.zeros((self.units, self.units)).astype(int)
        for i in FreeUnits:
            for j in FreeUnits:
                w[i, j] = self.weights[i, j]
        b = np.copy(self.bias)
        s = np.copy(self.states)
        s = s.astype(int)
        # For Now, the DA only performs correctly if we set the initial stats to all zeros
        #s = np.zeros((self.units)).astype(int)

        h = np.matmul(self.weights, self.states[0:self.bmunits])  # + bias when bias is used
        delta_e = np.multiply((1 - 2 * self.states[0:self.bmunits]),
                              h)  # shows the change of overall energy when the state of each unit changes
        # we want the initial probability of changing a unit's state to be high
        p = 0.7  # initial probability
        temps = -delta_e / (np.log(1 / p - 1))
        tmp_st = round(np.amax(temps))
        # print("tmp_st = ", tmp_st)
        if tmp_st < 20:
            tmp_st = 20
        # if tmp_st > 100000:
        #     print("Temperature is too big!!!!!!!")

        tmp_interval = len(FreeUnits)
        Tmin = 0.1
        iterations = tmp_interval*np.log(Tmin/tmp_st)/np.log(1-tmp_decay)

        # set anneal parameters
        pa['tmp_st'] = tmp_st
        pa['tmp_decay'] = tmp_decay
        pa['tmp_interval'] = tmp_interval

        args['state_i'] = ''.join([chr(ord('0') + i) for i in s.tolist()])
        args['bias'] = b.tolist()
        args['weight'] = w.reshape((1, self.units * self.units))[0].tolist()
        args['num_iteration'] = int(round(iterations))

        da.setAnnealParameter(pa)
        res = da.doAnneal(args)
        self.anneal_time['on_annealer'] = self.anneal_time['on_annealer'] + da._rpc_time()['elapsed_time']
        index = np.argmin(res['eg_min_o_n'])
        self.states = np.array([int(x) for x in res['state_min_o_n'][index]])
        # print(res['eg_min_o_n'])

