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
# from python_fjda_wrapper import fjda_wrapper
#
# da = fjda_wrapper.fjda_wrapper()
from python_fjda import fjda
da = fjda.fjda()


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
        'num_run': 20, #it seems that in the first run the minimum energy is found. So no need to more run
        'ope_mode': 1,
    }



class BMDA(Base.BaseBM):

    def anneal(self, clamp, tmp_st, tmp_decay, tmp_interval, iterations):
        self.bias = np.zeros((self.units)).astype(int)
        # Set bias for unused bits to -2**25 -1,
        self.bias[self.bmunits:] = (-2 ** 25) * np.ones(1024 - self.bmunits)
        if clamp == Clamp.VISIBLE_UNITS:
            numUnitsToSelect = self.hidden
        elif clamp == Clamp.NONE:
            numUnitsToSelect = self.bmunits
        else:  # we want to clamp the input units only, but not the output units
            numUnitsToSelect = self.hidden + self.output

        # Set bias for clamped nodes. Negative big number for units with state=0
        # and a big Positive number for units with state=1
        self.bias[numUnitsToSelect:self.bmunits] = (-1 + 2 * self.states[numUnitsToSelect:self.bmunits]) * (2 ** 25)

        for i in range(numUnitsToSelect):
            for j in range(numUnitsToSelect, self.bmunits):
                # modelParameters.bias[i] is added to the right side of the assignment
                self.bias[i] = self.bias[i] + self.weights[i, j] * self.states[j]

        w = np.zeros((self.units, self.units)).astype(int)
        w[0:numUnitsToSelect, 0:numUnitsToSelect] = self.weights[0:numUnitsToSelect, 0:numUnitsToSelect]

        #print(self.weights[0:30,40:50])
        #print(self.weights[0:30, 76:86])
        b = np.copy(self.bias)
        s = np.copy(self.states)
        s = s.astype(int)
        c = np.array(0)
        # set anneal parameters

        pa['tmp_st'] = tmp_st
        pa['tmp_decay'] = tmp_decay
        pa['tmp_interval'] = tmp_interval

        args['eg_start'] = c.tolist()
        args['state_i'] = ''.join([chr(ord('0') + i) for i in s.tolist()])
        args['bias'] = b.tolist()
        args['weight'] = w.reshape((1, self.units * self.units))[0].tolist()
        args['num_iteration'] = iterations
        # import pdb
        # pdb.set_trace()
        da.setAnnealParameter(pa)
        res = da.doAnneal(args)
        self.states = np.array([int(x) for x in res['state_min_o_n'][0]])
        print(res['eg_min_o_n'])

        ####################################################################
        # temperature = tmp_st
        # step = np.round(iterations/tmp_interval)
        # for i in range(np.int(step)):
        #     for j in range(tmp_interval):
        #         self.propagate(clamp, temperature)
        #     temperature = temperature * (1 - tmp_decay)











