from __future__ import division
import argparse
import numpy as np
import pickle
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from TrainingData import TrainingData

##########################################################
# Parse arguments
##########################################################

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N",
                        type = int,
                        default = 6,
                        help = "The size of image pattern (or a two dimensional data) (rows=cols=N)")
    parser.add_argument("--input",
                        type = int,
                        default = 4,
                        help = "Size of input pattern (one dimensional data)")
    parser.add_argument("--hidden",
                        type = int,
                        default = 4,
                        help = "Number of hidden units")
    parser.add_argument("--iterations",
                        type = int,
                        default = 30000,
                        help = "number of training iterations in each epoch")
    parser.add_argument("--epochs",
                        type=int,
                        default=1,
                        help="The number of times training is done")
    parser.add_argument("--patterns",
                        type = int,
                        default = 100,
                        help = "The number of patterns used for training the BM")
    parser.add_argument("--batch_size",
                        type = int,
                        default = 10,
                        help = "The number of training patterns used for training in each epoch")
    parser.add_argument("--temp_st",
                        type = float,
                        default = 20.,
                        help = "Starting temperature of annealing cycle")
    parser.add_argument("--temp_decay",
                        type = float,
                        default = 0.16,
                        help = "Temperature decay factor (Tn+1 = Tn(1-temp_decay))")
    parser.add_argument("--temp_interval",
                        type = int,
                        default = 2,
                        help = "The number of iterations of annealing, spent on each temperature")
    parser.add_argument("--anneal_iterations",
                        type = int,
                        default = 20,
                        help = "The overall number of iterations spent on annealing")
    parser.add_argument("--co_occurrence_temp",
                        type = float,
                        default = 10.,
                        help = "The temperature used for sampling from model")
    parser.add_argument("--co_occurrence_epoch",
                        type = int,
                        default = 10,
                        help = "The number of samples collected from the model")
    parser.add_argument("--learning_rate",
                        type = int,
                        default = 2,
                        help = "Learning rate of weights")
    parser.add_argument("--learning_rate_decay",
                        type = float,
                        default = 0.,
                        help = "")
    parser.add_argument("--learning_rate_cycle",
                        type = int,
                        default = 100,
                        help = "")
    parser.add_argument("--run_reconstructions",
                        type = int,
                        default = 1,
                        help = "Either 1 (run the reconstruction of the trained network) or 0 (no reconstruction).")
    parser.add_argument("--save_fig",
                        type = int,
                        default = 1,
                        help = "Save the reconstructed images")
    parser.add_argument("--save",
                        type = int,
                        default = 1,
                        help = "Save the learned weights")
    parser.add_argument("--show_metrics",
                        type=int,
                        default=1,
                        help="Display a few metrics after training")
    parser.add_argument("--data",
                        choices = ["traffic", "small_traffic", "4bits", "MNIST", "BAS"],
                        default = "4bits",
                        help = "The data used for training the Boltzman Machine")
    parser.add_argument("--traffic_links_used",
                        type = int,
                        default = 100,
                        help = "The number of traffic links used for training and"
                               " testing the anomaly detector")
    parser.add_argument("--speed_threshold",
                        type = float,
                        default = 50.,
                        help = "The threshold used for binarizing the speed data")
    parser.add_argument("--anneal_method",
                        choices = ["DA", "DA_SIM"],
                        default="DA",
                        help="The Algorithm used for annealing")
    parser.add_argument("--weight_type",
                        default = "int",
                        help = "The type of weights (int or float")
    parser.add_argument("--tmpdir",
                        default = "tmp",
                        help = "The path to tmp folder to save parameters")
    parser.add_argument("--load",
                        default = None,
                        help = "The file from which the pre-trained weights and biases are loaded")
    parser.add_argument("--errors",
                        type = int,
                        default = 3,
                        help = "The number of errors applied to original data")
    parser.add_argument("--use_fantasy_particles",
                        type = bool,
                        default = "False",
                        help = "Whether use Fantasy particles or not")
    args = parser.parse_args()
    return args

#
# Utility function to display an array
# as an N x N binary image
#
def show_pattern(ax, v):
    ax.set_yticks([], [])
    ax.set_xticks([], [])
    ax.imshow(v.reshape(args.N, args.N), "binary")

args = get_args()
if args.load != None:
    print("Loading parameters from ", args.load)
    f = open(args.tmpdir + "/" + args.load, "rb")
    params = pickle.load(f)
    f.close()
    args_loaded = params['args']
    args_loaded.save = args.save
    args_loaded.run_reconstructions = args.run_reconstructions
    args_loaded.tmpdir = args.tmpdir
    args_loaded.load = args.load
    args_loaded.anneal_method = args.anneal_method
    args_loaded.temp_st = args.temp_st
    args_loaded.temp_decay = args.temp_decay
    args_loaded.temp_interval = args.temp_interval
    args_loaded.anneal_iterations = args.anneal_iterations
    args = args_loaded

print("arguments", args)
#
# Create sample set
#
TrainingData = TrainingData(N=args.N, input=args.input, ds_size=args.patterns, ds=args.data,
                             speed_threshold = args.speed_threshold)
#
# Init BM and train
#
dw = []
error = []
BM_args={}
if args.data == "BAS" or args.data=="MNIST":
    BM_args['input'] = args.N * args.N
    BM_args['output'] = args.N * args.N
else:
    BM_args['input'] = args.input
    BM_args['output'] = args.input#
BM_args['hidden'] = args.hidden#
BM_args['units'] = 1024 #
BM_args['learning_rate'] = args.learning_rate #
BM_args['learning_rate_decay'] = args.learning_rate_decay
BM_args['learning_rate_cycle'] = args.learning_rate_cycle
BM_args['tmp_st'] = args.temp_st #
BM_args['tmp_decay'] = args.temp_decay #
BM_args['tmp_interval'] = args.temp_interval
BM_args['anneal_iterations'] = args.anneal_iterations #
BM_args['recall_tmp_st'] = args.temp_st #
BM_args['recall_tmp_decay'] = args.temp_decay #
BM_args['recall_tmp_interval'] =2000*args.temp_interval
BM_args['recall_iterations'] = (2000*args.anneal_iterations + 2000*args.temp_interval*3) #
BM_args['co_occurrence_temp'] = args.co_occurrence_temp
BM_args['co_occurrence_epoch'] = args.co_occurrence_epoch
BM_args['save'] = args.save
BM_args['weight_type'] = args.weight_type

if args.anneal_method == "DA":
    import BMDA
    BM = BMDA.BMDA(BM_args)
else:
    import BMDA_SIM
    BM = BMDA_SIM.BMDA_SIM(BM_args)
# import pdb
# pdb.set_trace()
start = time.time()
if None == args.load:
    for e in range(args.epochs):
        V = TrainingData.get_batch(batch_size=args.batch_size)
        print(V.shape)
        _dw, _error = BM.train(V,
                                iterations=args.iterations, FP = args.use_fantasy_particles)
        dw.append(_dw)
        if len(_error) > 0:
            print("_error is empty")
            error.append(_error)
    #
    # Allow the model to finalize the training
    #
    BM.postTraining()
else:
    print("Loading parameters from ", args.load)
    f = open(args.tmpdir + "/" + args.load, "rb")
    params = pickle.load(f)
    f.close()
    w = params['W']
    params['W']=w//10
    BM.setParameters(params)
    dw = params['dw']
    error = params['error']

print("total training time=", time.time()-start)
BM.print_weights()
timeStamp = str(int(time.time()))
if args.save == 1 and args.load == None:
    params = BM.getParameters()
    params['args'] = args
    params['dw'] = dw
    params['error'] = error
    outfile = args.tmpdir + "/BM_param_" + timeStamp + ".pkl"
    outfiletxt = args.tmpdir + "/BM_param_" + timeStamp + ".txt"
    f = open(outfile, "wb")
    pickle.dump(params, f)
    f.close()
    f = open(outfiletxt, "w+")
    f.write(str(args))
    f.write("\n==================================================================\nWeights:\n")
    f.write(str(params['W']))
    f.write("\n==================================================================\nweight change:\n")
    f.write(str(params['dw']))
    f.write("\n==================================================================\nerrors:\n")
    f.write(str(params['error']))
    f.close()
    print("Saved parameters in ", outfile)



if args.run_reconstructions:
    if args.data == "small_traffic" or args.data == "4bits":

        data = TrainingData.get_test_data()
        print("testData: ", data)
        for i in range(10):
            recovered = BM.recall(data)
            print("recovered", recovered)
    elif args.data == "traffic":
        data = TrainingData.get_test_data()
        print("testData: ", data)
        for i in range(1):
            recovered = BM.recall(data)
            print("recovered", recovered)
    elif args.data == "BAS":
        tests = 8
        cols = 4
        fig = plt.figure(figsize=(5, cols * tests / 5))
        #
        # Determine a sample set that we use for testing
        #
        I = TrainingData.get_batch(batch_size=tests)
        #
        # Now plot the original patterns
        #
        for t in range(tests):
            show_pattern(fig.add_subplot(tests, cols, cols * t + 1), I[t, :])
        #
        # Flip some bits at random in each
        # of the rows
        #
        sample = np.copy(I)
        for t in range(tests):
            for i in range(args.errors):
                field = np.random.randint(0, args.N * args.N)
                sample[t, field] = (1 if I[t, field] == 0 else 0)
        #
        # Sample
        #
        print("Sampling reconstructions")
        R = np.asarray(BM.recall(sample))

        ###############################################
        #Compare the energy of the resulting state and the correct state
        ##########################################
        parameters = BM.getParameters()
        W = np.asarray(parameters['W'])
        #print(R)
        #print(Complete)
        matmul = np.matmul(R, W)
        Energy1 = np.matmul(matmul, np.transpose(R))
        R = R[:,args.hidden:args.hidden+args.N*args.N]
        #
        # R2 = np.asarray(BM.recall_hidden(I))
        # matmul2 = np.matmul(R2, W)
        # Energy2 = np.matmul(matmul2, np.transpose(R2))
        #
        print("Energy of recovered states: ", np.diag(Energy1))
        reconstruction_error = np.linalg.norm(R - sample)
        print((reconstruction_error)**2/tests)
        # print("Energy of correct states: ", Energy2)
        ###############################################
        #
        # Display results
        #
        for t in range(tests):
            # Display distorted image
            show_pattern(fig.add_subplot(tests, cols, cols * t + 2), sample[t, :])
            # Display reconstructions
            show_pattern(fig.add_subplot(tests, cols, cols * t + 3), R[t, :])
        fig.suptitle("temp st, decay, interval=(%d %f %d) anneal_iterations=%d \
                     error=%d iterations=%d learning rate, decay, cycle(%f %f %d)"
                     %(args.temp_st, args.temp_decay, args.temp_interval, args.anneal_iterations,
                       args.errors, args.iterations, args.learning_rate, args.learning_rate_decay,
                       args.learning_rate_cycle))
        fig.tight_layout()
        if args.save_fig == 1:
            outfile = args.tmpdir + "/BM_recovered_" + timeStamp + ".png"
            print("Saving simulation results part I to ", outfile)
            fig.savefig(outfile)

if args.show_metrics == 1:
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 2, 1)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Change of weights")
    ax.plot(np.concatenate(dw))

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(np.concatenate(error), "y")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Reconstruction error")

    if args.save_fig == 1:
        outfile = args.tmpdir + "/BM_metrics_" + timeStamp + ".png"
        print("Saving simulation results part II to ", outfile)
        fig.savefig(outfile)