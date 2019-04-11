import pickle
import numpy as np

class Traffic_data:
    def __init__(self, links=100, time_period=0, T=50):
        #time_period is a number between 0 and 7, indicatind one of the 8 time intervals in each day.
        #time intervals are:
        # 0:  0:00 -  2:59
        # 1:  3:00 -  5:59
        # 2:  6:00 -  8:59
        # 3:  9:00 - 11:59
        # 4: 12:00 - 14:59
        # 5: 15:00 - 17:59
        # 6: 18:00 - 20:59
        # 7: 21:00 - 23:59
        self.links = links
        self.threshold = T
        self.time_period = time_period
        self.samples_per_interval = 36
        self.prepare_data()
    def prepare_data(self):
        #Apr 09-2019: choose the speed threshold automatically using the mean and standard deviation
        # of speed for each link and each time slot.
        pickle_out = open('TrafficData.pkl', 'rb')
        speedData = pickle.load(pickle_out)#this is an array of size (43,253,287) (
        # number of days, number of links, samples per day) from Dec 13, 2017 to Jan 24, 2018
        pickle_out.close()

        meanSpeed = np.mean(speedData, axis=0)
        stdSpeed = np.std(speedData, axis=0)
        speedThreshold = meanSpeed - stdSpeed # size: (253,287)
        start_time = self.time_period*self.samples_per_interval
        end_time = (self.time_period+1)*self.samples_per_interval

        # only use weekdays
        speedData = speedData[
                    [0, 1, 2, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 19, 20, 21, 22, 23, 26, 27, 28, 29, 30, 33, 34, 35,
                     36, 37, 40, 41, 42], :, :]#start_time: end_time]
        num_days = speedData.shape[0]
        #print(speedData.shape)

        for k in range(num_days):
            speedData[k,:,:] = (speedData[k,:,:]>speedThreshold).astype(int)

        speedData = speedData[:,:,start_time: end_time]
        raw_patterns = np.zeros((self.links, self.samples_per_interval * num_days), float)
        for k in range(num_days):
            raw_patterns[:, self.samples_per_interval * k: self.samples_per_interval * (k + 1)] = speedData[k, 0:self.links, 0:self.samples_per_interval]

        self.train = raw_patterns.T
        #self.train = self.train>self.threshold
        self.ds_size = self.samples_per_interval * num_days

        # Prepare the test data. Speed data consists of 22 days of speed measurements
        # from Jan 10, 2018 to Jan 31, 2018
        pickle_out =open("TestData.pkl", "rb")
        raw_speed_test = pickle.load(pickle_out)
        pickle_out.close()

        # we are only interested in week days for now
        raw_speed_test = raw_speed_test[[0, 1, 2, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 19, 20, 21], :, :]
        num_test_days = raw_speed_test.shape[0]
        for k in range(num_test_days):
            raw_speed_test[k,:,:] = raw_speed_test[k,:,:]>speedThreshold
        raw_speed_test = raw_speed_test[:, 0:self.links, start_time:end_time]

        speed_test = np.zeros((self.links, self.samples_per_interval * num_test_days), int)
        for k in range(num_test_days):
            speed_test[:, self.samples_per_interval*k : self.samples_per_interval* (k+1)] = raw_speed_test[k, 0:self.links, 0:self.samples_per_interval]
        self.test = (speed_test.T).astype(int)
        #self.test = self.test>self.threshold

        # Load the ground truth file.
        # This file consists of a three dimensional array of size
        # 22X253X287. first axis corresponds to 22 days of incident reports
        # from Jan 10, to Jan 31, 2018.
        # for each day there is an array of size 253X287. In this array,
        # 1 means an incident happened and 0 means no incident
        GT = open("ground_truth.pkl", "rb")
        ground_truth = pickle.load(GT)
        GT.close()
        sub_ground_truth = np.zeros((self.links, self.samples_per_interval * num_test_days), int)
        for k in range(num_test_days):
            sub_ground_truth[:, self.samples_per_interval*k : self.samples_per_interval* (k+1)] = ground_truth[k, 0:self.links, 0:self.samples_per_interval]
        self.ground_truth = sub_ground_truth.T