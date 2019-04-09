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
        speedData = pickle.load(pickle_out)#this is an array of size (43,253,287) (number of days, number of links, samples per day)
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
        num_time_points_used = end_time - start_time #speedData.shape[2]
        num_days = speedData.shape[0]
        print(speedData.shape)
        raw_patterns = np.zeros((self.links, num_time_points_used * num_days), float)
        for k in range(num_days):
            speedData[k,:,:] = (speedData[k,:,:]>speedThreshold).astype(int)
        speedData = speedData[:,:,start_time: end_time]
        for k in range(num_days):
            raw_patterns[:, self.samples_per_interval * k: self.samples_per_interval * (k + 1)] = speedData[k, 0:self.links, 0:self.samples_per_interval]

        self.train = raw_patterns.T
        #self.train = self.train>self.threshold
        self.ds_size = num_time_points_used * num_days
        # Prepare the test data
        pickle_out =open("test_week_speed_Jan023.pickle", "rb")
        raw_speed_test = pickle.load(pickle_out)
        pickle_out.close()
        speed_test = raw_speed_test>speedThreshold
        speed_test = speed_test[0:self.links, start_time:end_time]
        self.test = speed_test.T
        #self.test = self.test>self.threshold