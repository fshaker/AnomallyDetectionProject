import numpy as np
import os
import urllib
import pickle
from sklearn.datasets import fetch_mldata
from sklearn.datasets.base import get_data_home
from Traffic_data import Traffic_data
from BAS import BAS


class TrainingData:

    def __init__(self, N=28, input=100, ds_size=80, ds="MNIST", speed_threshold=50):
        self.ds = ds
        if ds == "BAS":
            bas_size=30
            data_home = get_data_home(data_home=None)
            data_home = os.path.join(data_home, 'BAS')
            datafile = os.path.join(data_home, 'BAS.pkl')
            if not os.path.exists(data_home):
                os.makedirs(data_home)
            if not os.path.exists(datafile):
                _BAS = BAS(N)
                BAS_data = _BAS.getSample(size=bas_size)
                f = open(datafile, "wb")
                pickle.dump(BAS_data, f)
                f.close()
            else:
                f = open(datafile, "rb")
                BAS_data = pickle.load(f)
                f.close()
            S = BAS_data[0:ds_size, None, :]
            self.S = np.concatenate(S, axis=0)
            self.ds_size = ds_size
            self.test_data = self.S
        elif ds == "4bits":
            self.S = np.array([[1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]])
            self.ds_size = 4
            self.test_data = self.S
        elif ds == "traffic":
            traffic_data = Traffic_data(input, 0, 50)
            self.S = traffic_data.train
            self.test_data = traffic_data.test
            self.ds_size = traffic_data.ds_size
        elif ds == "small_traffic":
            pickle_out = open("small_traffic_dataset.pkl", "rb")
            data = pickle.load(pickle_out)
            pickle_out.close()
            self.S = data["train"]>speed_threshold
            self.ds_size = self.S.shape[0]
            self.test_data = data["test"]>speed_threshold
        elif ds == "MNIST":
            if (N != 28):
                raise ValueError("Please use N = 28 for the MNIST data set")
            try:
                custom_data_home = "C:/Users/Hamco/scikit_learn_data"
                mnist = fetch_mldata('MNIST original', data_home=custom_data_home)
            except:
                print("Could not download MNIST data from mldata.org, trying alternative...")
                mnist_alternative_url = "https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat"
                data_home = get_data_home(data_home=None)
                data_home = os.path.join(data_home, 'mldata')
                if not os.path.exists(data_home):
                    os.makedirs(data_home)
                mnist_save_path = os.path.join(data_home, "mnist-original.mat")
                if not os.path.exists(mnist_save_path):
                    print("Downloading from ", mnist_alternative_url)
                    urllib.urlretrieve(mnist_alternative_url, mnist_save_path)
                print("Now calling fetch_mldata once more")
                mnist = fetch_mldata('MNIST original')
            label = mnist['target']
            mnist = mnist.data
            mnist = ((mnist / 255.0) + 0.5).astype(int)
            images = []
            for i in range(ds_size):
                digit = i % 10
                u = np.where(label == digit)[0]
                images.append(mnist[u[i // 10], None, :])
            self.S = np.concatenate(images, axis=0)
            self.resize(20,20)
            self.ds_size = ds_size
        else:
            raise ValueError("Unknown data set name")

    def get_batch(self, batch_size=10):
        if self.ds == "small_traffic" or self.ds == "4bits" or self.ds == "BAS":
            return self.S[0:batch_size,:]
        elif self.ds == "traffic":
            batch = []
            for i in range(batch_size):
                u = np.random.randint(low=0, high=self.ds_size)
                batch.append(self.S[u, None, :])
            return np.concatenate(batch, axis=0)
        else:
            images = []
            for i in range(batch_size):
                u = np.random.randint(low=0, high=self.ds_size)
                images.append(self.S[u, None, :])
            return np.concatenate(images, axis=0)

    def get_test_data(self):
        return self.test_data

    def resize(self, row, col):
        images = []
        for i in range(self.ds_size):
            image = self.S[i,:]
            small_image = image.reshape(28,28)
            small_image = imresize(small_image, (row, col))
            small_image = small_image.reshape(1, row*col)
            images.append(small_image[0,None,:])
        self.S = np.concatenate(images, axis=0)
