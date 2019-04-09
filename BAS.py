
import numpy as np


class BAS:

    def __init__(self, N=4):
        self.N = N

    def createMatrix(self, orientation=0, number=0):
        #
        # Create a 4x4 matrix out of the bars-and-stripes
        # collection
        #
        values = np.zeros((self.N, 1))
        for i in range(self.N):
            values[i] = (number >> i) % 2
        if (orientation == 0):
            return np.matmul(values, np.ones((1, self.N)))
        else:
            return np.transpose(np.matmul(values, np.ones((1, self.N))))

    def createVector(self, orientation=0, number=0):
        M = self.createMatrix(orientation=orientation, number=number)
        return M.reshape((self.N * self.N, 1))

    #
    # Return a matrix with a given number of
    # samples. The result will be stored in a
    # matrix with shape (size, N*N)
    #
    def getSample(self, size=30):
        if size > 2 * (2 ** self.N) - 2:
            raise ValueError("Cannot generate that many samples")
        if 0 != (size % 2):
            raise ValueError("Number of samples must be even")
        images = []
        for n in range(int(size / 2)):
            a = self.createVector(1, n + 1)
            images.append(a)
            b = self.createVector(0, n + 1)
            images.append(b)
        V = np.concatenate(images, axis=1)
        return np.transpose(V)