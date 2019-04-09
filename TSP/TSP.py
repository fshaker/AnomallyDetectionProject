import numpy as np
import math
from scipy.spatial import distance

djibouti_data = open('dj38.tsp.txt', 'r')
contents = djibouti_data.readlines()
print(len(contents))

coords = contents[10: ]
cities_coords = np.zeros((len(coords),2), dtype = float)
i=0
for line in coords:
    words = line.split()
    cities_coords[i,0] = float(words[1])
    cities_coords[i,1] = float(words[2])
    i = i + 1

def dist(inputcities):
    d = 0
    for i in range(inputcities.shape(0)):
        if i == inputcities.shape(0):
            d = d + distance.euclidean(inputcities[i,:], inputcities[0,:])
        else:
            d = d + distance.euclidean((inputcities[i,:], inputcities[i+1,:]))
    return d
# simulated anealing
T0 = 50
b =0.999
s = cities_coords

d = dist(s)


