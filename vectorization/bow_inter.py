import os
import sys
import random
import math
import numpy
import spams
#from learning import inter_kernel

# Slice trajectory with fixed window size
def slice(trajectory, w_len, interval=1):
    segments = list()
    for item in trajectory:
        temp = list()
        stamp_index = range(0, item.shape[0]-w_len+1, interval)
        for i in stamp_index:
            temp.append(item[i:i+w_len, :1])
        segments.append(numpy.hstack(temp))
    return numpy.hstack(segments)

# Compute velocity from trajectory
def diff(trajectory):
    for index, item in enumerate(trajectory):
        temp = numpy.zeros([item.shape[0]-1, 1])
        for i in range(1, item.shape[0]):
            temp[i-1, 0] = numpy.linalg.norm(item[i, :] - item[i-1, :])
        trajectory[index] = temp
    return trajectory

# Produce bow histogram
def bow(trajectory, D, a, w_len, interval=1):
    k = D.shape[1]
    histogram = numpy.zeros([len(trajectory), k])
    for index, item in enumerate(trajectory):
        temp = list()
        stamp_index = range(0, item.shape[0]-w_len+1, interval)
        for i in stamp_index:
            temp.append(item[i:i+w_len, :1])
        temp = numpy.hstack(temp)
        code = spams.lasso(numpy.asfortranarray(temp), D, lambda1=a, pos=True)
        code = numpy.sum(code.todense(), axis=1)
        histogram[index:index+1, :] += code.reshape([1, k])
        div = numpy.linalg.norm(histogram[index, :])
        if div > 0:
            histogram[index, :] = histogram[index, :] / div
    return histogram

# Set parameters
w_len = 40
k = 500
a = 1.0/math.sqrt(w_len)

'''
file = "submission_bow_inter.csv"

if __name__ == '__main__':
    # Write header to submission.csv
    f = open(file, "w")
    f.write("driver_trip,prob\n")
    f.close()

    # List drivers folder
    folder = os.listdir('drivers')

    # Iteratively process drivers' data
    for i, driver in enumerate(folder):
        # Join path
        path = os.path.join('drivers', driver)
        
        # Load 200 trajectory for each drivers
        test_trajectory = list()
        for j in range(1, 201):
            temp = numpy.genfromtxt(os.path.join(path, str(j)+'.csv'), 
                                    delimiter=',', skip_header=True)
            test_trajectory.append(temp)
        # Compute velocity
        test_trajectory = diff(test_trajectory)
        # Slice trajectory to multiple segments
        test_segments = slice(test_trajectory, w_len)
        # Learn dictionary
        D = spams.trainDL(numpy.asfortranarray(test_segments), 
                          K=k, lambda1=a, posAlpha=True, iter=-5)
        
        
        # Transform test trajectory to BoW
        test_histogram = bow(test_trajectory, D, a, w_len)
        mean_bow = numpy.mean(test_histogram, axis=0)
        K = inter_kernel(test_histogram, mean_bow.reshape([1, k]))
        
        # Write result to submission.csv
        f = open(file, "a")
        for j in range(1, 201):
            f.write(driver+'_'+str(j)+','+str(K[j-1, 0])+'\n')
        f.close()
'''
