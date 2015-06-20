# -*- coding: utf-8 -*-
"""
Created on Wed May 27 16:53:12 2015

@author: shih
"""
#import csv
import numpy, math, time 
import matplotlib.pyplot as plt

#--------transform x-y series to series of x_speed, y_speed, speed---------

def movingaverage(series, window_size):
  
    window = numpy.ones(int(window_size))/float(window_size)

    return numpy.convolve(series, window, 'same')
      

def interpolate(speed, mode=0):  
    # input: a list of speed array
    import numpy, csv, math
    '''        
        any pair larger than threshold should be interpolate
        PS. only happen at certain routes, may be no need to interpolate
    '''
    '''    
    with open('D:\dataSets\drivers\\' +str(driver)+ '\\' +str(tripNum)+ '.csv') as reader:
        trip = list(csv.reader(reader))
        del trip[0]
    '''    
    '''
    trip = numpy.array(trip,dtype=float)
    
    speed = trip_diff(trip)
    #mean = numpy.mean(speed)
    
    for i in range(5,len(speed)):
        if (speed[i]>15) & (speed[i-1]>15) &  ((speed[i] > 2*speed[i-1]) | (speed[i] < 0.5*speed[i-1])):            
            
            n= max(round(speed[i]/numpy.mean(speed[i-5:i]),0),1)
            #print [i+2, speed[i], speed[i-1] ]
            x_step = (trip[i+1][0] - trip[i][0])/n
            y_step = (trip[i+1][1] - trip[i][1])/n
            if x_step<10e-3:
                
                y = numpy.arange(trip[i][1], trip[i+1][1], y_step)
                x = numpy.ones((y.shape[0]))*trip[i][0]
            elif y_step <10e-3:
                x = numpy.arange(trip[i][0], trip[i+1][0], x_step)
                y = numpy.ones((x.shape[0]))*trip[i][1]
            else:
                x = numpy.arange(trip[i][0], trip[i+1][0], x_step)
                y = numpy.arange(trip[i][1], trip[i+1][1], y_step)
            print[x.shape, y.shape]
            if numpy.shape(x) != numpy.shape(y):
                length = min(x.shape[0], y.shape[0])
                x = x[0:length]
                y = y[0:length]
                
            temp = numpy.vstack((x, y))
            #print numpy.shape(temp)
            #print [i, step, x_step, y_step]
            #numpy.delete(trip,i,axis=0)
            #print [numpy.shape(trip[:i]), numpy.shape(numpy.transpose(temp)), numpy.shape(trip[i:])]
            trip = numpy.concatenate((trip[:i], numpy.transpose(temp), trip[i+1:]),axis=0)
            #print trip[i:i+(n+1)]
        else:
            continue
    '''    
    smooth = speed
    
    for j in range(5,len(speed)):
        if numpy.absolute(speed[j]-speed[j-1]) > 20 :
            #abnormal peak occur
            print [j, speed[j-3:j+3]]
            if mode:
                smooth = numpy.delete(speed,j)

            else:
                n= max(round(speed[j]/numpy.mean(speed[j-3:j]),0),1)
                
                step = round((speed[j+1] - speed[j-1])/n,1)
                temp = numpy.arange(speed[j-1], speed[j+1], step)
                print [n, step, temp]
                smooth = numpy.concatenate((smooth[:j-1], numpy.transpose(temp), smooth[j+1:]),axis=0)
                print smooth[j-2:j+n+2]
            raw_input("Press Enter to continue...") 
    return smooth
                   

def trip_diff(trip):     # input each trip as a float ndarray(numpy)
        
    trip = numpy.array(trip,dtype=float)
    speed = []
    '''
    #print len(trip)
    for i in range(1,len(trip)):
        #print [round(numpy.linalg.norm(trip[i] - trip[i-1]),2), trip[i] , trip[i-1]]      
        speed.append( round(numpy.linalg.norm(trip[i]-trip[i-1]),2))   
        #time.sleep(1)
    '''
    speed = numpy.round(numpy.sqrt(numpy.diff(trip[:,0])**2 + numpy.diff(trip[:,1])**2),2)
    #speed = movingaverage(speed, 5)
        
    return speed            


def trip2angle(trip):
    
    x_v = numpy.diff(trip[:,0])
    y_v = numpy.diff(trip[:,1])
    angle = [math.degrees(numpy.arctan(y/max(x, 0.01))) for x, y in zip(x_v, y_v)]
    #print angle
    #raw_input('hahahahahahhaahahhahahaha')
    # only handle rotation,  reflection will cause 'Complementary Remainder'
    # not rotation and reflection in preprocessing yet, so not negative angle
    return angle


def featEx(trip,segNum=1):     #numpy array, extract features and concanate them       
    #------------- transform to different type of data source ----------------
     
    #check trip length for numpy.split
    if (len(trip)%segNum)==0:
        #print 111
        segments = numpy.split(trip, segNum)
    elif (len(trip)%segNum) <  0.5*len(trip)/segNum:
        #print 222
        length = len(trip)- (len(trip)%segNum)
        sub_len = length/segNum
        #print (trip[:length,:]).shape
        segments = numpy.split(trip[:length :], segNum)
    else:
        #print 333
        length = len(trip)- (len(trip)%segNum)
        sub_len = length/segNum
        #print (trip[:length,:]).shape
        index =  range(0, len(trip), sub_len)
        index.append(len(trip)-1)
        segments = [trip[index[i]:index[i+1],:]  for i in range(len(index)-1)]

    # extract features from each segment
    jump = [0]
    features = [trip.shape[0]]  #1
    for j in range(segNum):
        distMove = 0
        dist2origin = [math.sqrt((trip[0,0])**2+(trip[0,1])**2)]
        trip = segments[j]
        for i in range(1,trip.shape[0]):
            temp = math.sqrt((trip[i,0])**2+(trip[i,1])**2)
            dist2origin.append(temp)
            temp = math.sqrt((trip[i,0]-trip[i-1,0])**2+(trip[i,1]-trip[i-1,1])**2)
            if temp > 50:
                jump.append(temp)

            distMove = distMove+temp

        features.extend([max(jump)])    #1
        dist2origin = numpy.asarray(dist2origin)
        speed = trip_diff(trip)     #ndarray

        accel = numpy.diff(speed)    #ndarray
        jerk = numpy.diff(accel)     #ndarray
        angle = trip2angle(trip)    #ndarray
        # ------------------- feature extraction ----------------------
        turnP1 = 0
        turnP2 = 0
        turnP3 = 0
        turnN1 = 0
        turnN2 = 0
        turnN3 = 0
        for i in range(3,len(angle)):
            if angle[i] - angle[i-1] > 45:
                turnP1 = turnP1 + 1
            elif angle[i] - angle[i-1] < -45:
                turnN1 = turnN1 + 1
            elif angle[i] - angle[i-2] > 45:
                turnP2 = turnP2 + 1
            elif angle[i] - angle[i-2] < -45:
                turnN2 = turnN2 + 1    
            elif angle[i] - angle[i-3] > 45:
                turnP3 = turnP3 + 1
            elif angle[i] - angle[i-3] < -45:
                turnN3 = turnN3 + 1
            else:
                continue

        distanceO = [max(dist2origin), numpy.mean(dist2origin), numpy.std(dist2origin)]
        pathEffi = math.sqrt((trip[0,0]-trip[trip.shape[0]-1,0])**2+
                            (trip[0,1]-trip[trip.shape[0]-1,1])**2) / max(distMove, 0.01)
        #angle_whole = math.degrees(numpy.arctan(abs(trip[0,1]-trip[trip.shape[0]-1,1])/max(abs(trip[0,0]-trip[trip.shape[0]-1,0]), 0.01) ))
        angles = [numpy.mean(angle), numpy.std(angle), sum(angle), turnP1, turnP2, turnP3, turnN1, turnN2, turnN3]
        #, turnP1, turnP2, turnP3, turnN1, turnN2, turnN3
        #turnEffi = angle_whole/max(sum(angle), 0.01)
        speed_s = [numpy.mean(speed), numpy.std(speed)]
        accel_s = [numpy.std(accel)]
        jerk_s  = [numpy.std(jerk)]
        features.extend([distMove, pathEffi])    #2
        features.extend(distanceO)      #3
        features.extend(angles)         #9
        features.extend(speed_s)        #2
        features.extend(accel_s)        #1
        features.extend(jerk_s)         #1

    features =  [round(features[x],2) for x in range(len(features))]
    return features


def quanFeat(trip):
    start = 0
    end = 0
    speed = movingaverage(trip_diff(trip),5)
    for i in range(trip.shape[0]-2):
        if speed[i] > 0.1:
            start = i
            break
    for i in range(trip.shape[0]-2,0,-1):
        if speed[i] > 0.1:
            end = i
            break
    #print [start, end]
    accel = movingaverage(numpy.diff(speed),5)   
    jerk = movingaverage(numpy.diff(accel),5)
    angle = movingaverage(trip2angle(trip),5)
    a_angle = movingaverage(numpy.diff(angle),5) 

    speed = numpy.percentile(speed, range(5,101,5))
    accel = numpy.percentile(accel, range(5,101,5))
    jerk = numpy.percentile(jerk, range(5,101,5))
    angle = numpy.percentile(angle, range(5,101,5))
    a_angle = numpy.percentile(a_angle, range(5,101,5))

    features = numpy.hstack((speed, accel, jerk, angle, a_angle))

    return features

#if __name__ == '__main__':


