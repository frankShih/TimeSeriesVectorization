# -*- coding: utf-8 -*-
"""
Created on Wed May 27 16:53:12 2015

@author: shih
"""
#import csv


#--------transform x-y series to series of x_speed, y_speed, speed---------


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
    import numpy, time       
    
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
    
    return speed            

