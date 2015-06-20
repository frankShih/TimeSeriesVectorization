# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 15:11:03 2015

@author: shih
"""
import numpy

def euclideanMatch(driver, other):
    # input : list of legal driver, other drivers
    total = driver + other
    kernelM = numpy.zeros((len(total),len(total)))
    for i in range(len(total)):
        for j in range(len(total)):
            temp1 = numpy.asarray(total[i])
            temp2 = numpy.asarray(total[j])
            if  temp1.shape[0] > temp2.shape[0]:
                #different lemngth -> add zeros
                temp2 = numpy.concatenate((temp2,numpy.zeros(temp1.shape[0] - temp2.shape[0])),0)            
                
            elif temp1.shape[0] < temp2.shape[0]:
                temp1 = numpy.concatenate((temp1,numpy.zeros(temp2.shape[0] - temp1.shape[0])),0)            
                
            else:
                print 'same length, doing onthing'
                

        kernelM[i,j] = numpy.linalg.norm(temp1-temp2)
        
    return kernelM[:len(driver),:], kernelM[len(driver):,:]
    
    
