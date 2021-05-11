# -*- coding: utf-8 -*-
"""
Created on Mon May 10 21:26:32 2021

@author: Iveta
"""

import os 
import numpy as np 
from sdtfile import * 
import matplotlib.pyplot as plt

def read_sdt(cwd, filename): 
    '''
    This function reads any Becker&Hickl single decay .sdt file, uses sdt 
    library (from sdtfile import *) to return the file in SdtFile 
    format. This file is later used in process_sdt() to read the individual
    decays and times in numpy array format.
    
    Parameters
    ----------
    cwd : the current working directory where your files are (obtain by os.getcwd())
    filename : string representing the name of the .sdt file 

    Returns
    -------
    data : data read by SdtFile

    '''
    cwd = cwd
    #path = os.path.join(cwd, folder)
    path = os.path.join(cwd, filename)
    data = SdtFile(path)
    return data

def read_gfactor(cwd): 
    '''
    This function reads the Gfactor single decay .sdt file into an SdtFile 
    type. It searches the current working directory for a file containing
    the 'gfactor' key word => the Gfactor file HAS TO CONTAIN 'gfactor' in
    its filename. If more than one file contain 'gfactor' in their filename, 
    the second file will be read. 

    Parameters
    ----------
    cwd : the current working directory where your files are (obtain by os.getwd())
    
    Returns
    -------
    data : data read by SdtFile

    '''
    for file in os.listdir(cwd): 
        if 'gfactor' in file and file.endswith('.sdt'): 
            path = os.path.join(cwd, file)
            print(file)
    data = SdtFile(path)
    return data
    
def process_sdt(data): 
    '''
    This function takes the single SdtFile anisotropy file, 
    and reads the perpendicular (1), parallel (2) decays 
    and the time bins into numpy arrays. 
    It assumes the time is in nanoseconds, and uses that to
    create a time vector of nanoseconds.

    Parameters
    ----------
    data :SdtFile anisotropy data with two channels (one for 
    each component - parallel and perpendicular)

    Returns
    -------
    perpdecay : uint16 array of the perpendicular decay histogram
    pardecay : uint16 array of the parallel decay histogram
    time : float64 array of the times in seconds
    bins: number of time bins 

    '''
    perpdecay = np.array(data.data[1]).flatten()     ##  M2 - perpendicular 
    pardecay = np.array(data.data[0]).flatten()     ## M1 - parallel 
    time = np.array(data.times[0])    
    bins = len(time)
    ns = 1000000000
    time = np.multiply(time,ns)   ## convert to ns 
    return perpdecay,  pardecay, time, bins


## plot decay in log 
def plot_decay(perpdecay, pardecay, time, scale, title = 'Fluorescence decay'): 
    '''
    Plots the perpendiculal and parallel decays.

    Parameters
    ----------
    perpdecay : perpendicular component of data as numpy arrays 
    pardecay : parallel component of data as numpy arrays
    time: time vector as numpy array
    scale : if want log scale, pass "log" as argument; the default is not log (i.e. linear)
    title : any string you want as title of your plot; The default is 'Fluorescence decay'.

    Returns
    -------
    None. Just shows the plot

    '''
    fig = plt.figure()
    plt.plot(time, perpdecay, label = 'perpendicular')
    plt.plot(time,pardecay, label = 'parallel')
    plt.xlabel('Time(ns)')
    plt.ylabel('Decay')
    plt.suptitle(title, fontsize= 16)
    plt.legend(loc = 'upper right')
    if scale == 'log':
        plt.yscale('log')
    plt.show()
    

def find_nearest_neighbor_index(array, points):
    '''
    
    Helper function for background_subtract(). 
    Takes the time array and the points chosen 
    by the user, and finds the indexes of the nearest
    time bin that the user-clicked point corresponds to. 
    
    '''
    store = []
    for point in points:
        distance = (array - point) ** 2
        idx = np.where(distance == distance.min()) 
        idx = idx[0][0] # take first index
        store.append(idx)
    return store

def background_subtract(perpdecay, pardecay, time):
    '''
    This function plots logarithmically the two decays 
    components. The user has to choose two points on 
    the plot to show the area where the background counts are 
    (i.e. before the peak) 
    The average photon intensity/counts within this time range 
    is calculated from the perpendicular decay (arbitrary choice) 
    and subtracted from each time bin in each decay.
    
    Requires find_nearest_neighbor_index() function. 

    Parameters
    ----------
    perpdecay : perpendicular component of data as numpy arrays
    pardecay : parallel component of data as numpy array
    time: time vector as numpy array

    Returns
    -------
    BG_indices : TYPE
        DESCRIPTION.
    perpdecay : background subtracted perpendicular decay data as numpy array 
    pardecay : background subtracted parallel decay data as numpy array 
    BG : the averaged background used for the subtraction 

    '''
    
    # Plot decay
    x = time
    y1 = perpdecay
    y2 = pardecay
    #plt.switch_backend('Qt5Agg')
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.xlabel('Time (ns)') 
    plt.ylabel('Intensity (photons)') 
    plt.yscale('log')
    plt.title('Select background range (2 points)') 
    plt.show() 
    
    # Select background region and draw on plot
    boundaries = plt.ginput(2)
    
    # Update L with x_points array
    boundaries_x = []
    for i in range(len(boundaries)):
        boundaries_x.append(boundaries[i][0])
        plt.axvline(boundaries_x[i], color='r')  # add vertical line 
    plt.close()    
    # Get indices for the background delimiters- i.e. the number of counts in these bins 
    BG_indices = find_nearest_neighbor_index(time, boundaries_x)
    # Calculate background by averaging - use perpendicular decay 
    # BG_sum - sum of intensities in this time window (BG_indices[0] and [1])
    BG_sum = sum([perpdecay[i] for i in range(BG_indices[0], BG_indices[1] + 1)])
    # BG_indices[1] - BG_indices[0] + 1 => how many time bins selected? 
    BG = BG_sum / (BG_indices[1] - BG_indices[0] + 1)

    # Subtract background from decay
    perpdecay = perpdecay - BG
    pardecay = pardecay - BG
    
    plot_decay(perpdecay, pardecay, time, 'log', 'background corrected')
    
    # Replace all negative values by 0
    #perpdecay = np.where(perpdecay < 0, 0, perpdecay)
    #pardecay = np.where(pardecay < 0, 0, pardecay)
   
    return  BG_indices, perpdecay, pardecay, BG


def get_peaks(perp, par): 
    '''
    Finds the time bin index of the peak for both the perpendicular 
    and the parallel decay. 

    Parameters
    ----------
    perpdecay : perpendicular component of data as numpy arrays
    pardecay : parallel component of data as numpy array

    Returns
    -------
    peak_perp : index of peak 
    peak_par : index of peak 

    '''
    peak_perp = np.argmax(perp)
    peak_par = np.argmax(par)
    return peak_perp, peak_par




def align_peaks(perpdecay,pardecay, time): 
    '''
    Takes three arguments: the arrays of perpendicular and parallel decays and 
    the time array. 
    
    Then calculates the shift between the peaks of the two decays and modifies
    whichever is the second peak to align to the first one. 
    It shows a plot of the aligned peaks. 
    
    Returns four variables: 
        the decays of the aligned peaks, 
        the peak index 
        the shift (in index values)
    
    '''
    # find INDEX of max value of each decay 
    maxindexperp = np.argmax(perpdecay)
    maxindexpar = np.argmax(pardecay)
    shift = maxindexpar - maxindexperp
    
    if shift > 0:  # if perpendicular decay first => shift the parallel one 
        pardecay = np. pad(pardecay, (0, shift), mode = 'constant')[shift:]
        peak_index = np.argmax(pardecay)
    else:    # and if par decay first, shift the perp one by *shift* 
        perpdecay = np.pad(perpdecay,(0, abs(shift)), mode = 'constant')[abs(shift):] 
        peak_index = np.argmax(perpdecay)
            
    
    plot_decay(perpdecay, pardecay, time, 'log', 'Peaks aligned' )
    return perpdecay, pardecay, peak_index, shift

def get_gfactor(Gfact, t, peak_idx):
    '''
    This function takes the G factor array and makes an interactive plot,
    starting from just after the peak. The user has to choose two points where
    the G factor array has a somewhat constant fluctuation (about the middle
    of the time series). The G factor value is calculated by taking an average 
    from the user-selected range.                                                      
                                                                
    Requires the find_nearest_neighbor_index() function. 
    
    Parameters
    ----------
    Gfact : Gfactor array obtained by the element-wise division of the parallel
    by the perpendicular decay
    
    t : time array 
    
    peak_idx : time index of the peak 

    Returns
    -------
    Gvalue : a single float G-value. 

    '''
    plt.plot(t, Gfact)
    plt.xlim([t[peak_idx],50])
    plt.ylim([-5,5])
    plt.suptitle('Select G factor range')
    
    # select area that os 
    boundaries = plt.ginput(2)
    
    # Update L with x_points array
    boundaries_x = []
    for i in range(len(boundaries)):
        boundaries_x.append(boundaries[i][0])
        plt.axvline(boundaries_x[i], color='r')  # add vertical line 
    plt.close()    
    G_indices = find_nearest_neighbor_index(t, boundaries_x)
    G_sum = sum([Gfact[i] for i in range(G_indices[0], G_indices[1] + 1)])
    # BG_indices[1] - BG_indices[0] + 1 => how many time bins selected? 
    Gvalue = G_sum / (G_indices[1] - G_indices[0] + 1)
    return Gvalue

# some small helpers functions 

def count_zeros(perpdecay, pardecay, bins = 4096):
    '''
    Returns the number of zero values in the perpendicular 
    and parallel decay, respectivaly. Requires the number
    of time bins - default value is 4096.

    '''
    zerosperp = bins - np.count_nonzero(perpdecay) #3316
    zerospar = bins - np.count_nonzero(pardecay) #3313
    return zerosperp, zerospar

def count_negatives(perpdecay, pardecay): 
    '''
    Returns the number of negative values in the perpendicular 
    and parallel decay, respectivaly. 
    '''
    negativeperp = np.sum(np.array(perpdecay) < 0)
    negativepar = np.sum(np.array(pardecay) < 0)
    return negativeperp, negativepar