# -*- coding: utf-8 -*-
"""
Created on Mon May 10 21:26:32 2021

@author: Iveta
"""

'''

Experimental assumptions under this work: 
    sdt.data[0] - M2 TCSPC card - perpendicular decay 
    sdt.data[1] - M1 TCSPC card - parallel decay 
    in case of lifetime data where the parallel detector has been shut down, sdt.data[1] will be empty 

The data has to be reshaped in (t,x,y) format, in accordance with numpy array shape convention. 

Future: 
    0. parametrise the functions so they can easily be done for anisotropy as well as lifetime decays 
    1. control when matplotlib plots close and stay open  
    2. intergrate G factor in the main code 
    3. interactive plots - draw vertical lines and save them after they are closed 
    4. log file containing the following info 
    - chosen indexes 
    - 
'''

import os 
import numpy as np 
from sdtfile import * 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def create_lifetime_decay_img(perp, par, G): 
    '''
    This function takes the perpendicular and parallel images 
    and creates a lifetime image by pixel-wise application 
    of the formula Ipar + 2 * G * Iperp. 
    It is important the input image shape is (t,x,y), i.e. the time 
    channel is first and both images have identical shape. 
    
    Parameters
    ----------
    perp : perpendicular image, a 3D array of shape (t,x,y)
    par : parallel image, a 3D array of shape (t,x,y)
    
    Returns
    -------
    lifetime : a single 3D array (image) containing lifetime decays 
    in each pixel. 
    '''
    if perp.shape != par.shape: 
        print('Per and par images have uneven size, check transform!')
        #break 
    else: 
        pass 
    dims = perp.shape
    t = dims[0]
    x = dims[1]
    y = dims[2]
    lifetime = np.empty_like(perp)
    for i in range(x):  
        for j in range(y): 
            lifetime[:,i,j] = np.add(par[:,i,j], (2.0*G*perp[:,i,j]))
    return lifetime 

def bin_image(data):
    '''
    This function takes an image stack (3D array) and performs a 
    3x3 binning (or bin=1 by SPCImage / Becker & Hickl convention), 
    by summing the photons of the immediate neighbouring pixels in each
    stack/slice/time channel. 
    The image data has to be of the shape (t,x,y), i.e. the time
    channel dimension has to be first. This can be achieved by 
    data.transform((2,0,1))

    Parameters
    ----------
    data : image data, a 3D array (e.g. x, y and time dimension)
    xdim : size of x dimension of image 
    ydim : size of y dimension of image 
    t : number of stacks/slices/time channels 
        
    Returns
    -------
    binned : 3x3 binned image ; array of float64 

    '''
    dims = data.shape 
    t = dims[0]
    x = dims[1]
    y = dims[2]
    print(f'Image dimensions x by y by t: {x,y,t}')
    #print(f'Image.shape: {data.shape}')
    binned = np.zeros_like(data)   # create empty array of same dims to hold the data
    #print(f'Empty bin dimensions: {binned.shape}')
    #print(binned)
    for i in range(x):  # iterate columns 
        #print(f'x = {i} of total {x} pixels ')
        for j in range(y):   # iterate along rows
            #print(f'y = {j} of total {y} pixels ')
            if i == 0:  # entire first row  
                if j == 0:   # top left corner 
                    binned[:,i,j] = np.sum([[data[:,i,j] for j in range(j,j+2)] for i in range(i, i+2)], axis=0).sum(axis=0) 
                elif j == y-1: # top right corner
                    binned[:,i,j] = np.sum([[data[:,i,j] for j in range(j-1,j+1)] for i in range(i, i+2)], axis=0).sum(axis=0)
                else: # on edge (non-corner) of first row 
                    binned[:,i,j] = np.sum([[data[:,i,j] for j in range(j-1,j+2)] for i in range(i, i+2)], axis=0).sum(axis=0) 
                        
            elif i == x-1 :  # entire last row  
                if j == 0:   # bottom left corner 
                    binned[:,i,j] = np.sum([[data[:,i,j] for j in range(j,j+2)] for i in range(i-1, i+1)], axis=0).sum(axis=0) 
                elif j == y-1:  # bottom right corner
                    binned[:,i,j] = np.sum([[data[:,i,j] for j in range(j-1,j+1)] for i in range(i-1, i+1)], axis=0).sum(axis=0)  
                else: # on edge (non-corner) of last row 
                    binned[:,i,j]= np.sum([[data[:,i,j] for j in range(j-1,j+2)] for i in range(i-1, i+1)], axis=0).sum(axis=0)  
                
            elif j == 0: # entire first column, corners should be caught by now 
                binned[:,i,j] = np.sum([[data[:,i,j] for j in range(j,j+2)] for i in range(i-1, i+2)], axis=0).sum(axis=0)   
                
            elif j == y-1: # entire last column 
                binned[:,i,j]= np.sum([[data[:,i,j] for j in range(j-1,j+1)] for i in range(i-1, i+2)], axis=0).sum(axis=0)    
                
            else:      # if on the inside of matrix
                binned[:,i,j]= np.sum([[data[:,i,j] for j in range(j-1,j+2)] for i in range(i-1, i+2)], axis=0).sum(axis=0)
    return binned 

def projection(data, xdim, ydim, t):
    '''
    This function takes an image stack (3D array) and sums up all 
    values in all time bins into a single plane i.e. creates a 
    projection of the 3D image. 

    Parameters
    ----------
    data : image data, a 3D array (e.g. x, y and time dimension)
    xdim : size of x dimension of image 
    ydim : size of y dimension of image 
    t : number of stacks/slices/time channels 
        
    Returns
    -------
    binned : 3x3 binned image ; array of float64 

    '''
    bins = t
    x  = xdim
    y = ydim
    #print(f'Image dimensions from params: {x,y,t}')
    #print(f'Image.shape: {data.shape}')
    projection = np.empty((bins,x,y))   # create empty array of same dims to hold the data
    #print(f'Empty bin dimensions: {binned.shape}')
    #print(binned)
    for i in range(x):  # iterate columns 
        for j in range(y):   # iterate along rows
            projection[:,i,j] = np.sum(data[:,i,j])  # [:] - sums all of them along the time axis                 
    return projection[0,:,:] 

def convolve_IRF(IRF,decay,time): 
    '''
    This function convoles a lifetime decay with an instrument response function. 

    Parameters
    ----------
    IRF : numpy array of instrument response function
    decay : numpy array of decay we want to convolve 
    time : numpy array of time vector in nanoseconds 

    Returns
    -------
    convolved : the original convolved decay 
    convolvedNorm : the convolved decay normalised to the decay counts

    '''
    dx = time[1] - len[0]   # in ns, difference between each bin 
    convolved = np.convolve(IRF, decay)[:len(time)]*dx 
    convolvedNorm = convolved*(max(decay)/max(convolved))  # for the plotting normalise
    plt.plot(time, decay, label = 'raw decay')
    plt.plot(time, convolvedNorm, label = 'convolved decay')
    plt.legend(loc = 'upper right')
    plt.show()    
    return convolved, convolvedNorm
    
def read_sdt(cwd, filename): 
    '''
    This function reads any Becker&Hickl .sdt file, uses sdt 
    library (from sdtfile import *) to return the file in SdtFile 
    format. This file is later used in process_sdt() to read the individual
    decays and times in numpy array format.
    
    Parameters
    ----------
    cwd : the current working directory where your files are (obtain by os.getcwd())
    filename : string representing the name of the .sdt file 

    Returns
    -------
    data : data read by SdtFile - SdtFile object 

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


def process_sdt_image(data): 
    '''
    This function takes the SdtObject (raw anisotropy image data) 
    and returns the perpendicular and parallel matrix decays as 3D
    numpy arrays (third dimension is the photon count in each of the time bins).

    Parameters
    ----------
    data : SdtObject (image with parallel and perpendicular decay component)

    Returns
    -------
    perpimage : uint16 : 3D numpy array holding perpendicular image data 
    parimage : uint16 : 3D numpy array holding perpendicular image data 
    time : float64 1D numpy array holding the time in seconds 
    timens : float64 1D numpy array holding the time in nanoseconds 
    bins : scalar, number of time bins 

    '''
    perpimage = np.array(data.data[0]) # return an x by y by t matrix 
    parimage = np.array(data.data[1])
    time = np.array(data.times[0])   # return array of times in seconds 
    bins = len(time)
    ns = 1000000000
    timens = np.multiply(time,ns)  # time array in nanoseconds 
    return perpimage, parimage, time, timens, bins 
    

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
    #plt.savefig('Gfactor_range')
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
    plt.close()
    # Replace all negative values by 0
    #perpdecay = np.where(perpdecay < 0, 0, perpdecay)
    #pardecay = np.where(pardecay < 0, 0, pardecay)
   
    return  BG_indices, perpdecay, pardecay, BG

def background_subtract_lifetime(data, time):
    '''
    This function plots logarithmically the single decay. 
    The user has to choose two points on 
    the plot to show the area where the background counts are 
    (i.e. before the peak) 
    The average photon intensity/counts within this time range 
    is calculated and subtracted from each time bin in the decay.
    
    Requires find_nearest_neighbor_index() function. 

    Parameters
    ----------
    data : numpy array that holds the data 
    time: time vector as numpy array

    Returns
    -------
    BG_indices : 
    data : background subtracted perpendicular decay data as numpy array 
    BG : the averaged background used for the subtraction 

    '''
    
    # Plot decay
    x = time
    y = data
    #plt.switch_backend('Qt5Agg')
    plt.plot(x, y)
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
    #plt.savefig('Gfactor_range')
    plt.close()    
    # Get indices for the background delimiters- i.e. the number of counts in these bins 
    BG_indices = find_nearest_neighbor_index(time, boundaries_x)
    # Calculate background by averaging - use perpendicular decay 
    # BG_sum - sum of intensities in this time window (BG_indices[0] and [1])
    BG_sum = sum([data[i] for i in range(BG_indices[0], BG_indices[1] + 1)])
    # BG_indices[1] - BG_indices[0] + 1 => how many time bins selected? 
    BG = BG_sum / (BG_indices[1] - BG_indices[0] + 1)

    # Subtract background from decay
    data = data - BG
    
    #plot_decay(perpdecay, pardecay, time, 'log', 'background corrected')
    plt.close()
    # Replace all negative values by 0
    #perpdecay = np.where(perpdecay < 0, 0, perpdecay)
    #pardecay = np.where(pardecay < 0, 0, pardecay)
   
    return  BG_indices, data, BG

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
    print(f'Perpendicular peak at index {peak_perp}')
    print(f'Parallel peak at index {peak_par}')
    return peak_perp, peak_par


def align_peaks(perpdecay,pardecay, time): 
    '''
    Takes three arguments: the arrays of perpendicular and parallel decays and 
    the time array. 
    
    Then calculates the shift between the peaks of the two decays and modifies
    whichever is the second peak to align to the first one. 
    It shows a plot of the aligned peaks. 
    
    Returns four variables: 
        the decays of the peaks,aligned  
        the peak index - index of the maximum counts 
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
    plt.pause(5)
    plt.close()
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
    plt.ylim([0,2])
    plt.suptitle('Select G factor range')
    plt.ylabel('I(par)/I(perp)')
    plt.xlabel('Time (ns')
    
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


def plot_lifetimes(DOPCtotal, DPPCtotal, time, scale = True, title = 'Lifetime decays'): 
    '''
    Accepts the total intensity of two different samples (here DOPC and DPPC:Chol, in this order), 
    calculated by Ipar + 2*G*Iperp, normalizes according to the decay with the greater intensity
    and plots them on a logarithmic scale.
    
    It is possible to plot the non-normalized decays by passing scale = False, as well as 
    to change the default title, by passing title = 'My title'. 
    '''
    if scale: 
        # find which one has the greater intensity and scale the other decay 
        # according to it 
        if DOPCtotal.max() > DPPCtotal.max(): 
            scaler = MinMaxScaler(feature_range = (DOPCtotal.min(), DOPCtotal.max()))
            DPPCtotal = scaler.fit_transform(DPPCtotal.reshape(-1,1))
        else: 
            scaler = MinMaxScaler(feature_range = (DPPCtotal.min(), DPPCtotal.max()))
            DOPCtotal = scaler.fit_transform(DOPCtotal.reshape(-1,1))
    plt.plot(time, DOPCtotal, label = 'DOPC')
    plt.plot(time, DPPCtotal, label = 'DPPC:Chol')
    plt.legend(loc = 'upper right')
    plt.suptitle(title, fontsize = 16)
    plt.yscale('log')
    plt.ylabel('Normalized counts')
    plt.xlabel('Time (ns')
    plt.show()
    
    

def plot_anisotropy_decay(perp, par, G, time, bgcorr = True, align = True, from_peak = True, 
                         title = 'Anisotropy decay'):
    '''
    This function takes the perp and parallel component, as well as the time vector 
    and the G value. After background subtraction (requires user input) and peak alignment, 
    it calculates the anisotropy decay as per [(Ipar - GIperp)/Itotal] and plots it from 
    the peak onwards (pass from_peak = False to plot the entire decay). 
    
    Returns the full anisotropy decay decay, as well as the peak index of the aligned decays. 
    '''
    # first bg correction 
    if bgcorr: 
        _, perp, par, _ = background_subtract(perp, par, time)
    else: 
        perp = perp 
        par = par
    # second peak alignment 
    if align: 
        perp,par, peak_idx, shift = align_peaks(perp,par,time)
    else: 
        perp = perp
        par = par
        peak_idx, _ = get_peaks(perp,par)
    total = np.add(par, (2*G*perp))
    r_decay = np.divide(np.subtract(par, G*perp),total)
    plt.plot(time, r_decay)
    if from_peak: 
        plt.xlim(time[peak_idx], 50)  # plot only from peak 
    plt.ylim(-0.5,1)
    plt.suptitle(title, fontsize = 16)
    plt.xlabel('Time (ns')
    plt.ylabel('Anisotropy (r)')
    #plt.close()
    return r_decay, peak_idx
    
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