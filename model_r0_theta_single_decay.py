# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 10:08:36 2021

@author: Iveta
"""

import numpy as np
from scipy.signal import savgol_filter
import scipy.optimize
from anisotropy_functions import align_peaks
import matplotlib.pyplot as plt
def model_sim_r0_tau_single_decay(perp, par, time, G, end_idx, theta, r0, SG_window, SG_polynomial = 4):     
    #bounds = bounds
    ''' 
    This model fits simultaneously both perpendicular and parallel decays 
    to the respective integrated intensity function for r0
    For parallel: 1/3 *T * (1 + 2r0)*exp(t/theta)
    For perpendicular: 1/3 * T * (1-r0)*exp(t/theta)
    '''
    
    total_decay = np.add(par, (2*G*perp))
    r_0 = r0  # initial params 
    theta_0 = theta
    
    #r_decay = np.empty_like(perp, dtype = float) # empty image to hold anisotropy decays
    #perp_decay = np.empty_like(perp, dtype = float) # empty image to hold anisotropy decays
    #par_decay = np.empty_like(par, dtype = float) # empty image to hold anisotropy decays
        
    ######### 
       
    # remove 0s 

    # smooth - get single decay 
    perp_s = savgol_filter(perp, 9, 4)
    #plt.pause(10)
    #plt.close()
    par_s = savgol_filter(par, 9, 4)
    #plt.pause(10)
    #plt.close()
    total_s = savgol_filter(total_decay, 9, 4)
    #plt.pause(10)
    #plt.close()
               
    perp_s = np.where(perp_s < 1, 1, perp_s)
    par_s = np.where(par_s < 1, 1, par_s)
      
    # normalize smoothed with height of decay to ensure ratio is kept  
    perp_s = perp_s * (max(perp)/max(perp_s))
    par_s = par_s * (max(par)/max(par_s))
    total_s = total_s * (max(total_decay)/max(total_s))

    #print_decay_stats(perp_s)
    # print_decay_stats(par_s)

    # get weights 
    #weights_perp = 1/np.sqrt(perp_s)
    #weights_par = 1/np.sqrt(par_s)
    #weights_total =np.divide(1, np.sqrt(totalimage_s))
    
    # align peaks of smoothed decay
    perp_s, par_s, peak_index, _ = align_peaks(perp_s, par_s, time, plot = False)
    #plt.pause(10)
    #plt.close()
    perp_s, total_s, peak_index, _ = align_peaks(par_s, total_s, time, plot = False)                
    #plt.pause(10)
    #plt.close()
     
    # choose selected time range for entire image - here raw
    y_perp = perp[peak_index:]
    y_par = par[peak_index:]
    total = total_decay[peak_index:]   
    t = time[peak_index:]

    # model functions - one for perp and one for par 
    # get total intensity at given time point from totalimage_s[t]
    # t,total = given vars; r0, theta - to be optimized 
    model_func_par = lambda t, total, p1, p2:  (total/3)*(1+2*p1)*np.exp(-t/p2)
    model_func_perp = lambda t, total, p1, p2: (total/3)*(1-p1)*np.exp(-t/p2) 
    
    #objective function to minimize 
    objective = lambda p, t, total, y_perp, y_par: norm((model_func_par(t,total, p[0], p[1]) - y_par)**2) + norm((model_func_perp(t,total, p[0], p[1]) - y_perp)**2)
    
    '''
    p
    p[0] = _r0 
    p[1] =  theta_0
    
    ''' 
    
    # initial guesses 
    p10, p20 = r_0, theta_0 
    
    #method = 'BFGS'
    res = minimize(objective, x0 = [p10,p20], method='BFGS', 
                   args = (t, total, y_perp, y_par))
    
    print(res.message)
    
    if res.hess_inv is None:  
        print('unsuccessful')
    else: 
        r0 = res.x[0]   # r0 
        theta = res.x[1]  # tau    
        cov_x = res.hess_inv   
    
    #r0 = res.x[0]
    #theta = res.x[1]
    #cov_x = res.hess_inv
    # plot model: 
    y_perp_fit = (total/3)*(1-r0)*np.exp(-t/theta) 
    y_par_fit = (total/3)*(1+2*r0)*np.exp(-t/theta)
    
    fig, ax = plt.subplots(1,2)    
    ax[0].plot(t, y_perp_fit, 'r', label = 'fit')
    ax[0].plot(t, perp[peak_index:end_idx], label = 'raw')
    ax[0].set_title('Perpendicular decay')
    ax[1].plot(t, y_par_fit, 'r', label = 'fit')
    ax[1].plot(t, par[peak_index:end_idx], label = 'raw')
    ax[1].set_title('Parallel decay')
    for ax in ax.flat: 
        ax.set(xlabel = 'Time (ns)', ylabel = 'Counts')
    plt.show()
            
    return r0, theta, res, cov_x