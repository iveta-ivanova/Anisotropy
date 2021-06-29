# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 09:37:11 2021

@author: Iveta
"""
import numpy as np 
from scipy.signal import savgol_filter
import scipy.optimize

def model_biexp_tau_image(image, timens, mask, A, tau1, tau2, SG_window = 5, SG_polynomial = 2): 
    '''
    This function uses the least squared metho to fit a biexponential decay 
    (A*exp(t/tau1) + (1-A)*exp(t/tau2) 
    to individual pixels of a single TCSPC image (lifetime, or single anisotropy component). 
    A 3D mask is used to apply a pre-defined photon count threshold, 
    Weights are obtained by applying a Savitzky-Golay filter to each decay. 
    
    Arguments: 
        
        image - TCSPC image (3D numpy array) as proccessed from process_sdt_img
        timens - 2D array holding the time vector in ns 
        mask - 3D mask (boolean 3D array) to indicate which pixels should be modelled 
        A - initial guess for preexponential factor, scalar 
        tau1 - initial guess for long lifetime component, scalar
        tau2 - initial guess for short lifetime component, scalar
        SG_window - window for Savitzky-Golay filter (default = 5)
        SG_polynoial - polynomial order for S.G. filter (default =2))

    Returns: 
        decay_fit - 3D array holding the fitted function (from the peak)
        A_img - 2D array holding the pre-exponential factor
        tau1_img - 2D array holding the long lifetime compoent
        tau2_img - 2D array holding the short lifetime component 
        chi2_img - 2D array holding the pre-expo
        
    '''
    # initial params 
    A_0 = A
    tau1_0 = tau1
    tau2_0 = tau2 
    
    t,x,yz = image.shape
    
    # 3d image to hold the fit 
    decay_fit = np.empty_like(image, dtype = float) 
    
    # 2d to hold the params - tau1, tau2, A 
    tau1_img = np.empty((image.shape[1], image.shape[2]))
    tau2_img = np.empty((image.shape[1], image.shape[2]))
    A_img = np.empty((image.shape[1], image.shape[2]))
    chi2_img = np.empty((image.shape[1], image.shape[2]))
    
   
    for i in range(x):  # create anisotropy and lifetime images 
        for j in range(yz):   
            if mask[:, i, j].any() == False:   # if not segmented - set to 0 or NaN?  
                #print(f'Pixel at {i},{j} is False.')    
                   # 3d image to hold the fit 
                decay_fit[:, i, j] = np.nan
                
                # 2d to hold the params - tau1, tau2, A 
                tau1_img[i,j] = np.nan
                tau2_img[i,j] = np.nan
                A_img[i,j] = np.nan
                chi2_img[i,j] = np.nan
            else: 
                image_s = savgol_filter(image[:,i,j], window_size = SG_window, polynomial = SG_polynomial, plot = False)
                image_s = np.where(image_s < 0.1, 0.1, image_s)
                w = 1/np.sqrt(image_s)
                
                peak_idx = np.argmax(image[:,i,j])
                
                # chop of data from peak 
                t = peak_idx-1
                x = timens[t:]
                y = image[t:,i,j]
                w = w[t:]
                
                model_fcn = lambda x,p1,p2,p3: p1*np.exp(np.divide(x,p2)) + (1-p1)*np.exp(np.divide(x,p3))
                '''
                p1 = A 
                p2 = tau1 
                p2 = tau2 
                '''
                
                err_func = lambda p, x, y, w: (y - model_fcn(x, p[0], p[1], p[2])) * w
                
                # initial params - see in arguments 
                p01, p02, p03 = A_0, tau1_0, tau2_0
                
                params_fit, cov_x, infodict, mesg, ier = scipy.optimize.leastsq(err_func, 
                                                                                x0 = [p01,p02,p03],
                                                                                args = (x,y,w),
                                                                                full_output = True)
                ''' 
                params_fit[0] = A 
                params_fit[1] = tau1 
                params_fit[2] = tau2 
                
                '''
                s_sq = None
                if (len(y) > len([p01, p02, p03])) and cov_x is not None:
                    print(f' Pixels {i} x {j} - convergence reached')
                    s_sq = (err_func(params_fit, x, y, w)**2).sum()/(len(y)-len([p01, p02, p03]))
                    cov_x = cov_x * s_sq
                    
                else:
                    cov_x = np.inf
                    print(f'At pixel {i} x {j} convergence not reached')
                  
                # populate 
                decay_fit[t:,i,j] = model_fcn(x, params_fit[0], params_fit[1],params_fit[2])                
                A_img[i,j] = params_fit[0]
                tau1_img[i,j] = params_fit[1]
                tau2_img[i,j] = params_fit[2]
                chi2_img[i,j] = s_sq
                
    return decay_fit, A_img, tau1_img, tau2_img, chi2_img