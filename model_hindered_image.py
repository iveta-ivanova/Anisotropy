# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 10:03:13 2021

@author: Iveta
"""

import numpy as np
from scipy.signal import savgol_filter
import scipy.optimize
from anisotropy_functions import align_peaks

def model_hindered_image(perpimage, parimage, timens, G, limit_idx, mask, theta, r0, rinf, model = 'hindered'):
    t,x,y = perpimage.shape
    if perpimage.shape == parimage.shape:
        t,x,y = perpimage.shape
    else: 
        print('Images have uneven size or shape!')
    
    tau_img = np.empty_like(perpimage)   # empty image to hold 
    ani_img = np.empty_like(perpimage, dtype = float) # empty image to hold anisotropy decays
    r_fit_img = np.empty_like(perpimage) # this will hold the fit 
    r0_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    rinf_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    theta_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    #chi2_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    #r0_err_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    #rinf_err_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    #theta_err_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    #chi2_err_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
       
    # use mask to eliminate the low intensity values     
    for i in range(x):  # create anisotropy and lifetime images 
        for j in range(y):   # align peaks in-place
            if mask[:, i, j].any() == False:   # if not segmented - set to 0 
                tau_img[:,i,j] = 0
                ani_img[:,i,j] = 0
            else: 
                print(f'Start analysis of pixel position {i,j}')
                perpimage[:,i,j], parimage[:,i,j], peak_index, _ = align_peaks(perpimage[:,i,j], parimage[:,i,j], timens, plot = False)                 
                print("Peaks aligned")
                # select range between peak and user selected index 
                #perpimage = perpimage[peak_index:limit_idx,i,j] # all pixels, selected time bin range 
                #parimage = parimage[peak_index:limit_idx,i,j]  # all pixels, selected time bin range
                #t = timens[peak_index:limit_idx]      # single decay, selected range  
                
                # now calculate anisotropy decay of each pixel - full range
                #print(f'Shapes : tau_img: {tau_img.shape}, perp image: {perpimage.shape}, par image: {parimage.shape}')
                tau_img[:,i,j] = np.add(parimage[:,i,j], (2*G*perpimage[:,i,j])) # calculate lifetime 
                ani_img[:,i,j] = np.divide(np.subtract(parimage[:,i,j], G*perpimage[:,i,j]),tau_img[:,i,j])   # calculate anisotropy image
                
                # now choose range between peak and user-selected peak 
                tau_img = tau_img[peak_index:limit_idx,:,:]
                ani_img = ani_img[peak_index:limit_idx,:,:]
                
                print(f'This anisotropy image has a total of {sum(np.isnan(ani_img[:,i,j]))} NaN values, i.e. {((sum(np.isnan(ani_img[:,i,j]))/ani_img.shape[0])*100).round(1)} % of all bins.')
                # select range between peak and user selected index 
                #r = ani_img[peak_index:limit_idx,i,j]  # single decay, selected range 
                #t = timens[peak_index:limit_idx]      # single decay, selected range  
                
                # smoothing 
                varians = savgol_filter(ani_img[:,i,j], window_length = 35 , polyorder = 2)
                #varians = np.where(varians < 0.1, 0.1, varians)
                w = 1/np.sqrt(varians)   # weights  
                               
                # initial parameters 
                theta_0 = theta 
                r0_0 = r0 
                rinf_0 = rinf 
                
                '''
                Constant parameters: 
                    r
                    t
                    w*
                
                Parameters we are optimising (independent/decision/design variables) : 
                    p0 = r0
                    p1 = rinf
                    p2 = theta
                    
                '''
                # arguments we pass : t + params  that will be optimized when this is finished
                model_func = lambda t, r0, rinf, theta: (r0-rinf)*np.exp(-np.divide(t,theta)) + rinf
                                    
                # arguments we pass: r, t, w + optimizing params as a list p
                # calculate residuals - difference between real decay r and the proposed model 
                # weighted by the weights we calculated above
                # p is a list of independent vars that will be defined later 
                error_func = lambda p, r, t, w: (r - model_func(t, p[0], p[1], p[2]))*w
            
                # initial guesses - p - independent variables
                #p = [r0, rinf, theta]
                p0,p1,p2 = r0_0, rinf_0, theta_0
            
                # minimise sum of squares 
                # x0 - starting estimate - initial guesses 
                # args = the additional arguments to the function 
                full_output = scipy.optimize.leastsq(error_func, x0 = [p0, p1, p2],
                                              args = (r,t,w),
                                              full_output = True)
                
                params_fit, cov_x, infodict, mesg, ier = full_output
                
                if cov_x is None: 
                    print('NLLS could not converge')      
                else: 
                
                    r0, rinf, theta = params_fit   # the solutions found
                    r0_img[i,j] = r0
                    rinf_img[i,j] = rinf
                    theta_img[i,j] = theta
                    '''
                    Solutions : 
                        params_fit[0] = r0 
                        params_fit[1] = rinf
                        params_fit[2] = theta
                    
                    '''
                    
                    #print('full_output - cov_x: {}'.format(cov_x))
                    
                    ## Estimate fit_parameter errors
                    if (len(r) > len([p0, p1, p2])) and cov_x is not None:
                        # calculate reduced chi square (s_sq)
                        s_sq = (error_func(params_fit, r, t, w)**2).sum()/(len(r)-len([p0, p1, p2]))
                        cov_x = cov_x * s_sq
                    else:
                        cov_x = np.inf
                        s_sq = None
                    
                    error = []
                    
                    for i in range(len(params_fit)): 
                        try: 
                            error.append(np.absolute(cov_x[i][i])**0.5)
                        except: 
                            error.append(0.00)
                                                
                    #pfit_leastsq = params_fit
                    #perr_leastsq = np.array(error)
                    
                    #print("\n# Fit parameters and parameter errors from leastsq method :")
                    #print("pfit = ", pfit_leastsq)
                    #print("perr = ", perr_leastsq)
                    #print('r0 = {} with error = {}'.format(pfit_leastsq[0],perr_leastsq[0]))
                    #print('rinf = {} with error = {}'.format(pfit_leastsq[1],perr_leastsq[1]))
                    #print('Theta = {} with error = {}'.format(pfit_leastsq[2],perr_leastsq[2]))
                    #print("Reduced chi-square = ", s_sq)
                    
                    # finally, generate fit decay 
                    r_fit_img[:,i,j] = model_func(t, params_fit[0], params_fit[1], params_fit[2])
                    
                    # plot 
                    #plt.plot(t, r)
                    #plt.plot(t, r_fit, '-r')
                    #plt.ylim(-0.5, 1.0)
                    #plt.xlabel('Time (ns)', fontsize = 16)
                    #plt.ylabel('Anisotropy (r)', fontsize = 16)
                    #plt.suptitle(title, fontsize = 16)
                    #plt.show()
                
            # filter out by the mask - fit only higher intensity values 
    return ani_img, tau_img, rinf_img, r0_img, theta_img, r_fit_img