# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 09:54:02 2021

@author: Iveta
"""
import numpy as np
from scipy.signal import savgol_filter
from anisotropy_functions import align_peaks
from scipy.optimize import minimize
from scipy.linalg import norm


def model_r0_theta_image(perpimage, parimage, totalimage, time, G, mask, theta, r0):
    #bounds = bounds 
    t,x,y = perpimage.shape
    # tau image will already be calculated and directly inserted as argument     
    r_0 = r0
    theta_0 = theta
    #empty images
        
    #3D images to hold the decays and fits 
    r_decay = np.empty_like(perpimage, dtype = float) # empty image to hold anisotropy decays
    perp_fit = np.empty_like(perpimage) # this will hold the fit 
    par_fit = np.empty_like(perpimage) # this will hold the fit 
    
    #2D images to hold the parameter maps 
    r0_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    theta_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    chi2_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    
    #2D images to hold the errors 
    #r0_err_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    #theta_err_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    #chi2_err_img = np.empty((perpimage.shape[1], perpimage.shape[2]))
    
    # count successful fits 
    counter_success = 0 
    counter_True = 0 
    for i in range(x):  # create anisotropy and lifetime images 
        for j in range(y):   
            if mask[:, i, j].any() == False:   # if not segmented - set to 0 or NaN?  
                #print(f'Pixel at {i},{j} is False.')    
                r_decay[:,i,j] = np.nan 
                perp_fit[:,i,j] = np.nan
                par_fit[:,i,j] = np.nan
                chi2_img[i,j] = np.nan
                r0_img[i,j] = np.nan
                theta_img[i,j] = np.nan
                
            else: # fit 
                # smooth - get single decay 
                #print(f'Pixels {i} x {j} : Pre-savgol shapes: {perpimage.shape} and {parimage.shape}')
                counter_True += 1
                perpimage_s = savgol_filter(perpimage[:,i,j], 5, 4, plot = False)
                parimage_s = savgol_filter(parimage[:,i,j], 5, 4, plot = False)
                totalimage_s = savgol_filter(totalimage[:,i,j], 5, 4, plot = False)
                #normalize smoothed with height of decay to ensure ratio is kept  
                #perpimage_s = perpimage_s * (max(perpimage[:,i,j])/max(perpimage_s))
                #parimage_s = parimage_s * (max(parimage[:,i,j])/max(parimage_s))
                #totalimage_s = totalimage_s * (max(totalimage[:,i,j])/max(totalimage_s))
                
                ## take away 0s 
                perpimage_s = np.where(perpimage_s < 1, 1, perpimage_s) 
                parimage_s = np.where(parimage_s < 1, 1, parimage_s) 
                
                # get weights 
                w_perp = 1/np.sqrt(perpimage_s)
                w_par = 1/np.sqrt(parimage_s)
                #w_total = 1/np.sqrt(totalimage_s)
                
                # align peaks of smoothed decay 
                perpimage[:,i,j], parimage[:,i,j], peak_index, _ = align_peaks(perpimage[:,i,j], parimage[:,i,j], time, plot = False)
                #perpimage, totalimage, peak_index, _ = align_peaks(parimage_s, totalimage_s, timens, plot = False)                
                 
                # choose selected time range for given pixel of the image 
                y_perp = perpimage[peak_index-1:, i,j]   
                y_par = parimage[peak_index-1:,i,j]
                total = totalimage[peak_index-1:,i,j]   
                t = time[peak_index-1:]
                w_perp = w_perp[peak_index-1:]   
                w_par = w_par[peak_index-1:]   

                # model functions - one for perp and one for par 
                # get total intensity at given time point from totalimage_s[t]
                # t,total = given vars; r0, theta - to be optimized 
                #dt = t[1] - t[0]
                #model_func_par = lambda IRF, t, total, p1, p2:  (max((total/3)*(1+2*p1)*np.exp(-t/p2)) / max(convolve(IRF,((total/3)*(1+2*p1)*np.exp(-t/p2)))[:len(t)]*dt))*convolve(IRF,((total/3)*(1+2*p1)*np.exp(-t/p2)))[:len(t)]*dt
                #model_func_perp = lambda IRF, t, total, p1, p2:  (max((total/3)*(1-p1)*np.exp(-t/p2)) / max(convolve(IRF,((total/3)*(1-p1)*np.exp(-t/p2)))[:len(t)]*dt))*convolve(IRF,((total/3)*(1-p1)*np.exp(-t/p2)))[:len(t)]*dt                                                 
                
                
                model_func_par = lambda t, total, p1, p2:  (total/3)*(1+2*p1)*np.exp(-t/p2)
                model_func_perp = lambda t, total, p1, p2: (total/3)*(1-p1)*np.exp(-t/p2)
                
                #objective function to minimize 
                objective = lambda p, t, total, y_perp, y_par, w_perp, w_par: norm((y_par - model_func_par(t,total, p[0], p[1]))*w_par)**2 + norm((y_perp - model_func_perp(t,total, p[0], p[1]))*w_perp)**2
                
                '''
                p
                p[0] = _r0 
                p[1] =  theta_0
                
                ''' 
                
                # initial guesses 
                p10, p20 = r_0, theta_0 
                
                #options={'disp': True}
                res = minimize(objective, x0 = [p10,p20], method='BFGS', args = (t, total, y_perp, y_par, w_perp, w_par))
                
                print(f'Decay model fit at pixel {i} and {j}: {res}')
                if res.success is True: 
                    counter_success += 1 
                if res.hess_inv is None:  
                    print('Hess does not converge')
                else: 
                    r0_img[i,j] = res.x[0]   # r0  - 2d image 
                    theta_img[i,j] = res.x[1]  # tau - 2d image    
                
                # calculate decay in each pixel, no weights here applied 
                #print(f'Shapes of y_perp { y_perp.shape }, y_par { y_par.shape } and total { total.shape }')
                    #print(f'Shapes y_par: {y_par.shape} y_perp: {y_perp.shape} total: {total.shape}')
                    r_decay[peak_index-1:,i,j] = np.divide(np.subtract(y_par, G*y_perp),total)                
                
                # what will be minimized ; pass in real arguments in model function
                #objective1 = lambda t, total, y_perp, p, weights_perp: (perpimage_s - model_func_perp(time, total, p[0], p[1])) * weights_perp
                #objective2 = lambda t, total, y_par, p, weights_par: (parimage_s - model_func_par(time, total, p[0], p[1])) * weights_par
                    #par_fit[peak_index-1:end_idx,i,j] = (max((total/3)*(1+2*res.x[0])*np.exp(-t/res.x[1])) / max(convolve(IRF,(total/3)*(1+2*res.x[0])*np.exp(-t/res.x[1]))[:len(t)]*dt))*convolve(IRF,(total/3)*(1+2*res.x[0])*np.exp(-t/res.x[1]))[:len(t)]*dt
                    #perp_fit[peak_index-1:end_idx,i,j] =  (max((total/3)*(1-res.x[0])*np.exp(-t/res.x[1])) / max(convolve(IRF,(total/3)*(1-res.x[0])*np.exp(-t/res.x[1]))[:len(t)]*dt))*convolve(IRF,(total/3)*(1-res.x[0])*np.exp(-t/res.x[1]))[:len(t)]*dt                                                 
                    par_fit[peak_index-1:,i,j] = (total/3)*(1+2*res.x[0])*np.exp(-t/res.x[1])
                    perp_fit[peak_index-1:,i,j] = (total/3)*(1-res.x[0])*np.exp(-t/res.x[1])
                #print(f'Chi-squared = {objective(res.x, t, total, y_perp, y_par)**2}')
                    chi2_img[i,j] = np.sum(objective(res.x, t, total, y_perp, y_par, w_perp, w_par)**2)/(len(y_perp)-len([p10, p20])) 
    return r0_img, theta_img, r_decay[peak_index-1:], par_fit[peak_index-1:], perp_fit[peak_index-1:], chi2_img, t, counter_True, counter_success