#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 12:05:46 2023

@author: coletamburri
"""

import DKISTanalysis
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update(matplotlib.rcParamsDefault)

# establish plotting methods and formats
# plt.rcParams['text.usetex']=True
# plt.rcParams['font.family']='sans-serif'
# plt.rcParams['font.sans-serif'] = ['Helvetica']
# plt.rcParams['axes.labelsize'] = 25
# plt.rcParams['lines.linewidth'] = 2
# matplotlib.rc('xtick', labelsize=20) 
# matplotlib.rc('ytick', labelsize=20) 


muted = DKISTanalysis.color_muted2()

path = '/Volumes/ViSP_Aug_15_Aug_25_22/pid_1_84/'
folder1 = 'AZVXV'

dir_list2 = DKISTanalysis.pathdef(path,folder1)

# Stonyhurst lon/lat from JHv
lon = 58.57 #degrees
lat = -29.14 #degrees

wl = 396.8

limbdarkening = 0.57 # for Ca II H

hpc1_arcsec, hpc2_arcsec, x_center, y_center, z, rho, mu = \
    DKISTanalysis.spatialinit(path,folder1,dir_list2,lon,lat,wl,limbdarkening)

image_data_arr_arr,i_file_raster1, for_scale, times_raster1 = \
    DKISTanalysis.fourstepprocess(path,folder1,dir_list2)

spatial_range, dispersion_range = DKISTanalysis.spatialaxis(i_file_raster1)

#only for 19 August observations, really - the QS will be different for others
nonflare_average = np.load('/Users/coletamburri/Desktop/'+\
                           'bolow_nonflare_average.npy')
nonflare_stdevs = np.load('/Users/coletamburri/Desktop/'+\
                          'bolow_nonflare_stdevs.npy')
nonflare_fitvals = np.load('/Users/coletamburri/Desktop/'+\
                           'bolow_nonflare_fit_vals.npy')
nonflare_multfact = np.load('/Users/coletamburri/Desktop/'+\
                            'bolow_nonflare_mult_fact.npy')
                            
scaled_flare_time, bkgd_subtract_flaretime = DKISTanalysis.scaling(for_scale,nonflare_multfact,limbdarkening,nonflare_average)

DKISTanalysis.pltsubtract(dispersion_range,nonflare_average,scaled_flare_time,muted)

#variation in intensity value corresponding to wavelengths; PTE

# comment out if don't need PTE (shouldn't, usually)
stdevs_flaretime, ptes_flaretime = DKISTanalysis.deviations(bkgd_subtract_flaretime,nonflare_average,nonflare_stdevs)

DKISTanalysis.pltptes(ptes_flaretime,image_data_arr_arr_raster1)

# equivalent widths, effective widths, widths
caII_low = 480
caII_high = 660
hep_low = 700
hep_high = 850

sample_flaretime = bkgd_subtract_flaretime[0,:,1350]

# following line only if need to calculate continuum window independently of width determiation; 
# exists within width determination script as well

#nolines, cont_int_array, cont_int_wave_array = DKISTanalysis.contwind(sample_flaretime,dispersion_range,maxinds,avgs,low,high)

ew_CaII_all_fs = np.zeros((len(scaled_flare_time)-5,np.shape(bkgd_subtract_flaretime)[2]))
ew_hep_all_fs = np.zeros((len(scaled_flare_time)-5,np.shape(bkgd_subtract_flaretime)[2]))
eqw_CaII_all_fs = np.zeros((len(scaled_flare_time)-5,np.shape(bkgd_subtract_flaretime)[2]))
eqw_hep_all_fs = np.zeros((len(scaled_flare_time)-5,np.shape(bkgd_subtract_flaretime)[2]))
width_CaII_all_fs = np.zeros((len(scaled_flare_time)-5,np.shape(bkgd_subtract_flaretime)[2]))
width_hep_all_fs = np.zeros((len(scaled_flare_time)-5,np.shape(bkgd_subtract_flaretime)[2]))

maxinds = []

ew_CaII_all_fs, ew_hep_all_fs, eqw_CaII_all_fs,\
    eqw_hep_all_fs, width_CaII_all_fs, width_hep_all_fs = \
        DKISTanalysis.widths_strengths(ew_CaII_all_fs,eqw_CaII_all_fs,width_CaII_all_fs,
                         ew_hep_all_fs,eqw_hep_all_fs,width_hep_all_fs,
                         caII_low,caII_high,hep_low,hep_high,scaled_flare_time,bkgd_subtract_flaretime,
                         dispersion_range)
        
# Gaussian fitting
# automate for all timesteps
storeamp1 = []
storemu1 = []
storesig1 = []
storeamp2 = []
storemu2 = []
storesig2 = []

sel = bkgd_subtract_flaretime[0,caII_low:caII_high,1350]-min(bkgd_subtract_flaretime[0,caII_low:caII_high,1350])
selwl = dispersion_range[caII_low:caII_high]
        
storeamp1,storeamp2,storesig1,storesig2,storemu1,storemu2 = \
    DKISTanalysis.gauss2fit(storeamp1,storemu1,storesig1,storeamp2,storemu2,
                            storesig2,bkgd_subtract_flaretime,dispersion_range,
                            DKISTanalysis.double_gaussian_fit,times_raster1,
                            caII_low,caII_high,DKISTanalysis.double_gaussian,DKISTanalysis.gaussian,
                            selwl,sel)

store_ten_width = []
store_quarter_width = []
store_half_width = []

store_ten_width, store_quarter_width, store_half_width = \
    DKISTanalysis.perclevels(bkgd_subtract_flaretime,dispersion_range,caII_low,
                             caII_high,store_ten_width,store_quarter_width,
                             store_half_width)
    




