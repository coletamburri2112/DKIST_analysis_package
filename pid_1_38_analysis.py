#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4 December 2023
Author: Cole Tamburri, University of Colorado Boulder, National Solar 
Observatory, Laboratory for Atmospheric and Space Physics

Description of script: 
    Used to process and perform preparatory analysis on DKIST ViSP L1 data, 
    .fits file format.  Intensity calibration via quiet Sun also necessary as 
    input, performed via another script.  Here, observations are calibrated 
    using the factors determined through that external process, and adjustments
    made for limb darkening, etc., if necessary. Line widths and strengths are 
    calculated. Emission line modeling is performed in order to track flare 
    dynamics.
    
    Applied to pid_1_38, observations from 20 April 2022, Ca II 8542

"""

# package initialize
import dkistpkg_ct as DKISTanalysis
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

# color scheme for plotting
muted = DKISTanalysis.color_muted2()

# path and file ID for ViSP data
path = '/Users/coletamburri/Desktop/DKIST_Data/pid_1_38/ViSP/'
folder1 = 'Flare_Files'

# list of files in directory for DKIST/ViSP
dir_list2 = DKISTanalysis.pathdef(path,folder1)

# Stonyhurst lon/lat position of the AR from JHv
lon = -27.69 #degrees
lat = 18.29 #degrees

wl = 854.2 # central wavelength, Ca II H, in nm

# spatial coordinates
hpc1_arcsec, hpc2_arcsec, x_center, y_center, z, rho, mu, \
    doppshnonrel, doppshrel = \
    DKISTanalysis.spatialinit(path,folder1,dir_list2,lon,lat,wl)

# get limb darkening coefficient 
clv_corr = DKISTanalysis.limbdarkening(wl, mu=mu, nm=True)
    # for Ca II H (require mu value for determination, be sure to specify
    # correct wl units)
    
# process multi-step raster
image_data_arr_arr, for_scale, rasterpos, times = \
    DKISTanalysis.multistepprocess(path,folder1,dir_list2,div=10,startstep=0)
    
# spatial and dispersion axes for single observation (single slit step)
spatial_range, dispersion_range = \
    DKISTanalysis.spatialaxis(path,folder1,dir_list2)

# load QS intensity calibration results here!
nonflare_average = np.load('/Users/coletamburri/Desktop/'+\
                           'bolow_nonflare_average.npy')
nonflare_stdevs = np.load('/Users/coletamburri/Desktop/'+\
                          'bolow_nonflare_stdevs.npy')
nonflare_fitvals = np.load('/Users/coletamburri/Desktop/'+\
                           'bolow_nonflare_fit_vals.npy')
nonflare_multfact = np.load('/Users/coletamburri/Desktop/'+\
                            'bolow_nonflare_mult_fact.npy')

# intensity calibration, background subtraction                            
scaled_flare_time, bkgd_subtract_flaretime = \
    DKISTanalysis.scaling(for_scale, nonflare_multfact, clv_corr,
                          nonflare_average)

# plot intensity calibrated, background-subtracted spectra
DKISTanalysis.pltsubtract(dispersion_range,nonflare_average,scaled_flare_time,
                          muted)

# variation in intensity value corresponding to wavelengths; PTE; to test
# for variations in pseudo-continuum.  If PTE high, cannot be explained by the
# solar activity difference between quiet sun and flare-time

# comment out if don't need PTE (shouldn't, usually)
# stdevs_flaretime, ptes_flaretime = \
    # DKISTanalysis.deviations(bkgd_subtract_flaretime,nonflare_average,
                             # nonflare_stdevs)

# DKISTanalysis.pltptes(ptes_flaretime,image_data_arr_arr_raster1)

# equivalent widths, effective widths, widths
caII_8542_low = 500
caII_8542_high = 600

# spatial index corresponding to part of observation of interest
sliceind = 1600

# put bkgd_subtract_flaretime here, when ready
sample_flaretime = image_data_arr_arr[0,:,sliceind]

# perform following line only if need to calculate continuum window 
# independently of width determiation; 
# exists within width determination script as well

#nolines, cont_int_array, cont_int_wave_array = \
    # DKISTanalysis.contwind(sample_flaretime,dispersion_range,maxinds,avgs,
                           # low,high)

# line widths, strengths initial arrays
ew_CaII_all_fs = np.zeros((len(scaled_flare_time)-5,np.shape(bkgd_subtract_flaretime)[2]))
eqw_CaII_all_fs = np.zeros((len(scaled_flare_time)-5,np.shape(bkgd_subtract_flaretime)[2]))
width_CaII_all_fs = np.zeros((len(scaled_flare_time)-5,np.shape(bkgd_subtract_flaretime)[2]))

maxinds = []

# line widths, strength determination
ew_CaII_all_fs, eqw_CaII_all_fs,\
    width_CaII_all_fs = \
        DKISTanalysis.widths_strengths_oneline(ew_CaII_all_fs,eqw_CaII_all_fs,width_CaII_all_fs,
                             caII_8542_low,caII_8542_high,
                             scaled_flare_time,bkgd_subtract_flaretime,
                             dispersion_range,deg=6,low0=105,high0=134,low1=185,high1=202,
                             low2=290,high2=306,low3=357,high3=370,low4=446,high4=464,
                             low5=616,high5=645,low6=692,high6=715)
        
# Gaussian fitting
# automate for all timesteps
storeamp1 = []
storemu1 = []
storesig1 = []
storeamp2 = []
storemu2 = []
storesig2 = []


sel = bkgd_subtract_flaretime[0,caII_low:caII_high,sliceind]-\
    min(bkgd_subtract_flaretime[0,caII_low:caII_high,sliceind])
selwl = dispersion_range[caII_low:caII_high]
        
# width determination
store_ten_width = []
store_quarter_width = []
store_half_width = []

store_ten_width, store_quarter_width, store_half_width = \
    DKISTanalysis.perclevels(bkgd_subtract_flaretime,dispersion_range,caII_low,
                             caII_high,store_ten_width,store_quarter_width,
                             store_half_width)
    
# output fit parameters
fits_1g,fits_2g,fits_2gneg = \
    DKISTanalysis.fittingroutines(bkgd_subtract_flaretime,dispersion_range,
                                  times_raster1, caII_8542_low, caII_8542_high,
                                  DKISTanalysis.double_gaussian, 
                                  DKISTanalysis.gaussian, 
                                  selwl,sel,[1.1,854.28,0.05],
                                  [.4,854.275,0.025,0.4,854.295,0.025],
                                  [.5e6,396.85,0.015,-1e6,396.85,0.015],
                                  pid='pid_1_38', date = '04/20/2022',
                                  line = 'Ca II 854.2',nimg = 7, kernind = sliceind)
# plot results of Gaussian fitting
DKISTanalysis.pltfitresults(bkgd_subtract_flaretime,dispersion_range,
                            DKISTanalysis.double_gaussian,
                            DKISTanalysis.gaussian,times_raster1,muted,caII_8542_low,caII_8542_high,fits_1g,fits_2g,fits_2gneg,
                            pid='pid_1_38', date = '04202022',line = 'Ca 854.2',
                            nimg = 7, nrol=2,ncol=4,
                            note=', first try pid_1_38',lim=0.1,lamb0=854.2,kernind=1600)
    



