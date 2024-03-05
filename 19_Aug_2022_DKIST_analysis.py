#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2 December 2023
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

"""
end=0
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
path = '/Volumes/ViSP_External/pid_1_84/'
folder1 = 'AZVXV'
folder2 = 'ANYDJ' # for QS calibration

# list of files in directory for DKIST/ViSP
dir_list2 = DKISTanalysis.pathdef(path,folder1)
dir_list3 = DKISTanalysis.pathdef(path,folder2)

# Stonyhurst lon/lat position of the AR from JHv
lon = 58.57 #degrees
lat = -29.14 #degrees

# Stonyhurst lon/lat position of the AR from JHv for QS
lon = 60 #degrees
lat = -28 #degrees


wl = 396.847 # central wavelength, Ca II H

# spatial coordinates
hpc1_arcsec, hpc2_arcsec, x_center, y_center, z, rho, mu, doppshrel,\
    doppshnonrel = \
    DKISTanalysis.spatialinit(path,folder1,dir_list2,lon,lat,wl)
    
# spatial coordinates, qs    
hpc1_arcsecqs, hpc2_arcsecqs, x_centerqs, y_centerqs, zqs, rhoqs, muqs, doppshrelqs,\
    doppshnonrelqs = \
    DKISTanalysis.spatialinit(path,folder2,dir_list3,lon,lat,wl)

# get limb darkening coefficient 
clv_corr = DKISTanalysis.limbdarkening(wl, mu=mu, nm=True)
    # for Ca II H (require mu value for determination, be sure to specify
    # correct wl units)
    
clv_corrqs = DKISTanalysis.limbdarkening(wl, mu=muqs, nm=True)
    # for Ca II H (require mu value for determination, be sure to specify
    # correct wl units)
    
startstepqs = 250

# process multi-step raster - for qs time
image_data_arr_arr_qs, rasterpos_qs, times_qs = \
    DKISTanalysis.multistepprocess(path,folder2,dir_list3,div=10,
                                   startstep=startstepqs)
    
# for pid_1_84 - process four-step raster
image_data_arr_arr,i_file_raster1, for_scale, times_raster1 = \
    DKISTanalysis.fourstepprocess(path,folder1,dir_list2)
    
# spatial and dispersion axes for single observation (single slit step)
spatial_range, dispersion_range = DKISTanalysis.spatialaxis(path,folder1,
                                                            dir_list2,line='Ca II H')

# # old code, when basing QS on 15 August 2022 observations
# #only for 19 August observations, really - the QS will be different for others
# nonflare_average = np.load('/Users/coletamburri/Desktop/'+\
#                            'DKIST_Data/bolow_nonflare_average.npy')
# nonflare_stdevs = np.load('/Users/coletamburri/Desktop/'+\
#                           'DKIST_Data/bolow_nonflare_stdevs.npy')
# nonflare_fitvals = np.load('/Users/coletamburri/Desktop/'+\
#                            'DKIST_Data/bolow_nonflare_fit_vals.npy')
# nonflare_multfact = np.load('/Users/coletamburri/Desktop/'+\
#                             'DKIST_Data/bolow_nonflare_mult_fact.npy')
    
#qs calibration
wlsel, ilamsel = DKISTanalysis.load_fts(dispersion_range)
wlsel=wlsel/10

#wlsel = wlsel/10
space_and_time_averaged_qs = \
    DKISTanalysis.comp_fts_to_qs(wlsel,ilamsel,dispersion_range, 
                                 image_data_arr_arr_qs, lowint = -500,highint=-1,
                                 timelow=20,timehigh=46)

# telluric lines for comparison (or other absorption lines if telluric not 
# available, as is the case for the Ca II H window).  Most of the next steps
# are not used for pid_1_38, but include in any case to model the use of lines
line1 = 396.7432
line2 = 396.926

# indices of telluric lines in spectrum - lower
lowinds = [400,625]

# indices of telluric lines in spectrum - upper
highinds = [450,690]

# dispersion range and multiplicative ratio from intensity calibration
new_dispersion_range, rat = \
    DKISTanalysis.calib_qs_shift(wlsel,ilamsel,dispersion_range,
                                 space_and_time_averaged_qs, 
                                 line1,line2,lowinds,highinds)
    
# # but for pid_1_38, observations are not in a quiet sun region, so this function
# # really just takes a single index to compare the atlas and the observations
# # and uses that as the multiplicative factor
# calibration, calibrated_qs,new_dispersion_range2= \
#     DKISTanalysis.get_calibration_singleval(dispersion_range,\
#                                             space_and_time_averaged_qs,\
#                                                 wlsel,ilamsel,\
#                                                     limbdark_fact=clv_corrqs,
#                                                     noqs_flag = 1)

#original, based on 15 August 2022 QS
## simplify the quiet sun for comparison to flaretime
#nonflare_average_avg = np.mean(nonflare_average,axis=1)
cont_mult_facts,fit_vals,new_dispersion_range=DKISTanalysis.get_calibration_poly(dispersion_range,
                                                   space_and_time_averaged_qs,wlsel,ilamsel,DKISTanalysis.find_nearest,
                                                   line1,line2,lowinds,highinds,limbdark_fact=clv_corrqs,
                                                   noqs_flag=2)
# load QS intensity calibration results here!
calibrated_qs=fit_vals*space_and_time_averaged_qs
nonflare_average_avg = calibrated_qs
nonflare_multfact = fit_vals

yconv=DKISTanalysis.psf_adjust(wlsel,ilamsel,0.075,new_dispersion_range,calibrated_qs,clv_corrqs,45,DKISTanalysis.gaussian_psf)

#show comparison of atlas to qs
fig,ax=plt.subplots();ax.plot(new_dispersion_range,calibrated_qs,label='visp');ax.plot(new_dispersion_range,yconv*clv_corrqs,label='convolved');ax.plot(wlsel,clv_corrqs*ilamsel,label='raw');ax.set_xlim([396.6,397.2]);ax.set_ylim([0,0.6e6]);ax.legend();plt.show()#nonflare_multfact=np.full(len(dispersion_range), calibration[0])

#do another iteration of this step, after peforming the PSF convolution
cont_mult_facts,fit_vals,new_dispersion_range=DKISTanalysis.get_calibration_poly(new_dispersion_range,
                                                   space_and_time_averaged_qs,new_dispersion_range,yconv,DKISTanalysis.find_nearest,
                                                   line1,line2,lowinds,highinds,limbdark_fact=clv_corrqs,
                                                   noqs_flag=2)
calibrated_qs=fit_vals*space_and_time_averaged_qs
nonflare_average_avg = calibrated_qs
nonflare_multfact = fit_vals
# intensity calibration, background subtraction                            
scaled_flare_time, bkgd_subtract_flaretime = \
    DKISTanalysis.scaling(for_scale, nonflare_multfact,clv_corr,
                          nonflare_average_avg,end=end)

caII_low_foravg = 525
caII_high_foravg = 600

spacelow = 1000
spacehigh = -1

#indices of max intensity in each frame
maxindices = DKISTanalysis.maxintind(dispersion_range,bkgd_subtract_flaretime,
                                     caII_low_foravg,caII_high_foravg,
                                     spacelow,spacehigh)

# plot intensity calibrated, background-subtracted spectra
DKISTanalysis.pltsubtract(dispersion_range,nonflare_average_avg,
                          scaled_flare_time,muted,maxindices,end=end)

# variation in intensity value corresponding to wavelengths; PTE; to test
# for variations in pseudo-continuum.  If PTE high, cannot be explained by the
# solar activity difference between quiet sun and flare-time

# comment out if don't need PTE (shouldn't, usually)
# stdevs_flaretime, ptes_flaretime = \
    # DKISTanalysis.deviations(bkgd_subtract_flaretime,nonflare_average,
                             # nonflare_stdevs)

# DKISTanalysis.pltptes(ptes_flaretime,image_data_arr_arr_raster1)

# equivalent widths, effective widths, widths
caII_low = 480
caII_high = 650
hep_low = 700
hep_high = 850

sample_flaretime = bkgd_subtract_flaretime[0,:,maxindices[0]]

# perform following line only if need to calculate continuum window 
# independently of width determiation; 
# exists within width determination script as well

#nolines, cont_int_array, cont_int_wave_array = \
    # DKISTanalysis.contwind(sample_flaretime,dispersion_range,maxinds,avgs,
                           # low,high)

# line widths, strengths initial arrays
ew_CaII_all_fs = np.zeros((len(scaled_flare_time)-5,
                           np.shape(bkgd_subtract_flaretime)[2]))
ew_hep_all_fs = np.zeros((len(scaled_flare_time)-5,
                          np.shape(bkgd_subtract_flaretime)[2]))
eqw_CaII_all_fs = np.zeros((len(scaled_flare_time)-5,
                            np.shape(bkgd_subtract_flaretime)[2]))
eqw_hep_all_fs = np.zeros((len(scaled_flare_time)-5,
                           np.shape(bkgd_subtract_flaretime)[2]))
width_CaII_all_fs = np.zeros((len(scaled_flare_time)-5,
                              np.shape(bkgd_subtract_flaretime)[2]))
width_hep_all_fs = np.zeros((len(scaled_flare_time)-5,
                             np.shape(bkgd_subtract_flaretime)[2]))


# line widths, strength determination
ew_CaII_all_fs, ew_hep_all_fs, eqw_CaII_all_fs,\
    eqw_hep_all_fs, width_CaII_all_fs, width_hep_all_fs = \
        DKISTanalysis.widths_strengths(ew_CaII_all_fs,eqw_CaII_all_fs,
                                       width_CaII_all_fs,ew_hep_all_fs,
                                       eqw_hep_all_fs,width_hep_all_fs,caII_low,
                                       caII_high,hep_low,hep_high,
                                       scaled_flare_time,
                                       bkgd_subtract_flaretime, 
                                       dispersion_range)
        
# Gaussian fitting
# automate for all timesteps
storeamp1 = []
storemu1 = []
storesig1 = []
storeamp2 = []
storemu2 = []
storesig2 = []

# spatial index corresponding to part of observation of interest
sliceind = maxindices[0]
sel = bkgd_subtract_flaretime[0,caII_low:caII_high,sliceind]-\
    min(bkgd_subtract_flaretime[0,caII_low:caII_high,sliceind])
selwl = new_dispersion_range[caII_low:caII_high]
        
parameters=[2e6,396.82,0.015,6e6,396.86,0.015]
# fitting of spectral lines, e.g. double-Gaussian (for Ca II H)
storeamp1,storeamp2,storesig1,storesig2,storemu1,storemu2 = \
    DKISTanalysis.gauss2fit(storeamp1,storemu1,storesig1,storeamp2,storemu2,
                            storesig2,bkgd_subtract_flaretime,dispersion_range,
                            DKISTanalysis.double_gaussian_fit,maxindices,times_raster1,
                            caII_low,caII_high,DKISTanalysis.double_gaussian,
                            DKISTanalysis.gaussian,selwl,sel,
                            parameters = parameters)

# width determination
store_ten_width = []
store_quarter_width = []
store_half_width = []

store_ten_width, store_quarter_width, store_half_width = \
    DKISTanalysis.perclevels(bkgd_subtract_flaretime,dispersion_range,caII_low,
                             caII_high,store_ten_width,store_quarter_width,
                             store_half_width)

# output fit parameters
fits_1g,fits_2g,fits_2gneg,params2gaussnew,stopind = \
    DKISTanalysis.fittingroutines(bkgd_subtract_flaretime,dispersion_range,
                                  times_raster1, caII_low, caII_high,
                                  DKISTanalysis.double_gaussian, 
                                  DKISTanalysis.gaussian, 
                                  selwl,sel,[4e6,396.85,0.02],
                                  [2e6,396.82,0.015,6e6,396.86,0.015],
                                  [.5e6,396.85,0.015,-1e6,396.85,0.015],
                                  maxindices,pid='pid_1_84', date = '08/09/2022',
                                  line = 'Ca II H',nimg = 7,
                                  inds=[380,390,400,410,450,480,647,700,820,850,900],deg=7)

# plot results of Gaussian fitting

note = ', adjust with mu angle'
DKISTanalysis.pltfitresults(bkgd_subtract_flaretime,dispersion_range,
                            DKISTanalysis.double_gaussian,
                            DKISTanalysis.gaussian,times_raster1,muted,
                            caII_low,caII_high,fits_1g,fits_2g,fits_2gneg,maxindices,
                            mu, pid='pid_1_84', date = '08092022',line = 'Ca II H',
                            nimg = 7, nrow=2,ncol=4,lamb0=wl,note=note,yhigh=7.5e6,
                            inds=[380,390,400,410,450,480,647,700,820,850,900],deg=7)
    




