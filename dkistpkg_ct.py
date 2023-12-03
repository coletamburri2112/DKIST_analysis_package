# -*- coding: utf-8 -*-
"""
2 December 2023
Author: Cole Tamburri, University of Colorado Boulder, National Solar 
Observatory, Laboratory for Atmospheric and Space Physics

Description of script: 
    Main working functions for analysis package of DKIST data, applied first to 
    pid_1_84 ("flare patrol"), with PI Kowalski and Co-Is Cauzzi, Tristain, Notsu, 
    Kazachenko, and (unlisted) Tamburri.  Also applied to pid_2_11, with nearly 
    identical science objectives.  See READMe for details.  
    
    Includes intensity calibration, Gaussian fitting, and co-alignment routes 
    between ViSP and VBI, and, externally, SDO/HMI.  This code was developed with 
    the ViSP Ca II H and VBI blue continuum, TiO, and H-alpha channels as priority, 
    and using HMI in the continuum and 304 Angstrom bandpasses, but there is room 
    for application to other channels and science objectives.


"""

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import os
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
import scipy.integrate as integrate
import sunpy
import sunpy.coordinates
import sunpy.map
from sunpy.net import Fido
from sunpy.net import attrs as a
import astropy.units as u

# from sunpy.net import Fido, attrs as a
# import pandas as pd
# from astropy.utils.data import get_pkg_data_filename
# import shutil
# import fitsio
# import matplotlib.animation as animat
# import ffmpeg
# import latex
# import radx_ct
# import math as math
# import scipy.special as sp
# from scipy.stats import skewnorm
# from lmfit.models import SkewedGaussianModel
# import matplotlib
# from matplotlib import animation
# from lmfit import Model
# from pylab import *
# from astropy.coordinates import SkyCoord
# from astropy.time import Time
# from astropy.visualization import ImageNormalize, SqrtStretch



# define functions to be used for line fitting
def gaussian(x, c1, mu1, sigma1):
    res = c1 * np.exp( - (x - mu1)**2.0 / (2.0 * sigma1**2.0) )
    return res

def gaussfit(params,selwl,sel):
    fit = gaussian( selwl, params )
    return (fit - sel)

def double_gaussian( x, c1, mu1, sigma1, c2, mu2, sigma2 ):
    res =   (c1 * np.exp( - (x - mu1)**2.0 / (2.0 * sigma1**2.0) )) \
          + (c2 * np.exp( - (x - mu2)**2.0 / (2.0 * sigma2**2.0) ))
    return res

def double_gaussian_fit( params ,selwl,sel):
    fit = double_gaussian( selwl, params )
    return (fit - sel)

def numgreaterthan(array,value):
    count = 0
    for i in array:
        if i > value:
            count = count + 1
    return count


def color_muted2():
    #define colors for plotting
    
    #  0= indigo
    # 1 = cyan
    # 2 = teal
    # 3 = green
    # 4 = olive
    # 5= sand
    # 6 = rose
    # 7 = wine
    # 8 = purple
    # 9=grey
    
    muted =['#332288', '#88CCEE', '#44AA99','#117733','#999933','#DDCC77', '#CC6677','#882255','#AA4499','#DDDDDD']

    return muted

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx],idx


def pathdef(path,folder1):
    
    # define path for DKIST data; assumes L1 .fits files; 

    dir_list = os.listdir(path+folder1)
    
    #redefine list of directory contents
    dir_list2 = []
    for i in range(len(dir_list)):
        filename = dir_list[i]
        
        # modify based on filename structure and Stokes file preference
        if filename[-5:] == '.fits' and '_I_' in filename:
            dir_list2.append(filename)
    
    dir_list2.sort()
    
    return dir_list2

def spatialinit(path,folder1,dir_list2,lon,lat,wl,limbdarkening):
    
    # initialize spatial parameters, mu angle for use in determining limb darkening

    i_file_raster1 = fits.open(path+folder1+'/'+dir_list2[0]) #first image frame
    d = 150.84e9 #average distance to sun, in meters
    solrad = sunpy.sun.constants.radius.value
    
    # Coordinates from DKIST are not correct, but define them anyways as a starting
    # point.  Will co-align later in routine.
    
    hpc1_arcsec = i_file_raster1[1].header['CRVAL2'] # first axis coords
    hpc2_arcsec = i_file_raster1[1].header['CRVAL3'] # second axis corods
    
    # image center
    x_center = d*np.cos(hpc1_arcsec/206265)*np.sin(hpc2_arcsec/206265) # m
    y_center = d*np.sin(hpc1_arcsec/206265) # m
    z = solrad - d*np.cos(hpc1_arcsec/206265)*np.cos(hpc1_arcsec/206265) # m
    
    # to mu value
    rho = np.sqrt(x_center**2+y_center**2)
    mu = np.sqrt(1-(rho/solrad)**2)
    
    rotrate = 13.2 # approx. rotation rate; degrees of longitude per day
    
    #convert to cylindrical coordinates
    rcyl = solrad*np.cos(-lat*2*np.pi/360) #radius of corresponding cylinder
    circyl = 2*np.pi*solrad
    dratecyl = rotrate*circyl/360
    dratecylms = dratecyl/24/3600 # meters/day to meters/s
    
    # potential redshift effects due to solar rotation
    # testing impact of this on spectra to account for perceived wl shift in
    # data? Seems to account for shift in pid_1_84 by chance, uncertain if this
    # is the solution to the issue
    redsh = dratecylms*np.cos((90-lon)*2*np.pi/360) #redshift value
    
    # doppler shift in line due to rotation
    doppshnonrel = wl*(1+(redsh/3e8)) - wl #non-relativistic
    doppshrel = wl*np.sqrt(1-redsh/3e8)/np.sqrt(1+redsh/3e8) - wl #relativistic
        
    return hpc1_arcsec, hpc2_arcsec, x_center, y_center, z, rho, mu, \
        doppshnonrel, doppshrel



def fourstepprocess(path,folder1,dir_list2):
    
    # Simplest initial processing of data, when only a four-step raster; will 
    # certainly need to be generalized to observations with more raster steps in
    # ViSP observations

    image_data_arrs_raster1 = []
    image_data_arrs_raster2 = []
    image_data_arrs_raster3 = []
    image_data_arrs_raster4 = []
    rasterpos1 = []
    rasterpos2 = []
    rasterpos3 = []
    rasterpos4 = []
    times_raster1 = []
    times_raster2 = []
    times_raster3 = []
    times_raster4 = []

    image_data_arrs = []
    
    #four raster steps; make array for each
    for i in range(0,len(dir_list2),4):
        i_file_raster1 = fits.open(path+folder1+'/'+dir_list2[i])
        i_file_raster2 = fits.open(path+folder1+'/'+dir_list2[i+1])
        i_file_raster3 = fits.open(path+folder1+'/'+dir_list2[i+2])
        i_file_raster4 = fits.open(path+folder1+'/'+dir_list2[i+3])
        
        times_raster1.append(i_file_raster1[1].header['DATE-BEG'])
        times_raster2.append(i_file_raster2[1].header['DATE-BEG'])
        times_raster3.append(i_file_raster3[1].header['DATE-BEG'])
        times_raster4.append(i_file_raster4[1].header['DATE-BEG'])
        
        i_data_raster1 = i_file_raster1[1].data[0]
        i_data_raster2 = i_file_raster2[1].data[0]
        i_data_raster3 = i_file_raster3[1].data[0]
        i_data_raster4 = i_file_raster4[1].data[0]
        
        # collect observations belonging to the same slit position
        image_data_arrs_raster1.append(i_data_raster1)
        image_data_arrs_raster2.append(i_data_raster2)
        image_data_arrs_raster3.append(i_data_raster3)
        image_data_arrs_raster4.append(i_data_raster4)
        
        rasterpos1.append(i_file_raster1[1].header['CRPIX3'])
        rasterpos2.append(i_file_raster2[1].header['CRPIX3'])
        rasterpos3.append(i_file_raster3[1].header['CRPIX3'])
        rasterpos4.append(i_file_raster4[1].header['CRPIX3'])
        
        #array including all raster positions
        image_data_arrs.append(image_data_arrs_raster1)
        image_data_arrs.append(image_data_arrs_raster2)
        image_data_arrs.append(image_data_arrs_raster3)
        image_data_arrs.append(image_data_arrs_raster4)
        
    # array version of images corresponding to each slit position
    image_data_arr_arr_raster1 = np.array(image_data_arrs_raster1)
    image_data_arr_arr_raster2 = np.array(image_data_arrs_raster2)
    image_data_arr_arr_raster3 = np.array(image_data_arrs_raster3)
    image_data_arr_arr_raster4 = np.array(image_data_arrs_raster4)
    
    # all rasters
    image_data_arr_arr = np.array(image_data_arrs)
    
    # for intensity calibration purposes, only images from first raster pos
    for_scale = image_data_arr_arr_raster1[:,:,:]
    
    # uncomment second and third lines to return individual raster arrays
    return image_data_arr_arr, i_file_raster1, for_scale, times_raster1#,\
        # image_data_arr_arr_raster1, image_data_arr_arr_raster2,\
        # image-data_arr-arr_raster3, image_data_arr_arr_raster3


def spatialaxis(i_file_raster1):
    
    # find the axes of ViSP observations based on values given in L1 header;
    # spectral axis can be trusted as long as DKIST data set caveats have been
    # accounted for ( https://nso.atlassian.net/wiki/spaces/DDCHD/pages/1959985377/DKIST+Data+Set+Caveats+ViSP+VBI ).
    # Spatial coordiantes should not be trusted; only to be used for co-align
    # routines with SDO and between ViSP/VBI
    
    #crval1,cdelt1
    hdul1 = i_file_raster1
    centerlambda = hdul1[1].header['CRVAL1']
    deltlambda = hdul1[1].header['CDELT1'] 
    nlambda = hdul1[1].header['NAXIS2']
    
    dispersion_range = np.linspace(centerlambda-deltlambda*(nlambda-1)/2,
                                   centerlambda+deltlambda*(nlambda-1)/2,nlambda)
    
    centerspace = hdul1[1].header['CRVAL2']
    deltspace = hdul1[1].header['CDELT2'] #this actually is fine
    nspace = hdul1[1].header['NAXIS1']
    spatial_range = np.linspace(centerspace-deltspace*(nspace-1)/2,
                                centerspace+deltspace*(nspace-1)/2,nspace)
    
    return spatial_range, dispersion_range

       

def scaling(for_scale,nonflare_multfact,limbdarkening,nonflare_average,
            limbd = 1):
    # Scaling relative to QS values.  For this, require inputs of "nonflare" -
    # this can take the form of off-kernel observations.  In our case, was a disk
    # center observation, hence the allowance for limb darkening correction.  
    # Disk-center QS were compared to the Neckel-Hamburg disk center atlas, and a
    # multiplicative factor applied to give intensity values of observations. Based
    # on the spectral range being studied, may (1) have a single scaling factor, as
    # is found by e.g. Rahul Yadav, or (2) a variable scaling factor, as applied 
    # below, found for the Ca II H window by Cole Tamburri.  Solution to this 
    # theoretical dilemma not found as of 2 Dec 2023
    
    # set limbd to 0 if QS/nonflare values used for calibration were not at
    # disk center (or if limb darkening accounted for at a previoius time)
    
    # multiply dispersion dimension in each frame by fit_vals
    # gives us intensity calibrated spectra during flare time
    scaled_flare_time = np.zeros(np.shape(for_scale))
    
    for i in range(np.shape(for_scale)[0]):
        for k in range(np.shape(for_scale)[2]):
            # intensity calibration factor
            if limbd == 1:
                scaled_flare_time[i,:,k] = nonflare_multfact*for_scale[i,:,k]/\
                    limbdarkening
            elif limbd == 0:
                scaled_flare_time[i,:,k] = nonflare_multfact*for_scale[i,:,k]              
            
    end = 5
    bkgd_subtract_flaretime = np.zeros(np.shape(for_scale))

    # Subtracted averaged, scaled data cube 
    # from each time step in scaled data cube
    for i in range(np.shape(scaled_flare_time)[0]):
        bkgd_subtract_flaretime[i,end:,:] = scaled_flare_time[i,end:,:]-nonflare_average[:-end,:]
        
    return scaled_flare_time, bkgd_subtract_flaretime
    
                            
def pltsubtract(dispersion_range,nonflare_average,scaled_flare_time,muted,end=5,pid='pid_1_84'):
    
    # plotting routines to compare flare-time with non-flare spectra
    fig,ax = plt.subplots(figsize=(10,5))
    ax.plot(dispersion_range[end:]*10,nonflare_average[:-end,1350],\
            color=muted[4],label='Non-Flare')
    ax.plot(dispersion_range[end:]*10,scaled_flare_time[0,end:,1350],\
            color=muted[7],label='Flare-Time')
    ax.plot(dispersion_range[end:]*10,scaled_flare_time[0,end:,1350]-\
            nonflare_average[:-end,1350],color=muted[6],label='Flare-Only')
    ax.grid()
    ax.set_ylim([0,5e6])
    ax.legend(loc=0)
    ax.set_title('Non-Flare Estimate vs. Flare-time ',fontsize=25)
    ax.set_xlabel(r'Wavelength [$\mathring A$]',fontsize=15)
    ax.set_ylabel(r'Intensity [$W/cm^2/sr/\mathring A$]',fontsize=15)
    plt.show()
    
    fig.savefig('/Users/coletamburri/Desktop/DKIST_analysis_package/'+pid+\
                '/pltprofile.png')

    return None

def deviations(bkgd_subtract_flaretime,nonflare_average,nonflare_stdevs,end=5):
    
    # in pid_1_84, there is a strange difference between flare-time spectra and
    # nonflare, even outside of the main wings of Ca II H; testing for the
    # flare-time variations in these parts of the spectra and comparing to the
    # variation in the nonflare average in order to test if this variation is
    # due to flaring activity or something else in the spectra
    
    stdevs_flaretime = np.zeros([np.shape(nonflare_average)[0]-5,np.shape(nonflare_average)[1]])

    for i in range(np.shape(bkgd_subtract_flaretime)[1]-5):
        for j in range(np.shape(bkgd_subtract_flaretime)[2]):
            stdevs_flaretime[i,j] = np.std(bkgd_subtract_flaretime[:,i+end,j])
            
    # Compute PTE in dispersion dimension for all spatial locations
    ptes_flaretime = np.zeros([np.shape(nonflare_average)[0]-5,\
                               np.shape(nonflare_average)[1]]) # just one spatial location
    totnum = np.shape(ptes_flaretime)[1] # total number of spatial points
    for i in range(np.shape(nonflare_stdevs)[0]-5):
        for j in range(np.shape(nonflare_stdevs)[1]):
            stdev_inquestion_flaretime = stdevs_flaretime[i,j]
            stdev_inquestion_nonflare = nonflare_stdevs[i,:]
            num_gt = numgreaterthan(stdev_inquestion_nonflare,\
                                    stdev_inquestion_flaretime)
            pte = num_gt/totnum
            ptes_flaretime[i,j] = pte

    return stdevs_flaretime, ptes_flaretime


def pltptes(ptes_flaretime,image_data_arr_arr_raster1,pid='pid_1_84'):
    
    # plot probabity to exceed the variation in nonflare spectra at each
    # wavlength in flare-time - high probability to exceed means that variation
    # can be explained by the same variations seen in the quiet sun, low PTE 
    # means that variation can only be explained by something specific to the
    # flare data - either the flare itself, or something specific to that set, 
    # if different from the dataset used to determine QS
    
    fig,[ax,ax1] = plt.subplots(2,1,figsize=(7,12))
    im = ax.pcolormesh(np.log(ptes_flaretime))
    #cbar = plt.colorbar(im, shrink=0.9, pad = 0.05)
    #position=fig1.add_axes([0.93,0.1,0.02,0.35])
    #fig.colorbar(im, orientation="vertical",ax=ax,cax=position)
    ax1.set_xlabel('Spatial',fontsize=13)
    ax.set_ylabel('Dispersion',fontsize=13)
    ax.set_xlabel('Spatial',fontsize=13)
    ax.set_title('P.T.E. standard deviation in non-flare spectrum',fontsize=15)
    ax1.set_title('First Image Frame - Intensity',fontsize=15)
    ax1.pcolormesh(image_data_arr_arr_raster1[0,:,:])
    ax.axvline(1350)
    ax1.axvline(1350)
    ax.axvline(300)
    ax1.axvline(300)
    
    fig.savefig('/Users/coletamburri/Desktop/DKIST_analysis_package/'+pid+\
                '/pltpptes.png')

    return None

#definition of window for continuum

def contwind(sample_flaretime,dispersion_range,maxinds,scaled_flare_time,
             caII_low,caII_high,deg=8,low0=29,high0=29,low1=60,high1=150,
             low2=265,high2=290,low3=360,high3=400,low4=450,high4=480,
             low5=845,high5=870,low6=945,high6=965):
    
    # Define continuum (or "pseudo-continuum") window locations outside of the
    # main lines; this is used to isolate the line, to determine line flux,
    # strength, and for modeling.  In the Ca II window, this "pseudo-continuum"
    # is likely affected by the line itself - a problem to be discussed at a
    # later date
    
    # Default limits are for Ca II H in pid_1_84.  Select six (or fewer? then
    # comment extra lines) windows to fit a polynomial to as an estimate of
    # the pseudo-continuum with absorption (telluric or otherwise) and emission
    # lines removed
    
    avgs = []
    for i in range(len(scaled_flare_time)):
        snapshot = scaled_flare_time[i,:,:]
        average = np.mean(snapshot,axis=0)
        avgs.append(average)
    
    maxind = np.argmax(avgs[i][caII_low:caII_high])
    maxinds.append(maxind)
    contwind0_1 = sample_flaretime[low0:high0]
    contwind0_1_wave = dispersion_range[low0:high0]
    contwind1 = np.mean(sample_flaretime[low1:high1])
    contwind1_wave = np.mean(dispersion_range[low1:high1])
    contwind2 = np.mean(sample_flaretime[low2:high2])
    contwind2_wave = np.mean(dispersion_range[low2:high2])
    contwind3 = np.mean(sample_flaretime[low3:high3])
    contwind3_wave = np.mean(dispersion_range[low3:high3])
    contwind4 = np.mean(sample_flaretime[low4:high4])
    contwind4_wave = np.mean(dispersion_range[low4:high4])
    contwind5 = np.mean(sample_flaretime[low5:high5])
    contwind5_wave = np.mean(dispersion_range[low5:high5])
    contwind6 = np.mean(sample_flaretime[low6:high6])
    contwind6_wave = np.mean(dispersion_range[low6:high6])
    
    cont_int_array = [contwind0_1,contwind1,contwind2,contwind3,contwind4,
                      contwind5,contwind6]
    cont_int_wave_array = [contwind0_1_wave,contwind1_wave,contwind2_wave,
                           contwind3_wave,contwind4_wave,contwind5_wave,contwind6_wave]
    
    # polynomial fit of degree deg; deg = 8 likely oversolves?
    p = np.poly1d(np.polyfit(cont_int_wave_array,cont_int_array,deg))
    nolines = p(dispersion_range)
    
    return nolines, cont_int_array, cont_int_wave_array

def widths_strengths(ew_CaII_all_fs,eqw_CaII_all_fs,width_CaII_all_fs,
                     ew_hep_all_fs,eqw_hep_all_fs,width_hep_all_fs,
                     caII_low,caII_high,hep_low,hep_high,
                     scaled_flare_time,bkgd_subtract_flaretime,
                     dispersion_range,deg=6,low0=29,high0=29,low1=60,high1=150,
                     low2=265,high2=290,low3=360,high3=400,low4=450,high4=480,
                     low5=845,high5=870,low6=945,high6=965):
    
    # determine equivanet widths, effective widths, line widths for Ca II line
    
    # Uses pseudo-continuum polynomial determination similar to that described
    # in function above; see that doc for description
    
    avgs = []
    for i in range(len(scaled_flare_time)):
        snapshot = scaled_flare_time[i,:,:]
        average = np.mean(snapshot,axis=0)
        avgs.append(average)
    
    # "eq" means for use in equivalent width determination - equivalent width
    # determination does not use the background-subtracted values, but the raw
    # intensity-calibrated spectra (see description of effective vs. equivalent
    # with for justification)
    for j in range(np.shape(bkgd_subtract_flaretime)[2]):
        for i in range(len(scaled_flare_time)-5):
            sample_flaretime = bkgd_subtract_flaretime[i,:,j]
            foreqw = scaled_flare_time[i,:,j]
            contwind0_1 = sample_flaretime[low0:high0]
            contwind0_1eq = foreqw[low0:high0]
            contwind0_1_wave = dispersion_range[low0:high0]
            contwind1eq = np.mean(foreqw[low1:high1])
            contwind1 = np.mean(sample_flaretime[low1:high1])
            contwind1_wave = np.mean(dispersion_range[low1:high1])
            contwind2eq = np.mean(foreqw[low2:high2])
            contwind2 = np.mean(sample_flaretime[low2:high2])
            contwind2_wave = np.mean(dispersion_range[low2:high2])
            contwind3eq = np.mean(foreqw[low3:high3])
            contwind3 = np.mean(sample_flaretime[low3:high3])
            contwind3_wave = np.mean(dispersion_range[low3:high3])
            contwind4eq = np.mean(foreqw[low4:high4])
            contwind4 = np.mean(sample_flaretime[low4:high4])
            contwind4_wave = np.mean(dispersion_range[low4:high4])
            contwind5eq = np.mean(foreqw[low5:high5])
            contwind5 = np.mean(sample_flaretime[low5:high5])
            contwind5_wave = np.mean(dispersion_range[low5:high5])
            contwind6eq = np.mean(foreqw[low6:high6])
            contwind6 = np.mean(sample_flaretime[low6:high6])
            contwind6_wave = np.mean(dispersion_range[low6:high6])
    
            cont_int_arrayeqw = [contwind0_1eq,contwind1eq,contwind2eq,
                                 contwind3eq,contwind4eq,contwind5eq,contwind6eq]
    
            cont_int_array = [contwind0_1,contwind1,contwind2,contwind3,
                              contwind4,contwind5,contwind6]
            cont_int_wave_array = [contwind0_1_wave,contwind1_wave,
                                   contwind2_wave,contwind3_wave,
                                   contwind4_wave,contwind5_wave,
                                   contwind6_wave]
    
            deg = 6
    
            p = np.poly1d(np.polyfit(cont_int_wave_array,cont_int_array,deg))
    
            peq = np.poly1d(np.polyfit(cont_int_wave_array,cont_int_arrayeqw,
                                       deg))
    
            nolines = p(dispersion_range)
            nolineseq = peq(dispersion_range)
    
            maxind = np.argmax(sample_flaretime)
            maxint = sample_flaretime[maxind]
            maxcont = nolines[maxind]
    
            integrand = (sample_flaretime-nolines)/(maxint-maxcont)
            normflux = np.divide(foreqw,nolineseq)
    
            integrand2 = 1 - normflux
    
            #equivalent width determination, efefctive width determinatino
            ew_caII = integrate.cumtrapz(integrand[caII_low:caII_high],
                                         dispersion_range[caII_low:caII_high])\
                [-1]
            eqw_caII = integrate.cumtrapz(integrand2[caII_low:caII_high],
                                          dispersion_range[caII_low:caII_high])\
                [-1]
            maxind_Hep = np.argmax(sample_flaretime[hep_low:hep_high])
            maxint_Hep = sample_flaretime[maxind_Hep+hep_low]
            maxcont_Hep = nolines[maxind_Hep+hep_low]
    
            integrand_Hep = (sample_flaretime[hep_low:hep_high]-
                             nolines[hep_low:hep_high])/\
                (maxint_Hep-maxcont_Hep)
    
            ew_Hep = integrate.cumtrapz(integrand_Hep,
                                        dispersion_range[hep_low:hep_high])[-1]
            eqw_Hep = integrate.cumtrapz(integrand2[hep_low:hep_high],
                                         dispersion_range[hep_low:hep_high])[-1]
    
            ew_CaII_all_fs[i,j]=ew_caII
            ew_hep_all_fs[i,j]=ew_Hep
            eqw_CaII_all_fs[i,j]=eqw_caII
            eqw_hep_all_fs[i,j]=eqw_Hep
            
            caII_isolate = sample_flaretime[caII_low:caII_high]
            mincaIIH = min(caII_isolate)
            maxcaIIH = max(caII_isolate)
            meancaIIH = (maxcaIIH+mincaIIH)/2
    
            caIIHmidlow, caIImidlowindex = \
                find_nearest(caII_isolate[:round(len(caII_isolate)/2)],meancaIIH)
            caIIHmidhigh,caIImidhighindex = \
                find_nearest(caII_isolate[round(len(caII_isolate)/2):],meancaIIH)
    
            widthAng_caII = dispersion_range[caII_low+\
                                             round(len(caII_isolate)/2)+\
                                                 caIImidhighindex-1]-\
                dispersion_range[caII_low+caIImidlowindex-1] 
    
            hep_isolate = sample_flaretime[hep_low:hep_high]
            minhep = min(hep_isolate)
            maxhep = max(hep_isolate)
            meanhep = (maxhep+minhep)/2
    
            hepmidlow,hepmidlowindex = \
                find_nearest(hep_isolate[:round(len(hep_isolate)/2)],meanhep)
            hepmidhigh,hepmidhighindex = \
                find_nearest(hep_isolate[round(len(hep_isolate)/2):],meanhep)
    
            widthAng_hep = dispersion_range[hep_low+round(len(hep_isolate)/2)+\
                                            hepmidhighindex-1]-\
                dispersion_range[hep_low+hepmidlowindex-1] 
    
            width_CaII_all_fs[i,j]=widthAng_caII
            width_hep_all_fs[i,j]=widthAng_hep
        
    return ew_CaII_all_fs, ew_hep_all_fs, eqw_CaII_all_fs, eqw_hep_all_fs,\
        width_CaII_all_fs, width_hep_all_fs

# NOTE: Add plotting routine for widths?

def gauss2fit(storeamp1,storemu1,storesig1,storeamp2,storemu2,storesig2,
              bkgd_subtract_flaretime,dispersion_range, double_gaussian_fit,
              times_raster1,caII_low,caII_high,double_gaussian,gaussian,selwl,sel,
              pid='pid_1_84',parameters = [2e6,396.82,0.01,2e6,396.86,0.015]):
    fig, ax = plt.subplots(3,4,figsize=(30,30))
    
    # Original script for double-Gaussian fitting and plotting; no error metrics,
    # just visualization, limited room for model selection.  Not for
    # current use; for better option, look to functions "fittingroutines"
    # and "pltfitresults"

    fig.suptitle('Ca II H line evolution, 19-Aug-2022, Raster 1, Kernel Center'
                 ,fontsize=35)
    
    for i in range(np.shape(bkgd_subtract_flaretime)[0]):
        selwl = dispersion_range[caII_low,caII_high]
        sel = bkgd_subtract_flaretime[i,caII_low:caII_high,1350]-\
            min(bkgd_subtract_flaretime[i,caII_low:caII_high,1350])
        fit = leastsq(double_gaussian_fit,parameters,(selwl,sel))
        [c1,mu1,sigma1,c2,mu2,sigma2] = fit[0]
        storeamp1.append(c1)
        storemu1.append(mu1)
        storesig1.append(sigma1)
        storeamp2.append(c2)
        storemu2.append(mu2)
        storesig2.append(sigma2)
        ax.flatten()[i].plot(selwl,sel)
        ax.flatten()[i].plot(selwl, double_gaussian( selwl, fit[0] ))
        ax.flatten()[i].plot(selwl,gaussian(selwl,fit[0][0:3]),c='g')
        ax.flatten()[i].plot(selwl,gaussian(selwl,fit[0][3:6]),c='g')
        ax.flatten()[i].axis(ymin=0,ymax=4.3e6)
        ax.flatten()[i].set_title(times_raster1[i],fontsize=30)
        ax.flatten()[i].grid()
    ax.flatten()[-1].axis('off')
    
    fig.savefig('/Users/coletamburri/Desktop/DKIST_analysis_package/'+pid+\
                '/gaussianfits.png')
    
    #save
    
    return storeamp1, storeamp2, storesig1, storesig2, storemu1, storemu2

def fittingroutines(bkgd_subtract_flaretime,dispersion_range,
                    times_raster1, line_low, line_high,
                    double_gaussian, gaussian, selwl,sel,paramsgauss,
                    params2gauss,params2gaussneg,pid='pid_1_84',
                    date = '08/09/2022',line = 'Ca II H',nimg = 7,
                    kernind = 1350):
    # More flexible line fitting routines; currently for Ca II H as observed
    # in pid_1_84, but flexible for application to other lines.  Currently also
    # only includes functinoality for single Gaussian and double Gaussian fits;
    # additions welcome (skewed Gauss? Lorentz? Voigt? etc.)
    
    # Returns, via scipy.optimize.curve_fit, both the fit parameters and the 
    # error metrics for each model
    
    fits_1g = []
    fits_2g = []
    fits_2gneg = []
    
    for i in range(nimg):
        selwl = dispersion_range[line_low:line_high]
        sel = bkgd_subtract_flaretime[i,line_low:line_high,kernind]-\
            min(bkgd_subtract_flaretime[i,line_low:line_high,kernind])
        
        fit1g, fit1gcov = curve_fit(gaussian,selwl,sel,p0=paramsgauss)
        fit2g, fit2gcov = curve_fit(double_gaussian,selwl,sel, p0=params2gauss)
        #fit2gneg, fit2gnegcov = curve_fit(double_gaussian,selwl,\ 
            #sel,p0=params2gaussneg,maxfev=5000)
            
        fits_1g.append([fit1g,fit1gcov])
        fits_2g.append([fit2g,fit2gcov])
        #fits_2gneg.append([fit2gneg,fit2gnegcov])
            
    return fits_1g, fits_2g, fits_2gneg

def pltfitresults(bkgd_subtract_flaretime,dispersion_range,double_gaussian,
                  gaussian,times_raster1,muted,
                  line_low,line_high,fits_1g,fits_2g,fits_2gneg,
                  pid='pid_1_84',
                  date = '08092022',line = 'Ca II H',nimg = 7,
                  kernind = 1350,nrol=2,ncol=4,lamb0 = 396.85,c=2.99e5,
                  note=''):
    
    # plotting of the output of "fittingroutines"; can expand to beyond first
    # few image frames.  Tested 1 Dec 2023 for pid_1_84 Ca II H but not beyond
    # this.
    
    fig, ax = plt.subplots(2,4,figsize=(30,15))
    fig.suptitle(line+' evolution w/ fitting, '+date+note,fontsize=20)
    
    selwl = dispersion_range[line_low:line_high]
    
    selwlshift = selwl-lamb0
    
    selwlvel = (selwl/lamb0-1)*c
    
    def veltrans(x):
        return (((x+lamb0)/lamb0)-1)*c
    
    def wltrans(x):
        return (((x/c)+1)*lamb0)-lamb0
    
    for i in range(nimg):
        
        
        sel = bkgd_subtract_flaretime[i,line_low:line_high,kernind]-\
            min(bkgd_subtract_flaretime[i,line_low:line_high,kernind])
        
        if i == 0:
            maxprofile = max(sel)
        else:
            maxprofilenow = max(sel)
            if maxprofilenow > maxprofile:
                maxprofile = maxprofilenow
            
        fit1g = fits_1g[i][0]
        fit2g = fits_2g[i][0]
        #fit2gneg = fits_2gneg[i][0]
        
        gaussfity = gaussian(selwl,fit1g[0],fit1g[1],fit1g[2])
        gauss2fity = double_gaussian(selwl,fit2g[0],fit2g[1],fit2g[2],\
                                     fit2g[3],fit2g[4],fit2g[5])
            
        comp1fity = gaussian(selwl,fit2g[0],fit2g[1],fit2g[2])
        comp2fity = gaussian(selwl,fit2g[3],fit2g[4],fit2g[5])
        #gauss2negfity = double_gaussian(selwl,fit2gneg[0],fit2gneg[1],\
                                        # fit2gneg[2],fit2gneg[3],fit2gneg[4],
                                        # fit2gneg[5])
    
        ax.flatten()[i].plot(selwlshift,sel,label='data')
        #ax.flatten()[i].plot(selwlshift,gaussfity,label='G1')
        ax.flatten()[i].plot(selwlshift,gauss2fity,label='G2',color=muted[2])
        ax.flatten()[i].plot(selwlshift,comp1fity,label='G2,C1',color=muted[4])
        ax.flatten()[i].plot(selwlshift,comp2fity,label='G2,C2',color=muted[6])
        #ax.flatten()[i].plot(selwl,gauss2negfity,label='Gauss2neg')
        ax.flatten()[i].legend()
        ax.flatten()[i].axis(ymin=0,ymax=maxprofile+0.3e6)
        ax.flatten()[i].axvline(0,linestyle='dotted')
        secaxx = ax.flatten()[i].secondary_xaxis('top', functions=(veltrans,wltrans))
        ax.flatten()[i].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))    
    secaxx = ax.flatten()[0].secondary_xaxis('top', functions=(veltrans,wltrans))
    secaxx.set_xlabel(r'Velocity $[km\; s^{-1}]$')
    ax.flatten()[0].set_xlabel(r' $\lambda$ - $\lambda_0$ [nm]')
    ax.flatten()[0].set_ylabel(r'Intensity (- $I_{min}$) $[W\; cm^{-2} sr^{-1} \AA^{-1}]$')


    
    plt.subplots_adjust(wspace=0.4,hspace=0.4)
    plt.show()
    
    fig.savefig('/Users/coletamburri/Desktop/DKIST_analysis_package/'+\
                pid+'/fits'+date+note+'.png')
    
    return None


def perclevels(bkgd_subtract_flaretime,dispersion_range,caII_low,caII_high,
               store_ten_width,store_quarter_width,store_half_width):
    
    # Initial code, yet to be expanded to other wavelength ranges or proposals,
    # for determining the line width at different heights along emission line
    # profile; useful in tracking different types of broadening (or different
    # atmospheric heights) - see Graham and Cauzzi, 2015, 2020
    
    for i in range(np.shape(bkgd_subtract_flaretime)[0]):
        sel = bkgd_subtract_flaretime[i,caII_low:caII_high,1350]-min(bkgd_subtract_flaretime[i,caII_low:caII_high,1350])
        selwl = dispersion_range[caII_low:caII_high]
        
        maxsel = np.max(sel)
        minsel = np.min(sel)
        
        ten_perc_level = 0.1*(maxsel-minsel)
        quarter_perc_level = 0.25*(maxsel-minsel)
        half_level = 0.5*(maxsel-minsel)
        
        lensel=len(sel)
        
        ten_lev_low, ten_ind_low = find_nearest(sel[0:round(lensel/2)],ten_perc_level)
        ten_lev_high, ten_ind_high = find_nearest(sel[round(lensel/2):],ten_perc_level)
        quarter_lev_low, quarter_ind_low = find_nearest(sel[0:round(lensel/2)],quarter_perc_level)
        quarter_lev_high, quarter_ind_high = find_nearest(sel[round(lensel/2):],quarter_perc_level)
        half_lev_low, half_ind_low = find_nearest(sel[0:round(lensel/2)],half_level)
        half_lev_high, half_ind_high = find_nearest(sel[round(lensel/2):],half_level)
        
        ten_perc_width = dispersion_range[round(lensel/2)+ten_ind_high]-dispersion_range[ten_ind_low]
        quarter_perc_width = dispersion_range[round(lensel/2)+quarter_ind_high]-dispersion_range[quarter_ind_low]
        half_perc_width = dispersion_range[round(lensel/2)+half_ind_high]-dispersion_range[half_ind_low]
        
        store_ten_width.append(ten_perc_width)
        store_quarter_width.append(quarter_perc_width)
        store_half_width.append(half_perc_width)
        
    return store_ten_width, store_quarter_width, store_half_width

# Co-alignment routines

def space_range(hdul1):
    
    # Define spatial range for co-alignment given in L1 headers
    
    x_cent = hdul1[1].header['CRVAL2']
    y_cent = hdul1[1].header['CRVAL3']
    
    x_delt = hdul1[1].header['CDELT2']
    y_delt = hdul1[1].header['CDELT3']
    
    nspace = hdul1[1].header['NAXIS1']
    
    #x_range is helioprojective latitude position along slit
    #y_range is helioprojective longitude position of raster step
    x_range = np.linspace(x_cent-x_delt*(nspace-1)/2,x_cent+x_delt*(nspace-1)/2,nspace)
    y_range = np.linspace(y_cent-y_delt*(nspace-1)/2,y_cent+y_delt*(nspace-1)/2,nspace)
    
    arcsec_slit = np.linspace(0,nspace*x_delt,nspace)
    return x_cent, y_cent, x_delt, y_delt, x_range, y_range, arcsec_slit, nspace

def vispranges(hdul1,spatial_range,nslitpos=4):
    
    # Define spatial and wavelength ranges for ViSP; this takes all 
    # slit positions in a single raster scan and uses that as a second spatial
    # axis for the "ViSP image" which will be used to co-align with VBI
    
    slitlen = hdul1[1].header['CDELT2']*len(spatial_range) #in arcsec
    rastersize = hdul1[1].header['CDELT3']*nslitpos
    
    raster_range = [0,hdul1[1].header['CDELT3'],hdul1[1].header['CDELT3']*2,hdul1[1].header['CDELT3']*3,hdul1[1].header['CDELT3']*4]
    spatial_range2 = np.insert(spatial_range,0,spatial_range[0]-(spatial_range[1]-spatial_range[0]))
    
    return spatial_range2, raster_range

def imgprep(path,folder1,dir_list2):
    
    # Prepare initial image in the ViSP set, for comparison to VBI. Could be 
    # any, but make sure the correct timestamp
    
    image_data_arrs0 = []
    rasterpos = []
    times = []
    
    image_data_arrs = []
    image_date = []

    for i in range(0,round(len(dir_list2)),1):
        i_file = fits.open(path+folder1+'/'+dir_list2[i])
        
        times.append(i_file[1].header['DATE-BEG'])
        
        lammin = i_file[1].header['WAVEMIN']
        lamcen = i_file[1].header['CRVAL1']
        
        i_data = i_file[1].data[0]
        
        image_data_arrs0.append(i_data)
    
        rasterpos.append(i_file[1].header['CRPIX3'])
        
    return image_data_arrs0

def line_avg(image_data_arrs0,lowind,highind,nslit,nwave):
    caiiavgs = np.zeros((nslit,nwave))
    
    # define the boundaries (in dispersion direction) for the line and get an 
    # "average" intensity; could also use line flux for all positions along slit?
    # This simply gives an idea, when averaged lines from all slit positions are
    # plotted side-by-side, for the kernel locations.  Should use a low and high
    # index which approximately straddles the main line; Ca II H in the case of
    # the code originally developed for pid_1_84
    
    for i in range(nslit):
        caiiavgs[i,:] = np.mean(image_data_arrs0[i][lowind:highind,:],0)
            
    return caiiavgs

def pltraster(caiiavgs,raster_range,spatial_range2,pid='pid_1_84'):
    
    # plot the intensity images for the ViSP scan
    
    X,Y = np.meshgrid(raster_range,spatial_range2)
    fig,ax = plt.subplots()
    ax.pcolormesh(X,Y,np.transpose(caiiavgs),cmap='gray')
    ax.set_aspect('equal')
    
    plt.show()
    fig.savefig('/Users/coletamburri/Desktop/DKIST_analysis_package/'+pid+'/initslit.png')
    
    return None


def vbi_process(path_vbi,folder1_vbi):
    
    # Process VBI data similarly to ViSP above; only intensity files
    
    dir_list_vbi = os.listdir(path_vbi+folder1_vbi)

    dir_list2_vbi = []
    
    for i in range(len(dir_list_vbi)):
        filename = dir_list_vbi[i]
        if filename[-5:] == '.fits' and '_I_' in filename:
            dir_list2_vbi.append(filename)
    
    dir_list2_vbi.sort()
    
    dir_list2_vbi
    
    hdul1_vbi = fits.open(path_vbi+folder1_vbi+'/'+dir_list2_vbi[0])
    dat0_vbi = hdul1_vbi[1].data[0,:,:]
    
    xcent = hdul1_vbi[1].header['CRVAL1']
    xnum = hdul1_vbi[1].header['NAXIS1']
    xdelt = hdul1_vbi[1].header['CDELT1']
    
    ycent = hdul1_vbi[1].header['CRVAL2']
    ynum = hdul1_vbi[1].header['NAXIS2']
    ydelt = hdul1_vbi[1].header['CDELT2']
    
    xarr = np.linspace(((xcent-xdelt/2)-((xnum-1)/2)*xdelt),((xcent-xdelt/2)+((xnum-1)/2)*xdelt),xnum)
    yarr = np.linspace(((ycent-ydelt/2)-((ynum-1)/2)*ydelt),((ycent-ydelt/2)+((ynum-1)/2)*ydelt),ynum)
    
    vbi_X,vbi_Y = np.meshgrid(np.flip(xarr),yarr)
    
    dat0_vbi = hdul1_vbi[1].data[0,:,:]
    
    return vbi_X,vbi_Y,hdul1_vbi, dat0_vbi

def plt_precoalign(vbi_X, vbi_Y, hdul1_vbi, visp_X, visp_Y, vispimg,matplotlib, 
                   dat0_vbi,pid='pid_1_84'):
    
    # Plot VBI and ViSP, prior to co-alignment, together.  Use ginput to ID 
    # points for similar structures.  The result of the process which this
    # starts will be ViSP axes transformed into the VBI coordinate system; 
    # which is still not correct, but we can use to simultaneously get ViSP
    # and VBI in the SDO image frame
    
    # VBI is much higher-resolution, obviously, so easier/more effective in 
    # comparison to appropriate SDO bandpass.  
    
    # Minor errors in comparison of features lead to big errors in
    # transformation matrix, particularly farther from the features; so (1) 
    # choose features wisely (localized, bright); (2) choose bandpasses wisely
    # (similar atmospheric height, contribution function); (3) select points
    # in uncaffeinated state (jitters)


    fig,ax=plt.subplots(1,2,figsize=(10,5),sharey=True)
    ax[0].pcolormesh(vbi_X,vbi_Y,dat0_vbi,cmap='hot')
    ax[0].set_aspect('equal')
    ax[0].grid()
    ax[1].pcolormesh(visp_X,visp_Y,np.transpose(vispimg),cmap='hot')
    ax[1].set_aspect('equal')
    ax[1].grid()
    #ax[1].invert_xaxis()
    plt.tight_layout()
    
    plt.show()
    
    # ginput - press on vbi first, then visp, and so on, 
    # until three pairs collected
    # be VERY careful in selection of points! Require basis vectors
    # for each coordinate system; points should not be
    # colinear
    
    matplotlib.use('Qt5Agg')
    aa = plt.ginput(6,timeout = 120)
    
    fig.savefig('/Users/coletamburri/Desktop/DKIST_analysis_package/'+pid+'pre_coalign.png')
    
    return aa

def vbi_visp_transformation(aa, visp_X,visp_Y,nslit,nwave,vbi_X,vbi_Y,dat0_vbi,
                            vispimg,matplotlib,vbiband='H-alpha',
                            vispband='CaII H 396.8 nm',pid='pid_1_84'):
    
    # Simple transformation matrix between ViSP and VBI using output of ginput
    # process in function above
    
    # visp points
    A1 = np.array(aa[1])
    B1 = np.array(aa[3])
    C1 = np.array(aa[5])
    
    # vbi points
    A2 = np.array(aa[0])
    B2 = np.array(aa[2])
    C2 = np.array(aa[4])
    
    ViSP1 = B1 - A1
    ViSP2 = C1 - A1
    VBI1 = B2 - A2
    VBI2 = C2 - A2
    
    # insert test for linear combination
    
    # create basis matrices
    
    VBI_base = np.column_stack((VBI1,VBI2))
    
    ViSP_base = np.column_stack((ViSP1, ViSP2))
    
    # change of basis matrix
    
    COB = np.matmul(VBI_base, np.linalg.inv(ViSP_base))
    
    ViSP_points = [visp_X,visp_Y]
    
    new_ViSP = np.zeros(np.shape(ViSP_points))
        
    for i in range(nslit+1):
        for j in range(nwave+1):
            point_x = visp_X[i,j]
            point_y = visp_Y[i,j]
            
            point = [point_x,point_y]
            ViSPvec = point - A1
            
            A2_1 = np.matmul(COB,ViSPvec)+A2
            
            new_ViSP[:,i,j] = A2_1
            
    visp_X_new = new_ViSP[0,:,:]
    visp_Y_new = new_ViSP[1,:,:]
    
    # plot new axes
    
    fig,ax=plt.subplots(1,2,figsize=(10,5),sharey=True)
    ax[0].pcolormesh(vbi_X,vbi_Y,dat0_vbi,cmap='hot')
    ax[0].set_aspect('equal')
    #ax[0].invert_xaxis()
    ax[0].grid()
    ax[1].pcolormesh(visp_X_new,visp_Y_new,np.transpose(vispimg),cmap='hot')
    ax[1].set_aspect('equal')
    ax[1].grid()
    #ax[1].invert_xaxis()
    # custom_xlim = (-455,-410)
    # custom_ylim = (285,332)
    
    # # Setting the values for all axes.
    # plt.setp(ax, xlim=custom_xlim, ylim=custom_ylim)
    
    plt.tight_layout()
    
    plt.show()
    
    fig.savefig('/Users/coletamburri/Desktop/DKIST_analysis_package/'+pid+'postcalib.png')
    
    # plot overlay
    
    verts=np.array([[visp_X_new[-1,-1],visp_Y_new[-1,-1]],[visp_X_new[-1,0],visp_Y_new[-1,0]],[visp_X_new[0,0],visp_Y_new[0,0]],[visp_X_new[0,-1],visp_Y_new[0,-1]]])
    
    fig,ax = plt.subplots(1,1,figsize = (5,5))
    ax.pcolormesh(vbi_X,vbi_Y,dat0_vbi,cmap='gray')
    ax.pcolormesh(visp_X_new,visp_Y_new,np.transpose(vispimg),cmap='hot',alpha = 0.3)
    boxy = matplotlib.patches.Polygon(verts,fill=False,edgecolor='black',lw=3)
    ax.add_patch(boxy)
    ax.set_title('VBI '+vbiband+' and ViSP '+vispband,fontsize=15)
    ax.set_xlabel('Heliocentric Longitude',fontsize=10)
    ax.set_ylabel('Heliocentric Latitude',fontsize=10)
    
    plt.show()
    
    fig.savefig('/Users/coletamburri/Desktop/DKIST_analysis_package/'+pid+'postcalib_overlay.png')
    
    
    return visp_X_new, visp_Y_new

def query_sdo(start_time, email, cutout, matplotlib, 
              lowerx,upperx,lowery,uppery,
              wavelength = 304,
              timesamp=2,passband = '304'):
    
    # Use sunpy Fido to query different SDO bandpasses; on 2 Dec 2023 only 304 
    # and HMI continuum functionality, but can be expanded with a little work
    
    if passband == '304':
        query = Fido.search(
            a.Time(start_time - 0.01*u.h, start_time + .1*u.h),
            a.Wavelength(wavelength*u.angstrom),
            a.Sample(timesamp*u.h),
            a.jsoc.Series.aia_lev1_euv_12s,
            a.jsoc.Notify(email),
            a.jsoc.Segment.image,
            cutout,
        )

    if passband == 'cont':
        query = Fido.search(
            a.Time(start_time - 0.1*u.h, start_time + .09*u.h),
            a.Sample(timesamp*u.h),
            a.jsoc.Series.hmi_ic_45s,
            a.jsoc.Notify(email),
            a.jsoc.Segment.continuum,
            cutout,
        )
        
            
    files = Fido.fetch(query)
    
    sdohdul = fits.open(files.data[0])
    
    xr = np.linspace(lowerx,upperx,np.shape(sdohdul[1].data)[1])
    yr = np.linspace(lowery,uppery,np.shape(sdohdul[1].data)[0])
    
    X3,Y3 = np.meshgrid(xr,yr)
    
    # three points (remember which features - choose same features and same 
    # order when using points_vbi later!)

    fig,ax = plt.subplots()
    
    ax.pcolormesh(X3,Y3,sdohdul[1].data)

    plt.show()
    
    matplotlib.use('Qt5Agg')

    bb = plt.ginput(3,timeout = 60)
    
    return query, bb

def points_vbi(vbi_X,vbi_Y,dat0_vbi,matplotlib):
    
    # VBI coordinates, same features as in query_sdo ginput 
    
    fig,ax1 = plt.subplots()
    ax1.pcolormesh(vbi_X,vbi_Y,dat0_vbi,cmap='gray')
    #ax1.set_xticklabels([])
    #ax1.set_yticklabels([])
    ax1.set_aspect('equal')
    ax1.invert_xaxis()

    matplotlib.use('Qt5Agg')

    cc = plt.ginput(3,timeout=60)
    
    return cc

def vbi_to_sdo(bb, cc, vbi_X, vbi_Y):
    
    # Similar transformation, using basis vectors, as in visp to vbi 
    # transformation above
    
    A1 = np.array(bb[0])
    B1 = np.array(bb[1])
    C1 = np.array(bb[2])
    
    A2 = np.array(cc[0])
    B2 = np.array(cc[1])
    C2 = np.array(cc[2])
    
    #basis vectors
    SDO1 = B1 - A1
    SDO2 = C1 - A1
    VBI1 = B2 - A2
    VBI2 = C2 - A2
    
    SDO_base = np.column_stack((SDO1,SDO2))
    VBI_base = np.column_stack((VBI1,VBI2))

    COB2 = np.matmul(SDO_base,np.linalg.inv(VBI_base))
    
    VBI_points = [vbi_X,vbi_Y]
    new_VBI = np.zeros(np.shape(VBI_points))

    for i in range(np.shape(vbi_X)[0]):
        for j in range(np.shape(vbi_X)[1]):
            point_x = vbi_X[i,j]
            point_y = vbi_Y[i,j]
            
            point = [point_x,point_y]
            VBIvec = point - A2
            
            A2_1 = np.matmul(COB2,VBIvec)+A1
            
            new_VBI[:,i,j] = A2_1
            
    vbi_X_new = new_VBI[0,:,:]
    vbi_Y_new = new_VBI[1,:,:]
    
    return vbi_X_new, vbi_Y_new, COB2, A2, A1

def visp_sdo_trans(visp_X_new,visp_Y_new, COB2, A2, A1, nspace = 2544, nwave=4):
    
    # Finally, ViSP into the coordinate system defined by SDO

    ViSP_points = [visp_X_new,visp_Y_new]
    print(np.shape(ViSP_points))

    new_ViSP = np.zeros(np.shape(ViSP_points))
    
    for i in range(nspace+1):
        for j in range(nwave+1):
            point_x = visp_X_new[i,j]
            point_y = visp_Y_new[i,j]
            
            point = [point_x,point_y]
            ViSPvec = point-A2
            
            A2_1 = np.matmul(COB2,ViSPvec)+A1
            
            new_ViSP[:,i,j] = A2_1

    visp_X_new2 = new_ViSP[0,:,:]
    visp_Y_new2 = new_ViSP[1,:,:]
    
    return visp_X_new2, visp_Y_new2

def plt_final_coalign(vbi_X_new, vbi_Y_new, dat0_vbi2, 
                      visp_X_new2, visp_Y_new2, vispdat,
                      dat0_vbi, VBIpass1 = 'TiO',VBIpass2 = 
                      'H-alpha',ViSPpass = 'Ca II H',
                      obstimestr = '19 August 2022 20:42 UT',pid='pid_1_84'):
    
    # Plot final co-alignment
    
    fig,ax = plt.subplots(1,2,figsize = (10,5))
    ax[0].pcolormesh(vbi_X_new,vbi_Y_new,dat0_vbi2,cmap='gray')
    ax[0].pcolormesh(visp_X_new2,visp_Y_new2,np.transpose(vispdat),cmap='hot',alpha = 0.3)
    ax[0].patch.set_edgecolor('black')  
    ax[1].pcolormesh(vbi_X_new,vbi_Y_new,dat0_vbi,cmap='gray')
    ax[1].pcolormesh(visp_X_new2,visp_Y_new2,np.transpose(vispdat),cmap='hot',alpha = 0.3)
    ax[1].patch.set_edgecolor('black')  
    ax[0].set_title('VBI '+VBIpass1+ ' and ViSP '+ViSPpass,fontsize=15)
    ax[1].set_title('VBI '+VBIpass2+ ' and ViSP '+ViSPpass,fontsize=15)
    ax[0].set_xlabel('Helioprojective Longitude [arcsec]',fontsize=12)
    ax[0].set_ylabel('Helioprojective Latitude [arcsec]',fontsize=12)
    ax[1].set_xlabel('Helioprojective Longitude [arcsec]',fontsize=12)
    ax[1].set_ylabel('Helioprojective Latitude [arcsec]',fontsize=12)
    ax[0].set_box_aspect(1)
    ax[1].set_box_aspect(1)
    ax[0].grid()
    ax[1].grid()
    fig.suptitle(obstimestr+ ' Co-aligned',fontsize=20)
    
    fig.tight_layout()

    plt.show()
    
    fig.savefig('/Users/coletamburri/Desktop/DKIST_analysis_package/'+pid+'finalcoalign.png')
    
    return None





    
    

    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    







