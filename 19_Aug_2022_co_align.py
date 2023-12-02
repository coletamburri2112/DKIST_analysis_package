#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 11:29:17 2023

@author: coletamburri
"""

import DKISTanalysis
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.visualization import ImageNormalize, SqrtStretch

import sunpy.coordinates
import sunpy.map
from sunpy.net import Fido
from sunpy.net import attrs as a

muted = DKISTanalysis.color_muted2()

pid = 'pid_1_84'

path = '/Volumes/ViSP_Aug_15_Aug_25_22/pid_1_84/'
folder1 = 'AZVXV'

path_vbi = '/Volumes/VBI_Aug_15_Aug_25_22/pid_1_84/'
folder1_vbi = 'BXWNO'

folder2_vbi = 'BYMOL'

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

spatial_range2, raster_range = DKISTanalysis.vispranges(i_file_raster1,spatial_range)

x_cent, y_cent, x_delt, y_delt, x_range, y_range, arcsec_slit, nspace = \
    DKISTanalysis.space_range(i_file_raster1)
    
image_data_arrs0 = DKISTanalysis.imgprep(path,folder1,dir_list2)

caiiavgs = DKISTanalysis.line_avg(image_data_arrs0,500,600,4,nspace)
    
DKISTanalysis.pltraster(caiiavgs,raster_range,spatial_range2)

vbi_X, vbi_Y, hdul1_vbi, dat0_vbi = DKISTanalysis.vbi_process(path_vbi,folder1_vbi)

X,Y = np.meshgrid(raster_range,spatial_range2)

aa = DKISTanalysis.plt_precoalign(vbi_X,vbi_Y,hdul1_vbi,X,Y,caiiavgs,matplotlib,dat0_vbi)

visp_X_new, visp_Y_new = DKISTanalysis.vbi_visp_transformation(aa,X,Y,nspace,4,vbi_X,vbi_Y,dat0_vbi,caiiavgs,matplotlib)

# now vbi to SDO calibration

start_time = Time('2022-08-19T20:42:30', scale='utc', format='isot')

lowerx = 675
upperx = 725

lowery = -500
uppery = -425

bottom_left = SkyCoord(lowerx*u.arcsec, lowery*u.arcsec, obstime=start_time, 
                       observer="earth", frame="helioprojective")
top_right = SkyCoord(upperx*u.arcsec, uppery*u.arcsec, obstime=start_time, 
                     observer="earth", frame="helioprojective")
    
cutout = a.jsoc.Cutout(bottom_left, top_right=top_right, tracking=True)

query, bb = DKISTanalysis.query_sdo(start_time, "cole.tamburri@colorado.edu", cutout, matplotlib, wavelength = 304,
              timesamp=2,passband='cont')

#load continuum VBI?

vbi_X2, vbi_Y2, hdul1_vbi2, dat0_vbi2 = DKISTanalysis.vbi_process(path_vbi,folder2_vbi)

cc = DKISTanalysis.points_vbi(vbi_X2,vbi_Y2,dat0_vbi2,matplotlib)

vbi_X_new, vbi_Y_new, COB2, A2, A1 = DKISTanalysis.vbi_to_sdo(bb,cc,vbi_X,vbi_Y)

visp_X_new2, visp_Y_new2 = DKISTanalysis.visp_sdo_trans(visp_X_new,visp_Y_new, COB2, A2, A1, nspace = 2544, nwave=4)

DKISTanalysis.plt_final_coalign(vbi_X_new, vbi_Y_new, dat0_vbi2, 
                      visp_X_new2, visp_Y_new2, caiiavgs,
                      dat0_vbi)
    
    