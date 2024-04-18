#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 15:10:52 2021

@author: mik and nandinihazra
"""


#IMPORT STUFF
import numpy as np
import matplotlib.pyplot as plt
import sewpy
from astropy.table import Table
from astropy.stats import median_absolute_deviation as mad
from astropy.coordinates import SkyCoord
from astropy import units as u
import matplotlib.transforms as transforms

import pandas as pd 
import astropy.io 
from astropy.stats import sigma_clipped_stats,mad_std
import astropy.wcs as WCS
from astropy.visualization import simple_norm
from photutils.psf import extract_stars
from astropy.io import ascii
from astropy.nddata import NDData#contenitore numpy dove ci vanno anche l'astrometria.
#https://docs.astropy.org/en/stable/api/astropy.nddata.NDData.html#astropy.nddata.NDData


from photutils.detection import IRAFStarFinder
from photutils.psf import IntegratedGaussianPRF, DAOGroup
from photutils.background import MMMBackground, MADStdBackgroundRMS
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.stats import gaussian_fwhm_to_sigma
from photutils.psf import IterativelySubtractedPSFPhotometry
from photutils.psf import BasicPSFPhotometry
from photutils.psf import EPSFBuilder
from photutils.psf import EPSFModel
from astropy.io import fits
from photutils.psf import prepare_psf_model
#import scipy as sp
#import matplotlib.patches as patches
#from astropy.modeling import models, fitting
#import astropy as ap
#import math
"""
import os
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from astropy.io import fits

from astropy.table import Table
import warnings
import pyregion
import time
"""

#IMPORT GALAXY OBJECT
import init as gal

def get_hsc_mask(hdu,img_ext) :
    flags=gal.flags
    flags_header=gal.flags_header
    mask = 0
    for i in range(len(flags)):
	    mask += flags[i] << hdu[img_ext+1].header[flags_header[i]]
    instr_mask=((hdu[img_ext+1].data & mask)>0).astype(np.uint8)
    
    return instr_mask

def get_custom_mask(maskpath,mask_ext=0) :
    
    print("You have chosen the custom mask option, (masked pixels should be equal to 1)")
    print(f"Your mask was read from: {maskpath}, HDU {mask_ext}")
    mhdu=fits.open(maskpath, do_not_scale_image_data=True)
    print("WARNING: MASK WAS INVERTED SO THAT MASKED PIXELS==1")
    instr_mask=(1-mhdu[mask_ext].data).astype(np.uint8)
    return instr_mask
     

def get_plots(catfilepath,gxyra,gxydec,gxyid,gxy_name) :    
    
    font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 16}

    plt.rc('font', **font)
    tsex = Table.read(catfilepath, format='ascii.sextractor')
    
    #Clean the table
    tsex=tsex[(tsex['MAG_AUTO']<50) & (tsex['MAG_APER']<50)& (tsex['MAG_APER_1']<50)& (tsex['MAG_APER_2']<50)]

    #print(np.median(tsex[tsex['MAG_AUTO']<=20]['MAGERR_AUTO']))
    #print(np.median(tsex[tsex['CLASS_STAR']>=0.9]['FWHM_IMAGE']))
    
    
    #Fig 1
    
    fig1 = plt.figure(figsize=(14, 16))
    fig1.subplots_adjust(wspace=0.05, left=0.1, right=0.95,
                        bottom=0.15, top=0.9)
    plt.subplot(211)
    plt.scatter(tsex['MAG_AUTO'],tsex['FWHM_IMAGE'])
    cond=(tsex['CLASS_STAR']>=0.8)
    plt.scatter(tsex[cond]['MAG_AUTO'],tsex[cond]['FWHM_IMAGE'])
    plt.xlabel('MAG_AUTO')
    plt.ylabel('FWHM (pix)')
    plt.ylim(0,7)
    # plt.title(gxy_name)
    
    #plt.xlim(14,28)

    plt.subplot(212)
    plt.scatter(tsex['MAG_AUTO'],tsex['CLASS_STAR'])
    plt.xlabel('MAG_AUTO')
    plt.ylabel('CLASS_STAR')
    plt.savefig(f'../OUTPUT/plots/{gxyid}/{gxy_name}_part2_phot.jpeg',dpi=300)
    # plt.title(gxy_name)
    #plt.show(block=False)
    plt.clf()
    # plt.close(fig1)
    
    #plt.ylim(0,)
    # plt.scatter(tsex['MAG_AUTO'],tsex['FLUX_RADIUS'])
    # #plt.show(block=False)
    # #plt.clf()
    
    
    cs=tsex['CLASS_STAR']
    mag4=tsex['MAG_APER']
    mag6=tsex['MAG_APER_1']
    mag8=tsex['MAG_APER_2']
    #mag12=tsex['MAG_APER_4']
    
    cond=(tsex['CLASS_STAR']>=0.8) & (tsex['FLAGS']==0)

    rgc=np.sqrt((tsex['ALPHA_J2000'][(cond)]-gxyra)**2+(tsex['DELTA_J2000'][(cond)]-gxydec)**2)*3600 #Galactocentric distance in arcsecs
    hist=np.histogram(rgc, bins=20)
    width = 0.8 * (hist[1][1] - hist[1][0])
    center = (hist[1][:-1] + hist[1][1:]) / 2
    plt.bar(center, hist[0], align='center', width=width,color='C0')
    # plt.scatter(rgc,tsex['MAG_AUTO'])
    plt.xlabel('Galactocentric radius (arcsec)')
    plt.ylabel('N: compact')
    #plt.show(block=False)
    plt.savefig(f'../OUTPUT/plots/{gxyid}/{gxy_name}_part2_rgc.jpeg',dpi=300)
    plt.show()
    plt.clf()
    plt.close(fig1)
    
    
    #Fig 2
   
    fig2 = plt.figure(figsize=(14, 16))
    fig2.subplots_adjust(wspace=0.05, left=0.1, right=0.95,
                        bottom=0.15, top=0.9)
    #cmap='hsv'
    
    # cond=((mag8<26) & (cs>=0.75))
    cond=(cs>=0.8)
    plt.subplot(211)
    plt.title(gxy_name+' Concentration indices')
    plt.ylim(-1,2)
    plt.scatter(mag8,mag4-mag6)
    plt.scatter(mag8[cond],mag4[cond]-mag6[cond])
    plt.xlabel('MAG APER 8')
    plt.ylabel('MAG APER 4-6')
    plt.grid()
    
    
    plt.subplot(212)
    plt.ylim(-1,2)
    plt.scatter(mag8,mag4-mag8)
    plt.scatter(mag8[cond],mag4[cond]-mag8[cond])
    plt.xlabel('MAG APER 8')
    plt.ylabel('MAG APER 8-12')
    plt.grid()
    
    plt.savefig(f'../OUTPUT/plots/{gxyid}/{gxy_name}_part2_ci.jpeg',dpi=300)
    plt.show()
    plt.clf()
    plt.close(fig2)

    
def corr_catalog(catfilepath,apercolname,aper_corr,ext_corr) :

    tsex=Table.read(catfilepath, format='ascii')
    tsex[apercolname]=tsex[apercolname]+aper_corr-ext_corr
    
    return tsex

def sbf_match_catalogs(cat1path,cat2path,rad_asec):
    
    
    cs_lim=0.8
    font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 26}

    plt.rc('font', **font)
    
    #Read the 2 catalogs 
    tsex1 = Table.read(cat1path, format='ascii')
    tsex2 = Table.read(cat2path, format='ascii')
    
    skycoord_1=SkyCoord(tsex1['ALPHA_J2000'],tsex1['DELTA_J2000'],unit=(u.deg, u.deg))
    skycoord_2=SkyCoord(tsex2['ALPHA_J2000'],tsex2['DELTA_J2000'],unit=(u.deg, u.deg))
    idx1, idx2, d2d, d3d = skycoord_2.search_around_sky(skycoord_1,rad_asec*u.arcsec)
    #print(len(idx1),len(idx2),len(d2d), idx1)
    
    # CATALOGS OF ALL SOURCES
    tmatch1=tsex1[idx1]
    tmatch2=tsex2[idx2]
    
    # clean=((tmatch1['MAG_AUTO']<threshold) & (tmatch1['MAG_APER_1']<threshold) & 
    #        (tmatch1['MAG_APER_11']<threshold) & (tmatch2['MAG_AUTO']<threshold) & 
    #        (tmatch2['MAG_APER_1']<threshold) & (tmatch2['MAG_APER_11']<threshold))
    # tmatch1=tmatch1[clean]
    # tmatch2=tmatch2[clean]

    # CHOOSE PSF CANDIDATES
    # col_aper=(tmatch2['MAG_APER_1']-tmatch1['MAG_APER_1'])
    # ci1=(tmatch1['MAG_APER']-tmatch1['MAG_APER_2'])
    # ci2=(tmatch2['MAG_APER']-tmatch2['MAG_APER_2'])
    #plt.scatter(tmatch2['MAG_APER_1'],ci2,marker='.')
    # plt.xlim(-1.25,2.25)
    #plt.ylim(0.3,0.6)
    # plt.vlines(col_low,17,31.5)
    # plt.vlines(col_high,17,31.5)
    # #plt.show(block=False)
    # #plt.clf()
    # psf_sel=((col_aper<=col_high) & (col_aper>=col_low) & 
    # psf_sel=((tmatch1['CLASS_STAR']>=cs_lim) & (tmatch2['CLASS_STAR']>=cs_lim) &
            # (ci1<=0.8) & (ci1>=0.5) & (ci2>=0.3) & (ci2<=0.6))
    
    # t1_psf=tmatch1[psf_sel]
    # t2_psf=tmatch2[psf_sel]
    # print(len(t1_psf),len(t2_psf))

    return tmatch1,tmatch2


def gclf_match_catalogs(cat1path,cat2path,rad_asec,col_low=0.3,col_high=0.7,cs_lim=0.8,calc_ci='False'):
    
    
    #sep_asec=0.12
    font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 26}

    plt.rc('font', **font)
    
    #Read the 2 catalogs 
    tsex1 = Table.read(cat1path, format='ascii')
    tsex2 = Table.read(cat2path, format='ascii')
    
    skycoord_1=SkyCoord(tsex1['ALPHA_J2000'],tsex1['DELTA_J2000'],unit=(u.deg, u.deg))
    skycoord_2=SkyCoord(tsex2['ALPHA_J2000'],tsex2['DELTA_J2000'],unit=(u.deg, u.deg))
    
    idx1, idx2, d2d, d3d = skycoord_2.search_around_sky(skycoord_1,rad_asec*u.arcsec)
    print(len(idx1),len(idx2),len(d2d))
    
    # CATALOGS OF ALL SOURCES
    tmatch1=tsex1[idx1]
    tmatch2=tsex2[idx2]
    
    
    # CHOOSE GC CANDIDATES
    col_aper=(tmatch2['MAG_APER_1']-tmatch1['MAG_APER_1'])
    ci1=(tmatch1['MAG_APER']-tmatch1['MAG_APER_2'])
    ci2=(tmatch2['MAG_APER']-tmatch2['MAG_APER_2'])
    # plt.scatter(col_aper,tmatch1['MAG_APER_1'],marker='.')
    # plt.xlim(-1.25,2.25)
    # plt.ylim(31.5,17)
    # plt.vlines(col_low,17,31.5)
    # plt.vlines(col_high,17,31.5)
    # #plt.show(block=False)
    # #plt.clf()

    ci_low=0.25
    ci_hi=0.75
    threshold=50
    if (calc_ci=='True'):
        tclean=tsex1[(tsex1['MAG_AUTO']<threshold) & (tsex1['MAG_APER_1']<threshold) & (tsex1['MAG_APER_11']<threshold)]
        tclean_cpt=tclean[(tclean['CLASS_STAR']>cs_lim)& (tclean['FLAGS']==0)]
        ci_cpt=(tclean_cpt['MAG_APER']-tclean_cpt['MAG_APER_2'])
        ci_hi=np.median(ci_cpt)+mad(ci_cpt)*1.48*5
        ci_low=np.median(ci_cpt)-mad(ci_cpt)*1.48*5
        print(ci_low,ci_hi)
    
    print(col_low,col_high)
    print(np.median(col_aper))

    # gc_sel=((tmatch1['CLASS_STAR']>=cs_lim) & (tmatch2['CLASS_STAR']>=cs_lim) &
    #         (ci1<=ci_hi) & (ci1>=ci_low) & (ci2>=ci_low) & (ci2<=ci_hi))

    gc_sel=((col_aper<=col_high) & (col_aper>=col_low) & 
            (tmatch1['CLASS_STAR']>=cs_lim) & (tmatch2['CLASS_STAR']>=cs_lim) &
            (ci1<=ci_hi) & (ci1>=ci_low) & (ci2>=ci_low) & (ci2<=ci_hi))
    
    tmatch1_gc=tmatch1[gc_sel]
    #print(np.max(col_aper[gc_sel]))
    tmatch2_gc=tmatch2[gc_sel]
    print(len(tmatch1_gc),len(tmatch2_gc))

    return tmatch1_gc,tmatch2_gc,tmatch1,tmatch2

def basic_match_catalogs(cat1_tab,cat2_tab,rad_asec,x_1='ALPHA_J2000',y_1='DELTA_J2000',x_2='ALPHA_J2000',y_2='DELTA_J2000'):
    
    
    font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 26}

    plt.rc('font', **font)
    
    #Read the 2 catalogs 
    tsex1 = cat1_tab#Table.read(cat1path, format='ascii')
    tsex2 = cat2_tab#Table.read(cat2path, format='ascii')
    
    skycoord_1=SkyCoord(tsex1[x_1],tsex1[y_1],unit=(u.deg, u.deg))
    skycoord_2=SkyCoord(tsex2[x_2],tsex2[y_2],unit=(u.deg, u.deg))
    
    idx1, idx2, d2d, d3d = skycoord_2.search_around_sky(skycoord_1,rad_asec*u.arcsec)
    print("number of matched sources")
    print(len(idx1),len(idx2))
    
    # CATALOGS OF ALL SOURCES
    tmatch1=tsex1[idx1]
    tmatch2=tsex2[idx2]
    
    return tmatch1,tmatch2
    
def gclf_aper_corr(catfilepath, rad_asec, gxy_name) :
    threshold=50.0
    csmin=0.75
    mfaint=27.
    mbright=20.
    
    
    font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 16}

    plt.rc('font', **font)
    tsex = Table.read(catfilepath, format='ascii.sextractor')
    ttmp=tsex[(tsex['MAG_APER_11']>mbright) &(tsex['MAG_APER_11']<mfaint) & (tsex['CLASS_STAR']>csmin)] 
    apc_rough=ttmp['MAG_APER_1']-ttmp['MAG_APER_11']
    none, apc_med, apc_std=sigma_clipped_stats(apc_rough,sigma=1)
    print('Median ApC starting value', apc_med, apc_std)
     
    skycoord_full=SkyCoord(tsex['ALPHA_J2000'],tsex['DELTA_J2000'],unit=(u.deg, u.deg))
    
    tsel=tsex[(tsex['CLASS_STAR']>csmin) & (tsex['FLAGS']==0) & (tsex['MAG_AUTO']<mfaint) & (tsex['MAG_AUTO']>mbright) ]
    tsel=tsel[((tsel['MAG_APER_1']-tsel['MAG_APER_11'])<(apc_med+apc_std))&((tsel['MAG_APER_1']-tsel['MAG_APER_11'])>(apc_med-apc_std))]
    skycoord_sel=SkyCoord(tsel['ALPHA_J2000'],tsel['DELTA_J2000'],unit=(u.deg, u.deg))
    idxsel, idxfull, d2d, d3d = skycoord_full.search_around_sky(skycoord_sel,rad_asec*u.arcsec)
    
    un, c = np.unique(idxsel, return_counts=True)
    dup = un[c > 1]
    tnew=np.delete(tsel,dup)
    tnew=tnew[(tnew['MAG_APER_11']<threshold)& (tnew['MAG_AUTO']<25.0) & (tnew['MAG_AUTO']>19.0)]
    print(len(tsex),len(tnew))

    #mag4=tnew['MAG_APER']
    mag6=tnew['MAG_APER_1']
    mag32=tnew['MAG_APER_11']
    
    tbright=tsex[(tsex['MAG_AUTO']<threshold) & (tsex['MAG_APER_1']<threshold) & (tsex['MAG_APER_11']<threshold)]
    tbright_cpt=tbright[(tbright['CLASS_STAR']>csmin)& (tbright['FLAGS']==0)]

    mag6_full=tbright_cpt['MAG_APER_1']
    apers=[4,6,8,10,12,14,16,20,24,26,28,32,48,56,64]
    
    #Curve of growth for entire sample and the selected objects
    cog_full=[]
    cog_sel=[]

    for i in range(1,len(apers)) :
        mag_i_str='MAG_APER_'+str(i)
        mag_i_full=tbright_cpt[mag_i_str]
        mag_i= tnew[mag_i_str]
        cog_full.append(np.median(mag6_full[mag_i_full<threshold]-mag_i_full[mag_i_full<threshold]))
        cog_sel.append(np.median(mag6[mag_i<threshold]-mag_i[mag_i<threshold]))
    
    asym=np.median(cog_sel[7:11]) #Asymptote : median of apertures 24 to 48
    ax1 = plt.subplot(121)
    plt.title(gxy_name)
    plt.xlabel('Aperture in pixels')
    plt.ylabel('Delta mag (m_6-m_aper)')
    plt.scatter(apers[1:],cog_full,color='lightgrey')
    plt.scatter(apers[1:],cog_sel,color='C1')
    plt.axhline(asym,linestyle='--',color='C0')
    trans = transforms.blended_transform_factory(ax1.get_yticklabels()[0].get_transform(), ax1.transData)
    plt.text(0.5,asym-0.05, "{:4.3f}".format(asym), color="C0",ha="right", va="center",transform=trans)

    
    ax2=plt.subplot(122)
    plt.scatter(tbright_cpt['MAG_AUTO'],tbright_cpt['MAG_APER_1']-tbright_cpt['MAG_APER_11'],color='lightgrey',label='All bright obj')
    #plt.scatter(ttmp['MAG_AUTO'],ttmp['MAG_APER_1']-ttmp['MAG_APER_11'],color='lightgrey',label='All bright obj')

    plt.scatter(tnew['MAG_AUTO'],mag6-mag32,color='C1',label='Isolated obj')
    plt.xlabel('Magnitude')
    plt.ylabel('Delta mag (m_6-m_32)')
    #plt.ylim(np.median(mag6-mag32)-2.0*np.std(mag6-mag32),np.median(mag6-mag32)+2.0*np.std(mag6-mag32))
    plt.xlim(mbright-1,mfaint+1)
    plt.ylim(apc_med-20*apc_std,apc_med+20*apc_std)
    
    plt.axhline(np.median(mag6-mag32),color='C2')
    trans = transforms.blended_transform_factory(ax2.get_yticklabels()[0].get_transform(), ax2.transData)
    plt.text(0.5,np.median(mag6-mag32)+0.05, "{:4.3f}".format(np.median(mag6-mag32)), color="C2",ha="right", va="center",transform=trans)
    
    #tbright_cpt=ttmp#tbright[(tbright['CLASS_STAR']>csmin)& (tbright['FLAGS']==0) & (tbright['MAG_AUTO']<25.0) & (tbright['MAG_AUTO']>19.0)]
    tbright_cpt=tbright[(tbright['CLASS_STAR']>csmin)& (tbright['FLAGS']==0) & (tbright['MAG_APER_11']<mfaint) & (tbright['MAG_APER_11']>mbright)]
    med=sigma_clipped_stats(tbright_cpt['MAG_APER_1']-tbright_cpt['MAG_APER_11'],sigma=1.0)[1] #np.median(tbright_cpt['MAG_APER_1']-tbright_cpt['MAG_APER_11'])
    plt.axhline(med,color='C0')
    plt.text(0.5,med-0.05, "{:4.3f}".format(med), color="C0",ha="right", va="center",transform=trans)

    plt.axhline(np.median(mag6-mag32)-np.std(mag6-mag32),linestyle='--',color='lightblue')
    plt.axhline(np.median(mag6-mag32)+np.std(mag6-mag32),linestyle='--',color='lightblue')
    #plt.tight_layout()
    plt.legend()
    #plt.show(block=False)
    #plt.clf()

from scipy.optimize import curve_fit     
from astropy.stats import median_absolute_deviation as mad
import matplotlib.colors as colors

def sbf_aper_corr(catfilepath, rad_asec, gxyid, band,csmin,mfaint,mbright,threshold) :
    
    # threshold=50.0
    # csmin=0.75
    # mfaint=27.
    # mbright=20.
    # col_aper=(tmatch2['MAG_APER_1']-tmatch1['MAG_APER_1'])
    # ci1=(tmatch1['MAG_APER']-tmatch1['MAG_APER_2'])
    # ci2=(tmatch2['MAG_APER']-tmatch2['MAG_APER_2'])
    #plt.scatter(tmatch2['MAG_APER_1'],ci2,marker='.')
    # plt.xlim(-1.25,2.25)
    #plt.ylim(0.3,0.6)
    # plt.vlines(col_low,17,31.5)
    # plt.vlines(col_high,17,31.5)
    # #plt.show(block=False)
    # #plt.clf()
    # psf_sel=((col_aper<=col_high) & (col_aper>=col_low) & 
    # psf_sel=((tmatch1['CLASS_STAR']>=cs_lim) & (tmatch2['CLASS_STAR']>=cs_lim) &
            # (ci1<=0.8) & (ci1>=0.5) & (ci2>=0.3) & (ci2<=0.6))
    
    gxy_name=gxyid+'_'+band
    font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 16}
    plt.rc('font', **font)
    
    #load catalog
    tsex = Table.read(catfilepath, format='ascii')
    
    #Source selection ttmp: big aperture not too bright, not too faint, stars
    ttmp=tsex[(tsex['MAG_APER_11']>mbright) & (tsex['MAG_APER_11']<mfaint) & (tsex['CLASS_STAR']>csmin)] 
    
    #Rough aperture correction
    apc_rough=ttmp['MAG_APER_1']-ttmp['MAG_APER_11']
    none, apc_med, apc_std=sigma_clipped_stats(apc_rough,sigma=3)
    print('Median ApC starting value', apc_med, apc_std, '\n\n')

    #Coordinates rearrange
    skycoord_full=SkyCoord(tsex['ALPHA_J2000'],tsex['DELTA_J2000'],unit=(u.deg, u.deg))
    
    #Source selection clean: No aperture magnitude fainter than threshold
    tclean=tsex[(tsex['MAG_AUTO']<threshold) & (tsex['MAG_APER_1']<threshold) & (tsex['MAG_APER_2']<threshold) & (tsex['MAG_APER_11']<threshold)]
    
    #Source selection clean and compact: compact and no problems with sextractor
    tclean_cpt=tclean[(tclean['CLASS_STAR']>csmin)& (tclean['FLAGS']==0)]
    
    #Magnitude difference for aperture 2-0: Compact sources
    ci_cpt=(tclean_cpt['MAG_APER']-tclean_cpt['MAG_APER_2'])
    
    #Magnitude difference for aperture 2-0: clean sources
    ci=tclean['MAG_APER']-tclean['MAG_APER_2']
    
    #Maximum and minimum values of Magnitude difference for aperture 2-0
    ci_high=np.median(ci_cpt)+mad(ci_cpt)*1.48*3
    ci_low=np.median(ci_cpt)-mad(ci_cpt)*1.48*3
    
    
    #Conditions for aperture correction: clean, compact and adeguate for aperture correction
    apercorr_cond=((tclean['CLASS_STAR']>csmin)& (tclean['FLAGS']==0) & (ci>=ci_low) & (ci<=ci_high) &
                   ((tclean['MAG_APER_1']-tclean['MAG_APER_11'])<(apc_med+apc_std))&((tclean['MAG_APER_1']-tclean['MAG_APER_11'])>(apc_med-apc_std)))
    
    #Source seelection: clean, compact and adeguate for aperture correction
    tclean_cpt=tclean[apercorr_cond]
    
    #Source selection: MAG_AUTO bright, clean, compact and adeguate for aperture correction
    tbright_cpt=tclean_cpt[(tclean_cpt['MAG_AUTO']<mfaint) & (tclean_cpt['MAG_AUTO']>mbright)]

    #Parte inutile, solo rinominare selezione gia fatta
    tsel=tbright_cpt#tsex[(tsex['CLASS_STAR']>csmin) & (tsex['FLAGS']==0) & (tsex['MAG_AUTO']<mfaint) & (tsex['MAG_AUTO']>mbright) ]
    tsel=tsel[((tsel['MAG_APER_1']-tsel['MAG_APER_11'])<(apc_med+apc_std))&((tsel['MAG_APER_1']-tsel['MAG_APER_11'])>(apc_med-apc_std))]
    
    #Matching with the full sample
    skycoord_sel=SkyCoord(tsel['ALPHA_J2000'],tsel['DELTA_J2000'],unit=(u.deg, u.deg))
    idxsel, idxfull, d2d, d3d = skycoord_full.search_around_sky(skycoord_sel,rad_asec*u.arcsec)
    
    #Source selection: taking isolated sources, i.e. alone in the matching radius
    un, c = np.unique(idxsel, return_counts=True)
    non_dup = un[c == 1]
    #tnew=np.delete(tsel,dup)
    tnew=tsel[non_dup]
    print('len(tnew)',len(tnew), '\n\n')
    #tnew=tnew[(tnew['MAG_APER_11']<threshold)& (tnew['MAG_AUTO']<mfaint) & (tnew['MAG_AUTO']>mbright)]
    # print(len(tsex),len(tnew))
    
    #List definition
    mag4=tnew['MAG_APER']
    mag6=tnew['MAG_APER_1']
    mag32=tnew['MAG_APER_11']
    
    mag6_full=tclean_cpt['MAG_APER_1']
    print('len(mags)',len(mag4),len(mag6),len(mag32))
    print('len(mag6_full)',len(mag6_full),'\n\n')
    
    apers=[4,6,8,10,12,14,16,20,24,26,28,32]
    
    #Curve of growth for entire sample and the selected objects (clean and compact)
    
    
    cog_full=[]
    cog_sel=[]
    
    #Build the list of magnitude differences from the magnitude at 6 pixel 
    for i in range(1,len(apers)) :
        mag_i_str='MAG_APER_'+str(i)
        mag_i_full=tclean_cpt[mag_i_str]
        mag_i= tnew[mag_i_str]
        cog_full.append(np.median(mag6_full[mag_i_full<threshold]-mag_i_full[mag_i_full<threshold]))
        cog_sel.append(np.median(mag6[mag_i<threshold]-mag_i[mag_i<threshold]))
    
    def power_law(x,a,b):
         return a*(x**b)

    def inverse_power(y,*par):
        return (y / par[0]) ** (1 / par[1])
    
    
    #Curve of Growth fit 
    
    #Nandini method
    
    #Log function definition    
    sigmalin = lambda x,a,b: np.sum(10**(func_linear(x,a,b)))
    
    #Rearrange fit variables
    cog_sel=np.array(cog_sel)
    print('len(cog_sel)',len(cog_sel))
    print('cog_sel',cog_sel)
    d_cog=cog_sel[1:]-cog_sel[:-1]
    print('d_cog', d_cog, sum(d_cog))
    print(np.cumsum(d_cog))
    cond=(d_cog>0)
    x=np.array(apers[1:-1])[cond]
    print('x', x)
    y=np.log10(d_cog)[cond] #nandini here put log normal
    print('y', y, x*2)
    
    #Linear fit
    guess_a=-1.
    guess_b=-4
    popt_asymlin,pcov_asymlin = curve_fit(func_linear,x,y,p0=[guess_a,guess_b])
    
    plt.scatter(x,y)
    plt.plot(x, func_linear(x,*popt_asymlin), color='red', label='Fitted Line')
    # plt.plot(x, power_law(x,*popt_asympow), color='red', label='Fitted Line')
    plt.show()
    
    #Asymptote calculation
    asym=cog_sel[-1]+sigmalin(x*2,*popt_asymlin)
    
    print('testo',10**(func_linear(x,*popt_asymlin)), )
    
    print('sigmalin(x*2,*popt_asymlin)',sigmalin(x*2,*popt_asymlin),'\n\n')
    
    aper_corr=asym #np.median(mag6-mag32) #asym 
    
    print('Nandini aperture corr', aper_corr,'\n\n')
    
    
    #Gabriele method
    
    #Power law fit
    
    popt_asympow,pcov_asympow = curve_fit(power_law,x,d_cog[cond])
    #print('popt_asympow',popt_asympow, np.sum(d_cog))
   
    #Aperture value for the difference <threshold
    
    #threshold
    thr=0.001
    
    #inverse function to find the asymptote aperture
    
    aper_asym=inverse_power(thr,*popt_asympow)
    print('Aperture at asymptote',aper_asym)
    
    apers=np.array(apers)
    spacing=np.mean(apers[1:]-apers[:-1])
    
    #arrange generated aperture
    apertures=np.arange(max(apers), aper_asym,0.001)
    print(power_law(apertures,*popt_asympow ),apertures)
    print('spacing',spacing, '\n\n')
    
    asymptote=np.trapz(np.array(power_law(apertures,*popt_asympow )),apertures) #Divided by 2 considering trapezoidal integral method
    aper_corr=cog_sel[-1]+asymptote
    print('Gabriele aperture corr', aper_corr,'\n\n')
    
    #test 
    # a=power_law(x*2,*popt_asympow )
    # aper_corr_gab=cog_sel[-1]+ np.sum(a)
    # print(a)
    # print('Gabriele aperture corr', aper_corr_gab,'\n\n')
    
    print('Final aper_corr', aper_corr,'\n\n')
    
    
    fwhm=np.median(tnew['FWHM_IMAGE'])
    sfwhm=np.std(tnew['FWHM_IMAGE'])
    #tnew.write("test_tnew_"+gname+".cat",format='ascii',overwrite=True)
    #tbright_cpt.write("test_tbrightcpt_"+gname+".cat",format='ascii',overwrite=True)
    print('Median FWHM +/- err',fwhm,sfwhm,'\n\n')
    
    #Again conditions for final catalog corrected for aperture
    ci=tsex['MAG_APER']-tsex['MAG_APER_2']
    ci_high=np.median(ci_cpt)+mad(ci_cpt)*1.48*3
    ci_low=np.median(ci_cpt)-mad(ci_cpt)*1.48*3
    cpt_cond=((tsex['CLASS_STAR']>csmin) & (ci>=ci_low) & (ci<=ci_high))
    
    #Application of aperture correction
    mag_corr=np.where(cpt_cond,tsex['MAG_APER_1']-aper_corr,tsex['MAG_AUTO'])#=tsex[np.where(cpt_cond)]['MAG_APER_1']+aper_corr
    tsex['MAG_CORR']=mag_corr
    
    
    
    
    #Plots
    
    #Power law fit of the decrement in the magnitude differences
    
    
    plt.scatter(x,d_cog[cond])
    
    plt.plot(apertures, power_law(apertures,*popt_asympow ), color='blue', label='Fitted Line')
    plt.xlabel('Aperture')
    plt.ylabel('Decrement in $\Delta mag$ ')
    # plt.plot(x, power_law(x,*popt_asympow), color='red', label='Fitted Line')
    plt.plot((0,60),(0,0), ls='--', color='r')
    plt.plot((aper_asym,aper_asym),(0,d_cog[cond][0]), color='black', ls='--')
    plt.show()
    
    
    
    
    fig = plt.figure(figsize=(10,6))
    fig.subplots_adjust(wspace=0.4,hspace=0.4, left=0.15, right=0.98,
                            bottom=0.15, top=0.9)
    font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 20}
    plt.rc('font', **font)
    
    
    
    
    
    #Curve of growth

    ax1 = plt.subplot(121)
    plt.title(gxy_name)
    plt.xlabel('Aperture (in pix)')
    plt.ylabel('$\Delta mag$')
    plt.scatter(apers[1:],cog_full,color='lightcoral',marker='X',s=400,label='Bright')
    plt.scatter(apers[1:],cog_sel,color='darkgreen',s=150,label='Bright isolated')
    plt.axhline(asym,linestyle='--',color='darkgreen')
    plt.axhline(aper_corr,linestyle='--',color='red')
    trans = transforms.blended_transform_factory(ax1.get_yticklabels()[0].get_transform(), ax1.transData)
    plt.text(0.2,asym-0.05, "{:4.3f}".format(asym), color="darkgreen",ha="right", va="center",transform=trans)
    plt.legend()
    
    ax2=plt.subplot(122)
    plt.scatter(tclean_cpt['MAG_AUTO'],tclean_cpt['MAG_APER_1']-tclean_cpt['MAG_APER_11'],color='lightcoral',s=200,label='Bright')
    #plt.scatter(ttmp['MAG_AUTO'],ttmp['MAG_APER_1']-ttmp['MAG_APER_11'],color='lightgrey',label='All bright obj')

    plt.scatter(tnew['MAG_AUTO'],mag6-mag32,color='darkgreen',label='Bright isolated')
    plt.xlabel('Magnitude')
    plt.ylabel('Delta mag (m_6-m_32)')
    #plt.ylim(np.median(mag6-mag32)-2.0*np.std(mag6-mag32),np.median(mag6-mag32)+2.0*np.std(mag6-mag32))
    plt.xlim(mbright-1,mfaint+1)
    plt.ylim(apc_med-20*apc_std,apc_med+20*apc_std)
    
    plt.axhline(np.median(mag6-mag32),color='C2')
    trans = transforms.blended_transform_factory(ax2.get_yticklabels()[0].get_transform(), ax2.transData)
    plt.text(0.5,np.median(mag6-mag32)+0.005, "{:4.3f}".format(np.median(mag6-mag32)), color="C2",ha="right", va="center",transform=trans)
    
    #tbright_cpt=ttmp#tclean(tclean'CLASS_STAR']>csmin)& (tclean'FLAGS']==0) & (tclean'MAG_AUTO']<25.0) & (tclean'MAG_AUTO']>19.0)]
    med=sigma_clipped_stats(tbright_cpt['MAG_APER_1']-tbright_cpt['MAG_APER_11'],sigma=1.0)[1] #np.median(tbright_cpt['MAG_APER_1']-tbright_cpt['MAG_APER_11'])
    plt.axhline(med,color='C0')
    plt.text(0.5,med-0.005, "{:4.3f}".format(med), color="C0",ha="right", va="center",transform=trans)

    # plt.axhline(np.median(mag6-mag32)-np.std(mag6-mag32),linestyle='--',color='lightblue')
    # plt.axhline(np.median(mag6-mag32)+np.std(mag6-mag32),linestyle='--',color='lightblue')
    #plt.tight_layout()
    plt.legend()
    
    # #plt.savefig('/home/hazra/Documents/presentations/hsc_apercorr_g.jpeg',dpi=300)
    #plt.show(block=False)
    #plt.clf()
    
    
    # plt.scatter(tsex['MAG_AUTO'],tsex['MAG_AUTO']-tsex['MAG_APER_1'])
    # #plt.show(block=False)
    # #plt.clf()
    plt.savefig(f'../OUTPUT/plots/{gxyid}/{gxy_name}_apercorr.jpeg',dpi=300)
    plt.show()
    plt.clf()
    plt.close(fig)

    #mag_corr=tsex['MAG_AUTO']
    #print(mag_corr)
   
    # plt.scatter(tsex['MAG_CORR'],tsex['MAG_AUTO']-tsex['MAG_CORR'])
    # #plt.show(block=False)
    # #plt.clf()
    return aper_corr, tsex

def apply_aper_corr(catfilepath, csmin, threshold, aper_corr):

    tsex = Table.read(catfilepath, format='ascii')
    tclean=tsex[(tsex['MAG_AUTO']<threshold) & (tsex['MAG_APER_1']<threshold) & (tsex['MAG_APER_2']<threshold)]
    tclean_cpt=tclean[(tclean['CLASS_STAR']>csmin)& (tclean['FLAGS']==0)]
    ci_cpt=(tclean_cpt['MAG_APER']-tclean_cpt['MAG_APER_2'])
    # print(len(tsex),len(tclean),len(tclean_cpt))

    ci=tsex['MAG_APER']-tsex['MAG_APER_2']
    ci_high=np.median(ci_cpt)+mad(ci_cpt)*1.48*3
    ci_low=np.median(ci_cpt)-mad(ci_cpt)*1.48*3
    cpt_cond=((tsex['CLASS_STAR']>csmin) & (ci>=ci_low) & (ci<=ci_high))

    mag_corr=np.where(cpt_cond,tsex['MAG_APER_1']-aper_corr,tsex['MAG_AUTO'])#=tsex[np.where(cpt_cond)]['MAG_APER_1']+aper_corr
    magerr_corr=np.where(cpt_cond,tsex['MAGERR_APER_1'],tsex['MAGERR_AUTO'])#=tsex[np.where(cpt_cond)]['MAG_APER_1']+aper_corr
    tsex['MAG_CORR']=mag_corr
    tsex['MAGERR_CORR']=magerr_corr
    # plt.scatter(tsex['MAG_CORR'],tsex['MAG_AUTO']-tsex['MAG_CORR'])
    # #plt.show(block=False)
    # #plt.clf()
    return tsex




def visualize(data,figpath):
    mean, median, std = sigma_clipped_stats(data)
    fig = plt.figure(figsize=(7,7))
    fig.subplots_adjust(wspace=0.4,hspace=0.4, left=0.1, right=0.96,
                            bottom=0.15, top=0.9)
    font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 20}

    plt.rc('font', **font)
    
    # plt.title('IC 0745: $i$')
    plt.imshow(data, vmin = median - 2*std, vmax = median + 10*std,origin='lower', cmap='inferno')
    plt.colorbar()
    
    plt.savefig(figpath,dpi=300)
    plt.show()
    plt.clf()
    plt.close(fig)
    #plt.show(block=False)


def build_psf(respath,res_ext,corrcatpath,cscut,cilow,cihigh,mcutfaint,mcutbright,rad_asec):

        
    # def visualize_2D(data):
    #     mean, median, std = sigma_clipped_stats(data)
    #     plt.figure(figsize=(7,7))
    #     plt.imshow(data, vmin = median - 2*std, vmax = median + 10*std,origin='lower')
    #     plt.colorbar()
    #     plt.title('PSF Model')
    #     #plt.savefig('PSF.png',dpi=250,overwrite=True,bbox_inches='tight')
        
    #     #plt.show(block=False)
        
    hdul= astropy.io.fits.open(respath, ignore_missing_end = True)
    hdul.info()
    h=hdul[res_ext].header
    data=hdul[res_ext].data
    wcs=astropy.wcs.WCS(h)
    
    sex=ascii.read(corrcatpath)
    threshold=50.0 #threshold magnitude to clean catalog
    sex=sex[(sex['MAG_AUTO']<threshold) & (sex['MAG_APER']<threshold) & (sex['MAG_APER_1']<threshold) & (sex['MAG_APER_11']<threshold)]
    sex=sex[(sex['MAG_CORR']<mcutfaint)&(sex['MAG_AUTO']>mcutbright)]
    
    
    #iniziamo dando i nomi alle colonne
    ra=sex['ALPHA_J2000']
    dec=sex['DELTA_J2000']
    mag4=sex['MAG_APER']
    magau=sex['MAG_AUTO']
    ci64=sex['MAG_APER_1']-sex['MAG_APER']#concentration index 6-4
    cs=sex['CLASS_STAR']
    
    
    # cscut=0.8
    # cilow=0.25
    # cihigh=0.75
    # mcutbright=15
    # mcutfaint=23.5
    sel=((cs>cscut)&(magau<mcutfaint)&(magau>mcutbright)&(sex['FLAGS']==0))#&(ci64<cihigh)&(ci64>cilow))
    #selection on compactness and magnitude
    
    plt.scatter(magau,sex['MAG_APER']-sex['MAG_APER_2'],color='red',s=0.5,label='Cut Class Star')
    plt.scatter(sex['MAG_AUTO'][sel],sex['MAG_APER'][sel]-sex['MAG_APER_2'][sel],s=0.7,label='Cut Magnitude')
    
    #plt.xlim([20,27])
    #plt.ylim([-1.15,0.15])
    plt.ylabel('Concentration index 4-8 pix [mag]')
    plt.xlabel('mag_auto [mag]')
    #plt.show(block=False)
    #plt.clf()
    
    # print("magau[sel]")
    # print(magau[sel])
    
    plt.figure(figsize=(10,7))
    
    plt.scatter(ra,dec, s=0.1,color='red',label='All catalog ')
    plt.scatter(ra[sel],dec[sel],s=2,marker='s',label='Selected Sources')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
    #plt.show(block=False)
    
    sex_brightcpt=sex[sel]
    s=sex.to_pandas()
    #altri=s.drop(s[(s['CLASS_STAR']>cscut)&(magau<mcutfaint)&(magau>mcutbright)&((s['MAG_APER']-s['MAG_APER_2'])<cihigh)&((s['MAG_APER']-s['MAG_APER_2'])>cilow)].index)
    others=s.drop(s[sel].index)
    
    # print("others")
    # print(others)
    
    coord=SkyCoord(others['ALPHA_J2000'],others['DELTA_J2000'],unit=(u.deg, u.deg))
    coord_brightcpt =SkyCoord(sex_brightcpt['ALPHA_J2000'],sex_brightcpt['DELTA_J2000'],unit=(u.deg, u.deg))
    
    
    idxcb, idxcoord, d2d, d3d = coord.search_around_sky(coord_brightcpt, rad_asec*u.arcsec)
    print("len(d2d),d2d,idxcb")
    print(len(d2d),d2d,idxcb)
    ind=pd.DataFrame(idxcb)
    ind=ind.drop_duplicates()
    print("len(ind)")
    print(len(ind))
    
    sex_brightcpt.remove_rows(ind[0])
    coord2=SkyCoord(sex_brightcpt['ALPHA_J2000'],sex_brightcpt['DELTA_J2000'],unit=(u.deg, u.deg))
    idx_brightcpt, d2d_brightcpt, d3d_brightcpt = coord2.match_to_catalog_sky(coord2,nthneighbor=2)
    
    print("d2d_brightcpt")
    print(d2d_brightcpt.arcsec)
    
    plt.hist(d2d_brightcpt.arcsec, histtype='step')
    plt.xlabel('separation [arcsec]')
    plt.tight_layout()
    
    sex_psf=sex_brightcpt[(d2d_brightcpt.arcsec>rad_asec*10.)] #further selection on isolated
    # print("sex_psf")
    # print(sex_psf)
    print("Before and after nthneighbor=2 selection")
    print(len(sex_brightcpt),len(sex_psf))
    ra_psf=sex_psf['ALPHA_J2000']
    dec_psf=sex_psf['DELTA_J2000']
    
    plt.figure(figsize=(10,7))
    
    plt.scatter(ra,dec, s=0.1,color='red',label='All catalog')
    plt.scatter(ra_psf,dec_psf,s=2,marker='s',label='Selected isolated sources')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
    #plt.show(block=False)
    
    
    nddata = NDData(data=data,wcs=wcs)  
    position=SkyCoord(ra_psf,dec_psf,unit='deg')
    
    stars_tbl = Table()
    stars_tbl['skycoord']=position
    # stars_tbl
    
    psfsize=39
    stars = extract_stars(nddata, stars_tbl, size=psfsize)
    print(stars[0].shape)
    
    # for i in range(len(stars)):    
    #     visualize(stars[i])
     
    nrows = int(np.ceil(np.sqrt(len(stars))))
    ncols = int(np.ceil(np.sqrt(len(stars))))
    plt.figure(figsize=(nrows,ncols))
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(25, 25))
    ax = ax.ravel()
    for i in range(len(stars)):
    
        norm = simple_norm(stars[i], 'log', percent=99.)
        ax[i].imshow(stars[i], norm=norm, origin='lower', cmap='viridis')
    
    #sex_psf.remove_row(-4)
    
    position=SkyCoord(sex_psf['ALPHA_J2000'],sex_psf['DELTA_J2000'],unit='deg')
    stars_tbl = Table()
    stars_tbl['skycoord']=position
    print('Number of selected stars:',len(stars_tbl))
    
    stars = extract_stars(nddata, stars_tbl, size=psfsize)
    
    ##########################################
    # BUILDING THE LOCAL PSF
    
    
    epsf_builder = EPSFBuilder(oversampling=1, maxiters=50,progress_bar=False,norm_radius=30)  
    epsf, fitted_stars = epsf_builder.build_epsf(stars) 
    
    visualize(epsf.data)
    print(epsf.data.shape) 
    print(epsf.data.sum())
    
    x=np.arange(0,psfsize+2)
    y=epsf.data[int(psfsize/2),:]
    plt.plot(x,np.log(y))
    #plt.show(block=False)
    #plt.clf()
    #res=epsf.data
    
    ########################################### NEEDS FIXING:
    # sigma_psf=2.0
    # bkgrms = MADStdBackgroundRMS()
    # std = bkgrms(data)
    # iraffind = IRAFStarFinder(threshold=3.0*std,
    #                       fwhm=sigma_psf * gaussian_sigma_to_fwhm,
    #                       minsep_fwhm=0.01, roundhi=5.0, roundlo=-5.0,
    #                       sharplo=0.0, sharphi=2.0)
    # daogroup = DAOGroup(2.0 * sigma_psf * gaussian_sigma_to_fwhm)
    # mmm_bkg = MMMBackground()
    # fitter = LevMarLSQFitter()
    # # psf_model = IntegratedGaussianPRF(sigma=sigma_psf)
    # psf_model=epsf.data
    # photometry = IterativelySubtractedPSFPhotometry(finder=iraffind,
    #                                             group_maker=daogroup,
    #                                             bkg_estimator=mmm_bkg,
    #                                             psf_model=psf_model,
    #                                             fitter=LevMarLSQFitter(),
    #                                             niters=5, fitshape=(39,39))
    
    #result_tab = photometry(image=data)
    #residual_image = photometry.get_residual_image()
    #plt.imshow(residual_image)
    ###########################################################
    # Residui=fits.PrimaryHDU(data=epsf.data)
    # res_hdul=fits.HDUList([Residui])
    # res_hdul.info()
    # res_hdul.close()
    #res_hdul.writeto('testpsf.fits',overwrite=True)
    return epsf

from astropy.nddata.utils import Cutout2D
import time
from matplotlib.colors import LogNorm
from astropy.stats import SigmaClip
def twoband_psf(res1path,res_ext1,corrcat1path,res2path,res_ext2,corrcat2path,
                csmin,mcutfaint,mcutbright,rad_asec,threshold, psfsize,oversampling,
                gxyid,gxy_name1,gxy_name2, nthfactor=10, rgc_factor=20):

    gxyra=gal.gxyra
    gxydec=gal.gxydec
    r_e=gal.r_e # R_e in arcsec
    
    
    #Initialization catalogs
    
    #Residual catalog g
    hdul1= astropy.io.fits.open(res1path, ignore_missing_end = True)
    hdul1.info()
    h1=hdul1[res_ext1].header
    data1=hdul1[res_ext1].data
    wcs1=astropy.wcs.WCS(h1)
    
    #Corrected matched catalog g
    sex1=ascii.read(corrcat1path)
    sex1_cpt=sex1[(sex1['CLASS_STAR']>=csmin) & (sex1['FLAGS']==0)]
    
    
    #Residual catalog i
    hdul2= astropy.io.fits.open(res2path, do_not_scale_image_data = True)
    hdul2.info()
    h2=hdul2[res_ext2].header
    data2=hdul2[res_ext2].data
    wcs2=astropy.wcs.WCS(h2)
    
    #Corrected matched cagalog i
    sex2=ascii.read(corrcat2path)
    sex2_cpt=sex2[(sex2['CLASS_STAR']>=csmin) & (sex2['FLAGS']==0)]
    
    #Definitions of Concentration indexes
    ci1_cpt=(sex1_cpt['MAG_APER']-sex1_cpt['MAG_APER_2'])
    ci1=sex1['MAG_APER']-sex1['MAG_APER_2']
    ci1_high=np.median(ci1_cpt)+mad(ci1_cpt)*1.48*3
    ci1_low=np.median(ci1_cpt)-mad(ci1_cpt)*1.48*3
    
    ci2_cpt=(sex2_cpt['MAG_APER']-sex2_cpt['MAG_APER_2'])
    ci2=sex2['MAG_APER']-sex2['MAG_APER_2']
    ci2_high=np.median(ci2_cpt)+mad(ci2_cpt)*1.48*3
    ci2_low=np.median(ci2_cpt)-mad(ci2_cpt)*1.48*3
    print('Concentration index i band parameters',np.median(ci2_cpt),ci2_low, ci2_high,'\n\n')
    

    #Galactocentric distances
    rgc1=np.sqrt((sex1['ALPHA_J2000']-gxyra)**2+(sex1['DELTA_J2000']-gxydec)**2)*3600 #Galaxtocentric distance in arcsecs
    rgc2=np.sqrt((sex2['ALPHA_J2000']-gxyra)**2+(sex2['DELTA_J2000']-gxydec)**2)*3600 #Galaxtocentric distance in arcsecs
    
    
    
    #Stars selection conditions definition
    
    # ONLY select stars up to rgc_factor Half-light radii
    
    psf_sel=((sex2['MAG_CORR']<mcutfaint)&(sex2['MAG_CORR']>mcutbright) &
             (sex1['CLASS_STAR']>=csmin) & (sex2['CLASS_STAR']>=csmin)&
             (sex1['FLAGS']==0) & (sex2['FLAGS']==0) &
             (ci1>=ci1_low) & (ci1<=ci1_high) &
             (ci2>=ci2_low) & (ci2<=ci2_high) &
             (sex1['MAG_AUTO']<threshold) & (sex1['MAG_APER']<threshold) & (sex1['MAG_APER_1']<threshold) &
             (sex2['MAG_AUTO']<threshold) & (sex2['MAG_APER']<threshold) & (sex2['MAG_APER_1']<threshold) &
              (rgc1<=rgc_factor*r_e) & (rgc2<=rgc_factor*r_e))

    

      

    #Parameters initialization
    

    #qui dice psf size deve essere dispari, perchè?
    
    if (psfsize%2==0): 
        psfsize+=1
        print("SIZE OF PSF IN PIX:",psfsize,'\n\n')
    epsf_builder = EPSFBuilder(oversampling=oversampling, maxiters=50, progress_bar=True,norm_radius=psfsize,
                                sigma_clip=SigmaClip(sigma=10000, sigma_lower=10000, sigma_upper=10000, maxiters=1, cenfunc='median', stdfunc='std'))
    
    #### NORM_RADIUS has to be at least sqrt2* radius of PSF

    # Stars selection 

    # g band
    
    sex=sex1
    
    #Cleaning and parameters definition
    
    plt_clean_sex=(sex['MAG_AUTO']<threshold) & (sex['MAG_APER']<threshold) & (sex['MAG_APER_1']<threshold) & (sex['MAG_APER_11']<threshold)
    ra=sex['ALPHA_J2000']
    dec=sex['DELTA_J2000']
    mag4=sex['MAG_APER']
    magau=sex['MAG_AUTO']
    ci64=sex['MAG_APER_1']-sex['MAG_APER']  #concentration index 6-4
    cs=sex['CLASS_STAR']
        
    #Plot of initial star selection
    
    plt.scatter(magau[plt_clean_sex],sex['MAG_APER'][plt_clean_sex]-sex['MAG_APER_2'][plt_clean_sex],color='red',s=0.5)
    plt.scatter(sex['MAG_AUTO'][psf_sel],sex['MAG_APER'][psf_sel]-sex['MAG_APER_2'][psf_sel],s=1.0)
    plt.axvline(mcutbright,color='yellow')
    plt.axvline(mcutfaint,color='yellow')
    #plt.xlim([20,27])
    #plt.ylim([-1.15,0.15])
    plt.ylabel('Concentration index 4-8 pix [mag]')
    plt.xlabel('mag_auto [mag]')
    plt.show(block=False)
    plt.savefig(f'../OUTPUT/plots/{gxyid}/{gxy_name1}_psfsel_first.jpeg',dpi=300)
    plt.clf()
    plt.close()

    
    #Application of the selection conditions
    
    sex_brightcpt=sex[psf_sel]
    s=sex.to_pandas()
    
    #Objects not selected by the conditions
    others=s.drop(s[psf_sel].index)
    
    #Sky coordinates of the two catalogs
    coord=SkyCoord(others['ALPHA_J2000'],others['DELTA_J2000'],unit=(u.deg, u.deg))
    coord_brightcpt =SkyCoord(sex_brightcpt['ALPHA_J2000'],sex_brightcpt['DELTA_J2000'],unit=(u.deg, u.deg))
    
    # Select non-isolated stars
    
    idxcb, idxcoord, d2d, d3d = coord.search_around_sky(coord_brightcpt, rad_asec*u.arcsec)
    ind=pd.DataFrame(idxcb)
    ind=ind.drop_duplicates()
    
    #Remove the non-isolated stars from the catalog
    sex_brightcpt.remove_rows(ind[0])
    
    #Match the catalog with itself to find a second closest neighbor
    coord2=SkyCoord(sex_brightcpt['ALPHA_J2000'],sex_brightcpt['DELTA_J2000'],unit=(u.deg, u.deg))
    idx_brightcpt, d2d_brightcpt, d3d_brightcpt = coord2.match_to_catalog_sky(coord2,nthneighbor=2)
    
    #Further selection on isolated: the closest neighbor must be farther than a specified value
    
    sex_psf=sex_brightcpt[(d2d_brightcpt.arcsec>rad_asec*nthfactor)] 
    
    #Catalog handling
    savepath=f'../OUTPUT/{gxyid}/{gxy_name1}_psf.cat'
    sex_psf.write(savepath,format='ascii',overwrite=True)
    sex_psf=sex_psf.to_pandas()
    sex_psf=sex_psf.sort_values('MAG_CORR')
    
    ra_psf=sex_psf['ALPHA_J2000']
    dec_psf=sex_psf['DELTA_J2000']
    
    #Plot the positions of the stars 
    plt.figure(figsize=(10,7))
    
    plt.scatter(ra,dec, s=0.1,color='red',label='All catalog')
    plt.scatter(ra_psf,dec_psf,s=2,marker='s',label='Selected isolated sources')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(f'../OUTPUT/plots/{gxyid}/{gxy_name1}_psfsel_isolated.jpeg',dpi=300)
    plt.show()
    plt.clf()
    plt.close()

    #Creation of the NDDdata to extract the cutout of the stars with photutils extract_stars
    
    nddata = NDData(data=data1,wcs=wcs1)  
    position=SkyCoord(ra_psf,dec_psf,unit='deg')
    
    stars_tbl = Table()
    stars_tbl['skycoord']=position
    # stars_tbl=stars_tbl[:8]
    print('Number of selected stars:',len(stars_tbl),'\n\n')
   
    #Stars cut-out
    
    stars = extract_stars(nddata, stars_tbl, size=psfsize)
    
    print(stars[0].shape)
    
    # for i in range(len(stars)):    
    #     visualize(stars[i])
    
    #Plot of the selected stars
    nrows = int(np.ceil(np.sqrt(len(stars))))
    ncols = int(np.ceil(np.sqrt(len(stars))))
    plt.figure(figsize=(nrows,ncols))
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(25, 25))
    ax = ax.ravel()
    for i in range(len(stars)):
    
        norm = simple_norm(stars[i], 'log', percent=99.)
        ax[i].imshow(stars[i], norm=norm, origin='lower', cmap='viridis')

    plt.show(block=False)
    plt.savefig(f'../OUTPUT/plots/{gxyid}/{gxy_name1}_psfstars.jpeg',dpi=300)
    plt.clf()
    plt.close(fig)

    
    
    stars = extract_stars(nddata, stars_tbl, size=psfsize)
    
    
    # Building of the PSF: g band
       
    epsf1, fitted_stars1 = epsf_builder.build_epsf(stars) 
    

    visualize(data=epsf1.data,figpath=f'../OUTPUT/plots/{gxyid}/{gxy_name1}_epsf.jpeg')
    print('Epsf g band data shape',epsf1.data.shape) 
    print('Epsf g band data sum',epsf1.data.sum(),'\n\n')
    
    
    
   
    # i band
    
    sex=sex2
    
    plt_clean_sex=(sex['MAG_AUTO']<threshold) & (sex['MAG_APER']<threshold) & (sex['MAG_APER_1']<threshold) & (sex['MAG_APER_11']<threshold)
    
    ra=sex['ALPHA_J2000']
    dec=sex['DELTA_J2000']
    magau=sex['MAG_AUTO']
    
    #Plot firs selection stars
    plt.scatter(magau[plt_clean_sex],sex['MAG_APER'][plt_clean_sex]-sex['MAG_APER_2'][plt_clean_sex],color='red',s=0.5)
    plt.scatter(sex['MAG_AUTO'][psf_sel],sex['MAG_APER'][psf_sel]-sex['MAG_APER_2'][psf_sel],s=1)
    # plt.axvline(mcutbright,color='yellow')
    # plt.axvline(mcutfaint,color='yellow')
    
    #plt.xlim([20,27])
    #plt.ylim([-1.15,0.15])
    plt.ylabel('Concentration index 4-8 pix [mag]')
    plt.xlabel('mag_auto [mag]')
    plt.savefig(f'../OUTPUT/plots/{gxyid}/{gxy_name2}_psfsel_first.jpeg',dpi=300)
    plt.show()
    plt.clf()
    plt.close()
    
    #Application of the selection conditions
    sex_brightcpt=sex[psf_sel]
    s=sex.to_pandas()
    
    #discrad the non-selected objects
    others=s.drop(s[psf_sel].index)
    
    #Sky coordinates of the two catalogs
    coord=SkyCoord(others['ALPHA_J2000'],others['DELTA_J2000'],unit=(u.deg, u.deg))
    coord_brightcpt =SkyCoord(sex_brightcpt['ALPHA_J2000'],sex_brightcpt['DELTA_J2000'],unit=(u.deg, u.deg))
    
    #Selection of non-isolated stars
    idxcb, idxcoord, d2d, d3d = coord.search_around_sky(coord_brightcpt, rad_asec*u.arcsec)
    print("Number of selected non-isolated stars", len(d2d), '\n\n')
    
    ind=pd.DataFrame(idxcb)
    ind=ind.drop_duplicates()
    
    
    #First removal of the non-isolated stars
    sex_brightcpt.remove_rows(ind[0])
    
    #Matching the catalog with itself to find the closest neighbor
    coord2=SkyCoord(sex_brightcpt['ALPHA_J2000'],sex_brightcpt['DELTA_J2000'],unit=(u.deg, u.deg))
    idx_brightcpt, d2d_brightcpt, d3d_brightcpt = coord2.match_to_catalog_sky(coord2,nthneighbor=2)
    
    #Further selection on isolated: the closest neighbor must be farther than a specified value

    sex_psf=sex_brightcpt[(d2d_brightcpt.arcsec>rad_asec*nthfactor)] #further selection on isolated
    
    #Catalogs handling
    savepath=f'../OUTPUT/{gxyid}/{gxy_name2}_psf.cat'
    sex_psf.write(savepath,format='ascii',overwrite=True)
    sex_psf=sex_psf.to_pandas()
    sex_psf=sex_psf.sort_values('MAG_CORR')

    print("Number of selected nthneighbor=2 stars", len(sex_brightcpt)-len(sex_psf),'\n\n')
    
    
    
    
    
   #Plot of the positions of the stars 
    plt.figure(figsize=(10,7))
    
    plt.scatter(ra,dec, s=0.1,color='red',label='All catalog')
    plt.scatter(ra_psf,dec_psf,s=2,marker='s',label='Selected isolated sources')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(f'../OUTPUT/plots/{gxyid}/{gxy_name2}_psfsel_isolated.jpeg',dpi=300)
    plt.clf()
    plt.close()

    #plt.show(block=False)
    
    #Creation of the NDDdata to extract the cutout of the stars with photutils extract_stars
    ra_psf=sex_psf['ALPHA_J2000']
    dec_psf=sex_psf['DELTA_J2000']
    
    nddata = NDData(data=data2,wcs=wcs2)  
    position=SkyCoord(ra_psf,dec_psf,unit='deg')
    
    stars_tbl = Table()
    stars_tbl['skycoord']=position
    # stars_tbl=stars_tbl[:8]
    print('Number of selected stars:',len(stars_tbl),'\n\n')
    
    #Cut-out
    stars = extract_stars(nddata, stars_tbl, size=psfsize)
    
    # for i in range(len(stars)):    
    #     visualize(stars[i])
     
    #Plot of the cut-out stars
    
    nrows = int(np.ceil(np.sqrt(len(stars))))
    ncols = int(np.ceil(np.sqrt(len(stars))))
    plt.figure(figsize=(nrows,ncols))
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(25, 25))
    ax = ax.ravel()
    for i in range(len(stars)):
    
        norm = simple_norm(stars[i], 'log', percent=99.)
        ax[i].imshow(stars[i], norm=norm, origin='lower', cmap='viridis')
        fits.writeto(f'../OUTPUT/{gxyid}/{gxy_name2}_psfstar_{i}.fits',stars[i].data.astype(np.float32),overwrite=True)


    plt.savefig(f'../OUTPUT/plots/{gxyid}/{gxy_name2}_psfstars.jpeg',dpi=300)
    plt.show()
    plt.clf()
    plt.close(fig)
    
    
    print('Number of selected stars',len(stars),'\n\n')
    
    
    # Building of the PSF
    
    
    epsf2, fitted_stars2 = epsf_builder.build_epsf(stars) 
    
    visualize(data=epsf2.data,figpath=f'../OUTPUT/plots/{gxyid}/{gxy_name2}_epsf.jpeg')
    print(epsf2.data.shape) 
    print(epsf2.data.sum())
    
        

    # print('NEED TO COMPARE PHOTOMETRY OF PSF WITH DAOPHOT')

    return epsf1,epsf2



def twoband_psf_VCC(res1path,res_ext1,corrcat1path,res2path,res_ext2,corrcat2path,
                csmin,mcutfaint,mcutbright,rad_asec,threshold, psfsize,oversampling,
                gxyid,gxy_name1,gxy_name2, nthfactor=10, rgc_factor=20, run_epsfbuild=True, seg_path=None):

    gxyra=gal.gxyra
    gxydec=gal.gxydec
    r_e=gal.r_e # R_e in arcsec
    
    #Residual in g
    hdul1= fits.open(res1path, do_not_scale_image_data = True)
    hdul1.info()
    h1=hdul1[res_ext1].header
    data1=hdul1[res_ext1].data
    wcs1=astropy.wcs.WCS(h1)
    
    #Corrected matched cat g
    sex1=Table.read(corrcat1path, format='ascii.commented_header')
    #Compact
    sex1_cpt=sex1[(sex1['CLASS_STAR']>=csmin) & (sex1['FLAGS']==0)]
    
    #Residuals in i
    
    hdul2= astropy.io.fits.open(res2path, do_not_scale_image_data = True)
    hdul2.info()
    h2=hdul2[res_ext2].header
    data2=hdul2[res_ext2].data
    wcs2=astropy.wcs.WCS(h2)
    
    #Corrected matche cat i
    
    sex2=Table.read(corrcat2path, format='ascii.commented_header')
    #Compact
    sex2_cpt=sex2[(sex2['CLASS_STAR']>=csmin) & (sex2['FLAGS']==0)]
    
    #Selection stars for psf 1
    ci1_cpt=(sex1_cpt['MAG_APER']-sex1_cpt['MAG_APER_2'])
    ci1=sex1['MAG_APER']-sex1['MAG_APER_2']
    ci1_high=np.median(ci1_cpt)+mad(ci1_cpt)*1.48*3
    ci1_low=np.median(ci1_cpt)-mad(ci1_cpt)*1.48*3
    
    ci2_cpt=(sex2_cpt['MAG_APER']-sex2_cpt['MAG_APER_2'])
    ci2=sex2['MAG_APER']-sex2['MAG_APER_2']
    ci2_high=np.median(ci2_cpt)+mad(ci2_cpt)*1.48*3
    ci2_low=np.median(ci2_cpt)-mad(ci2_cpt)*1.48*3
    # cpt_cond=((tclean['CLASS_STAR']>csmin) & (ci>=ci_low) & (ci<=ci_high))
    # apercorr_cond=((tclean['CLASS_STAR']>csmin)& (tclean['FLAGS']==0) & (ci>=ci_low) & (ci<=ci_high) &



    
    rgc1=np.sqrt((sex1['ALPHA_J2000']-gxyra)**2+(sex1['DELTA_J2000']-gxydec)**2)*3600 #Galaxtocentric distance in arcsecs
    rgc2=np.sqrt((sex2['ALPHA_J2000']-gxyra)**2+(sex2['DELTA_J2000']-gxydec)**2)*3600 #Galaxtocentric distance in arcsecs
    
    # PSF selection 2 (ONLY select stars up to rgc_factor Half-light radii)
    
    psf_sel=((sex2['MAG_CORR']<mcutfaint)&(sex2['MAG_CORR']>mcutbright) &
             (sex1['CLASS_STAR']>=csmin) & (sex2['CLASS_STAR']>=csmin)&
             (sex1['FLAGS']==0) & (sex2['FLAGS']==0) &
             (ci1>=ci1_low) & (ci1<=ci1_high) &
             (ci2>=ci2_low) & (ci2<=ci2_high) &
             (sex1['MAG_AUTO']<threshold) & (sex1['MAG_APER']<threshold) & (sex1['MAG_APER_1']<threshold) &
             (sex2['MAG_AUTO']<threshold) & (sex2['MAG_APER']<threshold) & (sex2['MAG_APER_1']<threshold) &
              (rgc1<=rgc_factor*r_e) & (rgc2<=rgc_factor*r_e))
    # print(np.min(rgc1[psf_sel]))
    


    #parameters initialization
    

    #qui dice psf size deve essere dispari, perchè?
    
    if (psfsize%2==0): 
        psfsize+=1
        print("SIZE OF PSF IN PIX:",psfsize,'\n\n')
    epsf_builder = EPSFBuilder(oversampling=oversampling, maxiters=50, progress_bar=True,norm_radius=psfsize,
                                sigma_clip=SigmaClip(sigma=10000, sigma_lower=10000, sigma_upper=10000, maxiters=1, cenfunc='median', stdfunc='std'))
    
    #### NORM_RADIUS has to be at least sqrt2* radius of PSF

    # Stars selection 

    # g band
    
    sex=sex1
    
    #Cleaning and parameters definition
    
    plt_clean_sex=(sex['MAG_AUTO']<threshold) & (sex['MAG_APER']<threshold) & (sex['MAG_APER_1']<threshold) & (sex['MAG_APER_11']<threshold)
    ra=sex['ALPHA_J2000']
    dec=sex['DELTA_J2000']
    mag4=sex['MAG_APER']
    magau=sex['MAG_AUTO']
    ci64=sex['MAG_APER_1']-sex['MAG_APER']  #concentration index 6-4
    cs=sex['CLASS_STAR']
        
    #Plot of initial star selection
    
    plt.scatter(magau[plt_clean_sex],sex['MAG_APER'][plt_clean_sex]-sex['MAG_APER_2'][plt_clean_sex],color='red',s=0.5)
    plt.scatter(sex['MAG_AUTO'][psf_sel],sex['MAG_APER'][psf_sel]-sex['MAG_APER_2'][psf_sel],s=1.0)
    plt.axvline(mcutbright,color='yellow')
    plt.axvline(mcutfaint,color='yellow')
    #plt.xlim([20,27])
    #plt.ylim([-1.15,0.15])
    plt.ylabel('Concentration index 4-8 pix [mag]')
    plt.xlabel('mag_auto [mag]')
    plt.show(block=False)
    plt.savefig(f'../OUTPUT/plots/{gxyid}/{gxy_name1}_psfsel_first.jpeg',dpi=300)
    plt.clf()
    plt.close()

    
    #Application of the selection conditions
    
    sex_brightcpt=sex[psf_sel]
    s=sex.to_pandas()
    
    #Objects not selected by the conditions
    others=s.drop(s[psf_sel].index)
    
    #Sky coordinates of the two catalogs
    coord=SkyCoord(others['ALPHA_J2000'],others['DELTA_J2000'],unit=(u.deg, u.deg))
    coord_brightcpt =SkyCoord(sex_brightcpt['ALPHA_J2000'],sex_brightcpt['DELTA_J2000'],unit=(u.deg, u.deg))
    
    # Select non-isolated stars
    
    idxcb, idxcoord, d2d, d3d = coord.search_around_sky(coord_brightcpt, rad_asec*u.arcsec)
    ind=pd.DataFrame(idxcb)
    ind=ind.drop_duplicates()
    
    #Remove the non-isolated stars from the catalog
    sex_brightcpt.remove_rows(ind[0])
    
    #Match the catalog with itself to find a second closest neighbor
    coord2=SkyCoord(sex_brightcpt['ALPHA_J2000'],sex_brightcpt['DELTA_J2000'],unit=(u.deg, u.deg))
    idx_brightcpt, d2d_brightcpt, d3d_brightcpt = coord2.match_to_catalog_sky(coord2,nthneighbor=2)
    
    #Further selection on isolated: the closest neighbor must be farther than a specified value
    
    sex_psf=sex_brightcpt[(d2d_brightcpt.arcsec>rad_asec*nthfactor)] 
    
    #Catalog handling
    savepath=f'../OUTPUT/{gxyid}/{gxy_name1}_psf.cat'
    sex_psf.write(savepath,format='ascii',overwrite=True)
    sex_psf=sex_psf.to_pandas()
    sex_psf=sex_psf.sort_values('MAG_CORR')
    
    ra_psf=sex_psf['ALPHA_J2000']
    dec_psf=sex_psf['DELTA_J2000']
    
    #Plot the positions of the stars 
    plt.figure(figsize=(10,7))
    
    plt.scatter(ra,dec, s=0.1,color='red',label='All catalog')
    plt.scatter(ra_psf,dec_psf,s=2,marker='s',label='Selected isolated sources')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(f'../OUTPUT/plots/{gxyid}/{gxy_name1}_psfsel_isolated.jpeg',dpi=300)
    plt.show()
    plt.clf()
    plt.close()

    #Creation of the NDDdata to extract the cutout of the stars with photutils extract_stars
    
    nddata = NDData(data=data1,wcs=wcs1)  
    position=SkyCoord(ra_psf,dec_psf,unit='deg')
    
    stars_tbl = Table()
    stars_tbl['skycoord']=position
    # stars_tbl=stars_tbl[:8]
    print('Number of selected stars:',len(stars_tbl),'\n\n')
   
    #Stars cut-out
    
    stars = extract_stars(nddata, stars_tbl, size=psfsize)
    
    print(stars[0].shape)
    
    # for i in range(len(stars)):    
    #     visualize(stars[i])
    
    #Plot of the selected stars
    nrows = int(np.ceil(np.sqrt(len(stars))))
    ncols = int(np.ceil(np.sqrt(len(stars))))
    plt.figure(figsize=(nrows,ncols))
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(25, 25))
    ax = ax.ravel()
    for i in range(len(stars)):
    
        norm = simple_norm(stars[i], 'log', percent=99.)
        ax[i].imshow(stars[i], norm=norm, origin='lower', cmap='viridis')

    plt.show(block=False)
    plt.savefig(f'../OUTPUT/plots/{gxyid}/{gxy_name1}_psfstars.jpeg',dpi=300)
    plt.show()
    plt.clf()
    plt.close(fig)


   
    
   
    # i band
    
    sex=sex2
    
    plt_clean_sex=(sex['MAG_AUTO']<threshold) & (sex['MAG_APER']<threshold) & (sex['MAG_APER_1']<threshold) & (sex['MAG_APER_11']<threshold)
    
    ra=sex['ALPHA_J2000']
    dec=sex['DELTA_J2000']
    magau=sex['MAG_AUTO']
    
    #Plot firs selection stars
    plt.scatter(magau[plt_clean_sex],sex['MAG_APER'][plt_clean_sex]-sex['MAG_APER_2'][plt_clean_sex],color='red',s=0.5)
    plt.scatter(sex['MAG_AUTO'][psf_sel],sex['MAG_APER'][psf_sel]-sex['MAG_APER_2'][psf_sel],s=1)
    # plt.axvline(mcutbright,color='yellow')
    # plt.axvline(mcutfaint,color='yellow')
    
    #plt.xlim([20,27])
    #plt.ylim([-1.15,0.15])
    plt.ylabel('Concentration index 4-8 pix [mag]')
    plt.xlabel('mag_auto [mag]')
    plt.savefig(f'../OUTPUT/plots/{gxyid}/{gxy_name2}_psfsel_first.jpeg',dpi=300)
    plt.show()
    plt.clf()
    plt.close()
    
    #Application of the selection conditions
    sex_brightcpt=sex[psf_sel]
    s=sex.to_pandas()
    
    #discrad the non-selected objects
    others=s.drop(s[psf_sel].index)
    
    #Sky coordinates of the two catalogs
    coord=SkyCoord(others['ALPHA_J2000'],others['DELTA_J2000'],unit=(u.deg, u.deg))
    coord_brightcpt =SkyCoord(sex_brightcpt['ALPHA_J2000'],sex_brightcpt['DELTA_J2000'],unit=(u.deg, u.deg))
    
    #Selection of non-isolated stars
    idxcb, idxcoord, d2d, d3d = coord.search_around_sky(coord_brightcpt, rad_asec*u.arcsec)
    print("Number of selected non-isolated stars", len(d2d), '\n\n')
    
    ind=pd.DataFrame(idxcb)
    ind=ind.drop_duplicates()
    
    
    #First removal of the non-isolated stars
    sex_brightcpt.remove_rows(ind[0])
    
    #Matching the catalog with itself to find the closest neighbor
    coord2=SkyCoord(sex_brightcpt['ALPHA_J2000'],sex_brightcpt['DELTA_J2000'],unit=(u.deg, u.deg))
    idx_brightcpt, d2d_brightcpt, d3d_brightcpt = coord2.match_to_catalog_sky(coord2,nthneighbor=2)
    
    #Further selection on isolated: the closest neighbor must be farther than a specified value

    sex_psf=sex_brightcpt[(d2d_brightcpt.arcsec>rad_asec*nthfactor)] #further selection on isolated
    
    #Catalogs handling
    savepath=f'../OUTPUT/{gxyid}/{gxy_name2}_psf.cat'
    sex_psf.write(savepath,format='ascii',overwrite=True)
    sex_psf=sex_psf.to_pandas()
    sex_psf=sex_psf.sort_values('MAG_CORR')

    print("Number of selected nthneighbor=2 stars", len(sex_brightcpt)-len(sex_psf),'\n\n')
    
    
    
    
    
   #Plot of the positions of the stars 
    plt.figure(figsize=(10,7))
    
    plt.scatter(ra,dec, s=0.1,color='red',label='All catalog')
    plt.scatter(ra_psf,dec_psf,s=2,marker='s',label='Selected isolated sources')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(f'../OUTPUT/plots/{gxyid}/{gxy_name2}_psfsel_isolated.jpeg',dpi=300)
    plt.clf()
    plt.close()

    #plt.show(block=False)
    
    #Creation of the NDDdata to extract the cutout of the stars with photutils extract_stars
    ra_psf=sex_psf['ALPHA_J2000']
    dec_psf=sex_psf['DELTA_J2000']
    
    nddata = NDData(data=data2,wcs=wcs2)  
    position=SkyCoord(ra_psf,dec_psf,unit='deg')
    
    stars_tbl = Table()
    stars_tbl['skycoord']=position
    # stars_tbl=stars_tbl[:8]
    print('Number of selected stars:',len(stars_tbl),'\n\n')
    
    #Cut-out
    stars = extract_stars(nddata, stars_tbl, size=psfsize)
    
    # for i in range(len(stars)):    
    #     visualize(stars[i])
    
    if seg_path is not None:
        hdum2= astropy.io.fits.open(seg_path, do_not_scale_image_data = True)
        mask2=hdum2[res_ext2].data
        ndmask = NDData(data=mask2,wcs=wcs2)  
        # masks= extract_stars(ndmask, stars_tbl, size=psfsize)

        segmask=np.ones(mask2.shape)
        # segmask[mask2 not in id_psf]=0
        data2=data2*segmask
    
    #Plot of the cut-outs
    
    nrows = int(np.ceil(np.sqrt(len(stars))))
    ncols = int(np.ceil(np.sqrt(len(stars))))
    plt.figure(figsize=(nrows,ncols))
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(25, 25))
    ax = ax.ravel()
    for i in range(len(stars)):
    
        # norm = simple_norm(stars[i], 'log', percent=99.)
        # ax[i].imshow(stars[i], norm=norm, origin='lower', cmap='viridis')
        ax[i].imshow(np.log10(stars[i]), vmin=-2,vmax=5, origin='lower', cmap='viridis')

        fits.writeto(f'../OUTPUT/{gxyid}/{gxy_name2}_psfstar_{i}.fits',stars[i].data.astype(np.float32),overwrite=True)


    plt.savefig(f'../OUTPUT/plots/{gxyid}/{gxy_name2}_psfstars.jpeg',dpi=300)
    plt.show()
    plt.clf()
    plt.close(fig)
    
    
    print(len(stars))
    
    print('Number of selected stars',len(stars),'\n\n')
    
    
    # Building of the PSF
    

    # epsf_builder = EPSFBuilder(oversampling=1, maxiters=50,progress_bar=False,norm_radius=30)  
    if (run_epsfbuild):
        epsf2, fitted_stars2 = epsf_builder.build_epsf(stars) 
        
        visualize(data=epsf2.data,figpath=f'../OUTPUT/plots/{gxyid}/{gxy_name2}_epsf.jpeg')
        print('Epsf g band data shape',epsf2.data.shape) 
        print('Epsf g band data sum',epsf2.data.sum(),'\n\n')
        
            

        # print('NEED TO COMPARE PHOTOMETRY OF PSF WITH DAOPHOT')
        
        return epsf2

def choose_stars(res1path,res_ext1,corrcat1path,res2path,res_ext2,corrcat2path,
                csmin,mcutfaint,mcutbright,rad_asec,threshold, psfsize,oversampling,
                gxyid,gxy_name1,gxy_name2, nthfactor=10, rgc_factor=20):

    gxyra=gal.gxyra
    gxydec=gal.gxydec
    r_e=gal.r_e # R_e in arcsec
    
    hdul1= fits.open(res1path, do_not_scale_image_data = True)
    hdul1.info()
    h1=hdul1[res_ext1].header
    data1=hdul1[res_ext1].data
    wcs1=astropy.wcs.WCS(h1)
    
    sex1=Table.read(corrcat1path, format='ascii.commented_header')
    sex1_cpt=sex1[(sex1['CLASS_STAR']>=csmin) & (sex1['FLAGS']==0)]
    
    # hdul2= astropy.io.fits.open(res2path, ignore_missing_end = True)
    hdul2= astropy.io.fits.open(res2path, do_not_scale_image_data = True)
    hdul2.info()
    h2=hdul2[res_ext2].header
    data2=hdul2[res_ext2].data
    wcs2=astropy.wcs.WCS(h2)
    
    sex2=Table.read(corrcat2path, format='ascii.commented_header')
    sex2_cpt=sex2[(sex2['CLASS_STAR']>=csmin) & (sex2['FLAGS']==0)]
    
    ci1_cpt=(sex1_cpt['MAG_APER']-sex1_cpt['MAG_APER_2'])
    ci1=sex1['MAG_APER']-sex1['MAG_APER_2']
    ci1_high=np.median(ci1_cpt)+mad(ci1_cpt)*1.48*3
    ci1_low=np.median(ci1_cpt)-mad(ci1_cpt)*1.48*3
    
    ci2_cpt=(sex2_cpt['MAG_APER']-sex2_cpt['MAG_APER_2'])
    ci2=sex2['MAG_APER']-sex2['MAG_APER_2']
    ci2_high=np.median(ci2_cpt)+mad(ci2_cpt)*1.48*3
    ci2_low=np.median(ci2_cpt)-mad(ci2_cpt)*1.48*3
    # cpt_cond=((tclean['CLASS_STAR']>csmin) & (ci>=ci_low) & (ci<=ci_high))
    # apercorr_cond=((tclean['CLASS_STAR']>csmin)& (tclean['FLAGS']==0) & (ci>=ci_low) & (ci<=ci_high) &



    
    rgc1=np.sqrt((sex1['ALPHA_J2000']-gxyra)**2+(sex1['DELTA_J2000']-gxydec)**2)*3600 #Galaxtocentric distance in arcsecs
    rgc2=np.sqrt((sex2['ALPHA_J2000']-gxyra)**2+(sex2['DELTA_J2000']-gxydec)**2)*3600 #Galaxtocentric distance in arcsecs
    # print(np.min(rgc1))
    #### ONLY select stars up to 20 Half-light radii
    
    psf_sel=((sex2['MAG_CORR']<mcutfaint)&(sex2['MAG_CORR']>mcutbright) &
             (sex1['CLASS_STAR']>=csmin) & (sex2['CLASS_STAR']>=csmin)&
             (sex1['FLAGS']==0) & (sex2['FLAGS']==0) &
             (ci1>=ci1_low) & (ci1<=ci1_high) &
             (ci2>=ci2_low) & (ci2<=ci2_high) &
             (sex1['MAG_AUTO']<threshold) & (sex1['MAG_APER']<threshold) & (sex1['MAG_APER_1']<threshold) &
             (sex2['MAG_AUTO']<threshold) & (sex2['MAG_APER']<threshold) & (sex2['MAG_APER_1']<threshold) &
              (rgc1<=rgc_factor*r_e) & (rgc2<=rgc_factor*r_e))
    # print(np.min(rgc1[psf_sel]))
    

    # print("(SOLVED)WARNING: THE PSF SELECTION ALSO NEEDS TO FACTOR IN THE GALACTOCENTRIC DISTANCE (Selected upto 20 half-light radii)")     

    ######### INITIALIZE PSF PARAMETERS
    #psfsize=64

    if (psfsize%2==0): 
        psfsize+=1
        print("SIZE OF PSF IN PIX:",psfsize)
    # epsf_builder = EPSFBuilder(oversampling=oversampling, maxiters=50, progress_bar=True,norm_radius=psfsize,
    #                             sigma_clip=SigmaClip(sigma=10000, sigma_lower=10000, sigma_upper=10000, maxiters=1, cenfunc='median', stdfunc='std'))
    # epsf_builder = EPSFBuilder(oversampling=oversampling, maxiters=50, progress_bar=True,norm_radius=psfsize/2,recentering_boxsize=9,
    #                             sigma_clip=SigmaClip(sigma=5, maxiters=3, cenfunc='median', stdfunc='std'))
    # epsf_builder = EPSFBuilder(oversampling=1, maxiters=50, progress_bar=True,norm_radius=psfsize,recentering_boxsize=9,
    #                             sigma_clip=SigmaClip(sigma=5, maxiters=3, cenfunc='median', stdfunc='std'))
    # epsf_builder = EPSFBuilder(oversampling=oversampling, maxiters=50, progress_bar=True,norm_radius=psfsize,recentering_boxsize=9,
    #                             sigma_clip=SigmaClip(sigma=5, maxiters=3, cenfunc='median', stdfunc='std'))
    #### NORM_RADIUS has to be at least sqrt2* radius of PSF

    
    ################################# BAND 1 
    sex=sex1

    plt_clean_sex=(sex['MAG_AUTO']<threshold) & (sex['MAG_APER']<threshold) & (sex['MAG_APER_1']<threshold) & (sex['MAG_APER_11']<threshold)
    ra=sex['ALPHA_J2000']
    dec=sex['DELTA_J2000']
    mag4=sex['MAG_APER']
    magau=sex['MAG_AUTO']
    ci64=sex['MAG_APER_1']-sex['MAG_APER']#concentration index 6-4
    cs=sex['CLASS_STAR']
        
    plt.scatter(magau[plt_clean_sex],sex['MAG_APER'][plt_clean_sex]-sex['MAG_APER_2'][plt_clean_sex],color='red',s=0.5)
    plt.scatter(sex['MAG_AUTO'][psf_sel],sex['MAG_APER'][psf_sel]-sex['MAG_APER_2'][psf_sel],s=1.0)
    plt.axvline(mcutbright,color='yellow')
    # print("%%%%%%%%%%%%%%%%%%%%%%%")
    # print(mcutbright,mcutfaint)
    plt.axvline(mcutfaint,color='yellow')
    #plt.xlim([20,27])
    #plt.ylim([-1.15,0.15])
    plt.ylabel('Concentration index 4-8 pix [mag]')
    plt.xlabel('mag_auto [mag]')
    #plt.show(block=False)
    plt.savefig(f'../OUTPUT/plots/{gxyid}/{gxy_name1}_psfsel_first.jpeg',dpi=300)
    plt.clf()
    plt.close()

    # print("magau[sel]")
    # print(magau[sel])
    
    # plt.figure(figsize=(10,7))
    
    # plt.scatter(ra,dec, s=0.1,color='red',label='All catalog ')
    # plt.scatter(ra[psf_sel],dec[psf_sel],s=2,marker='s',label='Selected Bright, Compact, Central')
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
    #plt.show(block=False)
    # plt.savefig(f'../OUTPUT/plots/{gxyid}/{gxy_name1}_psfsel.jpeg',dpi=300)
    # plt.clf()
    # plt.close()
    
    sex_brightcpt=sex[psf_sel]
    s=sex.to_pandas()
    others=s.drop(s[psf_sel].index)
    
   
    coord=SkyCoord(others['ALPHA_J2000'],others['DELTA_J2000'],unit=(u.deg, u.deg))
    coord_brightcpt =SkyCoord(sex_brightcpt['ALPHA_J2000'],sex_brightcpt['DELTA_J2000'],unit=(u.deg, u.deg))
    
    
    idxcb, idxcoord, d2d, d3d = coord.search_around_sky(coord_brightcpt, rad_asec*u.arcsec)
    ind=pd.DataFrame(idxcb)
    ind=ind.drop_duplicates()
    
    sex_brightcpt.remove_rows(ind[0])
    coord2=SkyCoord(sex_brightcpt['ALPHA_J2000'],sex_brightcpt['DELTA_J2000'],unit=(u.deg, u.deg))
    idx_brightcpt, d2d_brightcpt, d3d_brightcpt = coord2.match_to_catalog_sky(coord2,nthneighbor=2)
    
    
    sex_psf=sex_brightcpt[(d2d_brightcpt.arcsec>rad_asec*nthfactor)] #further selection on isolated
    # print("sex_psf")
    # print(sex_psf)
    # print("Before and after nthneighbor=2 selection")
    # print(len(sex_brightcpt),len(sex_psf))
    # print(sex_psf)
    savepath=f'../OUTPUT/{gxyid}/{gxy_name1}_psf.cat'
    sex_psf.write(savepath,format='ascii',overwrite=True)
    sex_psf=sex_psf.to_pandas()
    sex_psf=sex_psf.sort_values('MAG_CORR')
    
    ra_psf=sex_psf['ALPHA_J2000']
    dec_psf=sex_psf['DELTA_J2000']
    
    plt.figure(figsize=(10,7))
    
    plt.scatter(ra,dec, s=0.1,color='red',label='All catalog')
    plt.scatter(ra_psf,dec_psf,s=2,marker='s',label='Selected isolated sources')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(f'../OUTPUT/plots/{gxyid}/{gxy_name1}_psfsel_isolated.jpeg',dpi=300)
    plt.clf()
    plt.close()


    #plt.show(block=False)
    #plt.clf()
    
    nddata = NDData(data=data1,wcs=wcs1)  
    position=SkyCoord(ra_psf,dec_psf,unit='deg')
    
    stars_tbl = Table()
    stars_tbl['skycoord']=position
    # stars_tbl=stars_tbl[:8]
    print('Number of selected stars:',len(stars_tbl))
   
    stars = extract_stars(nddata, stars_tbl, size=psfsize)
    
    print(stars[0].shape)
    
    # for i in range(len(stars)):    
    #     visualize(stars[i])
     
    nrows = int(np.ceil(np.sqrt(len(stars))))
    ncols = int(np.ceil(np.sqrt(len(stars))))
    plt.figure(figsize=(nrows,ncols))
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(25, 25))
    ax = ax.ravel()
    for i in range(len(stars)):
    
        norm = simple_norm(stars[i], 'log', percent=99.)
        ax[i].imshow(stars[i], norm=norm, origin='lower', cmap='viridis')

    #plt.show(block=False)
    #plt.clf()
    plt.savefig(f'../OUTPUT/plots/{gxyid}/{gxy_name1}_psfstars.jpeg',dpi=300)
    plt.clf()
    plt.close(fig)

   
    #sex_psf.remove_row(-4)
    
    
    # stars = extract_stars(nddata, stars_tbl, size=psfsize)
    
    ##########################################
    # BUILDING THE LOCAL PSF
    
    # epsf_model= EPSFModel(stars[0],oversampling=4)    
    # epsf1, fitted_stars1 = epsf_builder.build_epsf(stars) 
    # epsf1.data=epsf1.data.reshape((psfsize,psfsize))

    # visualize(data=epsf1.data,figpath=f'../OUTPUT/plots/{gxyid}/{gxy_name1}_epsf.jpeg')
    # print(epsf1.data.shape) 
    # print(epsf1.data.sum())
    
    #res=epsf.data
    
   
    ################################################### BAND 2
    sex=sex2
    plt_clean_sex=(sex['MAG_AUTO']<threshold) & (sex['MAG_APER']<threshold) & (sex['MAG_APER_1']<threshold) & (sex['MAG_APER_11']<threshold)
    #iniziamo dando i nomi alle colonne
    ra=sex['ALPHA_J2000']
    dec=sex['DELTA_J2000']
    # mag4=sex['MAG_APER']
    magau=sex['MAG_AUTO']
    # ci64=sex['MAG_APER_1']-sex['MAG_APER']#concentration index 6-4
    # cs=sex['CLASS_STAR']
    
    plt.scatter(magau[plt_clean_sex],sex['MAG_APER'][plt_clean_sex]-sex['MAG_APER_2'][plt_clean_sex],color='red',s=0.5)
    plt.scatter(sex['MAG_AUTO'][psf_sel],sex['MAG_APER'][psf_sel]-sex['MAG_APER_2'][psf_sel],s=1)
    # plt.axvline(mcutbright,color='yellow')
    # plt.axvline(mcutfaint,color='yellow')
    
    #plt.xlim([20,27])
    #plt.ylim([-1.15,0.15])
    plt.ylabel('Concentration index 4-8 pix [mag]')
    plt.xlabel('mag_auto [mag]')
    plt.savefig(f'../OUTPUT/plots/{gxyid}/{gxy_name2}_psfsel_first.jpeg',dpi=300)
    plt.clf()
    plt.close()
    
    sex_brightcpt=sex[psf_sel]
    s=sex.to_pandas()
    others=s.drop(s[psf_sel].index)
    
    # print("others")
    # print(others)
    
    coord=SkyCoord(others['ALPHA_J2000'],others['DELTA_J2000'],unit=(u.deg, u.deg))
    coord_brightcpt =SkyCoord(sex_brightcpt['ALPHA_J2000'],sex_brightcpt['DELTA_J2000'],unit=(u.deg, u.deg))
    
    
    idxcb, idxcoord, d2d, d3d = coord.search_around_sky(coord_brightcpt, rad_asec*u.arcsec)
    print("len(d2d),d2d,idxcb")
    print(len(d2d),d2d,idxcb)
    ind=pd.DataFrame(idxcb)
    ind=ind.drop_duplicates()
    print("len(ind)")
    print(len(ind))
    
    sex_brightcpt.remove_rows(ind[0])
    coord2=SkyCoord(sex_brightcpt['ALPHA_J2000'],sex_brightcpt['DELTA_J2000'],unit=(u.deg, u.deg))
    idx_brightcpt, d2d_brightcpt, d3d_brightcpt = coord2.match_to_catalog_sky(coord2,nthneighbor=2)
    
    
    sex_psf=sex_brightcpt[(d2d_brightcpt.arcsec>rad_asec*nthfactor)] #further selection on isolated
    savepath=f'../OUTPUT/{gxyid}/{gxy_name2}_psf.cat'
    sex_psf.write(savepath,format='ascii',overwrite=True)
    sex_psf=sex_psf.to_pandas()
    sex_psf=sex_psf.sort_values('MAG_CORR')

    print("Before and after nthneighbor=2 selection")
    print(len(sex_brightcpt),len(sex_psf))
    ra_psf=sex_psf['ALPHA_J2000']
    dec_psf=sex_psf['DELTA_J2000']
    plt.figure(figsize=(10,7))
    
    plt.scatter(ra,dec, s=0.1,color='red',label='All catalog')
    plt.scatter(ra_psf,dec_psf,s=2,marker='s',label='Selected isolated sources')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(f'../OUTPUT/plots/{gxyid}/{gxy_name2}_psfsel_isolated.jpeg',dpi=300)
    plt.clf()
    plt.close()

    #plt.show(block=False)
    
    #*******************************************
    nddata = NDData(data=data2,wcs=wcs2)  
    position=SkyCoord(ra_psf,dec_psf,unit='deg')
    
    stars_tbl = Table()
    stars_tbl['skycoord']=position
    # stars_tbl=stars_tbl[:8]
    print('Number of selected stars:',len(stars_tbl))
    
    stars = extract_stars(nddata, stars_tbl, size=psfsize)
    
    # for i in range(len(stars)):    
    #     visualize(stars[i])
     
    nrows = int(np.ceil(np.sqrt(len(stars))))
    ncols = int(np.ceil(np.sqrt(len(stars))))
    plt.figure(figsize=(nrows,ncols))
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(25, 25))
    ax = ax.ravel()
    
    for i in range(len(stars)):
        norm = simple_norm(stars[i], 'log', percent=99.)
        ax[i].imshow(stars[i], norm=norm, origin='lower', cmap='viridis')
        fits.writeto(f'../OUTPUT/{gxyid}/{gxy_name2}_psfstar_{i}.fits',stars[i].data.astype(np.float32),overwrite=True)


    plt.savefig(f'../OUTPUT/plots/{gxyid}/{gxy_name2}_psfstars.jpeg',dpi=300)
    plt.clf()
    plt.close(fig)
    
    
    print(len(stars))
    
    ##########################################
    # BUILDING THE LOCAL PSF
    
    
    # epsf_builder = EPSFBuilder(oversampling=1, maxiters=50,progress_bar=False,norm_radius=30)  
    epsf2, fitted_stars2 = epsf_builder.build_epsf(stars) 
    
    visualize(data=epsf2.data,figpath=f'../OUTPUT/plots/{gxyid}/{gxy_name2}_epsf.jpeg')
    print(epsf2.data.shape) 
    print(epsf2.data.sum())
    
        

    # print('NEED TO COMPARE PHOTOMETRY OF PSF WITH DAOPHOT')
    ###########################################
    return epsf2


def test_psf(band, res_ext, epsf):

    ############# TESTING PSF
       
    print('TESTING PSF PHOT')

    gxyid=gal.gxy        
    # print(gxyid)
    # print(band)
    psfcatpath=f"../OUTPUT/{gxyid}/{gxyid}_{band}_psf.cat"
    # print(psfcatpath)
    respath="../OUTPUT/"+gxyid+'/'+gxyid+'_'+band+'_res.fits'
    sex_psf=ascii.read(psfcatpath)
    hdul= astropy.io.fits.open(respath, ignore_missing_end = True)
    hdul.info()
    h=hdul[res_ext].header
    data=hdul[res_ext].data
    wcs=astropy.wcs.WCS(h)

    t1=time.time()
    fwhm=np.median(sex_psf['FWHM_IMAGE'])
    #print(fwhm )
    # sigma_psf=2.0
    bkgrms = MADStdBackgroundRMS()
    cutout=Cutout2D(data,(1800,1430),500,wcs=wcs) # hdul[res_ext] #Cutout2D(data,(1800,1430),500,wcs=wcs )
    data=cutout.data
    std = bkgrms(data )
    iraffind = IRAFStarFinder(threshold=4.0*std,
                          fwhm=fwhm)
                          # minsep_fwhm=0.01, roundhi=5.0, roundlo=-5.0,
                          # sharplo=0.0, sharphi=2.0)
    daogroup = DAOGroup(2.0 * fwhm)
    mmm_bkg = MMMBackground()
    fitter = LevMarLSQFitter()
    fitshape=epsf.data.shape
    # psf_model = IntegratedGaussianPRF(sigma=fwhm1*gaussian_fwhm_to_sigma)
    psf_model=epsf  #prepare_psf_model(epsf)

    photometry = IterativelySubtractedPSFPhotometry(finder=iraffind,
                                                group_maker=daogroup,
                                                bkg_estimator=mmm_bkg,
                                                psf_model=psf_model,
                                                fitter=fitter,
                                                aperture_radius=2*fwhm,
                                                niters=5, fitshape=fitshape)
    # photometry = BasicPSFPhotometry(finder=iraffind,
    #                                             group_maker=daogroup,
    #                                             bkg_estimator=mmm_bkg,
    #                                             psf_model=psf_model,
    #                                             fitter=fitter,
    #                                             aperture_radius=2*fwhm ,
    #                                             fitshape=fitshape)
    
    result_tab  = photometry(image=data)
    
    
    residual_image = photometry.get_residual_image()
    print(fitter.fit_info['message'])
    # norm = simple_norm(residual_image, 'log', percent=99.)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    ax = ax.ravel()
    vmax=0.95*np.max(data)
    ax[0].imshow(cutout.data, norm=LogNorm(vmin=0.001, vmax=vmax), origin='lower', cmap='viridis')
    # norm = simple_norm(data3, 'log', percent=99.)    
    ax[1].imshow(residual_image, norm=LogNorm(vmin=0.001, vmax=vmax), origin='lower', cmap='viridis')
    
    #plt.show(block=False)
    #plt.clf()
    
    Residui=fits.PrimaryHDU(data=residual_image,header=cutout.wcs.to_header())
    res_hdul=fits.HDUList([Residui])
    res_hdul.info()
    res_hdul.close()
    name='../OUTPUT/'+gxyid+'/'+band+'_psftest_res.fits'

    res_hdul.writeto(name,overwrite=True)
    
    
    t2=time.time()
    tdel=(t2-t1)/60.
    print('Total time elapsed in min=',tdel)

    return result_tab

def get_stats(mdata):
    
    rows=mdata.shape[0]
    cols=mdata.shape[1]
    
    mean, median, std = sigma_clipped_stats(mdata[0:100,0:100], sigma=3.0)
    #print('LL corner Mean/Median/STD 3sigma',(mean, median, std))
    if (abs(median)>abs(std)) :
        print("WARNING : THE MEDIAN BACKGROUND IS LARGER THAN THE SIGMA")
    
    mean, median, std = sigma_clipped_stats(mdata[0:100,cols-100:cols], sigma=3.0)
    #print('LR corner Mean/Median/STD 3sigma',(mean, median, std))  
    if (abs(median)>abs(std)) :
        print("WARNING : THE MEDIAN BACKGROUND IS LARGER THAN THE SIGMA")
    
    mean, median, std = sigma_clipped_stats(mdata[rows-100:rows,0:100], sigma=3.0)
    #print('UL corner Mean/Median/STD 3sigma',(mean, median, std))
    if (abs(median)>abs(std)) :
        print("WARNING : THE MEDIAN BACKGROUND IS LARGER THAN THE SIGMA")
     
    mean, median, std = sigma_clipped_stats(mdata[rows-100:rows,cols-100:cols], sigma=3.0)
    #print('UR corner Mean/Median/STD 3sigma',(mean, median, std))  
    if (abs(median)>abs(std)) :
        print("WARNING : THE MEDIAN BACKGROUND IS LARGER THAN THE SIGMA")

    mean, median, std = sigma_clipped_stats(mdata, sigma=3.0)
    print("MEAN/MEDIAN/STD 3sigma of entire frame:", (mean, median, std))

def corner_bkg(mdata, boxsize=100):
    
    rows=mdata.shape[0]
    cols=mdata.shape[1]
    
    
    meanLL, medianLL, stdLL = sigma_clipped_stats(mdata[0:boxsize,0:boxsize], sigma=3.0)
    flagLL=0
    print('LL corner Mean/Median/STD 3sigma',(meanLL, medianLL, stdLL),'\n\n')
    if (abs(medianLL)>abs(stdLL)) :
        print("WARNING : THE MEDIAN BACKGROUND IS LARGER THAN THE SIGMA",'\n\n')
        flagLL=1
        
    
    meanLR, medianLR, stdLR = sigma_clipped_stats(mdata[0:boxsize,cols-boxsize:cols], sigma=3.0)
    flagLR=0
    print('LR corner Mean/Median/STD 3sigma',(meanLR, medianLR, stdLR),'\n\n')  
    if (abs(medianLR)>abs(stdLR)) :
        print("WARNING : THE MEDIAN BACKGROUND IS LARGER THAN THE SIGMA",'\n\n')
        flagLR=1
    
    meanUL, medianUL, stdUL = sigma_clipped_stats(mdata[rows-boxsize:rows,0:boxsize], sigma=3.0)
    flagUL=0
    print('UL corner Mean/Median/STD 3sigma',(meanUL, medianUL, stdUL),'\n\n')
    if (abs(medianUL)>abs(stdUL)) :
        print("WARNING : THE MEDIAN BACKGROUND IS LARGER THAN THE SIGMA",'\n\n')
        flagUL=1
     
    meanUL, medianUR, stdUR = sigma_clipped_stats(mdata[rows-boxsize:rows,cols-boxsize:cols], sigma=3.0)
    flagUR=0
    print('UR corner Mean/Median/STD 3sigma',(meanUL, medianUR, stdUR),'\n\n')  
    if (abs(medianUR)>abs(stdUR)) :
        print("WARNING : THE MEDIAN BACKGROUND IS LARGER THAN THE SIGMA")
        flagUR=1
    
    return np.array([medianLL,medianLR,medianUL,medianUR]), np.array([flagLL,flagLR,flagUL,flagUR])

def corner_bkg_iter(mdata, boxsize=100):
    import random
    
    rows=mdata.shape[0]
    cols=mdata.shape[1]
    median=[[],[],[],[],[]]
    for i in range(0,100):
        box_pos=random.randint(0,100)
        
        meanLL, medianLL, stdLL = sigma_clipped_stats(mdata[0:boxsize,0:boxsize], sigma=3.0)
        median[0].append(medianLL)
        flagLL=0
        
            
        
        meanLR, medianLR, stdLR = sigma_clipped_stats(mdata[0:boxsize,cols-boxsize:cols], sigma=3.0)
        median[1].append(medianLR)
        flagLR=0
          
        
        
        meanUL, medianUL, stdUL = sigma_clipped_stats(mdata[rows-boxsize:rows,0:boxsize], sigma=3.0)
        median[2].append(medianUL)
        flagUL=0
        
         
        meanUL, medianUR, stdUR = sigma_clipped_stats(mdata[rows-boxsize:rows,cols-boxsize:cols], sigma=3.0)
        median[3].append(medianUR)
        flagUR=0
        
    print(np.array([np.median(median[0]),np.median(median[1]),np.median(median[2]),np.median(median[3])]))
    return np.array([np.median(median[0]),np.median(median[1]),np.median(median[2]),np.median(median[3])]), np.array([flagLL,flagLR,flagUL,flagUR])

def corner_median(mdata, frac):
    
    rows=mdata.shape[0]
    sizerows=int(frac*rows)
    cols=mdata.shape[1]
    sizecols=int(frac*cols)
    print(sizerows, sizecols)
    
    meanLL, medianLL, stdLL = sigma_clipped_stats(mdata[0:sizerows,0:sizecols], sigma=5.0)
    print('LL corner Mean/Median/STD 3sigma',(meanLL, medianLL, stdLL))
    # if (abs(medianLL)>abs(stdLL)) :
    #     print("WARNING : THE MEDIAN BACKGROUND IS LARGER THAN THE SIGMA")
    
    meanLR, medianLR, stdLR = sigma_clipped_stats(mdata[0:sizerows,cols-sizecols:cols], sigma=5.0)
    print('LR corner Mean/Median/STD 3sigma',(meanLR, medianLR, stdLR))  
    # if (abs(medianLR)>abs(stdLR)) :
    #     print("WARNING : THE MEDIAN BACKGROUND IS LARGER THAN THE SIGMA")
    
    meanUL, medianUL, stdUL = sigma_clipped_stats(mdata[rows-sizerows:rows,0:sizecols], sigma=5.0)
    print('UL corner Mean/Median/STD 3sigma',(meanUL, medianUL, stdUL))
    # if (abs(medianUL)>abs(stdUL)) :
    #     print("WARNING : THE MEDIAN BACKGROUND IS LARGER THAN THE SIGMA")
     
    meanUL, medianUR, stdUR = sigma_clipped_stats(mdata[rows-sizerows:rows,cols-sizecols:cols], sigma=5.0)
    print('UR corner Mean/Median/STD 3sigma',(meanUL, medianUR, stdUR))  
    # if (abs(medianUR)>abs(stdUR)) :
    #     print("WARNING : THE MEDIAN BACKGROUND IS LARGER THAN THE SIGMA")
    
    return np.array([medianLL,medianLR,medianUL,medianUR])

def run_sewpy(workdir,fits_imgpath,weight_imgpath,conf_filepath,params_filepath,fwhm0,magzp,plate_scale) :
        
    sewpyconf={"PARAMETERS_NAME":params_filepath,
                          "FILTER_NAME":"../INPUT/UTILS/gauss_3.0_5x5.conv",
                          "STARNNW_NAME":"../INPUT/UTILS/default.nnw",
                          "SEEING_FWHM": fwhm0,
                          "MAG_ZEROPOINT": magzp,
                          "PIXEL_SCALE" : gal.plate_scale,
                          "WEIGHT_IMAGE":weight_imgpath}
    
    ################# Sextractor run 
    
    sew=sewpy.SEW(workdir=workdir, sexpath="sex", configfilepath=conf_filepath,
                  config=sewpyconf,loglevel=None)
    out=sew(fits_imgpath)
    print("The catalog is :"+out["catfilepath"])

# import time
def azimuthal_avg(arr_2D,binsize=1) :
    
    # start=time.time()
    X, Y=np.meshgrid(np.arange(arr_2D.shape[1]),np.arange(arr_2D.shape[0]))

    x0=arr_2D.shape[1]/2
    y0=arr_2D.shape[0]/2
    binsize=binsize
    
    R=np.sqrt((X-x0)**2+(Y-y0)**2)
    # print("AZIMUTHAL AVERAGE R TESTING")
    # print(np.max(R))

    rsamp=np.arange(0,R.max(),binsize)
    rmed=[np.mean([rsamp[n], rsamp[n+1]]) for n in np.arange(len(rsamp)-1)]
    
    
    
    flux=[]
    
    for n in np.arange(len(rsamp)-1):
        l= (R >=rsamp[n]) & (R <rsamp[n+1])
        # flux.append([np.median(arr_2D[l]), np.std(arr_2D[l]), mad(arr_2D[l])])
        flux.append([np.mean(arr_2D[l]), np.std(arr_2D[l]), mad(arr_2D[l])])
        # print(rsamp[n],np.median(arr_2D[l]) )
    
    # print(rmed,flux)
    # end=time.time()
    # print(f"Total time in azimuthal avg module= {end-start} sec")
    return rmed,np.array(flux)

def azimuthal_sum(arr_2D,binsize=1) :
    
    # start=time.time()
    X, Y=np.meshgrid(np.arange(arr_2D.shape[1]),np.arange(arr_2D.shape[0]))

    x0=arr_2D.shape[1]/2
    y0=arr_2D.shape[0]/2
    binsize=binsize
    
    R=np.sqrt((X-x0)**2+(Y-y0)**2)
    # print("AZIMUTHAL AVERAGE R TESTING")
    # print(np.max(R))

    rsamp=np.arange(0,R.max(),binsize)
    rmed=[np.mean([rsamp[n], rsamp[n+1]]) for n in np.arange(len(rsamp)-1)]
    
    
    
    flux=[]
    
    for n in np.arange(len(rsamp)-1):
        l= (R >=rsamp[n]) & (R <rsamp[n+1])
        flux.append(np.sum(arr_2D[l]))
        # print(rsamp[n],np.median(arr_2D[l]) )
    
    # print(rmed,flux)
    # end=time.time()
    # print(f"Total time= {end-start} sec")
    return rmed,np.array(flux)



def radial_profile(X,Y,x0,y0,binsize,r_x,r_y) : # r_x and r_y and the half-width of the frame in pixels, in x and y directions
  
    R=np.sqrt((X-x0)**2+(Y-y0)**2)
    rsamp=np.arange(0,R.max(),binsize)
    #print(rsamp)

    rmed=[np.mean([rsamp[n], rsamp[n+1]]) for n in np.arange(len(rsamp)-1)]
    #print(rmed)
    rad_prof=[]
    
    for n in np.arange(len(rsamp)-1):
        l= (R >=rsamp[n]) & (R <rsamp[n+1])
        area_missing_x=0.
        area_missing_y=0.
        if (rsamp[n]>r_x) : 
            theta_in=2.*np.arccos(r_x/rsamp[n])
            theta_out=2.*np.arccos(r_x/rsamp[n+1])
            area_missing_x=(rsamp[n+1]**2*(theta_out-np.sin(theta_out))-rsamp[n]**2*(theta_in-np.sin(theta_in)))
        if (rsamp[n]>r_y) : 
            theta_in=2.*np.arccos(r_y/rsamp[n])
            theta_out=2.*np.arccos(r_y/rsamp[n+1])
            area_missing_y=(rsamp[n+1]**2*(theta_out-np.sin(theta_out))-rsamp[n]**2*(theta_in-np.sin(theta_in)))
        
        area= (rsamp[n+1]**2-rsamp[n]**2)*np.pi#-(area_missing_x+area_missing_y)
        #print(area)
        rad_prof.append(len(R[l])/area)
    # fig, ax = plt.subplots()
    # for i in range(len(rsamp)):
    #     circles = plt.Circle((0, 0), rsamp[i],fill=False)
    #     ax.add_patch(circles)
    # ax.set_xlim(-1*r_x,r_x)
    # ax.set_ylim(-1*r_y,r_y)
    # plt.scatter(X-x0,Y-y0)
    # #plt.show(block=False)
    # #plt.clf()
    return np.array(rmed),np.array(rad_prof)


def radial_profile_loglin(X,Y,x0,y0,binsize,r_x,r_y) : # r_x and r_y and the half-width of the frame in pixels, in x and y directions
    

    print("############ TESTING LOG MACRO")
    R=np.log10(np.sqrt((X-x0)**2+(Y-y0)**2))
    rsamp=np.linspace(0.15,3.1,num=binsize)
    
    R=np.sqrt((X-x0)**2+(Y-y0)**2)
    rsamp=10**(rsamp)
    print(X.max(),rsamp)


    rmed=[np.mean([rsamp[n], rsamp[n+1]]) for n in np.arange(len(rsamp)-1)]
    print(rmed)
    rad_prof=[]
    
    for n in np.arange(len(rsamp)-1):
        l= (R >=rsamp[n]) & (R <rsamp[n+1])
        area_missing_x=0.
        area_missing_y=0.
        if (rsamp[n]>r_x) : 
            theta_in=2.*np.arccos(r_x/rsamp[n])
            theta_out=2.*np.arccos(r_x/rsamp[n+1])
            area_missing_x=(rsamp[n+1]**2*(theta_out-np.sin(theta_out))-rsamp[n]**2*(theta_in-np.sin(theta_in)))
        if (rsamp[n]>r_y) : 
            theta_in=2.*np.arccos(r_y/rsamp[n])
            theta_out=2.*np.arccos(r_y/rsamp[n+1])
            area_missing_y=(rsamp[n+1]**2*(theta_out-np.sin(theta_out))-rsamp[n]**2*(theta_in-np.sin(theta_in)))
        
        area= (rsamp[n+1]**2-rsamp[n]**2)*np.pi#-(area_missing_x+area_missing_y)
        #print(area)
        rad_prof.append(len(R[l])/area)
    # fig, ax = plt.subplots()
    # for i in range(len(rsamp)):
    #     circles = plt.Circle((0, 0), rsamp[i],fill=False)
    #     ax.add_patch(circles)
    # ax.set_xlim(-1*r_x,r_x)
    # ax.set_ylim(-1*r_y,r_y)
    # plt.scatter(X-x0,Y-y0)
    # #plt.show(block=False)
    # #plt.clf()
    return np.array(rmed),np.array(rad_prof)

def radial_snr(sig_2D,noise_2D,binsize=1) :
    
    start=time.time()
    dim0=np.min(sig_2D.shape[0],noise_2D.shape[0])
    dim1=np.min(sig_2D.shape[1],noise_2D.shape[1])
    X, Y=np.meshgrid(np.arange(dim1),np.arange(dim0))

    sx0=sig_2D.shape[1]/2
    sy0=sig_2D.shape[0]/2
    binsize=binsize
    
    R=np.sqrt((X-x0)**2+(Y-y0)**2)
    # print("AZIMUTHAL AVERAGE R TESTING")
    # print(np.max(R))

    rsamp=np.arange(0,R.max(),binsize)
    rmed=[np.mean([rsamp[n], rsamp[n+1]]) for n in np.arange(len(rsamp)-1)]
    
    
    
    flux=[]
    
    for n in np.arange(len(rsamp)-1):
        l= (R >=rsamp[n]) & (R <rsamp[n+1])
        # flux.append([np.median(arr_2D[l]), np.std(arr_2D[l]), mad(arr_2D[l])])
        flux.append([np.mean(arr_2D[l]), np.std(arr_2D[l]), mad(arr_2D[l])])
        # print(rsamp[n],np.median(arr_2D[l]) )
    
    # print(rmed,flux)
    end=time.time()
    print(f"Total time in azimuthal avg module= {end-start} sec")
    return rmed,np.array(flux)



# from sklearn.linear_model import LinearRegression as linreg
from sklearn.linear_model import HuberRegressor as huber
def sbf_ps_fit(k,E_k,P_k):
    
    '''
    P_0=0.00001
    P_1=0
    # Adopt a curve-of-decay analysis on P_k to determine P_1
    diff=P_k[:-1]-P_k[1:]
    cond=(diff>0)
    x=k[:-1][cond]
    y=np.log(diff[cond])
    sigmalin = lambda x,a,b: np.sum((func_linear(x,a,b)))
    popt_asymlin,pcov_asymlin = curve_fit(func_linear,x,y, p0=[0.,0.])
    
    taillen=int(0.1*len(y))
    tailval=np.median(y[-(taillen):])
    P_1=tailval-sigmalin(x*2,*popt_asymlin)
    print("FITTING PROCEDURE TESTING HERE")
    print(taillen,tailval)
    print(sigmalin(x*2,*popt_asymlin))
    # P_0*E_k+P_1
    '''

    
    '''
    ## CURVE FITTING

    popt_ps, pcov_ps=curve_fit(sbf_ps, E_k, P_k, p0=[30,70], bounds=(0,np.inf))
    print(popt_ps)
    fig = plt.figure(figsize=(7, 7))
    font = {'family' : 'serif',
            'weight' : 'normal',
            'size'   : 22}
    
    plt.rc('font', **font)
    fig.subplots_adjust(wspace=0.05,hspace=0.65, left=0.15, right=0.95,
                        bottom=0.15, top=0.9)

    # plt.plot(x,y,color='darkred',label='P_k ')
    # plt.axhline(sigmalin(x*2,*popt_asymlin))
    plt.plot(k,np.log10(P_k),color='darkred')#,label='P_k '+gxy_name2)
    plt.plot(k,np.log10(sbf_ps(E_k,*popt_ps)),color='indigo')#,label='P_k '+gxy_name2)
    # plt.plot(k,np.log10(utils.sbf_ps(E_k,P_0,P_1)),color='indigo',label='P_k '+gxy_name2)
    # plt.xlabel('$k$')
    # plt.ylabel('$log(P)$')
    plt.title(f"$i$: $P(k)=P0*E(k)+P1$",fontsize='small')
    # plt.legend()
    
    # plt.yscale('log')
    #plt.ylim(-5,1)
    # plt.grid(True)
    
    
    plt.savefig(f'../OUTPUT/plots/IC0745/i_powspecmatch_testfit.jpeg',dpi=300)
    #plt.show(block=False)
    plt.clf()

    return popt_ps#P_0, P_1
    '''

    

    '''
    ## LINEAR REGRESSION FIT

    X=E_k.reshape((-1,1))
    y=P_k
    reg=huber().fit(X,y)
    P_0=reg.coef_[0]
    P_1=reg.intercept_

    
    fig = plt.figure(figsize=(7, 7))
    font = {'family' : 'serif',
            'weight' : 'normal',
            'size'   : 22}
    
    plt.rc('font', **font)
    fig.subplots_adjust(wspace=0.05,hspace=0.65, left=0.15, right=0.95,
                        bottom=0.15, top=0.9)

    # plt.plot(x,y,color='darkred',label='P_k ')
    # plt.axhline(sigmalin(x*2,*popt_asymlin))
    # plt.plot(k,np.log10(P_k),color='darkred')#,label='P_k '+gxy_name2)
    # plt.plot(k,np.log10(sbf_ps(E_k,P_0,P_1)),color='indigo')#,label='P_k '+gxy_name2)
    # plt.plot(k,np.log10(reg.predict(X)))#,label='P_k '+gxy_name2)
    plt.plot(k,(P_k),color='darkred')#,label='P_k '+gxy_name2)
    plt.plot(k,(sbf_ps(E_k,P_0,P_1)),color='indigo')#,label='P_k '+gxy_name2)
    plt.plot(k,(reg.predict(X)))#,label='P_k '+gxy_name2)
    # plt.plot(k,np.log10(utils.sbf_ps(E_k,P_0,P_1)),color='indigo',label='P_k '+gxy_name2)
    # plt.xlabel('$k$')
    # plt.ylabel('$log(P)$')
    plt.yscale('log')
    plt.title(f"$i$: $P(k)=P0*E(k)+P1$",fontsize='small')
    # plt.legend()
    
    # plt.yscale('log')
    #plt.ylim(-5,1)
    # plt.grid(True)
    
    
    plt.savefig(f'../OUTPUT/plots/IC0745/i_powspecmatch_testfit.jpeg',dpi=300)
    #plt.show(block=False)
    plt.clf()

    return P_0, P_1
    '''

    ## ITERATIVE CURVE FITTING
    P_0=[]
    P_1=[]
    for i in range(int(len(E_k)/2)):
        
        # print(i)
        popt_ps, pcov_ps=curve_fit(sbf_ps, E_k[i:], P_k[i:], p0=[30,70], bounds=(0,np.inf))
        # print(popt_ps)
        P_0.append(popt_ps[0])
        P_1.append(popt_ps[1])
    
    
    
    # fig = plt.figure(figsize=(7, 7))
    # font = {'family' : 'serif',
    #         'weight' : 'normal',
    #         'size'   : 22}
    
    # plt.rc('font', **font)
    # fig.subplots_adjust(wspace=0.05,hspace=0.65, left=0.15, right=0.95,
    #                     bottom=0.15, top=0.9)

    # plt.plot(k,np.log10(P_k),color='darkred')#,label='P_k '+gxy_name2)
    # plt.plot(k,np.log10(sbf_ps(E_k,*popt_ps)),color='indigo')#,label='P_k '+gxy_name2)
    # plt.title(f"$i$: $P(k)=P0*E(k)+P1$",fontsize='small')
    
    # plt.savefig(f'../OUTPUT/plots/IC0745/i_powspecmatch_testfit.jpeg',dpi=300)
    # plt.clf()

    return np.array(P_0), np.array(P_1)


def sbf_ps(E_k,P_0,P_1):
    return P_0*E_k+P_1


def func_linear(x,a,b):
    return a*x+b

def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

def two_gaussian(x, a1, x01, sigma1, a2, x02, sigma2):
    return a1 * np.exp(-(x - x01)**2 / (2 * sigma1**2)) + a2 * np.exp(-(x - x02)**2 / (2 * sigma2**2))

def gaussian_2D(x, y, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))
    return g.ravel()    
def pritchet(m,m_lim,alpha) :
    return 0.5 * (1.- (alpha*(m-m_lim))/np.sqrt(1. + (alpha*(m-m_lim))**2))

def fermi(m,m_0,C,a,b):
    return (1+C*np.exp(b*(m-m_0)))/(1+np.exp(a*(m-m_0)))

def powerlaw(x,alpha,c):
    return x**alpha+c

def linfit_sbf(psf_k,p0,p1):
    return psf_k*p0+p1


def read_jwst_spot(bands):
    
    filepath='../INPUT/UTILS/JW29092022.color'
    AB_Vega={'F090W':0.5041199922561646, 'F150W':1.2431399822235107, 'F200W':1.705970048904419, 'F277W':2.315419912338257, 'F356W':2.823899984359741, 'F444W':3.2418100833892822 }
    
    tcol=Table.read(filepath, format='ascii.commented_header')
    
    for i in range(len(bands)-1):
        for j in range(1,len(bands)):
            tcol[f'{bands[i]}-{bands[j]}']=tcol[f'V{bands[j].upper()}']-tcol[f'V{bands[i].upper()}']+(AB_Vega[f'{bands[i].upper()}']-AB_Vega[f'{bands[j].upper()}'])

    # tcol=tcol[tcol['age']>=9.]
    return tcol    