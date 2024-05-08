#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 10:28:43 2021

@author: mik and nandinihazra
"""


#IMPORT STUFF



#import matplotlib.patches as patches
#from astropy.modeling import models, fitting
#from astropy.modeling.models import Sersic1D
#import astropy as ap
#import math

import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from astropy.io import fits
from matplotlib.colors import LogNorm
import matplotlib.colors as colors

from photutils.isophote import EllipseGeometry
#from photutils.isophote import EllipseSample
from photutils.aperture import EllipticalAperture
from photutils.isophote import Ellipse


from photutils.isophote import build_ellipse_model

from astropy.table import Table
import warnings
import regions
import sewpy
import time
from astropy.stats import median_absolute_deviation as mad
from shutil import copyfile
from shutil import rmtree
from photutils.psf import EPSFBuilder, IterativelySubtractedPSFPhotometry
from astropy.wcs.utils import skycoord_to_pixel
from astropy.nddata.utils import Cutout2D
from scipy.optimize import curve_fit 
import matplotlib.transforms as transforms

from regions import EllipseSkyRegion, CircleSkyRegion, CirclePixelRegion
from astropy.coordinates import SkyCoord
from astropy import units as u
# from astropy.convolution import convolve
from scipy.signal import convolve
from scipy.signal import find_peaks
# from statistics import mode
from scipy.stats import mode
import pyds9

#IMPORT GALAXY OBJECT
import init as gal

#IMPORT UTILS CLASS
import utils

'''
Masking and galaxy modeling
'''

# Modeling of the galaxy

def run_part1(rms=0, remove_bkg=False) :
    
    #Start the time 
    start_time = time.time()
        
    # Parameters initialization
    
    gxyid=gal.gxy
    gxy_name=gal.gxy_name
    gxyra=gal.gxyra
    gxydec=gal.gxydec
    fwhm0=gal.fwhm0
    
    #Galaxy model parameters
    minsma0=gal.minsma0
    maxsma0=gal.maxsma0
    sma0=gal.sma0        #First annulus for ellipse fitting
    eps0=gal.eps0        #Ellipticity
    pa0=gal.pa0         #Position angle
    agxy=gal.agxy        #Step for increasing SMA
    maxrad=gal.maxrad    #Max radius for the fitting
    minrad=gal.minrad    #Min radius for the fitting
    maskin=gal.maskin    #Inner radius of the mask selection (arcsec)
    maskout=gal.maskout  #Outer radius 
    nclip=gal.nclip
    sclip=gal.sclip
    maxit=gal.maxit
    print('Initial modeling parameters \n')
    print('Ellipticity',eps0, '\n' )
    print('Semi-Major Axis',sma0, 'Increasing of' ,agxy ,'\n' )
    print('Min and max radius of fitting',minrad, maxrad, '\n' )
    print('Position Angle',pa0, '\n \n' )
    
    #Magnitude cutoffs for sextractor catalog : all objects brighter than this will be masked
    
    magcut=gal.magcut           #The magnitude cutoff for the extended objects mask
    magcut_cpt=gal.magcut_cpt   #The magnitude cutoff for the compact objects mask
    
    #Fixing the center. pa and eps
    
    fix_center_fit=gal.fix_center
    fix_pa_fit=gal.fix_pa
    fix_eps_fit=gal.fix_eps
    print('Fixing the parameters? \n')
    print('Pos. Angle:',fix_pa_fit, 'Center:', fix_center_fit, 'Ellipticity:', fix_eps_fit, '\n\n'  )
    
    
    magzp=gal.magzp     #i-band zeropoint magnitude
    box=gal.box         #Box size in pixels within which to fit galaxy
    print('The box is:', box, '\n\n')
    
    img_ext=gal.img_ext #Image extension
    wht_ext=gal.wht_ext #Weight extension
            
    # Galaxy file paths
    
    gxy=gal.gxyfilepath
    wghtimg=gal.wghtfilepath
    
    #masking scale factors for compact and extended sources
    
    mask_scale_ext=gal.mask_scale_ext
    mask_scale_cpt=gal.mask_scale_cpt
    

    #Create output directories
    
    if not os.path.exists("../OUTPUT/"+gxyid):
        os.makedirs("../OUTPUT/"+gxyid)
    
    if not os.path.exists("../OUTPUT/plots/"+gxyid):
        os.makedirs("../OUTPUT/plots/"+gxyid)

    
    #Loading of the image
    
    hdu = fits.open(gxy, do_not_scale_image_data=True)
    hdu.info()
    
    image_data = hdu[img_ext].data
    galimg=f"../OUTPUT/{gxyid}/{gxy_name}.fits"
    
    
    #Load the weight map
    
    hduw = fits.open(wghtimg, do_not_scale_image_data=True)
    hduw.info()
    weight_data = hduw[wht_ext].data
    
    #Find coordinates of the centre of the image
    
    wgxy=WCS(hdu[img_ext].header)
    xgxy, ygxy = wgxy.wcs_world2pix(gxyra, gxydec, 1) # Get the galaxy center in pixels
    x0=int(xgxy)
    y0=int(ygxy)
    print(gxyra, gxydec)
    print('x0,y0',x0,y0)
    
    
    #Read instrument mask if needed (read_hsc_mask=1)
    
    flag_instr_mask=1
    
    if (gal.read_hsc_mask):
        
        print('Instrument Mask is considered','\n\n' )
        
        instr_mask=utils.get_hsc_mask(hdu,img_ext)
        gal_core=instr_mask[x0-box:x0+box,y0-box:y0+box]
        maskedpix=np.count_nonzero(gal_core==1)
        total_size=gal_core.size
        
        if (maskedpix >= 0.5*total_size) :
            print ("WARNING : Instrument mask covers 50% or more of the central part of the galaxy, hence we will be ignoring it")
            instr_mask=0
            flag_instr_mask=0

        sex_mask=1-instr_mask
    else:
        sex_mask=1
    
    print('Sex_mask is',sex_mask,'\n\n' )
    
    #Convert weight data appropriately according to instrument and mask with instrument mask
    
    #FOR NGVS DATA
    
    if(rms):
        
        print('Weight data converted with the square root','\n\n' )
        weight_data=sex_mask/np.square(weight_data)
    
    #FOR HSC DATA
    
    else:
        print('Weight data converted linearly','\n\n' )
        weight_data=sex_mask/weight_data
    
    fits.writeto(f"../OUTPUT/{gxyid}/weights.fits", weight_data, overwrite=True)
    hduw.close()
    
    
    #Weight(instrument mask) application to the image
    masked_data=image_data*sex_mask 
    masked_data=np.ma.masked_equal(masked_data,0) 
    utils.get_stats(masked_data)
    
    # Generate a Mask with SExtractor  - We mask only the 15 brightest objects within  a given region

    # Get a Mask of extended sources using Sextractor
    # Run sextractor to get background subtracted image
    # On background subtracted image, run sextractor to generate list of compact objects
    
    # Generate sextractor file using sewpy 
    
    sewpyconf={"PARAMETERS_NAME":"../INPUT/UTILS/default.param",
                          "FILTER_NAME":"../INPUT/UTILS/gauss_3.0_5x5.conv",
                          "STARNNW_NAME":"../INPUT/UTILS/default.nnw",
                          "SEEING_FWHM": fwhm0,
                          "MAG_ZEROPOINT": magzp,
                          "PIXEL_SCALE" : gal.plate_scale,
                          "WEIGHT_IMAGE":f"../OUTPUT/{gxyid}/weights.fits"}
    
    # Sextractor run for extended objects mask
    
    sew=sewpy.SEW(workdir=f'../OUTPUT/{gxyid}', sexpath="sex", configfilepath='../INPUT/UTILS/sewpy_lsst_extended.par',
                  config=sewpyconf,loglevel=None)
    
    out=sew(galimg) #The run is done on image_data, that is the image-background
    
    #Load the Sextractor file generated by sewpy
    
    tsex = Table.read(out["catfilepath"], format='ascii.sextractor')
    tsex=tsex[tsex['MAG_AUTO']<35.0]
    rgc=np.sqrt((tsex['ALPHA_J2000']-gxyra)**2+(tsex['DELTA_J2000']-gxydec)**2)*3600 #Galaxtocentric distance in arcsecs
    mag=tsex['MAG_AUTO']
    
    #Choose objects within a certain radius of the galaxy centre which are brighter than a cutoff magnitude set by the user###Plot class star vs mag auto

    tmask=tsex[(rgc>=maskin) & (rgc <=maskout) & (mag<magcut)]
    
    print('Generating the mask of all extended objects within',maskin, '<=Rgc(arcsec)<=',maskout, ' and brighter than ',magcut)
    print('Number of extended objects detected=', len(tsex),', masked=',len(tmask),'\n\n' )
    
    
    
    #Generate the regions to mask
    
    theta=list(map(lambda x: abs(x) if x<0 else 180-x, tmask['THETA_WORLD']))
    center=SkyCoord(tmask['ALPHA_J2000'],tmask['DELTA_J2000'],unit=(u.deg, u.deg))
    width=mask_scale_ext*tmask['A_WORLD']   #*3600/plate_scale #(possible scale factor)
    height=mask_scale_ext*tmask['B_WORLD']  #*3600/plate_scale #(possible scale factor)
    
    #Masking
    
    for j in range(len(tmask)):
        # print(width[j])
        # region=CircleSkyRegion(center[j], width[j])
        region=EllipseSkyRegion(center[j], width[j]*u.deg, height[j]*u.deg, angle=theta[j]*u.deg)
        region=((region.to_pixel(wgxy)).to_mask(mode='center')).to_image(image_data.shape)
        if (j==0):
            mask_reg_ext=region   #.multiply(resdata[i])#mask_reg*(1-region.multiply(resdata[i]))
        else:
            mask_reg_ext=np.logical_or(mask_reg_ext,region).astype(np.int32)
        
         
    # Sextractor run for background
    
    sew_mback=sewpy.SEW(workdir=f'../OUTPUT/{gxyid}', sexpath="sex", configfilepath='../INPUT/UTILS/sewpy_lsst_back.par',
                        config=sewpyconf,loglevel=None)
    
    out_mback=sew_mback(galimg)
    mback_image=f'../OUTPUT/mback.fits'
    
    # Sextractor run for compact objects mask
    
    sew_compact=sewpy.SEW(workdir=f'../OUTPUT/{gxyid}', sexpath="sex", configfilepath='../INPUT/UTILS/sewpy_lsst_compact.par',
                          config=sewpyconf,loglevel=None)
    
    out_compact=sew_compact(mback_image)
    
    #Load the Sextractor file generated by sewpy
    
    tsex_cpt = Table.read(out_compact["catfilepath"], format='ascii.sextractor')
    tsex_cpt=tsex_cpt[tsex_cpt['MAG_AUTO']<35.0]
    rgc=np.sqrt((tsex_cpt['ALPHA_J2000']-gxyra)**2+(tsex_cpt['DELTA_J2000']-gxydec)**2)*3600 #Galactocentric distance in arcsecs
    mag=tsex_cpt['MAG_AUTO']
    class_star=tsex_cpt['CLASS_STAR']
    flags=tsex_cpt['FLAGS']
    
    # Selection of FWHM objects (stars)
    
    fwhm_list=tsex_cpt[(class_star>0.9) & (mag<=20) & (flags==0)]['FWHM_IMAGE']
    print("Number of objects chosen to calculate FWHM: ",len(fwhm_list))
    
    if (len(fwhm_list)<20): print("Too few objects to calculate FWHM")
    
    fwhm=gal.plate_scale*np.median(fwhm_list)
    sfwhm=gal.plate_scale*np.std(fwhm_list)
    print('Median FWHM=%5.2f +/-%5.2f' %(fwhm, sfwhm))
    
    if (fwhm-fwhm0)/fwhm0>=0.25: print('WARNING THE IMAGE FWHM IS >25% DIFFERENT THAN THE INPUT VALUE') 
    gal.fwhm0=fwhm
    
    
    #Choose objects within a certain radius of the galaxy centre which are brighter than a cutoff magnitude set by the user###Plot class star vs mag auto
    
    tmask_cpt=tsex_cpt[(rgc>=maskin) & (rgc <=maskout) & (mag<magcut_cpt) & (class_star>0.8)]  
    
    print('Generating the mask of all compact objects within ',maskin,' <=Rgc(arcsec)<=',maskout,' and brighter than ',magcut_cpt)
    print('Number of compact objects detected=', len(tsex_cpt),', masked=',len(tmask_cpt),'\n\n' )
    
    #Generate the regions to mask
    
    center=SkyCoord(tmask_cpt['ALPHA_J2000'],tmask_cpt['DELTA_J2000'],unit=(u.deg, u.deg))
    radius=mask_scale_cpt*tmask_cpt['A_WORLD']#*3600/plate_scale
    
    #Masking
    
    for j in range(len(tmask_cpt)):
        
        region=CircleSkyRegion(center[j], radius[j]*u.deg)
        region=((region.to_pixel(wgxy)).to_mask(mode='center')).to_image(image_data.shape)
        
        if (j==0):
            mask_reg_cpt=region  #.multiply(resdata[i])#mask_reg*(1-region.multiply(resdata[i]))
        else:
            mask_reg_cpt=np.logical_or(mask_reg_cpt,region).astype(np.int32)
            

    
    
    #Generate the mask, and if both masks are empty then set masked_data to image_data
    
    if len(tmask_cpt)==0 and len(tmask)==0 :
        mask_data=0
        print ("The mask generated using sextractor was empty")
        
    elif len(tmask_cpt)>0 and len(tmask)>0 :
        warnings.filterwarnings("ignore")
    
        #Combine of the compact and extended objects masks
        
        mask_data=np.logical_or(mask_reg_ext,mask_reg_cpt).astype(np.int32)

    elif len(tmask)==0 :
        mask_data=mask_reg_cpt.astype(np.int32)
        
    elif len(tmask_cpt)==0 :
        mask_data=mask_reg_ext.astype(np.int32)



    #Combine generated and instrumental masks
    
    if (gal.read_hsc_mask) :
        mask_data=np.logical_or(mask_data,instr_mask).astype(np.int32)
    maskedpix=np.count_nonzero(mask_data==1)
    total_size=mask_data.size
    maskedpc=(maskedpix/total_size)*100
    print('Total percentage of pixels masked = %3.4f' % (maskedpc),'\n\n' )
    
    #Application of the final mask
    
    masked_data=image_data*(1-mask_data)
    masked_data=np.ma.masked_equal(masked_data,0)
    #plt.imshow(masked_data,origin='lower')
    #plt.imshow(mask_data[int(xgxy-box):int(xgxy+box),int(ygxy-box):int(ygxy+box)],origin='lower')#,norm=LogNorm(vmin=0.001, vmax=vmax),cmap='hsv')
    
    tmid=time.time()
    print ("SExtractor total runtime in seconds : %6.3f" %(tmid-start_time),'\n\n' )
    
    #Get the four corners background
    boxsize_bkg=100      #int(0.1*np.mean(masked_data.shape))
    
    med_corners,flags_corners=utils.corner_bkg(masked_data,boxsize_bkg)
    
    #Background removal if requested
    if (remove_bkg):
        #print(np.multiply(med_corners, flags_corners))
        #image_data=image_data-np.median(np.multiply(med_corners, flags_corners))
        #image_data=image_data-np.median(med_corners)
        masked_data=masked_data-np.median(med_corners)#0.163
        print(f"BACKGROUND SUBTRACTED={np.median(med_corners)}")
        print("WARNING: BKG SETUP FOR VCC1146 ONLY",'\n\n' )
    
    # Save background subtracted image data
    
    fits.writeto(galimg, image_data, header=hdu[img_ext].header, overwrite=True)


    
    #Dry ellipse model to get the best initialization parameters
    
    #Geometry modeling
    tini_g=time.time()
    geo0 = EllipseGeometry(x0=x0, y0=y0, sma=sma0, eps=eps0, pa=pa0, astep=agxy/3,
                    linear_growth=False, fix_center=False, fix_pa=False, fix_eps=False)
    tend_g=time.time()
    print('Initial Geometry model', "Fit time in seconds : %6.3f" %(tend_g-tini_g),'\n\n' )

    #Ellipse modeling
    tini_e=time.time()
    ellipse = Ellipse(masked_data, geometry=geo0,threshold=0.1)
    tend_e=time.time()
    print('Initial Ellipse model',"Fit time in seconds : %6.3f" %(tend_e-tini_e),'\n\n' )
    
    #Isophote generation
    tini_i=time.time()
    isolist = ellipse.fit_image(minsma=minsma0, maxsma=maxsma0, maxit=maxit, maxgerr=0.5, fflag=0.7,
                                 maxrit=None, sclip=sclip, nclip=nclip)
    
    t0=isolist.to_table()
    tend_i=time.time()
    print('Initial Isophotes',"Fit time in seconds : %6.3f" %(tend_i-tini_i),'\n\n' )
    
    if len(t0)>0:
        smagxy=np.median(t0['sma'])
        pagxy=float(str(np.median(t0['pa']))[0:4])*np.pi/180.
        epsgxy=np.median(t0['ellipticity'])
        xgxy=np.median(t0['x0'])
        ygxy=np.median(t0['y0'])
        print ('INITIAL galaxy median parameters : SMA0 = %5.0f, PA0 = %5.0f deg, ellip = %5.2f' %(smagxy, pagxy*180/np.pi, epsgxy),'\n\n' )
    else:   
        smagxy=sma0
        pagxy=pa0
        epsgxy=eps0
        xgxy=x0
        ygxy=y0
        print('Did not run the zero-th ellipse model')
        
    #Draw an ellipse with initial guesses
    aper = EllipticalAperture((xgxy, ygxy), smagxy, smagxy*(1 - epsgxy), pagxy)
    
    print(aper)
    
    
    # Galaxy final Modeling, starting from the initial values calculated before
    
    #Geometry modeling
    
    tini_g=time.time()
    geogxy = EllipseGeometry(x0=xgxy, y0=ygxy, sma=smagxy, eps=epsgxy, pa=pagxy, astep=agxy,
                               linear_growth=False, fix_center=fix_center_fit, fix_pa=fix_pa_fit, fix_eps=fix_eps_fit)
    tend_g=time.time()
    
    print('Final Geometry model', "Fit time in seconds : %6.3f" %(tend_g-tini_g),'\n\n' )
    
    #Ellipse model
    tini_e=time.time()
    ellipse = Ellipse(masked_data, geometry=geogxy,threshold=0.1)
    tend_e=time.time()
    print('Final Ellipse model', "Fit time in seconds : %6.3f" %(tend_e-tini_e),'\n\n' )
    
    #Isophote generation
    tini_i=time.time()
    isolist = ellipse.fit_image(minsma=minrad, maxsma=maxrad, maxit=maxit, maxgerr=0.5, fflag=0.5,
                                 maxrit=None, sclip=sclip, nclip=nclip)
    tend_i=time.time()
    print('Final Isophotes',"Fit time in seconds : %6.3f" %(tend_i-tini_i),'\n\n' )
    
    t=isolist.to_table()
    t['tflux_e']=isolist.tflux_e
    t['tflux_c']=isolist.tflux_c
    #t.show_in_browser()
    
    tend=time.time()
    print ("Fit time in seconds : %6.3f" %(tend-tmid),'\n\n' )
    
    
    #GET THE GALAXY MODEL
    
    if (len(t)>0) :
        
        
        smagxy=np.median(t['sma'])
        pagxy=float(str(np.median(t['pa']))[0:4])*np.pi/180.
        epsgxy=np.median(t['ellipticity'])
        xgxy=np.median(t['x0'])
        ygxy=np.median(t['y0'])
        
        print ('FINAL galaxy median parameters : SMA0 = %5.0f, PA0 = %5.0f deg, ellip = %5.2f' %(smagxy, pagxy*180/np.pi, epsgxy))
        isofilepath="../OUTPUT/"+gxyid+'/'+gxy_name+'.iso'
        t.write(isofilepath,format='ascii',overwrite=True)

        
        print("Building the model")
        tini_b=time.time()
        image_model = build_ellipse_model(image_data.shape, isolist, fill=(t['intens'][-1]), high_harmonics=gal.high_harmonics)
        # image_model= image_model.astype(np.float32)
        tend_b=time.time()
        print('Time to build the model: %6.3f' %(tend_b-tini_b),'\n\n' )

        #print('WARNING: there is a sky oversubtraction because of model fitting, by: %6.3f' %(t['intens'][-1]))
        print("Plotting residuals")
        
        
        # Plots initialization
        import matplotlib.patches as patches
        from PIL import Image
        
        fig = plt.figure(figsize=(14, 16))
        fig.subplots_adjust(wspace=0.05, left=0.1, right=0.95,
                            bottom=0.15, top=0.9)
        cmap='viridis'
        vmax=0.95*np.max(image_data[int(xgxy-box):int(xgxy+box),int(ygxy-box):int(ygxy+box)])
        vmin=0.001
        if (vmin>vmax) : vmax=10.*vmin
        
        plt.subplot(221)
        #plt.xlim(xgxy-box, xgxy+box)
        #plt.ylim(ygxy-box, ygxy+box)
        plt.imshow(image_data, origin='lower', norm=LogNorm(vmin=vmin, vmax=vmax), cmap=cmap)
        #plt.add_patch(rect)
        
        plt.subplot(222)
        plt.xlim(xgxy-box, xgxy+box)
        plt.ylim(ygxy-box, ygxy+box)
        plt.imshow(masked_data, origin='lower', norm=LogNorm(vmin=vmin, vmax=vmax),cmap=cmap)
        aper.plot(color='red')
        
        
        plt.subplot(223)
        plt.xlim(xgxy-box, xgxy+box)
        plt.ylim(ygxy-box, ygxy+box)
        plt.imshow(image_model, origin='lower', norm=LogNorm(vmin=vmin, vmax=4*vmax),cmap=cmap)
        
        
        plt.subplot(224)
        plt.xlim(xgxy-box, xgxy+box)
        plt.ylim(ygxy-box, ygxy+box)
        plmedian=np.median((image_data-image_model)[int(xgxy-box/5):int(xgxy+box/5),int(ygxy-box/5):int(ygxy+box/5)])
        plstd=np.std((image_data-image_model)[int(xgxy-box/5):int(xgxy+box/5),int(ygxy-box/5):int(ygxy+box/5)])
        plt.imshow((image_data-image_model), origin='lower',norm=LogNorm(vmin=vmin, vmax=vmax),cmap=cmap )
        plt.savefig(f'../OUTPUT/plots/{gxyid}/{gxy_name}_modelling.jpeg',dpi=300)
        plt.show()
        plt.clf()
        plt.close(fig)

        
        # Save model, mask, residual, and instrument mask (if read) in separate files
        
        fits.writeto("../OUTPUT/"+gxyid+'/'+gxy_name+'_mod.fits', image_model.astype(np.float32), header=hdu[img_ext].header, overwrite=True)
        fits.writeto("../OUTPUT/"+gxyid+'/'+gxy_name+'_res.fits', (image_data-image_model+t['intens'][-1]).astype(np.float32), header=hdu[img_ext].header,  overwrite=True)
        fits.writeto("../OUTPUT/"+gxyid+'/'+gxy_name+'_mask.fits', mask_data, header=hdu[img_ext].header,  overwrite=True)
        if (gal.read_hsc_mask and flag_instr_mask):
                fits.writeto("../OUTPUT/"+gxyid+'/'+gxy_name+'_instrmask.fits', instr_mask, header=hdu[img_ext].header,  overwrite=True)
        # Write the masked weight file to the output directory
        fits.writeto("../OUTPUT/"+gxyid+'/'+gxy_name+'_weight.fits', weight_data.astype(np.float32), header=hdu[wht_ext].header,  overwrite=True)
        
    else : print ("The fit returned an empty isophote list")
    
    #Close the HDU
    hdu.close()
    # os.remove(sewpytempimg)
    # os.remove(f"../OUTPUT/{gxyid}/weights.fits")
    os.remove(f"../OUTPUT/mback.fits")
    #os.remove(f"../OUTPUT/extapfile.fits")
    os.remove(f"../OUTPUT/cptapfile.fits")
    # plt.show(block=False)
    plt.clf()
    print("The total run for %s took %s seconds ---" % (gxy_name,(time.time() - start_time)))

    return fwhm





def run_part2(alpha) :
    
    #Start the time and see how long it takes 
    start_time = time.time()
    
    
    
    #Parameters initialization
    
    gxyid=gal.gxy
    band=gal.band
    gxy_name=gal.gxy_name
    gxyra=gal.gxyra
    gxydec=gal.gxydec
    
    magzp=gal.magzp #i-band zeropoint magnitude
    #img_ext=gal.img_ext #Image extension
    #wht_ext=gal.wht_ext #Weight extension
    
    #Load images
    
    resimg="../OUTPUT/"+gxyid+'/'+gxy_name+'_res.fits' #residual image file path
    modimg="../OUTPUT/"+gxyid+'/'+gxy_name+'_mod.fits' #model file path
    whtimg="../OUTPUT/"+gxyid+'/'+gxy_name+'_weight.fits' #masked weight file path
  
    res=fits.open(resimg, do_not_scale_image_data=True)
    resdata=res[0].data
    
    mod=fits.open(modimg, do_not_scale_image_data=True)
    moddata=mod[0].data

    wht=fits.open(whtimg, do_not_scale_image_data=True)
    whtdata=wht[0].data

    if not os.path.exists("../OUTPUT/part2_temp"):
            os.makedirs("../OUTPUT/part2_temp")
    if not os.path.exists("../OUTPUT/plots/"+gxyid):
        os.makedirs("../OUTPUT/plots/"+gxyid)
    
    #Correct the weight by the model, if alpha is not 0
    whtdata=1/(1/whtdata+alpha*moddata)   # 0.1, 1 #If alpha is 0 here does not do anything
    whtfilepath="../OUTPUT/"+gxyid+"/"+gxy_name+'_revwht.fits'
    fits.writeto(whtfilepath, whtdata.astype(np.float32), header=wht[0].header,  overwrite=True)
    
    #galaxy coordinates in residual image
    wgxy=WCS(res[0].header)
    xgxy, ygxy = wgxy.wcs_world2pix(gxyra, gxydec, 1) # Get the galaxy center in pixels
    x0=int(xgxy)
    y0=int(ygxy)
    
    fwhm0= gal.fwhm0
    plate_scale=gal.plate_scale
    
    
    # Generate sextractor file using sewpy 
    
    
    sewpyconf={"PARAMETERS_NAME":"../INPUT/UTILS/part2.param.lsst",
                          "FILTER_NAME":"../INPUT/UTILS/gauss_3.0_5x5.conv",
                          "STARNNW_NAME":"../INPUT/UTILS/default.nnw",
                          "SEEING_FWHM": fwhm0,
                          "MAG_ZEROPOINT": magzp,
                          "WEIGHT_IMAGE": whtfilepath,
                          "PIXEL_SCALE" : plate_scale,
                          "CHECKIMAGE_NAME":"../OUTPUT/"+gxyid+"/"+band+"_seg.fits"}
    
    #Run sextractor on residuals 
    
    sew=sewpy.SEW(workdir='../OUTPUT/part2_temp', sexpath="sex", configfilepath='../INPUT/UTILS/sewpy_lsst_finalcat.par',
                  config=sewpyconf,loglevel=None)
    out=sew(resimg)
    copyfile(out["catfilepath"],"../OUTPUT/"+gxyid+'/'+gxy_name+'.cat')
    
    catfilepath="../OUTPUT/"+gxyid+'/'+gxy_name+'.cat'
    utils.get_plots(catfilepath,gxyra,gxydec,gxyid,gxy_name)
    
    rmtree('../OUTPUT/part2_temp')
    print("The total run for part 2 for %s took %s seconds ---" % (gxy_name,(time.time() - start_time)))
    # return fwhm0
    #hdu.close()
    #os.remove(filename)


def run_part3(gxyid, band1,band2, ext_corr1, ext_corr2,magzp1,magzp2,fwhm1,fwhm2, match_scale,
                csmin, mfaint,mbright,threshold):

    
    
    '''
    estimate aper corr
    apply aper corr and gal extinction
    match catalogs
    calculate half-light radius of galaxy    
    
    '''
    
    #Start the time and see how long it takes 
    start_time = time.time()
    
    gxyid=gal.gxy
    #gxyra=gal.gxyra
    #gxydec=gal.gxydec
    
    # fwhm0= gal.fwhm0
    plate_scale=gal.plate_scale
    # img_ext=gal.img_ext #Image extension
    # wht_ext=gal.wht_ext #Weight extension
    # res_ext=0
    gxyid=gxyid.upper()
    
    #Read the catalogs in the 2 bands of interest
    
    gxy_name1=gxyid+'_'+band1
    print('Name g catalog', gxy_name1)
    cat1path="../OUTPUT/"+gxyid+'/'+gxy_name1+'.cat'
    # resimg1="../OUTPUT/"+gxyid+'/'+gxy_name1+'_res.fits' #residual image file path
    # modimg1="../OUTPUT/"+gxyid+'/'+gxy_name1+'_mod.fits' #model image file path
    # galimg1="../OUTPUT/"+gxyid+'/'+gxy_name1+'.fits' #galaxy image
    # # maskimg1="../OUTPUT/"+gxyid+'/'+gxy_name1+'_mask.fits' #mask image
    # t1 = Table.read(cat1path, format='ascii.sextractor')

    gxy_name2=gxyid+'_'+band2
    print('Name 1 catalog', gxy_name2)
    cat2path="../OUTPUT/"+gxyid+'/'+gxy_name2+'.cat'
    # resimg2="../OUTPUT/"+gxyid+'/'+gxy_name2+'_res.fits' #residual image file path
    # modimg2="../OUTPUT/"+gxyid+'/'+gxy_name2+'_mod.fits' #residual image file path
    # galimg2="../OUTPUT/"+gxyid+'/'+gxy_name2+'.fits' #galaxy image
    # # maskimg2="../OUTPUT/"+gxyid+'/'+gxy_name2+'_mask.fits' #mask image
    # t2 = Table.read(cat2path, format='ascii.sextractor')


    if not os.path.exists("../OUTPUT/plots/"+gxyid):
        os.makedirs("../OUTPUT/plots/"+gxyid)

    #Apply extinction corrections to the 2 catalogs
    
    corr_tsex1=utils.corr_catalog(cat1path, 'MAG_APER_1', 0, ext_corr1)
    cat1path="../OUTPUT/"+gxyid+'/'+gxy_name1+'_extcorr.cat'
    corr_tsex1.write(cat1path,format='ascii',overwrite=True)
    
    corr_tsex2=utils.corr_catalog(cat2path, 'MAG_APER_1', 0, ext_corr2)
    cat2path="../OUTPUT/"+gxyid+'/'+gxy_name2+'_extcorr.cat'
    corr_tsex2.write(cat2path,format='ascii',overwrite=True)
    

    #print(fwhm1,fwhm2)
    rad_asec=np.mean([fwhm1,fwhm2])*match_scale#1.2
    # threshold=35 # 50 HSC
    
    tmatch_all1,tmatch_all2=utils.sbf_match_catalogs(cat1path, cat2path, rad_asec)
    match1path='../OUTPUT/'+gxyid+'/'+gxy_name1+'_matched.cat'
    match2path='../OUTPUT/'+gxyid+'/'+gxy_name2+'_matched.cat'
    tmatch_all1.write(match1path,format='ascii',overwrite=True)
    tmatch_all2.write(match2path,format='ascii',overwrite=True)


    rad_asec=np.mean([fwhm1,fwhm2])*5.
    #print(fwhm1, fwhm2, plate_scale)



    #Compute and apply aperture correction to matched, clean catalogs
    
    aper_corr1, tcorr1=utils.sbf_aper_corr(match1path,rad_asec,gxyid, band1,csmin,mfaint,mbright,threshold)
    aper_corr2, tcorr2=utils.sbf_aper_corr(match2path,rad_asec,gxyid, band2,csmin,mfaint,mbright,threshold)
    # aper_corr1, tcorr1=utils.sbf_aper_corr(match1path,rad_asec,gxy_name1,csmin,mfaint,mbright,threshold)
    # aper_corr2, tcorr2=utils.sbf_aper_corr(match2path,rad_asec,gxy_name2,csmin,mfaint,mbright,threshold)
    match1path='../OUTPUT/'+gxyid+'/'+gxy_name1+'_matchedcorr.cat'
    match2path='../OUTPUT/'+gxyid+'/'+gxy_name2+'_matchedcorr.cat'
    tcorr1.write(match1path,format='ascii.commented_header',overwrite=True)
    tcorr2.write(match2path,format='ascii.commented_header',overwrite=True)


    
    #Effective raiuds calculation of g image
    sigmalin = lambda x,a,b: np.sum((utils.func_linear(x,a,b)))
    

    #Read isophotes
    isofilepath1="../OUTPUT/"+gxyid+'/'+gxy_name1+'.iso'
    isophotes1=Table.read(isofilepath1,format='ascii')
    # print(isophotes1)
    
    #Parameters arrangment
    sma=isophotes1['sma']*plate_scale
    tflux_e=isophotes1['tflux_e']
    clean_nan=(np.isfinite(tflux_e))
    sma=sma[(clean_nan)]
    tflux_e=tflux_e[(clean_nan)]

    
    # m_V=magzp_V-2.5*np.log10(tot_flux_V)
    # m_g=magzp_g-2.5*np.log10(tot_flux_g)
    # d_V=m_V[:-1]-m_V[1:]
    # print(tflux_e)
    
    #CoG of the isophotes
    diff=tflux_e[1:]-tflux_e[:-1]
    cond=(diff>0)
    x=np.array(sma[:-1])
    x=x[cond]
    diff=diff[cond]
    y=np.log(diff)
    guess_a=-1.
    guess_b=-4
    # print(diff[diff<=0])
    # plt.plot(x,diff)
    # plt.show()
    # plt.clf()
    
    #Linear fit
    popt_asymlin,pcov_asymlin = curve_fit(utils.func_linear,x,y,p0=[guess_a,guess_b])
    
    #Asymptote calculation
    asym=tflux_e[-1]+sigmalin(x*2,*popt_asymlin)
    re1=sma[np.abs(tflux_e-0.5*asym).argmin()]
    
    
    #Plot
    plt.plot(sma,tflux_e)
    ax=plt.gca()
    # ax.invert_yaxis()
    trans = transforms.blended_transform_factory(ax.get_yticklabels()[0].get_transform(), ax.transData)
    plt.text(0.8,asym/1.5, f"R_e={round(re1,3)} asec", color="red",ha="right", va="center",transform=trans)
    plt.axhline(asym,color='red')
    plt.xlabel("sma arcsec")
    plt.ylabel("$Flux$ "+band1)
    #plt.show(block=False)
    ##plt.clf()
    plt.savefig(f'../OUTPUT/plots/{gxyid}/{gxy_name1}_R_e.jpeg',dpi=300)
    plt.show()
    plt.clf()
    plt.close()
    
    #Calculate effective radius of galaxy
    print(asym)
    print("R_E in "+str(band1)+" is",sma[np.abs(tflux_e-0.5*asym).argmin()])
    
    
    
    #Effective radius calculation of i image
    
    #Read Isopghotes
    isofilepath2="../OUTPUT/"+gxyid+'/'+gxy_name2+'.iso'
    isophotes2=Table.read(isofilepath2,format='ascii')
    # print(isophotes1)
    sma=isophotes2['sma']*plate_scale
    tflux_e=isophotes2['tflux_e']
    clean_nan=(np.isfinite(tflux_e))
    sma=sma[(clean_nan)]
    tflux_e=tflux_e[(clean_nan)]

    
    #CoG of the isophotes
    diff=tflux_e[1:]-tflux_e[:-1]
    cond=(diff>0)
    x=np.array(sma[:-1])
    x=x[cond]
    diff=diff[cond]
    y=np.log(diff)
    guess_a=-1.
    guess_b=-4
    
    #Linear fit
    popt_asymlin,pcov_asymlin = curve_fit(utils.func_linear,x,y,p0=[guess_a,guess_b])
    
    #Asymptote estimation
    asym=tflux_e[-1]+sigmalin(x*2,*popt_asymlin)
    re2=sma[np.abs(tflux_e-0.5*asym).argmin()]
    # print(popt_asymlin)
    
    
    #Plot
    plt.plot(sma,tflux_e)
    ax=plt.gca()
    # ax.invert_yaxis()
    trans = transforms.blended_transform_factory(ax.get_yticklabels()[0].get_transform(), ax.transData)
    plt.text(0.8,asym/1.5, f"R_e={round(re2,3)} asec", color="red",ha="right", va="center",transform=trans)
    plt.axhline(asym,color='red')
    plt.xlabel("sma arcsec")
    plt.ylabel("$Flux$ "+band2)
    #plt.show(block=False)
    ##plt.clf()
    plt.savefig(f'../OUTPUT/plots/{gxyid}/{gxy_name2}_R_e.jpeg',dpi=300)
    plt.clf()
    plt.close()
    
    print(asym)
    print("R_E in "+str(band2)+" is",sma[np.abs(tflux_e-0.5*asym).argmin()])
    
    # R_E of galaxy: half-light radius
     
    r_e=np.mean([re1,re2])
    # r_e=re2
    print("Mean R_E is",r_e)


    gal.r_e=r_e
    



def run_part4(gxyid, band1,band2, magzp1,magzp2,fwhm1,fwhm2, ext_corr2):

    
    
    '''
    Subtract large scale background
    
    
    '''
    
    #Start the time and see how long it takes 
    # start_time = time.time()
    
    gxyid=gal.gxy
    gxyra=gal.gxyra
    gxydec=gal.gxydec
    gxycen=SkyCoord(gxyra.item(), gxydec.item(), unit=(u.deg, u.deg))

    # fwhm0= gal.fwhm0
    plate_scale=gal.plate_scale
    img_ext=gal.img_ext #Image extension
    wht_ext=gal.wht_ext #Weight extension
    res_ext=0
    gxyid=gxyid.upper()
    
    #Read the catalogs in the 2 bands of interest
    gxy_name1=gxyid+'_'+band1
    cat1path="../OUTPUT/"+gxyid+'/'+gxy_name1+'_matchedcorr.cat' #Matched corrected catalog
    resimg1="../OUTPUT/"+gxyid+'/'+gxy_name1+'_res.fits'     #residual image
    modimg1="../OUTPUT/"+gxyid+'/'+gxy_name1+'_mod.fits'     #model image
    galimg1="../OUTPUT/"+gxyid+'/'+gxy_name1+'.fits'         #galaxy image
    # maskimg1="../OUTPUT/"+gxyid+'/'+gxy_name1+'_mask.fits' #mask image  
    whtimg1="../OUTPUT/"+gxyid+"/"+gxy_name1+'_revwht.fits'  #Weight image
    t1 = Table.read(cat1path, format='ascii.commented_header')

    gxy_name2=gxyid+'_'+band2
    cat2path="../OUTPUT/"+gxyid+'/'+gxy_name2+'_matchedcorr.cat'#Matched corrected catalog
    resimg2="../OUTPUT/"+gxyid+'/'+gxy_name2+'_res.fits'    #residual image
    modimg2="../OUTPUT/"+gxyid+'/'+gxy_name2+'_mod.fits'    #model image
    galimg2="../OUTPUT/"+gxyid+'/'+gxy_name2+'.fits'        #galaxy image
    # maskimg2="../OUTPUT/"+gxyid+'/'+gxy_name2+'_mask.fits'#mask image
    whtimg2="../OUTPUT/"+gxyid+"/"+gxy_name2+'_revwht.fits' #Weight image
    t2 = Table.read(cat2path, format='ascii.commented_header')
    
    #Residual g cataog
    # img1=fits.open(galimg1, do_not_scale_image_data=True)
    res1=fits.open(resimg1, do_not_scale_image_data=True)
    resdata1=res1[res_ext].data
    reshdr1=res1[res_ext].header
    wcs1=WCS(reshdr1)
    
    #residual catalog i
    # img2=fits.open(galimg2, do_not_scale_image_data=True)
    res2=fits.open(resimg2, do_not_scale_image_data=True)
    res2.info()
    resdata2=res2[res_ext].data
    reshdr2=res2[res_ext].header
    wcs2=WCS(reshdr2)
    
    #Model catalog
    mod2=fits.open(modimg2, do_not_scale_image_data=True)
    mod2.info()
    moddata2=mod2[res_ext].data
    
    #Weight catalog
    wht2=fits.open(whtimg2, do_not_scale_image_data=True)
    wht2.info()
    whtdata2=wht2[wht_ext].data

    
    # GENERATE MASKS:

    print("Generating Catalog for P_r module")
   

    # Sextractor run for background g
    
    sewpyconf={"PARAMETERS_NAME":"../INPUT/UTILS/default.param.lsst",
                          "FILTER_NAME":"../INPUT/UTILS/gauss_3.0_5x5.conv",
                          "STARNNW_NAME":"../INPUT/UTILS/default.nnw",
                          "SEEING_FWHM": fwhm1,
                          "MAG_ZEROPOINT": magzp1,
                          "PIXEL_SCALE" : gal.plate_scale,
                          "BACK_SIZE" :      int(10*fwhm1/plate_scale),  #int(10*popt1[2]*2.355),   # ADD REF
                          "BACK_FILTERSIZE" :  2,
                        #   "WEIGHT_TYPE": "MAP_WEIGHT",                          
                        #   "WEIGHT_IMAGE": whtimg1,
                          "CHECKIMAGE_NAME": "../OUTPUT/"+gxyid+"/"+gxy_name1+"_sbkg.fits, ../OUTPUT/"+gxyid+"/"+gxy_name1+"_mbkg.fits"
                          }

    
    sew_mback=sewpy.SEW(workdir=f'../OUTPUT/{gxyid}', sexpath="sex", configfilepath='../INPUT/UTILS/sewpy_lsst_smoothbkg.par',
                        config=sewpyconf,loglevel=None)
    
    out_mback=sew_mback(resimg1)
    
    
    # Sextractor run for background i

    sewpyconf={"PARAMETERS_NAME":"../INPUT/UTILS/default.param.lsst",
                          "FILTER_NAME":"../INPUT/UTILS/gauss_3.0_5x5.conv",
                          "STARNNW_NAME":"../INPUT/UTILS/default.nnw",
                          "SEEING_FWHM": fwhm2,
                          "MAG_ZEROPOINT": magzp2,
                          "PIXEL_SCALE" : gal.plate_scale,
                          "BACK_SIZE" :        int(10*fwhm2/plate_scale),  #int(10*popt2[2]*2.355),
                          "BACK_FILTERSIZE" :  2,
                        #   "WEIGHT_TYPE": "MAP_WEIGHT",
                        #   "WEIGHT_IMAGE": whtimg2,
                          "CHECKIMAGE_NAME": "../OUTPUT/"+gxyid+"/"+gxy_name2+"_sbkg.fits, ../OUTPUT/"+gxyid+"/"+gxy_name2+"_mbkg.fits"
                          }
    print('BACK_SIZE',int(10*fwhm2/plate_scale),)
    
    sew_mback=sewpy.SEW(workdir=f'../OUTPUT/{gxyid}', sexpath="sex", configfilepath='../INPUT/UTILS/sewpy_lsst_smoothbkg.par',
                        config=sewpyconf,loglevel=None)
    
    out_mback=sew_mback(resimg2)
    




def run_part5(gxyid, band1,band2, ext_corr1, ext_corr2,magzp1,magzp2,fwhm1,fwhm2,
              mfaint,mbright,threshold,cscut,psf_rad_scale,nthfactor, psfsize,oversampling,
              rgc_factor):

    
    
    '''
    generate and test PSF
    '''
    
    
    
    #Start the time and see how long it takes 
    start_time = time.time()
    
    #Parameters initialization
    gxyid=gal.gxy

    plate_scale=gal.plate_scale
    img_ext=gal.img_ext #Image extension
    wht_ext=gal.wht_ext #Weight extension
    res_ext=0
    gxyid=gxyid.upper()
    
    #Read the catalogs in the 2 bands of interest
    
    #g
    gxy_name1=gxyid+'_'+band1
    match1path="../OUTPUT/"+gxyid+'/'+gxy_name1+'_matchedcorr.cat' #Matched corrected catalog
    mbkimg1="../OUTPUT/"+gxyid+'/'+gxy_name1+'_mbkg.fits'      #bkg-subtracted residual image file path
   
    mbk1=fits.open(mbkimg1, do_not_scale_image_data=True)
    # mask1=fits.open(maskimg1, do_not_scale_image_data=True)
    mbk1data=mbk1[res_ext].data
    # maskdata1=mask1[res_ext].data
    hdr1=mbk1[res_ext].header
    wcs1=WCS(hdr1)
    
    #i
    gxy_name2=gxyid+'_'+band2
    match2path="../OUTPUT/"+gxyid+'/'+gxy_name2+'_matchedcorr.cat'
    mbkimg2="../OUTPUT/"+gxyid+'/'+gxy_name2+'_mbkg.fits'      #bkg-subtracted residual image file path

    mbk2=fits.open(mbkimg2, do_not_scale_image_data=True)
    # mask1=fits.open(maskimg1, do_not_scale_image_data=True)
    mbk2data=mbk2[res_ext].data
    # maskdata1=mask1[res_ext].data
    hdr2=mbk2[res_ext].header
    wcs2=WCS(hdr2)

    if not os.path.exists("../OUTPUT/plots/"+gxyid):
        os.makedirs("../OUTPUT/plots/"+gxyid)

    
    # Generate EPSF
    rad_asec=np.mean([fwhm1,fwhm2])*psf_rad_scale
    
    # epsf1,epsf2=utils.twoband_psf(resimg1,res_ext,match1path,resimg2,res_ext,match2path,
    #                               cscut,mfaint,mbright,rad_asec,threshold,psfsize,oversampling,
    #                               gxyid,gxy_name1,gxy_name2,nthfactor, rgc_factor)
    
    epsf2=utils.twoband_psf_VCC(mbkimg1,res_ext,match1path,mbkimg2,res_ext,match2path,
                                  cscut,mfaint,mbright,rad_asec,threshold,psfsize,oversampling,
                                  gxyid,gxy_name1,gxy_name2,nthfactor, rgc_factor, 
                                  seg_path=f"../OUTPUT/{gxyid}/i_seg.fits")
    
    psfpath="../OUTPUT/"+gxyid+'/'+gxy_name2+'_epsf.fits'
    fits.writeto(psfpath, epsf2.data.astype(np.float32), overwrite=True)

    
    
    #Reshape the Epsf data to compensate the oversampling
    
    shape = np.array(epsf2.shape, dtype=float)
    newshape = oversampling * np.floor(shape / oversampling).astype(np.int32)
    # epsf1_data=cv.resize(epsf1.data,(psfsize+1,psfsize+1),4,4,interpolation=cv.INTER_AREA)
    psfdata=epsf2.data[:newshape[0],:newshape[1]]
    temp=psfdata.reshape((newshape[0] // oversampling , oversampling,
                               newshape[1] // oversampling, oversampling))
    epsf2_data=np.sum(temp, axis=(1,3))
    utils.visualize(data=epsf2_data, figpath=f'../OUTPUT/plots/{gxyid}/{gxy_name2}_rbpsf.jpeg')
    
    
    #Write the psf
    psfpath="../OUTPUT/"+gxyid+'/'+gxy_name2+'_psf.fits'
    fits.writeto(psfpath, epsf2_data.astype(np.float32), overwrite=True)
    
    
        
    
    r_epsf2=np.zeros(epsf2_data.shape)
    nx,ny=epsf2_data.shape
    x0,y0=nx/2,ny/2
    
    for i in range(ny):
        for j in range(nx):
            r_epsf2[i,j]=round(np.sqrt((j-x0)**2+(i-y0)**2))
            
    if not os.path.exists("../OUTPUT/plots/"+gxyid):
        os.makedirs("../OUTPUT/plots/"+gxyid)

    
    # Gaussian fit and FWHM
    
    p0=[0.9,0,1.]
    
    x2,y2=utils.azimuthal_avg(epsf2_data)
    popt2,pcov2=curve_fit(utils.gaussian, x2, y2[:,0],p0=p0)
    # print(popt2,popt2[2]*2.355)

    
    fig = plt.figure(figsize=(9,7))
    fig.subplots_adjust(wspace=0.4,hspace=0.4, left=0.15, right=0.98,
                        bottom=0.15, top=0.9)
    font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 20}

    plt.rc('font', **font)
    plt.scatter(r_epsf2.ravel(),epsf2_data.ravel(),color='C1')
    plt.plot(x2,y2[:,0])
    # plt.plot(x2,utils.gaussian(x2,*popt2),color='C1')
    plt.title(band2+" rebinned")
    plt.xlabel("Pix")
    plt.ylabel("Flux")
    plt.yscale('log')
    #plt.show(block=False)
    plt.savefig(f'../OUTPUT/plots/{gxyid}/{gxy_name2}_rebinnedpsf.jpeg',dpi=300)
    plt.show()
    plt.clf()
    plt.close()
    print('Gaussian Fit',popt2,popt2[2]*2.355,'\n\n')
    
    print("FWHM: comparing with PSF")
    print(f'g band FWHM:                                from photometry {fwhm1/plate_scale} pix')
    print(f'i band FWHM: EPSF fit {popt2[2]*2.355} pix, from photometry {fwhm2/plate_scale} pix \n\n')


    # Testing PSF
    
    ################################################
    '''

    tab1=utils.test_psf(band1,res_ext,epsf1)
    tab2=utils.test_psf(band2,res_ext,epsf2)

    
    ########################################################################
    #TESTING PHOTOMETRY
    
    tab1['MAG']=-2.5*np.log10(tab1['flux_fit'])+magzp1
    tab2['MAG']=-2.5*np.log10(tab2['flux_fit'])+magzp2
    tab1=tab1[~np.isnan(tab1['MAG'])]
    tab2=tab2[~np.isnan(tab2['MAG'])]

    tab1['ALPHA_J2000'],tab1['DELTA_J2000']=wcs1.pixel_to_world_values(tab1['x_fit'],tab1['y_fit'])
    tab2['ALPHA_J2000'],tab2['DELTA_J2000']=wcs2.pixel_to_world_values(tab2['x_fit'],tab2['y_fit'])
    
    tab1.write('../OUTPUT/'+gxyid+'/'+band1+'_dao.cat',format='ascii',overwrite=True)
    tab2.write('../OUTPUT/'+gxyid+'/'+band2+'_dao.cat',format='ascii',overwrite=True)
    
    print("Radius of matching in arcsec: ",rad_asec)
    tmatch_sex1,tmatch_dao1=utils.basic_match_catalogs(tcorr1,tab1,rad_asec)
    tmatch_sex2,tmatch_dao2=utils.basic_match_catalogs(tcorr2,tab2,rad_asec)

    tmatch_sex1.write('../OUTPUT/'+gxyid+'/'+band1+'_matched_sex.cat',format='ascii',overwrite=True)
    tmatch_sex2.write('../OUTPUT/'+gxyid+'/'+band2+'_matched_sex.cat',format='ascii',overwrite=True)
    tmatch_dao1.write('../OUTPUT/'+gxyid+'/'+band1+'_matched_dao.cat',format='ascii',overwrite=True)
    tmatch_dao2.write('../OUTPUT/'+gxyid+'/'+band2+'_matched_dao.cat',format='ascii',overwrite=True)
    
    # tmatch_sex1=Table.read('../OUTPUT/'+gxyid+'/'+band1+'_matched_sex.cat',format='ascii')
    # tmatch_sex2=Table.read('../OUTPUT/'+gxyid+'/'+band2+'_matched_sex.cat',format='ascii')
    # tmatch_dao1=Table.read('../OUTPUT/'+gxyid+'/'+band1+'_matched_dao.cat',format='ascii')
    # tmatch_dao2=Table.read('../OUTPUT/'+gxyid+'/'+band2+'_matched_dao.cat',format='ascii')
    # tmatch_sex1=tmatch_sex1[~np.isnan(tmatch_dao1['MAG'])]
    # tmatch_sex2=tmatch_sex2[~np.isnan(tmatch_dao2['MAG'])]
    # tmatch_dao1=tmatch_dao1[~np.isnan(tmatch_dao1['MAG'])]
    # tmatch_dao2=tmatch_dao2[~np.isnan(tmatch_dao2['MAG'])]

    cpt_cond=((tmatch_sex1['CLASS_STAR']>0.8))  
    plt.scatter(tmatch_dao1['MAG'],tmatch_sex1['MAG_CORR']-tmatch_dao1['MAG'])
    plt.scatter(tmatch_dao1['MAG'][cpt_cond],tmatch_sex1['MAG_CORR'][cpt_cond]-tmatch_dao1['MAG'][cpt_cond])
    plt.savefig(f'../OUTPUT/plots/{gxyid}/{gxy_name1}_testpsf.jpeg',dpi=300)
    #plt.show(block=False)
    plt.clf()
    cpt_cond=((tmatch_sex2['CLASS_STAR']>0.8))
    plt.scatter(tmatch_dao2['MAG'],tmatch_sex2['MAG_CORR']-tmatch_dao2['MAG'])
    plt.scatter(tmatch_dao2['MAG'][cpt_cond],tmatch_sex2['MAG_CORR'][cpt_cond]-tmatch_dao2['MAG'][cpt_cond])
    plt.savefig(f'../OUTPUT/plots/{gxyid}/{gxy_name2}_testpsf.jpeg',dpi=300)
    #plt.show(block=False)
    plt.clf()

    check_cond=(tmatch_sex1['MAG_APER_1']<=22.)
    cpt_cond=((tmatch_sex1['CLASS_STAR']>0.8))
    
    print(np.median(tmatch_sex1['MAG_CORR']-tmatch_dao1['MAG']),mad(tmatch_sex1['MAG_CORR']-tmatch_dao1['MAG']))
    print("Bright")
    print(np.median(tmatch_sex1['MAG_CORR'][check_cond]-tmatch_dao1['MAG'][check_cond]),mad(tmatch_sex1['MAG_CORR'][check_cond]-tmatch_dao1['MAG'][check_cond]))
    print("Compact")
    print(np.median(tmatch_sex1['MAG_CORR'][cpt_cond]-tmatch_dao1['MAG'][cpt_cond]),mad(tmatch_sex1['MAG_CORR'][cpt_cond]-tmatch_dao1['MAG'][cpt_cond]))
    
    check_cond=(tmatch_sex2['MAG_APER_1']<=22.)
    cpt_cond=((tmatch_sex2['CLASS_STAR']>0.8))
    print(np.median(tmatch_sex2['MAG_CORR']-tmatch_dao2['MAG']),mad(tmatch_sex2['MAG_CORR']-tmatch_dao2['MAG']))
    print("Bright")
    print(np.median(tmatch_sex2['MAG_CORR'][check_cond]-tmatch_dao2['MAG'][check_cond]),mad(tmatch_sex2['MAG_CORR'][check_cond]-tmatch_dao2['MAG'][check_cond]))
    print("Compact")
    print(np.median(tmatch_sex2['MAG_CORR'][cpt_cond]-tmatch_dao2['MAG'][cpt_cond]),mad(tmatch_sex2['MAG_CORR'][cpt_cond]-tmatch_dao2['MAG'][cpt_cond]))
    '''
    
    
    #######################################################################
    #plt.show()
    #plt.clf()
    


def run_pr_cat(gxyid, band1,band2, magzp1,magzp2,fwhm1,fwhm2, cutout_size, 
                csmin,mfaint,mbright,threshold, rfact=1.5):

    '''
    Generate cutouts and catalog for P_r module

    '''

    gxyid=gal.gxy
    gxyra=gal.gxyra
    gxydec=gal.gxydec
    gxycen=SkyCoord(gxyra.item(), gxydec.item(), unit=(u.deg, u.deg))

    # fwhm0= gal.fwhm0
    plate_scale=gal.plate_scale
    img_ext=gal.img_ext #Image extension
    wht_ext=gal.wht_ext #Weight extension
    res_ext=0
    gxyid=gxyid.upper()
    
    #Read the catalogs in the 2 bands of interest
    gxy_name1=gxyid+'_'+band1
    cat1path="../OUTPUT/"+gxyid+'/'+gxy_name1+'_matchedcorr.cat'  #Matched corrected catalog
    resimg1="../OUTPUT/"+gxyid+'/'+gxy_name1+'_res.fits'        #residual image file path
    # modimg1="../OUTPUT/"+gxyid+'/'+gxy_name1+'_mod.fits'      #model image file path
    # galimg1="../OUTPUT/"+gxyid+'/'+gxy_name1+'.fits'          #galaxy image
    # maskimg1="../OUTPUT/"+gxyid+'/'+gxy_name1+'_mask.fits'    #mask image  
    whtimg1="../OUTPUT/"+gxyid+"/"+gxy_name1+'_revwht.fits'
    t1 = Table.read(cat1path, format='ascii.commented_header')

    gxy_name2=gxyid+'_'+band2
    cat2path="../OUTPUT/"+gxyid+'/'+gxy_name2+'_matchedcorr.cat'
    resimg2="../OUTPUT/"+gxyid+'/'+gxy_name2+'_res.fits'    #residual image file path
    # modimg2="../OUTPUT/"+gxyid+'/'+gxy_name2+'_mod.fits'  #model image file path
    # galimg2="../OUTPUT/"+gxyid+'/'+gxy_name2+'.fits'      #galaxy image
    # maskimg2="../OUTPUT/"+gxyid+'/'+gxy_name2+'_mask.fits'#mask image
    whtimg2="../OUTPUT/"+gxyid+"/"+gxy_name2+'_revwht.fits'
    t2 = Table.read(cat2path, format='ascii.commented_header')
    
    #Residuals handling
    
    #img1=fits.open(galimg1, do_not_scale_image_data=True)
    res1=fits.open(resimg1, do_not_scale_image_data=True)
    resdata1=res1[res_ext].data
    reshdr1=res1[res_ext].header
    wcs1=WCS(reshdr1)

    #img2=fits.open(galimg2, do_not_scale_image_data=True)
    res2=fits.open(resimg2, do_not_scale_image_data=True)
    # res2.info()
    resdata2=res2[res_ext].data
    reshdr2=res2[res_ext].header
    wcs2=WCS(reshdr2)
    
    #Weight handling
    wht1=fits.open(whtimg1, do_not_scale_image_data=True)
    wht2=fits.open(whtimg2, do_not_scale_image_data=True)
    # wht2.info()
    whtdata1=wht1[wht_ext].data
    whtdata2=wht2[wht_ext].data

    rmin=rfact*gal.minrad #mask inner region of galaxy

      #If you want to include the instrument mask
    if (gal.read_hsc_mask):
        instrmaskimg1="../OUTPUT/"+gxyid+'/'+gxy_name1+'_instrmask.fits' #instrument mask image
        mask1=fits.open(instrmaskimg1, do_not_scale_image_data=True)
        maskdata1=mask1[res_ext].data
        instrmaskimg2="../OUTPUT/"+gxyid+'/'+gxy_name2+'_instrmask.fits' #instrument mask image
        mask2=fits.open(instrmaskimg2, do_not_scale_image_data=True)
        maskdata2=mask2[res_ext].data
    else:
        maskdata1=np.zeros(resdata1.shape)
        maskdata2=np.zeros(resdata2.shape)
    
    #Masking the central part
    gxycenpix = regions.PixCoord.from_sky(gxycen, wcs2)#mwcs2.wcs_world2pix(gxyra, gxydec, 1) # Get the galaxy center in pixels
    maskin=CirclePixelRegion(gxycenpix, rmin)
    maskin=(maskin.to_mask(mode='center')).to_image(resdata2.shape)
    print('Galaxy center:',gxycenpix)
    
    #Application of instrumental mask
    maskdata1=np.logical_or(maskdata1,maskin).astype(np.int32)
    maskdata2=np.logical_or(maskdata2,maskin).astype(np.int32)
    # fits.writeto("../OUTPUT/"+gxyid+'/'+gxy_name2+'_testmask.fits',maskdata2, overwrite=True)
    
    #Background-subtracted residual 
    res_mbkg_g=fits.open("../OUTPUT/"+gxyid+"/"+gxy_name1+"_mbkg.fits")
    res_mbkg_i=fits.open("../OUTPUT/"+gxyid+"/"+gxy_name2+"_mbkg.fits")
    mbkgdata1=res_mbkg_g[res_ext].data
    mbkgdata2=res_mbkg_i[res_ext].data

    
    
    #Mask application to background subtracted residual
    masked_data_g=mbkgdata1*(1-maskdata1)
    masked_data_i=mbkgdata2*(1-maskdata2)
    
    #cutout of the background-subtracted residual
    cutout_masked_g=Cutout2D(masked_data_g,gxycen,cutout_size,wcs=wcs1) #g
    res1[res_ext].header.update(cutout_masked_g.wcs.to_header())
    cutout_hdr_g=res1[res_ext].header
    
    cutout_masked=Cutout2D(masked_data_i,gxycen,cutout_size,wcs=wcs2)#i
    res2[res_ext].header.update(cutout_masked.wcs.to_header())
    cutout_hdr=res2[res_ext].header
    
    #cutout of the weight image
    
    wht_cutout_g=Cutout2D(whtdata1,gxycen,cutout_size,wcs=wcs1) #g
    wht1[wht_ext].header.update(wht_cutout_g.wcs.to_header())
    wcutout_hdr_g=wht1[wht_ext].header
    
    wht_cutout=Cutout2D(whtdata2,gxycen,cutout_size,wcs=wcs2) #i
    wht2[wht_ext].header.update(wht_cutout.wcs.to_header())
    wcutout_hdr=wht2[wht_ext].header
    
    #cutout of the mask
    mask_cutout_g=Cutout2D(1-maskdata1,gxycen,cutout_size,wcs=wcs1)
    mask_cutout=Cutout2D(1-maskdata2,gxycen,cutout_size,wcs=wcs2)
    # print(cutout_hdr)

    #Galaxy center in pixel for the bkg subtracted image
    xgxy, ygxy = WCS(cutout_hdr).wcs_world2pix(gxyra, gxydec, 1) # Get the galaxy center in pixels
    x0=int(xgxy)
    y0=int(ygxy)

    # Writing the catalogs
    
    #Instrument/basic mask
    fits.writeto("../OUTPUT/"+gxyid+'/'+gxy_name1+'_p_r_mask_cutout.fits',mask_cutout_g.data.astype(np.float32), header=cutout_hdr_g, overwrite=True)
    fits.writeto("../OUTPUT/"+gxyid+'/'+gxy_name2+'_p_r_mask_cutout.fits',mask_cutout.data.astype(np.float32), header=cutout_hdr, overwrite=True)
    
    # Masked bkg-subtracted residuals
    fits.writeto("../OUTPUT/"+gxyid+"/"+gxy_name1+"_cutout.fits",cutout_masked_g.data.astype(np.float32), header=cutout_hdr_g, overwrite=True)
    fits.writeto("../OUTPUT/"+gxyid+"/"+gxy_name2+"_cutout.fits",cutout_masked.data.astype(np.float32), header=cutout_hdr, overwrite=True)
    
    wht_cutout_path_g="../OUTPUT/"+gxyid+"/"+gxy_name1+"_whtcutout.fits"
    wht_cutout_path="../OUTPUT/"+gxyid+"/"+gxy_name2+"_whtcutout.fits"
    fits.writeto(wht_cutout_path_g,wht_cutout_g.data.astype(np.float32), header=wcutout_hdr_g, overwrite=True)

    fits.writeto(wht_cutout_path,wht_cutout.data.astype(np.float32), header=wcutout_hdr, overwrite=True)

    # Sextractor run on the bkg-subtracted cutout (needed for the P_r module)

    sewpyconf={"PARAMETERS_NAME":"../INPUT/UTILS/p_r.param.lsst",
                          "FILTER_NAME":"../INPUT/UTILS/gauss_3.0_5x5.conv",
                          "STARNNW_NAME":"../INPUT/UTILS/default.nnw",
                          "SEEING_FWHM": fwhm2,
                          "MAG_ZEROPOINT": magzp2,
                          "DETECT_THRESH":	1.5	,	
                          "ANALYSIS_THRESH":	3, #5,#		
                          "DETECT_MINAREA":	4, #2,#
                          "PIXEL_SCALE" : gal.plate_scale,
                          "BACK_SIZE" :       16,
                          "BACK_FILTERSIZE" :  3,#2, #
                          "WEIGHT_TYPE": "MAP_WEIGHT",
                          "WEIGHT_IMAGE": wht_cutout_path,
                          "CHECKIMAGE_NAME": "../OUTPUT/"+gxyid+"/"+gxy_name2+"_part4ap.fits"
                          }

    
    sew_mback=sewpy.SEW(workdir=f'../OUTPUT/{gxyid}', sexpath="sex", configfilepath='../INPUT/UTILS/sewpy_lsst_finalcat.par',#'../INPUT/UTILS/sewpy_lsst_compact.par',
                        config=sewpyconf,loglevel=None)
    
    out_mback_g=sew_mback("../OUTPUT/"+gxyid+"/"+gxy_name1+"_cutout.fits")
    out_mback=sew_mback("../OUTPUT/"+gxyid+"/"+gxy_name2+"_cutout.fits")
    
    t=Table.read(out_mback_g["catfilepath"], format='ascii.sextractor')
    x=t['X_IMAGE'] 
    y=t['Y_IMAGE'] 
    r=np.sqrt((x-x0)**2+(y-y0)**2)#tab['col4']
    # print(x0,y0)
    # print(x,y)
    # print(len(t))
    # print(min(r))
    t=t[(r>=rmin)]
    # print(len(t))
    tpath="../OUTPUT/"+gxyid+"/"+gxy_name1+"_p_r.cat"
    t.write(tpath,format='ascii.commented_header',overwrite=True)
    
    t=Table.read(out_mback["catfilepath"], format='ascii.sextractor')
    x=t['X_IMAGE'] 
    y=t['Y_IMAGE'] 
    r=np.sqrt((x-x0)**2+(y-y0)**2)#tab['col4']
    # print(x0,y0)
    # print(x,y)
    # print(len(t))
    # print(min(r))
    t=t[(r>=rmin)]
    # print(len(t))
    tpath="../OUTPUT/"+gxyid+"/"+gxy_name2+"_p_r.cat"
    t.write(tpath,format='ascii.commented_header',overwrite=True)
    
    # rad_asec=fwhm2*5.
    # aper_corr, tcorr=utils.sbf_aper_corr(tpath,rad_asec,gxyid, band2,csmin,mfaint,mbright,threshold)
    # tcorr.write(tpath,format='ascii.commented_header',overwrite=True)
    
    # tmask2 = Table.read("../OUTPUT/"+gxyid+"/"+gxy_name2+"_p_r.cat", format='ascii.commented_header')



def find_annuli(gxyid, band1, band2):


    '''
    Calculate SNR radially


    '''
  
    gxyid=gal.gxy
    gxyra=gal.gxyra
    gxydec=gal.gxydec
    gxycen=SkyCoord(gxyra.item(), gxydec.item(), unit=(u.deg, u.deg))

    # fwhm0= gal.fwhm0
    plate_scale=gal.plate_scale
    img_ext=gal.img_ext #Image extension
    wht_ext=gal.wht_ext #Weight extension
    res_ext=0
    gxyid=gxyid.upper()
    
    #Read the catalogs in the 2 bands of interest
    # gxy_name1=gxyid+'_'+band1
    # cat1path="../OUTPUT/"+gxyid+'/'+band1+'_matchedcorr.cat' #Matched corrected catalog
    # resimg1="../OUTPUT/"+gxyid+'/'+gxy_name1+'_res.fits' #residual image file path
    # modimg1="../OUTPUT/"+gxyid+'/'+gxy_name1+'_mod.fits' #model image file path
    # galimg1="../OUTPUT/"+gxyid+'/'+gxy_name1+'.fits' #galaxy image
    # maskimg1="../OUTPUT/"+gxyid+'/'+gxy_name1+'_mask.fits' #mask image  
    # whtimg1="../OUTPUT/"+gxyid+"/"+gxy_name1+'_revwht.fits'
    # t1 = Table.read(cat1path, format='ascii.commented_header')

    gxy_name2=gxyid+'_'+band2
    # cat2path="../OUTPUT/"+gxyid+'/'+band2+'_matchedcorr.cat'
    resimg2="../OUTPUT/"+gxyid+'/'+gxy_name2+'_res.fits' #residual image file path
    modimg2="../OUTPUT/"+gxyid+'/'+gxy_name2+'_mod.fits' #residual image file path
    galimg2="../OUTPUT/"+gxyid+'/'+gxy_name2+'.fits' #galaxy image
    # maskimg2="../OUTPUT/"+gxyid+'/'+gxy_name2+'_mask.fits' #mask image
    rmsimg2=gal.wghtfilepath
    # t2 = Table.read(cat2path, format='ascii.commented_header')

    # img1=fits.open(galimg1, do_not_scale_image_data=True)
    # res1=fits.open(resimg1, do_not_scale_image_data=True)
    # resdata1=res1[res_ext].data
    # reshdr1=res1[res_ext].header
    # wcs1=WCS(reshdr1)

    img2=fits.open(galimg2, do_not_scale_image_data=True)
    # res2=fits.open(resimg2, do_not_scale_image_data=True)
    # res2.info()
    # mod2=fits.open(modimg2, do_not_scale_image_data=True)
    # mod2.info()
    # moddata2=mod2[res_ext].data
    galdata2=img2[res_ext].data
    galhdr2=img2[res_ext].header
    wcs2=WCS(galhdr2)
    rms2=fits.open(rmsimg2, do_not_scale_image_data=True)
    # wht2.info()
    rmsdata2=rms2[wht_ext].data

    utils.radial_snr(galdata2,rmsdata2)

  


def run_part6(gxyid, band1,band2, magzp1,magzp2,fwhm1,fwhm2, 
                in_rad, out_rad, ext_corr2, cutout_size, r):

    
    
    '''

    Analyse power spectrum of masked image, psf
    
    
    '''
    

    
    #######  $$$$$$$$$$$$$$$
    # box=gal.box #Box size in pixels within which to fit galaxy    
    # x0=int(xgxy)
    # y0=int(ygxy)
    # maskin=gal.maskin   #Inner radius of the mask selection (arcsec)
    # maskout=gal.maskout #Outer radius 


    
    # gxycen=SkyCoord(gxyra.item(), gxydec.item(), unit=(u.deg, u.deg))
    # # print(gxycen)
    # rad2=0.017 #*u.deg
    # annreg=CircleSkyRegion(gxycen, rad)
    # annreg=((annreg.to_pixel(wcs2)).to_mask(mode='center')).to_image(resdata2.shape)

    ann=r+1

    gxyid=gal.gxy
    gxyra=gal.gxyra
    gxydec=gal.gxydec
    gxycen=SkyCoord(gxyra.item(), gxydec.item(), unit=(u.deg, u.deg))

    # fwhm0= gal.fwhm0
    plate_scale=gal.plate_scale
    img_ext=gal.img_ext #Image extension
    wht_ext=gal.wht_ext #Weight extension
    res_ext=0
    gxyid=gxyid.upper()
    
    #Read the catalogs in the 2 bands of interest
    # gxy_name1=gxyid+'_'+band1
    # cat1path="../OUTPUT/"+gxyid+'/'+band1+'_matchedcorr.cat' #Matched corrected catalog
    # resimg1="../OUTPUT/"+gxyid+'/'+gxy_name1+'_res.fits' #residual image file path
    # modimg1="../OUTPUT/"+gxyid+'/'+gxy_name1+'_mod.fits' #model image file path
    # galimg1="../OUTPUT/"+gxyid+'/'+gxy_name1+'.fits' #galaxy image
    # maskimg1="../OUTPUT/"+gxyid+'/'+gxy_name1+'_mask.fits' #mask image  
    # whtimg1="../OUTPUT/"+gxyid+"/"+gxy_name1+'_revwht.fits'
    # t1 = Table.read(cat1path, format='ascii.commented_header')

    gxy_name2=gxyid+'_'+band2
    # cat2path="../OUTPUT/"+gxyid+'/'+band2+'_matchedcorr.cat'
    resimg2="../OUTPUT/"+gxyid+'/'+gxy_name2+'_res.fits' #residual image file path
    modimg2="../OUTPUT/"+gxyid+'/'+gxy_name2+'_mod.fits' #residual image file path
    galimg2="../OUTPUT/"+gxyid+'/'+gxy_name2+'.fits' #galaxy image
    # maskimg2="../OUTPUT/"+gxyid+'/'+gxy_name2+'_mask.fits' #mask image
    whtimg2="../OUTPUT/"+gxyid+"/"+gxy_name2+'_revwht.fits'
    # t2 = Table.read(cat2path, format='ascii.commented_header')

    # img1=fits.open(galimg1, do_not_scale_image_data=True)
    # res1=fits.open(resimg1, do_not_scale_image_data=True)
    # resdata1=res1[res_ext].data
    # reshdr1=res1[res_ext].header
    # wcs1=WCS(reshdr1)

    # img2=fits.open(galimg2, do_not_scale_image_data=True)
    res2=fits.open(resimg2, do_not_scale_image_data=True)
    # res2.info()
    mod2=fits.open(modimg2, do_not_scale_image_data=True)
    # mod2.info()
    moddata2=mod2[res_ext].data
    resdata2=res2[res_ext].data
    reshdr2=res2[res_ext].header
    wcs2=WCS(reshdr2)
    wht2=fits.open(whtimg2, do_not_scale_image_data=True)
    # wht2.info()
    whtdata2=wht2[wht_ext].data

    
    
    # epsf_1=fits.open("../OUTPUT/"+gxyid+'/'+gxy_name1+'_psf.fits', do_not_scale_image_data=True)
    epsf_2=fits.open("../OUTPUT/"+gxyid+'/'+gxy_name2+'_psf.fits', do_not_scale_image_data=True)
    # epsf_2=fits.open("../OUTPUT/"+gxyid+'/'+gxy_name2+'_psf_new.fits', do_not_scale_image_data=True)
    # epsf_2=fits.open("../OUTPUT/"+gxyid+'/'+gxy_name2+'_psf_nosky.fits', do_not_scale_image_data=True)
    # epsf_2=fits.open("../OUTPUT/"+gxyid+'/psf_i3.fits', do_not_scale_image_data=True)
    # epsf_2=fits.open("../INPUT/IMAGES/"+gxyid+'/'+'psf.big_edit.fits', do_not_scale_image_data=True)
    # epsf_2=fits.open("../OUTPUT/"+gxyid+'/vcc1025_I.psf.fits', do_not_scale_image_data=True) #LAURA'S PSF
    # epsf_2=fits.open("../OUTPUT/"+gxyid+'/'+gxy_name2+'_psf_zbg.fits', do_not_scale_image_data=True)
    # epsf_2=fits.open("../OUTPUT/"+gxyid+'/'+gxy_name2+'_psf_zbg_msk.fits', do_not_scale_image_data=True)
    # epsf1_data=epsf_1[0].data
    epsf2_data=epsf_2[0].data

    if not os.path.exists("../OUTPUT/plots/"+gxyid):
        os.makedirs("../OUTPUT/plots/"+gxyid)

    # tmask2 = Table.read("../OUTPUT/"+gxyid+"/"+gxy_name2+"_p_r.cat", format='ascii.commented_header')
    # mask_stompout=fits.open("../OUTPUT/"+gxyid+"/"+gxy_name2+"_stompout.fits")
    mask_stompout=fits.open("../OUTPUT/"+gxyid+"/_stompout.fits")
    mask_i=1-mask_stompout[0].data  # FOR MIK'S MASK
    # mask_stompout=fits.open("../OUTPUT/"+gxyid+"/stompout.fits")
    # mask_i=mask_stompout[0].data  # FOR NH'S MASK
    mwcs2=WCS(mask_stompout[0].header)

    mask_scale=4.
    # t_arr=[tmask2]
    bands=np.array(['i'])
    wcs=np.array([wcs2])
    resdata=np.array([resdata2])
    reshdr=[reshdr2]
    mask_reg=[np.empty(resdata2.shape)]
    # radii=[rad2]

    # masked_data_reg=resdata2
    # for i in range(len(t_arr)):
        
        # tmask=t_arr[i]
        # annreg=CircleSkyRegion(gxycen, radii[i]*u.deg)
        # annreg=((annreg.to_pixel(wcs2)).to_mask(mode='center')).to_image(resdata2.shape)
        # annreg_in=CircleSkyRegion(gxycen, in_rad*u.deg)
        # annreg_in=((annreg_in.to_pixel(mwcs2)).to_mask(mode='center')).to_image(mask_i.shape)
        # annreg_out=CircleSkyRegion(gxycen, out_rad*u.deg)
        # annreg_out=((annreg_out.to_pixel(mwcs2)).to_mask(mode='center')).to_image(mask_i.shape)
        # annreg=annreg_out-annreg_in

        # rgc=np.sqrt((tmask['ALPHA_J2000']-gxyra)**2+(tmask['DELTA_J2000']-gxydec)**2)
        # cond=(rgc<=out_rad) & (rgc>=in_rad)
        # tmask=tmask[(cond)]
        
        # theta=list(map(lambda x: abs(x) if x<0 else 180-x, tmask['THETA_WORLD']))*u.deg
        # center=SkyCoord(tmask['ALPHA_J2000'],tmask['DELTA_J2000'],unit=(u.deg, u.deg))
        # # print(center)
        # width=mask_scale*tmask['A_WORLD']*u.deg#*3600/plate_scale
        # height=mask_scale*tmask['B_WORLD']*u.deg#*3600/plate_scale


        # print(bands[i],len(tmask))
        # for j in range(len(tmask)):
        #     # print(width[j])
        #     # region=CircleSkyRegion(center[j], width[j])
        #     region=EllipseSkyRegion(center[j], width[j], height[j],angle=theta[j])
        #     region=((region.to_pixel(wcs[i])).to_mask(mode='center')).to_image(resdata[i].shape)
        #     if (j==0):
        #         mask_reg[i]=region#.multiply(resdata[i])#mask_reg*(1-region.multiply(resdata[i]))
        #     else:
        #         mask_reg[i]=np.logical_or(mask_reg[i],region).astype(np.int32)
        #     # masked_data_reg=region.multiply(masked_data_reg)
        #     # regions.append(region.cutout(resdata[i]))
        # print(f"Writing mask file for {bands[i]}")
        # # print(WCS(reshdr[i]))
        # fits.writeto(f"../OUTPUT/{gxyid}/{bands[i]}_mask_reg.fits", mask_reg[i], header=reshdr[i],overwrite=True)

    # annreg_in=CircleSkyRegion(gxycen, in_rad*u.arcsec)
    # annreg_in=((annreg_in.to_pixel(mwcs2)).to_mask(mode='center')).to_image(mask_i.shape)
    # annreg_out=CircleSkyRegion(gxycen, out_rad*u.arcsec)
    # annreg_out=((annreg_out.to_pixel(mwcs2)).to_mask(mode='center')).to_image(mask_i.shape)
    # annreg=annreg_out-annreg_in

    gxycenpix = regions.PixCoord.from_sky(gxycen, mwcs2)#mwcs2.wcs_world2pix(gxyra, gxydec, 1) # Get the galaxy center in pixels
    annreg_in=CirclePixelRegion(gxycenpix, in_rad)
    annreg_in=(annreg_in.to_mask(mode='center')).to_image(mask_i.shape)
    annreg_out=CirclePixelRegion(gxycenpix, out_rad)
    annreg_out=(annreg_out.to_mask(mode='center')).to_image(mask_i.shape)
    annreg=annreg_out-annreg_in


    # res_mbkg_i=fits.open("../OUTPUT/"+gxyid+"/"+gxy_name2+"_mbkg.fits")
    res_mbkg_i=fits.open("../OUTPUT/"+gxyid+"/"+gxy_name2+"_cutout.fits")
    mbkgdata2=res_mbkg_i[res_ext].data
    # mbkgdata2=Cutout2D(mbkgdata2,gxycen,cutout_size,wcs=wcs2).data
    # print('WARNING WARNING: UNCOMMENT ABOVE LINE')

    # maskreg_comb=np.logical_and(mask_reg[0],mask_reg[1]).astype(np.int32)
    # masked_data_comb=mbkgdata2*(1-maskreg_comb)
    # fits.writeto("../OUTPUT/"+gxyid+"/i_masked_reg_combined.fits",masked_data_comb, header=res_mbkg_i[res_ext].header,overwrite=True)
    
    # mask_reg_i=mask_reg[0]
    mask_final=np.logical_or(mask_i,1-annreg).astype(np.int32)
    masked_data_i=mbkgdata2*(1-mask_final)
    fits.writeto(f"../OUTPUT/{gxyid}/i_masked_stompout_ann{ann}.fits",masked_data_i.astype(np.float32), header=res_mbkg_i[res_ext].header, overwrite=True)
    masked_data_i=np.ma.masked_equal(masked_data_i,0)
    mask_i=1-mask_final # THIS IS THE MASK VARIABLE BEING USED
    print(f"N_PIX={np.sum(mask_i)}")
    # npix=np.sum(mask_i)

    
        # #Combine masks
        # if (gal.read_hsc_mask) :
        #     mask_data=np.logical_or(mask_data,instr_mask).astype(np.int32)
        # maskedpix=np.count_nonzero(mask_data==1)
        # total_size=mask_data.size
        # maskedpc=(maskedpix/total_size)*100
        # print('Total percentage of pixels masked = %3.4f' % (maskedpc))
        # masked_data=image_data*(1-mask_data)
        # masked_data=np.ma.masked_equal(masked_data,0)

    
    
    '''
    ############################################
    # MASK using segmentation map
    seg1=fits.open("../OUTPUT/"+gxyid+"/"+band1+"_seg.fits", do_not_scale_image_data=True)
    seg2=fits.open("../OUTPUT/"+gxyid+"/"+band2+"_seg.fits", do_not_scale_image_data=True)

    from scipy.ndimage import gaussian_filter
    mask=np.logical_and(seg1[0].data,seg2[0].data).astype(np.int32)
    # smask=gaussian_filter(mask, sigma=3)

    from astropy.convolution import Gaussian2DKernel, convolve
    kernel=Gaussian2DKernel(x_stddev=int(2*popt2[2]*2.355))
    smask=convolve(mask,kernel)
    smask=np.where(smask>0.02, 1, 0)
    # print(smask)
    # print(np.median(smask[smask>0]), np.std(smask[smask>0]))
    fits.writeto("../OUTPUT/"+gxyid+"/mask_seg.fits",mask, overwrite=True)
    fits.writeto("../OUTPUT/"+gxyid+"/mask_seg_smooth.fits",smask, overwrite=True)

    masked_data2=resdata2*(1-smask)
    fits.writeto("../OUTPUT/"+gxyid+"/i_masked_seg.fits",masked_data2, overwrite=True)
    #############################################
    '''
    
   
    
    
    ###########################
    # MASKED POWER SPECTRUM

    # masked_data1=resdata1*(1-maskdata1)
    # masked_data1=np.ma.masked_equal(masked_data1,0)
    # cutout1= masked_data1 #Cutout2D(masked_data1,(1800,1430),500,wcs=wcs1)  
    # ps1=np.fft.fft2(cutout1.data)
    # ps1=np.fft.fftshift(ps1)
    # ps1=np.abs(ps1)**2
    
    masked_data2=masked_data_i#resdata2*(1-maskdata2)
    # masked_data2=np.ma.masked_equal(masked_data2,0)    
    # cutout2= Cutout2D(masked_data2,(1800,1430),500,wcs=wcs2)    #masked_data2 #  
    cutout2= masked_data2#Cutout2D(masked_data2,gxycen,cutout_size,wcs=wcs2)    #masked_data2 #  
    # print(cutout2.shape)
    # print("POWSPEC CUTOUT")
    ps2=np.fft.fft2(cutout2.data)
    ps2=np.fft.fftshift(ps2)
    # print(ps2.shape[0]*ps2.shape[1])
    ps2=np.abs(ps2)**2
    # print(np.sum(ps2))
    ps2=ps2/(ps2.shape[0]*ps2.shape[1]) # NORMALISE TO THE TOTAL NUMBER OF PIXELS
    # ps2=ps2/(npix) # NORMALISE TO THE TOTAL NUMBER OF UNMASKED PIXELS
    # print(np.sum(ps2))

    fits.writeto("../OUTPUT/"+gxyid+"/i_powspec_ann{ann}.fits",ps2.astype(np.float32), overwrite=True)
    
    # print(np.median(ps1),np.std(ps1),np.min(ps1))
    # print(np.median(ps2),np.std(ps2),np.min(ps2))
    
    
    fig = plt.figure(figsize=(18, 12))
    font = {'family' : 'serif',
            'weight' : 'normal',
            'size'   : 22}
    
    plt.rc('font', **font)
    fig.subplots_adjust(wspace=0.05,hspace=0.65, left=0.1, right=0.95,
                        bottom=0.15, top=0.9)
    
    cmap='viridis'   
    # plt.subplot(221)
    # vmin=np.min(ps1)
    # vmax=0.95*np.max(ps1)
    # plt.imshow(ps1, origin='lower', cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax))

    # plt.subplot(222)
    # vmin=0.001
    # vmax=0.95*np.max(resdata1)
    # plt.imshow(resdata1, origin='lower', cmap=cmap,norm=LogNorm(vmin=vmin, vmax=vmax))
    
    plt.subplot(232)
    vmin=np.min(ps2)
    vmax=0.95*np.max(ps2)
    # print("VMIN VMAX")
    # print(vmin,vmax)
    plt.imshow(ps2, origin='lower',cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax))
    
    plt.subplot(231)
    vmin=0.001
    vmax=0.95*np.max(cutout2.data)
    # print("VMIN VMAX")
    # print(vmin,vmax)
    plt.imshow(cutout2.data, origin='lower',cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax))
    # #plt.show(block=False)
    # ##plt.clf()
    
    print("AZIM AVG POWER SPECTRUM")
    # kmed1,flux1=utils.azimuthal_avg(ps1)
    kmed2,flux2=utils.azimuthal_avg(ps2)
    
    # np.savez(f"../OUTPUT/{gxyid}/{gxy_name1}_maskedazim.npz",kmed1=kmed1,flux1=flux1)
    # np.savez(f"../OUTPUT/{gxyid}/{gxy_name2}_maskedazim.npz",kmed2=kmed2,flux2=flux2)
    np.savetxt(f"../OUTPUT/{gxyid}/{gxy_name2}_maskedazim_ann{ann}.txt", np.c_[kmed2,flux2[:,0]])
    

    # fig = plt.figure(figsize=(12,7))
    # font = {'family' : 'serif',
    #         'weight' : 'normal',
    #         'size'   : 22}
    
    # plt.rc('font', **font)
    # fig.subplots_adjust(wspace=0.3,hspace=0.65,left=0.1, right=0.99,
    #                     bottom=0.15, top=0.9)
    # # plt.subplot(221)
    # plt.plot(kmed1,np.log10(flux1[:,0]),color='indigo',label=gxy_name1)
    # plt.xlabel('$k$')
    # plt.ylabel('$log(P)$')
    # plt.title("IC 0745: $g$",fontsize='small')
    # # plt.legend()
    
    plt.subplot(233)
    plt.plot(kmed2,np.log10(flux2[:,0]),color='indigo',label=gxy_name2)
    plt.xlabel('$k$')
    plt.ylabel('$log(P)$')
    plt.title(f"{gxyid} $i$: $P(k)$",fontsize='small')
    # # plt.legend()

    
    
    # # PSF POWER SPECTRUM
    # psf1_stitched=np.zeros(cutout1.shape)
    # psf1_stitched[int(np.floor(cutout1.shape[0]/2.-epsf1.data.shape[0])):int(np.ceil(cutout1.shape[0]/2.+epsf1.data.shape[0])),int(np.floor(cutout1.shape[1]/2.-epsf1.data.shape[1])):int(np.ceil(cutout1.shape[1]/2.-epsf1.data.shape[1]))]=epsf1.data
    
    #################################
    # epsf_2=fits.open(f'../OUTPUT/{gxyid}/{gxy_name2}_psf.fits', do_not_scale_image_data=True)
    # epsf_1=fits.open(f'../OUTPUT/{gxyid}/{gxy_name1}_psf.fits', do_not_scale_image_data=True)
    #################################
    
    
    # # psf1_stitched=np.pad(epsf1.data,((int(np.floor(cutout1.shape[0]/2.-epsf1.data.shape[0])),int(np.ceil(cutout1.shape[0]/2.+epsf1.data.shape[0]))),(int(np.floor(cutout1.shape[1]/2.-epsf1.data.shape[1])),int(np.ceil(cutout1.shape[1]/2.-epsf1.data.shape[1])))),'constant')
    # psf1_stitched=np.pad(epsf_1[0].data,((int(np.floor(cutout1.shape[0]/2.-epsf_1[0].data.shape[0])),int(np.ceil(cutout1.shape[0]/2.+epsf_1[0].data.shape[0]))),(int(np.floor(cutout1.shape[1]/2.-epsf_1[0].data.shape[1])),int(np.ceil(cutout1.shape[1]/2.-epsf_1[0].data.shape[1])))),'constant')
    # print(cutout1.shape, psf1_stitched.shape)
    # ps_psf1=np.fft.fft2(psf1_stitched)
    # ps_psf1=np.fft.fftshift(ps_psf1)
    # ps_psf1=np.abs(ps_psf1)**2
    # ps_path="../OUTPUT/"+gxyid+'/'+gxy_name1+"_psf_ps.fits"
    # fits.writeto(ps_path, ps_psf1, overwrite=True)
    
    
    # psf2_stitched=np.pad(epsf2.data,((int(np.floor(cutout2.shape[0]/2.-epsf2.data.shape[0])),int(np.ceil(cutout2.shape[0]/2.+epsf2.data.shape[0]))),(int(np.floor(cutout2.shape[1]/2.-epsf2.data.shape[1])),int(np.ceil(cutout2.shape[1]/2.-epsf2.data.shape[1])))),'constant')
    pad_ax0=(cutout2.shape[0]-epsf_2[0].data.shape[0])/2.
    pad_ax1=(cutout2.shape[1]-epsf_2[0].data.shape[1])/2.
    # print(pad_ax0,pad_ax1)
    
    # psf2_stitched=np.pad(epsf_2[0].data,((int(np.floor(cutout2.shape[0]/2.-epsf_2[0].data.shape[0])),int(np.ceil(cutout2.shape[0]/2.+epsf_2[0].data.shape[0]))),(int(np.floor(cutout2.shape[1]/2.-epsf_2[0].data.shape[1])),int(np.ceil(cutout2.shape[1]/2.-epsf_2[0].data.shape[1])))),'constant')
    psf2_stitched=np.pad(epsf_2[0].data,((int(pad_ax0),int(pad_ax1)),(int(pad_ax0),int(pad_ax1))),'constant')
    if (round(np.sum(psf2_stitched))!=1): psf2_stitched=psf2_stitched/np.sum(psf2_stitched)
    # print("&&&&&&&&&&&& ECCOMI***************")
    # print(np.sum(psf2_stitched))
    # print(psf2_stitched.shape)
    # print(epsf_2[0].data.shape, cutout2.shape, psf2_stitched.shape)
    # print((int(np.floor(cutout2.shape[0]/2.-epsf_2[0].data.shape[0])),int(np.ceil(cutout2.shape[0]/2.+epsf_2[0].data.shape[0]))),(int(np.floor(cutout2.shape[1]/2.-epsf_2[0].data.shape[1])),int(np.ceil(cutout2.shape[1]/2.-epsf_2[0].data.shape[1]))))
    # print((np.floor(cutout2.shape[0]/2.-epsf_2[0].data.shape[0])),(np.ceil(cutout2.shape[0]/2.+epsf_2[0].data.shape[0])))
    # print((np.floor(cutout2.shape[1]/2.-epsf_2[0].data.shape[1])),(np.ceil(cutout2.shape[1]/2.-epsf_2[0].data.shape[1])))    
    ps_psf2=np.fft.fft2(psf2_stitched)
    ps_psf2=np.fft.fftshift(ps_psf2)
    # print(np.sum(ps_psf2))
    ps_psf2=np.abs(ps_psf2)**2
    # print(np.sum(ps_psf2))
    # print("LOOK ABOVE")
    ps_psf2=ps_psf2/(ps_psf2.shape[0]*ps_psf2.shape[1]) # NORMALISE TO THE TOTAL NUMBER OF PIXELS
    print(f"Total power per pixel in PS of PSF: {np.sqrt(np.sum(ps_psf2))}")
    ps_path="../OUTPUT/"+gxyid+'/'+gxy_name2+"_psf_ps.fits"
    fits.writeto(ps_path, ps_psf2.astype(np.float32), overwrite=True)
    psf_stitch__path="../OUTPUT/"+gxyid+'/'+gxy_name2+"_psf_stitched.fits"
    fits.writeto(psf_stitch__path, psf2_stitched.astype(np.float32), overwrite=True)

    # print(np.median(ps_psf1),np.std(ps_psf1),np.min(ps_psf1))
    # print(np.median(ps_psf2),np.std(ps_psf2),np.min(ps_psf2))

    ### MULTIPLY MASK WITH MODEL, GENERATE POWSPEC OF SQRT OF THIS PRODUCT
    mask_cutout=mask_i#Cutout2D(mask_i, gxycen, cutout2.shape, wcs=wcs2)
    mod_cutout=Cutout2D(moddata2, gxycen, cutout2.shape, wcs=wcs2)
    mod_ann=mod_cutout.data*mask_cutout.data
    # print(np.sum(mask_cutout.data))
    print(f"<g> in annulus={np.sum(mod_ann)/np.sum(mask_cutout.data)}")
    # mod_ann=mod_cutout.data*(annreg)
    # print(f"<g> in annulus={np.median(mod_ann[mod_ann!=0])}")
    # mod_ann=mod_cutout.data*mask_cutout.data
    # print(f"<g> in annulus={np.median(mod_ann[mod_ann!=0])}")
    smaskmod=np.sqrt(mask_cutout.data*mod_cutout.data)
    # print(np.min(mask_cutout.data), np.min(mod_cutout.data))
    fits.writeto("../OUTPUT/"+gxyid+'/'+gxy_name2+"_smaskmod.fits", smaskmod.astype(np.float32), overwrite=True)
    smaskmod=np.ma.masked_equal(smaskmod,0)
    ps_mask=np.fft.fft2(smaskmod)
    # print(" **************** SHAPES******************")
    # print("SMASKMOD ", smaskmod.shape)
    # print("PS_MASK ", ps_mask.shape)
    # ps_mask=np.fft.fft2(mask_cutout.data)
    ps_mask=np.fft.fftshift(ps_mask)
    # print("FFTSHIFT ", ps_mask.shape)
    ps_mask=np.abs(ps_mask)**2
    ps_mask=ps_mask/(ps_mask.shape[0]*ps_mask.shape[1]) # NORMALISE TO THE TOTAL NUMBER OF PIXELS
    ps_path=f"../OUTPUT/{gxyid}/{gxy_name2}_smaskmod_ps_ann{ann}.fits"
    fits.writeto(ps_path, ps_mask.astype(np.float32), overwrite=True)
    
    # CONVOLVE PSF POWSPEC WITH MASK POWSPEC
    print("CONVOLVING PSF PS with MASK PS")
    teststart=time.time()

    # fits.writeto("../OUTPUT/"+gxyid+'/'+gxy_name2+"_testconv.fits", psf2_stitched, overwrite=True)

    # ps_psf_conv=np.fft.fft2(smaskmod*psf2_stitched)
    # ps_psf_conv=np.fft.fftshift(ps_psf_conv) 
    # ps_psf_conv=np.abs(ps_psf_conv)**2   
    ps_psf_conv=convolve(ps_mask, ps_psf2, mode='same', method='fft')
    ps_psf_path="../OUTPUT/"+gxyid+'/'+gxy_name2+"_ps_psfconv.fits"
    fits.writeto(ps_psf_path, ps_psf_conv.astype(np.float32), overwrite=True)
    # ps_psf_conv=convolve(ps_mask,ps_psf2,normalize_kernel=True)
    teststop=time.time()
    print(f"time taken in above step= {(teststop-teststart)/60.} min")


    plt.subplot(234)
    vmin=0.9*np.min(epsf_2[0].data)
    vmax=0.95*np.max(epsf_2[0].data)
    # print("VMIN VMAX 234")
    # print(vmin<=vmax)
    plt.imshow(epsf_2[0].data, origin='lower', cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax))
    plt.title("EPSF")

    plt.subplot(235)
    vmin=0.9*np.min(ps_psf_conv)
    vmax=0.95*np.max(ps_psf_conv)
    # print("VMIN VMAX 235")
    # print(vmin<=vmax)
    plt.imshow(ps_psf_conv, origin='lower',cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax))
    plt.title("PS PSF conv")


    print("AZIM AVG PSF CONV POWER SPECTRUM")
    # kmed_psf1,flux_psf1=utils.azimuthal_avg(ps_psf1)
    kmed_psf2,flux_psf2=utils.azimuthal_avg(ps_psf_conv)
    # np.savez(f"../OUTPUT/{gxyid}/{gxy_name1}_psfazim.npz",kmed_psf1=kmed_psf1,flux_psf1=flux_psf1)
    # np.savez(f"../OUTPUT/{gxyid}/{gxy_name2}_psfconvazim.npz",kmed_psf2=kmed_psf2,flux_psf2=flux_psf2)
    np.savetxt(f"../OUTPUT/{gxyid}/{gxy_name2}_psfconvazim_ann{ann}.txt", np.c_[kmed_psf2,flux_psf2[:,0]])
    
    plt.subplot(236)
    plt.plot(kmed_psf2,np.log10(flux_psf2[:,0]),color='darkred',label='PSF PS '+gxy_name2)
    plt.xlabel('$k$')
    plt.ylabel('$log(P)$')
    plt.title(f"{gxyid} $i$: $E(k)$",fontsize='small')
    # plt.legend()
    
    # plt.yscale('log')
    #plt.ylim(-5,1)
    # plt.grid(True)
    
    
    plt.savefig(f'../OUTPUT/plots/{gxyid}/{gxy_name2}_powspec_ann{ann}.jpeg',dpi=300)
    #plt.show(block=False)
    plt.clf()
    plt.close()
    

    fig = plt.figure(figsize=(18, 12))
    font = {'family' : 'serif',
            'weight' : 'normal',
            'size'   : 22}
    
    plt.rc('font', **font)
    fig.subplots_adjust(wspace=0.05,hspace=0.65, left=0.1, right=0.95,
                        bottom=0.15, top=0.9)
    
    cmap='viridis'   
    
    plt.subplot(121)
    vmin=0.9*np.min(epsf_2[0].data)
    vmax=0.95*np.max(epsf_2[0].data)
    # print("VMIN VMAX 121")
    # print(vmin<=vmax)
    plt.imshow(epsf_2[0].data, origin='lower', cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax))
    # plt.xlabel('$k$')
    # plt.ylabel('$log(P)$')
    plt.title(f"PSF",fontsize='small')
    # plt.legend()
    plt.subplot(122)
    vmin=0.9*np.min(psf2_stitched)
    vmax=0.95*np.max(psf2_stitched)
    # print("VMIN VMAX 122")
    # print(vmin<=vmax)
    plt.imshow(psf2_stitched, origin='lower', cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax))
    # plt.xlabel('$k$')
    # plt.ylabel('$log(P)$')
    plt.title(f"PSF stitched",fontsize='small')
    # plt.legend()
    
    # plt.yscale('log')
    #plt.ylim(-5,1)
    # plt.grid(True)
    
    
    plt.savefig(f'../OUTPUT/plots/{gxyid}/{gxy_name2}_psfstitch_ann{ann}.jpeg',dpi=300)
    #plt.show(block=False)
    plt.clf()
    plt.close()
    
    ############# AZIMUTHAL AVERAGE OF MODEL AND RESIDUALS
    # r_res1, f_res1=utils.azimuthal_avg(resdata1)
    # r_res2, f_res2=utils.azimuthal_avg(resdata2)
    
    # mod1=fits.open(modimg1, do_not_scale_image_data=True)
    # moddata1=mod1[res_ext].data
    # modhdr1=mod1[res_ext].header
    # wcs1=WCS(modhdr1)

    # mod2=fits.open(modimg2, do_not_scale_image_data=True)
    # moddata2=mod2[res_ext].data
    # modhdr2=mod1[res_ext].header
    # wcs2=WCS(modhdr2)
   
    # r_mod1, f_mod1=utils.azimuthal_avg(moddata1)
    # r_mod2, f_mod2=utils.azimuthal_avg(moddata2)
    
    # plt.plot(r_mod1,f_mod1/f_res1)
    # plt.plot(r_mod2,f_mod2/f_res2)
    #####################################
    
    ######## READ ISOLIST AND CALCULATE BACKGROUND & SNR
    
    # gal1=fits.open(galimg1, do_not_scale_image_data=True)
    # galdata1=gal1[res_ext].data
    # gal2=fits.open(galimg2, do_not_scale_image_data=True)
    # galdata2=gal2[res_ext].data

    # med_corner1=utils.corner_bkg(galdata1)
    # med_corner2=utils.corner_bkg(galdata2)
    
    # isofilepath1="../OUTPUT/"+gxyid+'/'+gxy_name1+'.iso'
    # isophotes1=Table.read(isofilepath1,format='ascii')
    
    # isofilepath2="../OUTPUT/"+gxyid+'/'+gxy_name2+'.iso'
    # isophotes1=Table.read(isofilepath2,format='ascii')
    
    #### HERE #####################

    # #plt.show(block=False)
    
    
    #############################################
    # FITTING AND MATCHING
    # print(kmed2)
    # print(kmed_psf2)
    k=np.array(kmed2)
    # print(len(k))
    P_k=np.array(flux2[:,0])
    # print(len(P_k))
    E_k=np.array(flux_psf2[:,0])
    cond_k=(k<450) #(int(cutout2.shape[0]/2.5)))
    print("&&&&&&&&&&&&&&&&&&&")
    # print(cond_k)
    # cond_k=(k>50) & (k<=256)
    # P_k=P_k[(cond_k)]
    # E_k=E_k[(cond_k)]
    # k=k[(cond_k)]

    np.savetxt(f'../OUTPUT/{gxyid}/{gxy_name2}_k_Pk_Ek_ann{ann}.txt', np.c_[k,P_k,E_k])
    
    P_0,P_1=utils.sbf_ps_fit(k[(cond_k)],E_k[(cond_k)],P_k[(cond_k)])
    # print(len(k[cond_k]))
    # print("YOU ARE HERE %%%%%%%%%%%%")
    cond_k=(k<(450/2))#(int(cutout2.shape[0]/(2.5*3))))
    clean_nan=(np.isfinite(P_0))
    P_0=np.round(P_0,3)

    k_clean=k[(cond_k)][(clean_nan)]
    P_1=P_1[(clean_nan)]
    P_0=P_0[(clean_nan)]

    # print("%%%%%%%%%%%%%%%%%%%%%")
    # print(len(P_0))
    # P_0,P_1=utils.sbf_ps_fit(k,E_k,P_k)
    # print(P_0,P_1)

    # arr_half=P_0#[(cond_k)]
    peaks, _=find_peaks(P_0)
    # print(P_0[peaks.astype(int)])

    P_0_final=mode(P_0[peaks.astype(int)], nan_policy='omit',keepdims=False).mode
    # print(peaks)
    # print(P_0[peaks.astype(int)])
    # print(P_0_final.mode)
    P_1_final=np.median(P_1[(P_0==P_0_final)])

    fig = plt.figure(figsize=(7,10))
    font = {'family' : 'serif',
            'weight' : 'normal',
            'size'   : 22}
    
    plt.rc('font', **font)
    # fig.subplots_adjust(wspace=0.05,hspace=0.35, left=0.2, right=0.95,
    #                     bottom=0.15, top=0.9)
    fig.subplots_adjust(wspace=0.35,hspace=0.12, left=0.15, right=0.95,
                        bottom=0.15, top=0.9)

    # for p_0 in P_0:

    plt.subplot(211)
    # plt.plot(k,np.log10(utils.sbf_ps(E_k,*popt_ps)),color='indigo',label='P_k '+gxy_name2)
    # plt.plot(k,np.log10(utils.sbf_ps(E_k,np.median(P_0),np.median(P_1))),color='indigo',label='P_k '+gxy_name2)
    # plt.plot(k,(P_k),color='indigo',label='P_k '+gxy_name2)
    # plt.plot(k,(utils.sbf_ps(E_k,P_0_final,P_1_final)),color='darkred',label='P_k '+gxy_name2)
    plt.plot(k,np.log10(P_k),color='indigo',label='P_k '+gxy_name2)
    plt.plot(k,np.log10(utils.sbf_ps(E_k,P_0_final,P_1_final)),color='darkred',label='P_k '+gxy_name2)
    plt.axvspan(int(cutout2.shape[0]/2),np.max(k),color='lightgrey',alpha=0.4)
    plt.xlabel('$k$')
    plt.ylabel('$P$')
    # plt.ylabel('$log(P)$')
    # plt.xlim(0,400) 
    plt.title(f"{gxyid} $i$: $P(k)=P0*E(k)+P1$",fontsize='small')
        # plt.legend()
    
    plt.subplot(212)
    plt.plot((P_0[P_0>0]),".",color='indigo')#,label='P_k '+gxy_name2)
    plt.plot(peaks,(P_0[peaks.astype(int)]),"x",color="gold")
    # plt.plot(np.log10(P_0[P_0>0]),".",color='indigo')#,label='P_k '+gxy_name2)
    # plt.plot(peaks,np.log10(P_0[peaks.astype(int)]),"x",color="gold")
    plt.axhline((P_0_final),color="darkred",linestyle="dotted", label='$P_0$')
    # plt.axhline(np.log10(P_0_final),color="darkred",linestyle="dotted", label='$P_0$')
    plt.legend()
    # plt.plot(k,np.log10(utils.sbf_ps(E_k,*popt_ps)),color='indigo',label='P_k '+gxy_name2)
    # plt.plot(k,np.log10(utils.sbf_ps(E_k,P_0,P_1)),color='indigo',label='P_k '+gxy_name2)
    plt.axvspan(int(cutout2.shape[0]/2),np.max(k),color='lightgrey',alpha=0.4)
    # plt.ylim(-10,)
    # plt.xlim(0,400) 
    plt.xlabel('$k_{start}$')
    plt.ylabel('$P_0$')
    # plt.ylabel('$log(P_0)$')
    # plt.xlim(0,np.max(k)) 
    
    # plt.yscale('log')
    #plt.ylim(-5,1)
    # plt.grid(True)
    
    
    plt.savefig(f'../OUTPUT/plots/{gxyid}/{gxy_name2}_powspecmatch_epsf_ann{ann}.jpeg',dpi=300)
    #plt.show(block=False)
    plt.clf()
    plt.close()

    '''
    ############### FIT BY HAND

    # interval_p1=(k>=500) & (k<=600)
    interval_p1=(k>=140) & (k<=170)
    p1=np.median(P_k[(interval_p1)])
    print(p1)

    # interval=(k>=200) & (k<=400)
    # interval=(k>=100) & (k<=200)
    interval=(k>=50) & (k<=120)

    alpha=np.median(P_k[(interval)]/E_k[(interval)])
    print(alpha)

    fig = plt.figure(figsize=(7, 7))
    font = {'family' : 'serif',
            'weight' : 'normal',
            'size'   : 22}
    
    plt.rc('font', **font)
    fig.subplots_adjust(wspace=0.05,hspace=0.65, left=0.15, right=0.95,
                        bottom=0.15, top=0.9)

    # plt.scatter(k,np.log10(P_k),color='plum',label='P_k '+gxy_name2)
    plt.scatter(k,np.log10(P_k),color='darkred',label='P_k '+gxy_name2)
    # plt.plot(k,np.log10(utils.sbf_ps(E_k,*popt_ps)),color='indigo',label='P_k '+gxy_name2)
    plt.axhline(np.log10(p1), color='indigo')
    plt.plot(k, np.log10(alpha*E_k+p1), color='indigo')
    plt.xlabel('$k$')
    plt.ylabel('$log(P)$')
    plt.title(f"{gxyid} $i$: $P(k)=P0*E(k)+P1$",fontsize='small')
    # plt.legend()
    
    # plt.yscale('log')
    #plt.ylim(-5,1)
    # plt.grid(True)
    
    
    plt.savefig(f'../OUTPUT/plots/{gxyid}/{gxy_name2}_powspecmatch_test.jpeg',dpi=300)
    #plt.show(block=False)
    plt.clf()
    '''


    ## https://ui.adsabs.harvard.edu/abs/2007ApJ...668..130C/abstract
    # CALCULATE m_i from P_0_final

    # P_r=0.051#P_0_final*0.05
    # m_i = -2.5*np.log10(P_0_final-P_r)+magzp2-ext_corr2#+2.5*np.log10(t_exp)

    print(f'P_0={P_0_final}')
    print(f'P_1={P_1_final}')
    # print(f'P_r={P_r}')
    # print(f'm_i={m_i}')

    return P_0_final,P_1_final



def run_part6_starpsf(gxyid, band1,band2, magzp1,magzp2,fwhm1,fwhm2, 
                in_rad, out_rad, ext_corr2, cutout_size, psf_arr, r):

    
    
    '''

    Analyse power spectrum of masked image, psf
    
    
    '''
    

    ann=r+1

    gxyid=gal.gxy
    gxyra=gal.gxyra
    gxydec=gal.gxydec
    gxycen=SkyCoord(gxyra.item(), gxydec.item(), unit=(u.deg, u.deg))

    # fwhm0= gal.fwhm0
    plate_scale=gal.plate_scale
    img_ext=gal.img_ext #Image extension
    wht_ext=gal.wht_ext #Weight extension
    res_ext=0
    gxyid=gxyid.upper()
    
    #Read the catalogs in the 2 bands of interest
    gxy_name1=gxyid+'_'+band1
    cat1path="../OUTPUT/"+gxyid+'/'+gxy_name1+'_matchedcorr.cat' #Matched corrected catalog
    resimg1="../OUTPUT/"+gxyid+'/'+gxy_name1+'_res.fits' #residual image file path
    modimg1="../OUTPUT/"+gxyid+'/'+gxy_name1+'_mod.fits' #model image file path
    galimg1="../OUTPUT/"+gxyid+'/'+gxy_name1+'.fits' #galaxy image
    maskimg1="../OUTPUT/"+gxyid+'/'+gxy_name1+'_mask.fits' #mask image  
    whtimg1="../OUTPUT/"+gxyid+"/"+gxy_name1+'_revwht.fits'
    t1 = Table.read(cat1path, format='ascii.commented_header')

    gxy_name2=gxyid+'_'+band2
    # cat2path="../OUTPUT/"+gxyid+'/'+band2+'_matchedcorr.cat'
    resimg2="../OUTPUT/"+gxyid+'/'+gxy_name2+'_res.fits' #residual image file path
    modimg2="../OUTPUT/"+gxyid+'/'+gxy_name2+'_mod.fits' #residual image file path
    galimg2="../OUTPUT/"+gxyid+'/'+gxy_name2+'.fits' #galaxy image
    # maskimg2="../OUTPUT/"+gxyid+'/'+gxy_name2+'_mask.fits' #mask image
    whtimg2="../OUTPUT/"+gxyid+"/"+gxy_name2+'_revwht.fits'
    # t2 = Table.read(cat2path, format='ascii.commented_header')

    
    
    res1=fits.open(resimg1, do_not_scale_image_data=True)
    # res2.info()
    mod1=fits.open(modimg1, do_not_scale_image_data=True)
    # mod2.info()
    moddata1=mod1[res_ext].data
    resdata1=res1[res_ext].data
    reshdr1=res1[res_ext].header
    wcs1=WCS(reshdr1)
    wht1=fits.open(whtimg1, do_not_scale_image_data=True)
    # wht2.info()
    whtdata1=wht1[wht_ext].data
    
    
    # img2=fits.open(galimg2, do_not_scale_image_data=True)
    res2=fits.open(resimg2, do_not_scale_image_data=True)
    # res2.info()
    mod2=fits.open(modimg2, do_not_scale_image_data=True)
    # mod2.info()
    moddata2=mod2[res_ext].data
    resdata2=res2[res_ext].data
    reshdr2=res2[res_ext].header
    wcs2=WCS(reshdr2)
    wht2=fits.open(whtimg2, do_not_scale_image_data=True)
    # wht2.info()
    whtdata2=wht2[wht_ext].data

    
    if not os.path.exists("../OUTPUT/plots/"+gxyid):
        os.makedirs("../OUTPUT/plots/"+gxyid)

    # tmask2 = Table.read("../OUTPUT/"+gxyid+"/"+gxy_name2+"_p_r.cat", format='ascii.commented_header')
    # mask_stompout=fits.open("../OUTPUT/"+gxyid+"/"+gxy_name2+"_stompout.fits")
    # mask_i=1-mask_stompout[0].data  # FOR MIK'S MASK
    mask_stompout_g=fits.open("../OUTPUT/"+gxyid+"/"+gxy_name1+"_stompout.fits")
    mask_stompout=fits.open("../OUTPUT/"+gxyid+"/"+gxy_name2+"_stompout.fits")
    
    mask_g=mask_stompout[0].data
    mask_i=mask_stompout[0].data  # FOR NH'S MASK
    # print('mask_i',mask_i)
    mwcs1=WCS(mask_stompout_g[0].header)
    mwcs2=WCS(mask_stompout[0].header)

    mask_scale=4.
    # t_arr=[tmask2]
    bands=np.array(['i'])
    wcs=np.array([wcs2])
    resdata=np.array([resdata2])
    reshdr=[reshdr2]
    mask_reg=[np.empty(resdata2.shape)]
    # radii=[rad2]

    #Creating the annulus region
    
    #i
    gxycenpix = regions.PixCoord.from_sky(gxycen, mwcs2)#mwcs2.wcs_world2pix(gxyra, gxydec, 1) # Get the galaxy center in pixels
    annreg_in=CirclePixelRegion(gxycenpix, in_rad)
    annreg_in=(annreg_in.to_mask(mode='center')).to_image(mask_i.shape)
    annreg_out=CirclePixelRegion(gxycenpix, out_rad)
    annreg_out=(annreg_out.to_mask(mode='center')).to_image(mask_i.shape)
    annreg=annreg_out-annreg_in
    
    #g
    gxycenpix_g = regions.PixCoord.from_sky(gxycen, mwcs1)#mwcs2.wcs_world2pix(gxyra, gxydec, 1) # Get the galaxy center in pixels
    annreg_in_g=CirclePixelRegion(gxycenpix_g, in_rad)
    annreg_in_g=(annreg_in_g.to_mask(mode='center')).to_image(mask_g.shape)
    annreg_out_g=CirclePixelRegion(gxycenpix_g, out_rad)
    annreg_out_g=(annreg_out_g.to_mask(mode='center')).to_image(mask_g.shape)
    


    # res_mbkg_i=fits.open("../OUTPUT/"+gxyid+"/"+gxy_name2+"_mbkg.fits")
    res_mbkg_g=fits.open("../OUTPUT/"+gxyid+"/"+gxy_name1+"_cutout.fits")
    res_mbkg_i=fits.open("../OUTPUT/"+gxyid+"/"+gxy_name2+"_cutout.fits")
    mbkgdata1=res_mbkg_g[res_ext].data
    mbkgdata2=res_mbkg_i[res_ext].data
    
    # mask_reg_i=mask_reg[0]
    
    #Masking the regions: creating the masked image
    
    mask_final=np.logical_or(mask_i,1-annreg).astype(np.int32)
    masked_data_i=mbkgdata2*(1-mask_final)
    fits.writeto(f"../OUTPUT/{gxyid}/i_masked_stompout_ann{ann}.fits",masked_data_i.astype(np.float32), header=res_mbkg_i[res_ext].header, overwrite=True)
    
    mask_final_g=np.logical_or(mask_g,1-annreg).astype(np.int32)
    masked_data_g=mbkgdata1*(1-mask_final_g)
    fits.writeto(f"../OUTPUT/{gxyid}/g_masked_stompout_ann{ann}.fits",masked_data_g.astype(np.float32), header=res_mbkg_i[res_ext].header, overwrite=True)

    # Load the FITS file into DS9
    # ds9 = pyds9.DS9()
    
    # fits_filename = f"../OUTPUT/{gxyid}/i_masked_stompout_ann{ann}.fits"
    # ds9.set(f'file {fits_filename}')
    
    masked_data_g=np.ma.masked_equal(masked_data_g,0)
    masked_data_i=np.ma.masked_equal(masked_data_i,0)
    # print('mask_final',mask_final)
    mask_g=1-mask_final_g # THIS IS THE MASK VARIABLE BEING USED
    mask_i=1-mask_final # THIS IS THE MASK VARIABLE BEING USED
    print(f"N_PIX={np.sum(mask_i)}\n\n")
    # npix=np.sum(mask_i)

    
        # #Combine masks
        # if (gal.read_hsc_mask) :
        #     mask_data=np.logical_or(mask_data,instr_mask).astype(np.int32)
        # maskedpix=np.count_nonzero(mask_data==1)
        # total_size=mask_data.size
        # maskedpc=(maskedpix/total_size)*100
        # print('Total percentage of pixels masked = %3.4f' % (maskedpc))
        # masked_data=image_data*(1-mask_data)
        # masked_data=np.ma.masked_equal(masked_data,0)


    #Power spectrum of the residual (masked) 

    masked_data2=masked_data_i#resdata2*(1-maskdata2)
    masked_data1=masked_data_g
    # masked_data2=np.ma.masked_equal(masked_data2,0)    
    # cutout2= Cutout2D(masked_data2,(1800,1430),500,wcs=wcs2)    #masked_data2 #  
    cutout1= masked_data1
    cutout2= masked_data2#Cutout2D(masked_data2,gxycen,cutout_size,wcs=wcs2)    #masked_data2 #  
    # print(cutout2.shape)
    # print("POWSPEC CUTOUT")
    ps2=np.fft.fft2(cutout2.data)
    ps2=np.fft.fftshift(ps2)
    # print(ps2.shape[0]*ps2.shape[1])
    ps2=np.abs(ps2)**2
    # print(np.sum(ps2))
    ps2=ps2/(ps2.shape[0]*ps2.shape[1]) # NORMALISE TO THE TOTAL NUMBER OF PIXELS
    # ps2=ps2/(npix) # NORMALISE TO THE TOTAL NUMBER OF UNMASKED PIXELS
    # print(np.sum(ps2))

    fits.writeto("../OUTPUT/"+gxyid+"/i_powspec_ann{ann}.fits",ps2.astype(np.float32), overwrite=True)
    
    # print(np.median(ps1),np.std(ps1),np.min(ps1))
    # print(np.median(ps2),np.std(ps2),np.min(ps2))
    
    
    
    print("AZIM AVG POWER SPECTRUM")
    # kmed1,flux1=utils.azimuthal_avg(ps1)
    
    #Azimuthal average power spectrum of the masked image
    
    kmed2,flux2=utils.azimuthal_avg(ps2)
    
    # np.savez(f"../OUTPUT/{gxyid}/{gxy_name1}_maskedazim.npz",kmed1=kmed1,flux1=flux1)
    # np.savez(f"../OUTPUT/{gxyid}/{gxy_name2}_maskedazim.npz",kmed2=kmed2,flux2=flux2)
    np.savetxt(f"../OUTPUT/{gxyid}/{gxy_name2}_maskedazim_ann{ann}.txt", np.c_[kmed2,flux2[:,0]])
    


    # MULTIPLY MASK WITH MODEL, GENERATE POWSPEC OF SQRT OF THIS PRODUCT
    
    #Power spectrum of the mask * model

    mask_cutout=mask_i#Cutout2D(mask_i, gxycen, cutout2.shape, wcs=wcs2)
    mask_cutout_g=mask_g
    
    # print('mask_cutout',mask_cutout)
   
    #Image cutout
    # img_cutout=Cutout2D(galimg2, gxycen, cutout2.shape, wcs=wcs2)
    # img_cutout_g=Cutout2D(moddata1, gxycen, cutout1.shape, wcs=wcs1)
    #Model Cutoutgalimg1
    mod_cutout=Cutout2D(moddata2, gxycen, cutout2.shape, wcs=wcs2)
    mod_cutout_g=Cutout2D(moddata1, gxycen, cutout1.shape, wcs=wcs1)
    
    mod_ann=mod_cutout.data*mask_cutout.data
    mod_ann_g=mod_cutout_g.data*mask_cutout_g.data
    # print(np.sum(mask_cutout.data))
    print(f"<g> in annulus={np.sum(mod_ann)/np.sum(mask_cutout.data)}")
    # mod_ann=mod_cutout.data*(annreg)
    # print(f"<g> in annulus={np.median(mod_ann[mod_ann!=0])}")
    # mod_ann=mod_cutout.data*mask_cutout.data
    # print(f"<g> in annulus={np.median(mod_ann[mod_ann!=0])}")
    
    
    
    smaskmod=abs(np.sqrt(mask_cutout.data*mod_cutout.data))
    smaskmod_g=abs(np.sqrt(mask_cutout_g.data*mod_cutout_g.data))
    
    # print('smaskmod',smaskmod)
    # print(np.min(mask_cutout.data), np.min(mod_cutout.data))
    fits.writeto("../OUTPUT/"+gxyid+'/'+gxy_name2+"_smaskmod.fits", smaskmod.astype(np.float32), overwrite=True)
    fits.writeto("../OUTPUT/"+gxyid+'/'+gxy_name1+"_smaskmod.fits", smaskmod_g.astype(np.float32), overwrite=True)
    
    #Colors
   
    smaskmod_g_col=smaskmod_g**2
    smaskmod_i_col=smaskmod**2
    # print(np.median(mg[smaskmod_g>0]-mi[smaskmod>0]), mad_std(mg[smaskmod_g>0]-mi[smaskmod>0]))
    print('Color Magnitude',-2.5*np.log10(np.sum(smaskmod_g_col)/np.sum(smaskmod_i_col)))
    
    color_gi=smaskmod_g-smaskmod

    fits.writeto("../OUTPUT/"+gxyid+'/'+band1+band2+"_color.fits", color_gi.astype(np.float32), overwrite=True)
    
    
    smaskmod=np.ma.masked_equal(smaskmod,0)
    smaskmod_g=np.ma.masked_equal(smaskmod_g,0)
    # print('smaskmod',smaskmod)
    ps_mask=np.fft.fft2(smaskmod)
    # print('ps_mask',ps_mask)
    
    
    # print(" **************** SHAPES******************")
    # print("SMASKMOD ", smaskmod.shape)
    # print("PS_MASK ", ps_mask.shape)
    # ps_mask=np.fft.fft2(mask_cutout.data)
    ps_mask=np.fft.fftshift(ps_mask)
    # print("FFTSHIFT ", ps_mask.shape)
    ps_mask=np.abs(ps_mask)**2
    ps_mask=ps_mask/(ps_mask.shape[0]*ps_mask.shape[1]) # NORMALISE TO THE TOTAL NUMBER OF PIXELS
    ps_path=f"../OUTPUT/{gxyid}/{gxy_name2}_smaskmod_ps_ann{ann}.fits"
    fits.writeto(ps_path, ps_mask.astype(np.float32), overwrite=True)
    
    
    # # PSF POWER SPECTRUM
    
    psf_arr=psf_arr-1
    p0=np.zeros(psf_arr.shape)
    p1=np.zeros(psf_arr.shape)
    ind=0
    
    for pnum in psf_arr:
        
        #Plot
        
        fig = plt.figure(figsize=(18, 12))
        font = {'family' : 'serif',
                'weight' : 'normal',
                'size'   : 22}
        
        plt.rc('font', **font)
        fig.subplots_adjust(wspace=0.05,hspace=0.65, left=0.1, right=0.95,
                            bottom=0.15, top=0.9)
        
        cmap='viridis'   
        # plt.subplot(221)
        # vmin=np.min(ps1)
        # vmax=0.95*np.max(ps1)
        # plt.imshow(ps1, origin='lower', cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax))

        # plt.subplot(222)
        # vmin=0.001
        # vmax=0.95*np.max(resdata1)
        # plt.imshow(resdata1, origin='lower', cmap=cmap,norm=LogNorm(vmin=vmin, vmax=vmax))
        
        plt.subplot(232)
        vmin=np.min(ps2)
        vmax=0.95*np.max(ps2)
        plt.imshow(ps2, origin='lower',cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax))
        
        plt.subplot(231)
        vmin=0.001
        vmax=0.95*np.max(cutout2.data)
        
        plt.imshow(cutout2.data, origin='lower',cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax))
        plt.subplot(233)
        plt.plot(kmed2,np.log10(flux2[:,0]),color='indigo',label=gxy_name2)
        plt.xlabel('$k$')
        plt.ylabel('$log(P)$')
        plt.title(f"{gxyid} $i$: $P(k)$",fontsize='small')
        plt.show()
        # # plt.legend()
    
    
        
        #Power spectrum of PSF
    
        epsf_2=fits.open(f"../OUTPUT/{gxyid}/{gxy_name2}_psfstar_{pnum}.fits", do_not_scale_image_data=True)
        epsf2_data=epsf_2[0].data


        pad_ax0=(cutout2.shape[0]-epsf_2[0].data.shape[0])/2.
        pad_ax1=(cutout2.shape[1]-epsf_2[0].data.shape[1])/2.
        # print(pad_ax0,pad_ax1)
        
        # psf2_stitched=np.pad(epsf_2[0].data,((int(np.floor(cutout2.shape[0]/2.-epsf_2[0].data.shape[0])),int(np.ceil(cutout2.shape[0]/2.+epsf_2[0].data.shape[0]))),(int(np.floor(cutout2.shape[1]/2.-epsf_2[0].data.shape[1])),int(np.ceil(cutout2.shape[1]/2.-epsf_2[0].data.shape[1])))),'constant')
        psf2_stitched=np.pad(epsf_2[0].data,((int(pad_ax0),int(pad_ax1)),(int(pad_ax0),int(pad_ax1))),'constant')
        if (round(np.sum(psf2_stitched))!=1): psf2_stitched=psf2_stitched/np.sum(psf2_stitched)
        ps_psf2=np.fft.fft2(psf2_stitched)
        ps_psf2=np.fft.fftshift(ps_psf2)
        # print(np.sum(ps_psf2))
        ps_psf2=np.abs(ps_psf2)**2
        # print(np.sum(ps_psf2))
        ps_psf2=ps_psf2/(ps_psf2.shape[0]*ps_psf2.shape[1]) # NORMALISE TO THE TOTAL NUMBER OF PIXELS
        
        print(f"Total power per pixel in PS of PSF: {np.sqrt(np.sum(ps_psf2))}\n\n")
        
        ps_path=f"../OUTPUT/{gxyid}/{gxy_name2}_psf_ps_star{pnum}.fits"
        fits.writeto(ps_path, ps_psf2.astype(np.float32), overwrite=True)
        psf_stitch__path=f"../OUTPUT/{gxyid}/{gxy_name2}_psf_stitched_star{pnum}.fits"
        fits.writeto(psf_stitch__path, psf2_stitched.astype(np.float32), overwrite=True)

        # CONVOLVE PSF POWSPEC WITH MASK POWSPEC (E(k))
        print("CONVOLVING PSF PS with MASK PS\n\n")
        teststart=time.time()
        
        
        ps_psf_conv=convolve(ps_mask, ps_psf2, mode='same', method='fft')
        # print('ps_psf_conv',ps_psf_conv)
        ps_psf_path=f"../OUTPUT/{gxyid}/{gxy_name2}_ps_psfconv_star{pnum}.fits"
        fits.writeto(ps_psf_path, ps_psf_conv.astype(np.float32), overwrite=True)
        # ps_psf_conv=convolve(ps_mask,ps_psf2,normalize_kernel=True)
        teststop=time.time()
        print(f"time taken in above step= {(teststop-teststart)/60.} min")

        #Plot 
        
        plt.subplot(234)
        vmin=0.9*np.min(epsf_2[0].data)
        vmax=0.95*np.max(epsf_2[0].data)
        print("VMIN VMAX 234")
        print(vmin<=vmax)
        plt.imshow(epsf_2[0].data, origin='lower', cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax))
        plt.title("EPSF")

        plt.subplot(235)
        vmin=0.9*np.min(ps_psf_conv)
        vmax=0.95*np.max(ps_psf_conv)
        # print("VMIN VMAX 235")
        # print(vmin<=vmax)
        plt.imshow(ps_psf_conv, origin='lower',cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax))
        plt.title("PS PSF conv")


        print("AZIM AVG PSF CONV POWER SPECTRUM\n\n")
        # kmed_psf1,flux_psf1=utils.azimuthal_avg(ps_psf1)
        kmed_psf2,flux_psf2=utils.azimuthal_avg(ps_psf_conv)
        # np.savez(f"../OUTPUT/{gxyid}/{gxy_name1}_psfazim.npz",kmed_psf1=kmed_psf1,flux_psf1=flux_psf1)
        # np.savez(f"../OUTPUT/{gxyid}/{gxy_name2}_psfconvazim.npz",kmed_psf2=kmed_psf2,flux_psf2=flux_psf2)
        np.savetxt(f"../OUTPUT/{gxyid}/{gxy_name2}_psfconvazim_ann{ann}_star{pnum}.txt", np.c_[kmed_psf2,flux_psf2[:,0]])
        
        
        #Plot
        
        plt.subplot(236)
        plt.plot(kmed_psf2,np.log10(flux_psf2[:,0]),color='darkred',label='PSF PS '+gxy_name2)
        plt.xlabel('$k$')
        plt.ylabel('$log(P)$')
        plt.title(f"{gxyid} $i$: $E(k)$",fontsize='small')
        # plt.legend()
        
        # plt.yscale('log')
        #plt.ylim(-5,1)
        # plt.grid(True)
        
        
        # plt.savefig(f'../OUTPUT/plots/{gxyid}/{gxy_name2}_powspec_ann{ann}_star{pnum}.jpeg',dpi=300)
        #plt.show(block=False)
        plt.show()
        plt.clf()
        plt.close()
        
        
        
        #Plot stitched psf
        
        fig = plt.figure(figsize=(18, 12))
        font = {'family' : 'serif',
                'weight' : 'normal',
                'size'   : 22}
        
        plt.rc('font', **font)
        fig.subplots_adjust(wspace=0.05,hspace=0.65, left=0.1, right=0.95,
                            bottom=0.15, top=0.9)
        
        cmap='viridis'   
        
        plt.subplot(121)
        vmin=0.9*np.min(epsf_2[0].data)
        vmax=0.95*np.max(epsf_2[0].data)
        # print("VMIN VMAX 121")
        # print(vmin<=vmax)
        plt.imshow(epsf_2[0].data, origin='lower', cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax))
        # plt.xlabel('$k$')
        # plt.ylabel('$log(P)$')
        plt.title(f"PSF",fontsize='small')
        # plt.legend()
        plt.subplot(122)
        vmin=0.9*np.min(psf2_stitched)
        vmax=0.95*np.max(psf2_stitched)
        # print("VMIN VMAX 122")
        # print(vmin<=vmax)
        plt.imshow(psf2_stitched, origin='lower', cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax))
        # plt.xlabel('$k$')
        # plt.ylabel('$log(P)$')
        plt.title(f"PSF stitched",fontsize='small')
        plt.show()
        # plt.legend()
        
        # plt.yscale('log')
        #plt.ylim(-5,1)
        # plt.grid(True)
        
        
        plt.savefig(f'../OUTPUT/plots/{gxyid}/{gxy_name2}_psfstitch_ann{ann}_star{pnum}.jpeg',dpi=300)
        #plt.show(block=False)
        plt.clf()
        plt.close()

        
        
        
        # Fit of the power spectrum
        
        # print(kmed2)
        # print(kmed_psf2)
        k=np.array(kmed2)
        
        print('max k',max(k),'\n\n')
        #print(len(k))
        P_k=np.array(flux2[:,0])
        # print(P_k)
        # print(len(P_k))
        E_k=np.array(flux_psf2[:,0])
        
        cond_k=(k<450) #(int(cutout2.shape[0]/2.5)))
        print('len(E_k)',len(E_k), len(E_k[(cond_k)]))
        print("&&&&&&&&&&&&&&&&&&&")
        # print(cond_k)
        # cond_k=(k>50) & (k<=256)
        # P_k=P_k[(cond_k)]
        # E_k=E_k[(cond_k)]
        # k=k[(cond_k)]

        np.savetxt(f'../OUTPUT/{gxyid}/{gxy_name2}_k_Pk_Ek_ann{ann}_star{pnum}.txt', np.c_[k,P_k,E_k])
        
    
        
        P_0,P_1=utils.sbf_ps_fit(k[(cond_k)],E_k[(cond_k)],P_k[(cond_k)])
        # print('P0',P_0,'\n\n')
        
        print('len p0',len(P_0))
        # print(len(k[cond_k]))
        # print("YOU ARE HERE %%%%%%%%%%%%")
        ##(int(cutout2.shape[0]/(2.5*3))))
        clean_nan=(np.isfinite(P_0))
        P_0=np.round(P_0,3)

        # k_clean=k[(cond_k)][(clean_nan)]
        P_1=P_1[(clean_nan)]
        P_0=P_0[(clean_nan)]
        # print(P_0)
        moving_median_p0, moving_mad_p0, rms_p0 = utils.moving_median_mad(P_0)
        # print('moving median', moving_mad_p0)
        P_0_gab=moving_median_p0[np.argmin(moving_mad_p0)]
        P_0_gab_rms=moving_median_p0[np.argmin(rms_p0)]

        # arr_half=P_0#[(cond_k)]
        peaks, _=find_peaks(P_0)
        print('len p0',len(P_0),'\n\n')
        # print('p0 and peaks',P_0, P_0[[peaks.astype(int)]],'\n\n')
        # print(P_0[peaks.astype(int)])

        P_0_final=mode(P_0[peaks.astype(int)], nan_policy='omit',keepdims=False).mode
        # print(peaks)
        # print(P_0[peaks.astype(int)])
        # print(P_0_final.mode)
        P_1_final=np.median(P_1[(P_0==P_0_final)])

        fig = plt.figure(figsize=(7,10))
        font = {'family' : 'serif',
                'weight' : 'normal',
                'size'   : 22}
        
        plt.rc('font', **font)
        # fig.subplots_adjust(wspace=0.05,hspace=0.35, left=0.2, right=0.95,
        #                     bottom=0.15, top=0.9)
        fig.subplots_adjust(wspace=0.35,hspace=0.12, left=0.15, right=0.95,
                            bottom=0.15, top=0.9)

        # for p_0 in P_0:

        plt.subplot(211)
        # plt.plot(k,np.log10(utils.sbf_ps(E_k,*popt_ps)),color='indigo',label='P_k '+gxy_name2)
        # plt.plot(k,np.log10(utils.sbf_ps(E_k,np.median(P_0),np.median(P_1))),color='indigo',label='P_k '+gxy_name2)
        # plt.plot(k,(P_k),color='indigo',label='P_k '+gxy_name2)
        # plt.plot(k,(utils.sbf_ps(E_k,P_0_final,P_1_final)),color='darkred',label='P_k '+gxy_name2)
        plt.plot(k,np.log10(P_k),color='indigo',label='P_k '+gxy_name2)
        plt.plot(k,np.log10(utils.sbf_ps(E_k,P_0_final,P_1_final)),color='darkred',label='P_k '+gxy_name2)
        plt.axvspan(int(cutout2.shape[0]/2),np.max(k),color='lightgrey',alpha=0.4)
        plt.axhline((np.log10(P_1_final)),color="darkred",linestyle="dotted", label='$P_0$')
        plt.xlabel('$k$')
        plt.ylabel('$P$')
        # plt.ylabel('$log(P)$')
        # plt.xlim(0,400) 
        plt.title(f"{gxyid} $i$: $P(k)=P0*E(k)+P1$",fontsize='small')
            # plt.legend()
        
        plt.subplot(212)
        plt.plot((P_0[P_0>0]),".",color='indigo')#,label='P_k '+gxy_name2)
        plt.plot(peaks,(P_0[peaks.astype(int)]),"x",color="gold")
        # plt.plot(np.log10(P_0[P_0>0]),".",color='indigo')#,label='P_k '+gxy_name2)
        # plt.plot(peaks,np.log10(P_0[peaks.astype(int)]),"x",color="gold")
        plt.axhline((P_0_final),color="darkred",linestyle="dotted", label='$P_0$ Peaks')
        plt.axhline((P_0_gab),color="darkblue",linestyle="dotted", label='$P_0$ MAD')
        plt.axhline((P_0_gab_rms),color="green",linestyle="dotted", label='$P_0$ RMS')
        # plt.axhline(np.log10(P_0_final),color="darkred",linestyle="dotted", label='$P_0$')
        plt.legend()
        # plt.plot(k,np.log10(utils.sbf_ps(E_k,*popt_ps)),color='indigo',label='P_k '+gxy_name2)
        # plt.plot(k,np.log10(utils.sbf_ps(E_k,P_0,P_1)),color='indigo',label='P_k '+gxy_name2)
        plt.axvspan(int(cutout2.shape[0]/2),np.max(k),color='lightgrey',alpha=0.4)
        plt.ylim(-1,5)
        # plt.xlim(0,400) 
        plt.xlabel('$k_{start}$')
        plt.ylabel('$P_0$')
        # plt.ylabel('$log(P_0)$')
        # plt.xlim(0,np.max(k)) 
        
        # plt.yscale('log')
        #plt.ylim(-5,1)
        # plt.grid(True)
        
        
        plt.savefig(f'../OUTPUT/plots/{gxyid}/{gxy_name2}_powspecmatch_epsf_ann{ann}_star{pnum}.jpeg',dpi=300)
        plt.show(block=False)
        plt.clf()
        plt.close()

    
        ## https://ui.adsabs.harvard.edu/abs/2007ApJ...668..130C/abstract
        # CALCULATE m_i from P_0_final

        # P_r=0.051#P_0_final*0.05
        # m_i = -2.5*np.log10(P_0_final-P_r)+magzp2-ext_corr2#+2.5*np.log10(t_exp)
        print(f'P_0={P_0_final}')
        print(f'P_1={P_1_final}')
        print('P0 gab is',P_0_gab,P_0_gab_rms)
        # print(f'P_r={P_r}')
        # print(f'm_i={m_i}')
        p0[ind]=P_0_final
        p1[ind]=P_1_final
        ind+=1
    
    print(f"ANNULUS #{ann}")
    print(f"P0= {p0}")
    print(f"P1= {p1}")
    return p0,p1

        # return P_0_final,P_1_final