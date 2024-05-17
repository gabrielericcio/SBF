#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 16:04 2023

@author: nandinihazra
"""
'''
#%%
from IPython import get_ipython
ipython = get_ipython()
ipython.run_line_magic('load_ext', 'autoreload')
ipython.run_line_magic('autoreload', '2')

#%%
'''
# #IMPORT STUFF

import init
import time
import sbf_classes_v1 as sbf
import numpy as np
import sbf_pr as p_r
from astropy.table import Table
from scipy import stats
t0=time.time()
# galaxies = ['ic0745','ngc4753','ngc5813','ngc5831','ngc5839']
# galaxies = ['vcc1025','vcc0828','vcc1146']
# galaxies = ['ngc3308']
galaxies = ['vcc0538']
# galaxies=['vcc0230']
# galaxies=['ngc1404']
# bands=['i']
#bands=['i']
bands=['g','i']
instr=['cfht']

################ Query extin ction
from astroquery.ipac.irsa.irsa_dust import IrsaDust
# IrsaDust.clear_cache()
R_V={'g': 3.237,'i': 1.684} #CCM value = a(lambda)+b(lambda)/R_v
# R_V=[1.684]
magzp={'g':30.,'i': 30.}
fwhm={'g':0.,'i': 0.}
alpha={'g':0.,'i': 0.}

#################################
# RUN PART 1 OF PROCEDURE : MODEL FITTING AND SUBTRACTION

bkg_val={'vcc0828': 2.783, 'vcc1146': 0.163}
rms=1 # Weight file is an RMS
for i in range(len(galaxies)) :
    
    for j in range(len(bands)) :
        galaxy=galaxies[i]
        passband=bands[j]
        gxy=galaxy+'_'+passband
        instrname=instr[0]+'_galinit'
        init_instr= getattr(init,instrname)
        init_instr(galaxy,passband)# instrument specific params
        init.default() # default for all galaxies
        init_spec = getattr(init, gxy) #specific initialisation for each galaxy
        init_spec()
        fwhm[bands[j]]=sbf.run_part1(rms=rms, remove_bkg=True,star_image=False) #IS BUILDING MODEL NOW
        #FWHM IS UPDATED CORRECTLY ONLY WHEN PART 1 IS RUN
        print('FWHM AAAAAA', fwhm)
        sbf.run_part2(alpha[bands[j]])
        
    table=IrsaDust.get_query_table(galaxies[i],section='ebv')
    
    E_BV=table['ext SandF mean']   
    
    A=[(R_V[band]*E_BV.value) for band in bands]
    print(A)
    #####

    

    #VCC0122:
    # fwhm={'g':0.66,'i': 0.59} # KEEP ONLY WHEN NOT RUNNING PART 1
    # fwhm={'g': 0.875, 'i': 0.618}  #vcc0033
    # fwhm={'g': 0.6488900000000001, 'i': 0.6096199999999999} #vcc0230
    # fwhm={'g': 0.8003600000000001, 'i': 0.63393} #vcc0140
    # fwhm={'g': 0.059519999999999997, 'i': 0.059519999999999997} #ngc1404
    # Parameters for the aperture correction
    
    match_scale=1.2 #Multiplicative factor to the mean FWHM of the matched objects, serve to compute the matching radius
    threshold=35    #Maximum magnitude for the sources
    csmin=0.9       #Minimum class star index for compact objects selection
    # mbright=19
    # mfaint=22
    

    #### PARAMS FOR EPSF
    cscut= 0.6#0.95 #both bands: for generating PSF
    mbright= 18#16 HSC
    mfaint= 20#21 #20.5 HSC
    # cilow=0.25 NOT USED
    # cihigh=0.75 NOT USED

    sbf.run_part3(galaxies[i],bands[0],bands[1],A[0],A[1],
                  magzp[bands[0]],magzp[bands[1]],fwhm[bands[0]],fwhm[bands[1]],
                  match_scale,csmin,mfaint,mbright,threshold)

    
    
    sbf.run_part4(galaxies[i],bands[0],bands[1],magzp[bands[0]],magzp[bands[1]],
                    fwhm[bands[0]],fwhm[bands[1]], A[1])
    
    
    # PSF Modelling:
        
    psf_rad_scale=3 # 5 FOR HSC.#4*plate_scale
    nthfactor=20 # nth neighbor selection radius= nthfactor*rad_asec
    rgc_factor=40

    psfsize=64
    oversampling=4
    
    sbf.run_part5(galaxies[i],bands[0],bands[1],A[0],A[1],magzp[bands[0]],
                    magzp[bands[1]],fwhm[bands[0]],fwhm[bands[1]],
                    mfaint,mbright,threshold,cscut,psf_rad_scale,nthfactor,
                    psfsize,oversampling,rgc_factor)
    
    
    # sbf.find_annuli(galaxies[i],bands[0],bands[1])

        
    # MASKING ANNULUS: Inner and outer radii--- in pix
    # VCC1025:
    # ORIGINAL    
    # in_rad=np.array([36,54,90,162,306,36,36])#*0.187#0.00187 # degrees 36 pix   0.0084 #
    # out_rad=np.array([54,90,162,306,594,600,300])#*0.187#0.0156 #0.0159 # cannulus 4 162 to 306 pix # degrees 300 pix : cannulus7 from Mik
    
    #NANDINI VCC1025
    # in_rad=np.array([236,254,290,162,306,36,36])#*0.187#0.00187 # degrees 36 pix   0.0084 #
    # out_rad=np.array([354,390,362,306,594,600,300])#*0.187#0.0156 #0.0159 # cannulus 4 162 to 306 pix # degrees 300 pix : cannulus7 from Mik
    
    #VCC1025
    # in_rad=np.array([36])#*0.187#0.00187 # degrees 36 pix   0.0084 #
    # out_rad=np.array([300])
    
    #VCC0140
    # in_rad=np.array([36])#*0.187#0.00187 # degrees 36 pix   0.0084 #
    # out_rad=np.array([240]) # 27.2 arcsec mik's 7th anullus

    #VCC0033
    
    # in_rad=np.array([36])#*0.187#0.00187 # degrees 36 pix   0.0084 #
    # out_rad=np.array([147]) # 27.2 arcsec mik's 7th anullus
    
    #VCC0230
    
    # in_rad=np.array([36])#*0.187#0.00187 # degrees 36 pix   0.0084 #
    # out_rad=np.array([103]) # 27.2 arcsec mik's 7th anullus

    #VCC0538
    
    in_rad=np.array([16])#*0.187#0.00187 # degrees 36 pix   0.0084 #
    out_rad=np.array([68]) # 27.2 arcsec mik's 7th anullus
    
    #NGC1404
    # in_rad=np.array([36])#*0.187#0.00187 # degrees 36 pix   0.0084 #
    # out_rad=np.array([450]) # 27.2 arcsec mik's 7th anullus

    cutout_size=1536


    sbf.run_pr_cat(galaxies[i],bands[0],bands[1],magzp[bands[0]],magzp[bands[1]],
                    fwhm[bands[0]],fwhm[bands[1]], cutout_size, csmin,mfaint,mbright,threshold)


    p_r_ann= p_r.main(galaxies[i], in_rad, out_rad)

#Star Selection

    #VCC1025
    # psf_stars= np.array([1,3,4,5])
    
    # #VCC1040
    # psf_stars= np.array([1,2,9,15])
    
    #VCC0033
    # psf_stars= np.array([1,2,7,8,12,13,14,17])
    
    #VCC0230
    # psf_stars= np.array([5,8,10])
    
    #VCC0538
    psf_stars= np.array([2,3,4])
    
    #NGC1404
    # psf_stars= np.array([14])
    
    p0=np.zeros(psf_stars.shape)
    p0_gab=np.zeros(psf_stars.shape)
    p0_mad=np.zeros(psf_stars.shape)
    p1=np.zeros(psf_stars.shape)
    p0_med=np.zeros(in_rad.shape)
    p0_gab_med=np.zeros(in_rad.shape)
    p1_med=np.zeros(in_rad.shape)
    
    for r in range(len(in_rad)):
         print(in_rad[r],out_rad[r])
        
     #   print(f"ANNULUS #{r+1}")
        # POWER SPECTRA ANALYSIS AND COMPARISON:
        # p0[r],p1[r]=sbf.run_part6(galaxies[i],bands[0],bands[1],magzp[bands[0]],magzp[bands[1]],
        #             fwhm[bands[0]],fwhm[bands[1]],in_rad[r], out_rad[r], A[1], cutout_size,r)


         p0,p1,p0_gab=sbf.run_part6_starpsf(galaxies[i],bands[0],bands[1],magzp[bands[0]],magzp[bands[1]],
                    fwhm[bands[0]],fwhm[bands[1]],in_rad[r], out_rad[r], A[1], cutout_size, psf_stars, r)
         print(p0)
         print(p1)
         p0_med[r]=np.median(p0)
         p0_gab_med[r]=np.median(p0_gab)
         p0_mad=stats.median_abs_deviation(p0_gab)
         p1_med[r]=np.median(p1)
    print("median P_0 by annulus:") 
    print(p0_gab_med,'+-', p0_mad )
   

    print("P_1 by annulus:") 
    print(p1_med)
   
    print("P_r by annulus:") 
    print(p_r_ann)
    print("median P_fluc by annulus:") 
    print(p0_med-p_r_ann)
    
    t=Table([p0_med,p1_med, p_r_ann,p0_med-p_r_ann], names=('P0','P1','Pr','Pfluc'))
    print(t)
    print('SBF Magnitude NAND', -2.5*np.log10(p0_med-p_r_ann)+30)
    
    dy_dx = -2.5 * 30 / (p0_gab_med-p_r_ann) * np.log10(np.e)  # Derivative dy/dx
    delta_y = np.abs(dy_dx) * p0_mad  # Propagated error
    print('SBF Magnitude Gab', -2.5*np.log10(p0_gab_med-p_r_ann)+30, '+-', delta_y)
    

t1=time.time()
t=(t1-t0)/60.0
print ("Total runtime = %f minutes" %t)

# %%
