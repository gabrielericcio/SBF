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

t0=time.time()
# galaxies = ['ic0745','ngc4753','ngc5813','ngc5831','ngc5839']
# galaxies = ['vcc1025','vcc0828','vcc1146']
galaxies = ['vcc1025']
# galaxies = ['vcc1146']
# galaxies=['vcc0828']
# bands=['i']
bands=['g','i']
instr=['cfht']

################ Query extinction
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
        # print(R_V[bands[j]])
        gxy=galaxy+'_'+passband
        instrname=instr[0]+'_galinit'
        init_instr= getattr(init,instrname)
        init_instr(galaxy,passband)# instrument specific params
        init.default() # default for all galaxies
        init_spec = getattr(init, gxy) #specific initialisation for each galaxy
        init_spec()
        fwhm[bands[j]]=sbf.run_part1(rms=rms, remove_bkg=True) #IS BUILDING MODEL NOW
        #FWHM IS UPDATED CORRECTLY ONLY WHEN PART 1 IS RUN
        print('FWHM AAAAAA', fwhm)
        #sbf.run_part2(alpha[bands[j]])
        
    table=IrsaDust.get_query_table(galaxies[i],section='ebv')
    
    E_BV=table['ext SandF mean']   
    
    A=[(R_V[band]*E_BV.value) for band in bands]
    print(A)
    #####

    

    #VCC0122:
    fwhm={'g':0.66,'i': 0.59} # KEEP ONLY WHEN NOT RUNNING PART 1

    # print(fwhm) 
    #### PARAMS FOR APER_CORR
    match_scale=1.2
    threshold=35
    csmin=0.8
    # mbright=19
    # mfaint=22
    match_scale=1.2 #scaling factor to be multiplied to mean FWHM in order to match band catalogs

    #### PARAMS FOR EPSF
    cscut= 0.6#0.95 #both bands: for generating PSF
    mbright= 18#16 HSC
    mfaint= 20#21 #20.5 HSC
    # cilow=0.25 NOT USED
    # cihigh=0.75 NOT USED

    #sbf.run_part3(galaxies[i],bands[0],bands[1],A[0],A[1],
                  #magzp[bands[0]],magzp[bands[1]],fwhm[bands[0]],fwhm[bands[1]],
                  #match_scale,csmin,mfaint,mbright,threshold)

    #sbf.run_part4(galaxies[i],bands[0],bands[1],magzp[bands[0]],magzp[bands[1]],
                    #fwhm[bands[0]],fwhm[bands[1]], A[1])
    ## PSF Modelling:
    psf_rad_scale=3 # 5 FOR HSC.#4*plate_scale
    nthfactor=20 # nth neighbor selection radius= nthfactor*rad_asec
    rgc_factor=40

    psfsize=64
    oversampling=4
    #sbf.run_part5(galaxies[i],bands[0],bands[1],A[0],A[1],magzp[bands[0]],
                    #magzp[bands[1]],fwhm[bands[0]],fwhm[bands[1]],
                    #mfaint,mbright,threshold,cscut,psf_rad_scale,nthfactor,
                    #psfsize,oversampling,rgc_factor)
    
    # sbf.find_annuli(galaxies[i],bands[0],bands[1])

        
    # MASKING ANNULUS: Inner and outer radii--- in pix
    # VCC1025:
    # ORIGINAL    
    # in_rad=np.array([36,54,90,162,306,36,36])#*0.187#0.00187 # degrees 36 pix   0.0084 #
    # out_rad=np.array([54,90,162,306,594,600,300])#*0.187#0.0156 #0.0159 # cannulus 4 162 to 306 pix # degrees 300 pix : cannulus7 from Mik
    in_rad=np.array([236,254,290,162,306,36,36])#*0.187#0.00187 # degrees 36 pix   0.0084 #
    out_rad=np.array([354,390,362,306,594,600,300])#*0.187#0.0156 #0.0159 # cannulus 4 162 to 306 pix # degrees 300 pix : cannulus7 from Mik

    #VCC0828:
    # in_rad=np.array([85,95,115,155,235,85,85])#*0.187#0.00187 # degrees 36 pix   0.0084 #
    # out_rad=np.array([95,115,155,235,395,385,235])#*0.187#0.0156 #0.0159 # cannulus 4 162 to 306 pix # degrees 300 pix : cannulus7 from Mik
    # in_rad=np.array([85,85])*0.187#0.00187 # degrees 36 pix   0.0084 #
    # out_rad=np.array([385,235])*0.187#0.0156 #0.0159 # cannulus 4 162 to 306 pix # degrees 300 pix : cannulus7 from Mik

    #VCC1146:
    # in_rad=np.array([40,62,106,194,370,40,40])
    # out_rad=np.array([62,106,194,370,722,740,420])

    cutout_size=1536


    #sbf.run_pr_cat(galaxies[i],bands[0],bands[1],magzp[bands[0]],magzp[bands[1]],
                    #fwhm[bands[0]],fwhm[bands[1]], cutout_size, csmin,mfaint,mbright,threshold)


    #p_r_ann= p_r.main(galaxies[i], in_rad, out_rad)

    
    #VCC1025
    p0_mik=np.array([0.5149,.5619,.4851,.6617,1.8124,.6618,.5477])
    p1_mik=np.array([0.03,0.05,0.11,0.33,1.32,1.88,0.51])

    #VCC0828
    # p0_mik=np.array([0.8288,.7557,.7348,.6755,1.4124,.8737,.7102])
    # p1_mik=np.array([0.01,.02,.07,.22,.63,.89,.31])


    #VCC1146
    # p0_mik=np.array([1.053,1.0262,1.0405,1.0583,1.4413,1.1364,1.0490])
    # p1_mik=np.array([0.04,.1,.22,.62,1.76,2.83,1.21])


    psf_stars= np.array([1,3,4,5])#([1,2,4,5]) #([1,2,4])#np.arange(1,8)# 
    p0=np.zeros(psf_stars.shape)
    p1=np.zeros(psf_stars.shape)
    p0_med=np.zeros(in_rad.shape)
    p1_med=np.zeros(in_rad.shape)
    
    #for r in range(len(in_rad)):
        # print(in_rad[r],out_rad[r])
        
     #   print(f"ANNULUS #{r+1}")
        # POWER SPECTRA ANALYSIS AND COMPARISON:
        # p0[r],p1[r]=sbf.run_part6(galaxies[i],bands[0],bands[1],magzp[bands[0]],magzp[bands[1]],
        #             fwhm[bands[0]],fwhm[bands[1]],in_rad[r], out_rad[r], A[1], cutout_size,r)


      #  p0,p1=sbf.run_part6_starpsf(galaxies[i],bands[0],bands[1],magzp[bands[0]],magzp[bands[1]],
                    #fwhm[bands[0]],fwhm[bands[1]],in_rad[r], out_rad[r], A[1], cutout_size, psf_stars, r)
       # print(p0)
        #print(p1)
        #p0_med[r]=np.median(p0)
        #p1_med[r]=np.median(p1)
    #print("median P_0 by annulus:") 
    #print(p0_med)
    #print("% differences:")
    #print((p0_med-p0_mik)/p0_mik*100)

    #print("P_1 by annulus:") 
    #print(p1_med)
    # print("% differences:")
    # print((p1_med-p1_mik)/p1_mik*100)
    #print("P_r by annulus:") 
    #print(p_r_ann)
    #print("median P_fluc by annulus:") 
    #print(p0_med-p_r_ann)
    
    #t=Table([p0_med,p1_med, p_r_ann,p0_med-p_r_ann], names=('P0','P1','Pr','Pfluc'))
    #print(t)

#t1=time.time()
#t=(t1-t0)/60.0
#print ("Total runtime = %f minutes" %t)

# %%
