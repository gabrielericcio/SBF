#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 12:07:18 2021

@author: nandinihazra

DESCRIPTION : this class contains functions to initialise all necessary
parameters needed to run the sbf_vX functions
"""
#IMPORT NECESSARY ROUTINES
import numpy as np
import astroquery
from astroquery.ipac.ned import Ned
from astropy.io import fits

class GalaxyInBand:

    def __init__(self, galaxy: str, passband: str):
        # self.galaxy=galaxy
        # self.passband=passband
        #GENERATE GALAXY IMAGE FILEPATH
        self.gxyfilepath='../INPUT/IMAGES/'+galaxy.upper()+'/'+galaxy.upper()+'_'+passband+'.fits'
            
        #WEIGHT IMAGE FILEPATH
        self.wghtfilepath='../INPUT/IMAGES/'+galaxy.upper()+'/'+galaxy.upper()+'_'+passband+'.fits'
        
        
        #INITIALIZE PARAMETERS
        self.gxy=galaxy.upper() #Name of the galaxy
        self.band=passband #Read the passband
        self.gxy_name=gxy+'_'+band #Complete name of the galaxy including the passband
        t=Ned.query_object(gxy) #Query the NASA/IPAC Extragalactic Database
        self.gxyra=t["RA"] #Initialise coordinates of the galaxy : Right Ascension
        self.gxydec=t["DEC"] #Initialise coordinates of the galaxy : Declination

class Telescope:
    
    def __init__(self, gal: GalaxyInBand, telescope: str):

        if telescope.casefol()=='HSC'.casefold():
            self.hsc_subaru()
    
    def hsc_subaru():
        pass

    
def hsc_galinit(galaxy,passband)  :  
    """
    Description : This function intitalises all the parameters specific to HSC

    Input Parameters
    ----------
    galaxy : string
        Input the name of the galaxy
    passband : string
        Input the passband of the galaxy image

    Returns
    -------
    The object itself with global parameters initialised

    """
    global gxyfilepath
    global wghtfilepath
    global gxy
    global band
    global gxy_name
    global gxyra
    global gxydec
    global magzp
    global magcut
    global magcut_cpt
    global img_ext
    global wht_ext
    global fwhm0
    global plate_scale
    #######################
    #Initialisation for reading LSST instrument mask
    
    global read_hsc_mask
    global flags_header
    global flags
    
    
    
    #GENERATE GALAXY IMAGE FILEPATH
    gxyfilepath='../INPUT/IMAGES/'+galaxy.upper()+'/'+galaxy.upper()+'_'+passband+'.fits'
        
    #WEIGHT IMAGE FILEPATH
    wghtfilepath='../INPUT/IMAGES/'+galaxy.upper()+'/'+galaxy.upper()+'_'+passband+'.fits'
    
    
    #INITIALIZE PARAMETERS
    gxy=galaxy.upper() #Name of the galaxy
    band=passband #Read the passband
    gxy_name=gxy+'_'+band #Complete name of the galaxy including the passband
    t=Ned.query_object(gxy) #Query the NASA/IPAC Extragalactic Database
    gxyra=t["RA"] #Initialise coordinates of the galaxy : Right Ascension
    gxydec=t["DEC"] #Initialise coordinates of the galaxy : Declination
    
    
    #######################################
    #############HSC specific initializations
    magzp=27 #i-band zeropoint magnitude
    
    #Magnitude cutoffs for sextractor catalog : all objects brighter than this will be masked
    magcut=22#The magnitude cutoff for the extended objects mask
    magcut_cpt=23 #The magnitude cutoff for the compact objects mask
    
    img_ext=1 #Image extension
    wht_ext=3 #Weight extension
    
    fwhm0=0.7 #Seeing FWHM
    plate_scale=0.1679 #Plate scale of images
    
    read_hsc_mask=1 #Instruct the routine whether to read the instrument mask
    
    #Initialise flag values to create instrument mask
    flag_bad  = 0                   # 0                                                  
    flag_bright_object = 0 # 9                                                   
    flag_clipped =    0   # 14                                                        
    flag_cr   =    1               # 3                                                  
    flag_crosstalk =  0   # 10                                                      
    flag_detected =   0   # 5                                                        
    flag_detected_neg = 0    # 6                                               
    flag_edge =      1             #   4                                                  
    flag_inexact_psf =  1    # 16                                                    
    flag_intrp=       0            #   2                                                  
    flag_not_deblended = 0      # 11                                                  
    flag_no_data =   1      # 8                                                         
    flag_rejected =  0    # 13                                                       
    flag_sat  =        0           #  1                                                  
    flag_sensor_edge =  0 # 15                                                    
    flag_suspect =    0   # 7                                                         
    flag_unmasked_nan =  0 # 12     
    
    #Create arrays of flags defined above to pass to utils.get_lsst_mask
    flags_header = [
    		'MP_BAD', #0                                                  
    		'HIERARCH MP_BRIGHT_OBJECT', #9                                             
    		'HIERARCH MP_CLIPPED', #14                                                  
    		'MP_CR', #3                                                  
    		'HIERARCH MP_CROSSTALK', #10                                                      
    		'HIERARCH MP_DETECTED', #5                                                        
    		'HIERARCH MP_DETECTED_NEGATIVE', #6                                               
    		'MP_EDGE', #4                                                  
    		'HIERARCH MP_INEXACT_PSF', #16                                                    
    		'MP_INTRP', #2                                                  
    		'HIERARCH MP_NOT_DEBLENDED', #11                                                 
    		'HIERARCH MP_NO_DATA', #8                                                      
    		'HIERARCH MP_REJECTED', #13                                                       
    		'MP_SAT', #1                                                  
    		'HIERARCH MP_SENSOR_EDGE', #15                                                    
    		'HIERARCH MP_SUSPECT', #7                                                         
    		'HIERARCH MP_UNMASKEDNAN' #12 
    	]
    
    flags = [flag_bad,flag_bright_object,flag_clipped,flag_cr,flag_crosstalk,
             flag_detected,flag_detected_neg,flag_edge,flag_inexact_psf,flag_intrp,
             flag_not_deblended,flag_no_data,flag_rejected,flag_sat,
             flag_sensor_edge,flag_suspect,flag_unmasked_nan
        ]
    """
    #Flags in numerical order 0-16
    flags = [flag_bad,flag_sat,flag_intrp,flag_cr,flag_edge,flag_detected,
             flag_detected_neg,flag_suspect,flag_no_data,flag_bright_object,
             flag_crosstalk,flag_not_deblended,flag_unmasked_nan,flag_rejected,
             flag_clipped,flag_sensor_edge,flag_inexact_psf
        ]
    """
def cfht_galinit(galaxy,passband)  :  
    """
    Description : This function intitalises all the parameters specific to HSC

    Input Parameters
    ----------
    galaxy : string
        Input the name of the galaxy
    passband : string
        Input the passband of the galaxy image

    Returns
    -------
    The object itself with global parameters initialised

    """
    global gxyfilepath
    global wghtfilepath
    global gxy
    global band
    global gxy_name
    global gxyra
    global gxydec
    global magzp
    global magcut
    global magcut_cpt
    global img_ext
    global wht_ext
    global fwhm0
    global plate_scale
    #######################
    #Initialisation for reading LSST instrument mask
    
    global read_hsc_mask
    global flags_header
    global flags
    
    
    
    #GENERATE GALAXY IMAGE FILEPATH
    gxyfilepath='../INPUT/IMAGES/'+galaxy.upper()+'/'+galaxy.upper()+'_'+passband+'.fits'
   
    
    #WEIGHT IMAGE FILEPATH
    wghtfilepath='../INPUT/IMAGES/'+galaxy.upper()+'/'+galaxy.upper()+'_'+passband+'_sig.fits'
    
    
    #INITIALIZE PARAMETERS
    gxy=galaxy.upper() #Name of the galaxy
    band=passband #Read the passband
    gxy_name=gxy+'_'+band #Complete name of the galaxy including the passband
    t=Ned.query_object(gxy) #Query the NASA/IPAC Extragalactic Database
    gxyra=t["RA"] #Initialise coordinates of the galaxy : Right Ascension
    gxydec=t["DEC"] #Initialise coordinates of the galaxy : Declination
    
    
    #######################################
    #############HSC specific initializations
    magzp=30 #i-band zeropoint magnitude #30
    
    #Magnitude cutoffs for sextractor catalog : all objects brighter than this will be masked
    magcut=23#The magnitude cutoff for the extended objects mask
    magcut_cpt=24 #The magnitude cutoff for the compact objects mask
    
    img_ext=0 #Image extension
    wht_ext=0 #Weight extension
    
    fwhm0=0.7 #Seeing FWHM
    plate_scale=0.187 #Plate scale of images #0.187 (arcsec/pixel)
    
    read_hsc_mask=0 #Instruct the routine whether to read the instrument mask
    
    
def hst_acs_galinit(galaxy,passband)  :  
    """
    Description : This function intitalises all the parameters specific to HSC

    Input Parameters
    ----------
    galaxy : string
        Input the name of the galaxy
    passband : string
        Input the passband of the galaxy image

    Returns
    -------
    The object itself with global parameters initialised

    """
    global gxyfilepath
    global wghtfilepath
    global gxy
    global band
    global gxy_name
    global gxyra
    global gxydec
    global magzp
    global magcut
    global magcut_cpt
    global img_ext
    global wht_ext
    global fwhm0
    global plate_scale
    #######################
    #Initialisation for reading LSST instrument mask
    
    global read_hsc_mask
    
    
    
    
    #GENERATE GALAXY IMAGE FILEPATH
    gxyfilepath='../INPUT/IMAGES/'+galaxy.upper()+'/'+galaxy.upper()+'_'+passband+'.fits'
        
    #WEIGHT IMAGE FILEPATH
    wghtfilepath='../INPUT/IMAGES/'+galaxy.upper()+'/'+galaxy.upper()+'_'+passband+'_sig.fits'
    
    
    #INITIALIZE PARAMETERS
    gxy=galaxy.upper() #Name of the galaxy
    band=passband #Read the passband
    gxy_name=gxy+'_'+band #Complete name of the galaxy including the passband
    t=Ned.query_object(gxy) #Query the NASA/IPAC Extragalactic Database
    gxyra=t["RA"] #Initialise coordinates of the galaxy : Right Ascension
    gxydec=t["DEC"] #Initialise coordinates of the galaxy : Declination
        
    #######################################
    #############HSC specific initializations
    magzp=26 #i-band zeropoint magnitude
    
    #Magnitude cutoffs for sextractor catalog : all objects brighter than this will be masked
    magcut=24#The magnitude cutoff for the extended objects mask
    magcut_cpt=25 #The magnitude cutoff for the compact objects mask
    
    img_ext=1 #Image extension
    wht_ext=2 #Weight extension
    
    fwhm0=0.13 #Seeing FWHM
    plate_scale=0.04 #Plate scale of images
    
    read_hsc_mask=0 #Instruct the routine whether to read the instrument mask
    
def jwst_galinit(galaxy,passband)  :  
    """
    Description : This function intitalises all the parameters specific to HSC

    Input Parameters
    ----------
    galaxy : string
        Input the name of the galaxy
    passband : string
        Input the passband of the galaxy image

    Returns
    -------
    The object itself with global parameters initialised

    """
    global gxyfilepath
    global wghtfilepath
    global gxy
    global band
    global gxy_name
    global gxyra
    global gxydec
    global magzp
    global magcut
    global magcut_cpt
    global img_ext
    global wht_ext
    global fwhm0
    global plate_scale
    #######################
    #Initialisation for reading LSST instrument mask
    
    global read_hsc_mask
    
    
    
    
    #GENERATE GALAXY IMAGE FILEPATH
    gxyfilepath='../INPUT/IMAGES/'+galaxy.upper()+'/'+galaxy.upper()+'_'+passband+'.fits'
        
    #WEIGHT IMAGE FILEPATH
    wghtfilepath='../INPUT/IMAGES/'+galaxy.upper()+'/'+galaxy.upper()+'_'+passband+'.fits'
    
    
    #INITIALIZE PARAMETERS
    gxy=galaxy.upper() #Name of the galaxy
    band=passband #Read the passband
    gxy_name=gxy+'_'+band #Complete name of the galaxy including the passband
    t=Ned.query_object(gxy) #Query the NASA/IPAC Extragalactic Database
    gxyra=t["RA"] #Initialise coordinates of the galaxy : Right Ascension
    gxydec=t["DEC"] #Initialise coordinates of the galaxy : Declination
        
    #######################################
    #############HSC specific initializations
    magzp=26.69	 #i-band zeropoint magnitude
    
    #Magnitude cutoffs for sextractor catalog : all objects brighter than this will be masked
    magcut=24#The magnitude cutoff for the extended objects mask
    magcut_cpt=25 #The magnitude cutoff for the compact objects mask
    
    img_ext=0 #Image extension
    wht_ext=0 #Weight extension
    
    fwhm0=0.033 #Seeing FWHM
    plate_scale=0.031 #Plate scale of images
    
    read_hsc_mask=0 #Instruct the routine whether to read the instrument mask

def hst_uvis_galinit(galaxy,passband)  :  
    """
    Description : This function intitalises all the parameters specific to HSC

    Input Parameters
    ----------
    galaxy : string
        Input the name of the galaxy
    passband : string
        Input the passband of the galaxy image

    Returns
    -------
    The object itself with global parameters initialised

    """
    global gxyfilepath
    global wghtfilepath
    global gxy
    global band
    global gxy_name
    global gxyra
    global gxydec
    global magzp
    global magcut
    global magcut_cpt
    global img_ext
    global wht_ext
    global fwhm0
    global plate_scale
    #######################
    #Initialisation for reading LSST instrument mask
    
    global read_hsc_mask
    
    
    
    
    #GENERATE GALAXY IMAGE FILEPATH
    gxyfilepath='../INPUT/IMAGES/'+galaxy.upper()+'/'+galaxy.upper()+'_'+passband+'.fits'
        
    #WEIGHT IMAGE FILEPATH
    wghtfilepath='../INPUT/IMAGES/'+galaxy.upper()+'/'+galaxy.upper()+'_'+passband+'.fits'
    
    
    #INITIALIZE PARAMETERS
    gxy=galaxy.upper() #Name of the galaxy
    band=passband #Read the passband
    gxy_name=gxy+'_'+band #Complete name of the galaxy including the passband
    t=Ned.query_object(gxy) #Query the NASA/IPAC Extragalactic Database
    gxyra=t["RA"] #Initialise coordinates of the galaxy : Right Ascension
    gxydec=t["DEC"] #Initialise coordinates of the galaxy : Declination
        
    #######################################
    #############HSC specific initializations
    magzp=26.08 #i-band zeropoint magnitude
    
    #Magnitude cutoffs for sextractor catalog : all objects brighter than this will be masked
    magcut=24#The magnitude cutoff for the extended objects mask
    magcut_cpt=25 #The magnitude cutoff for the compact objects mask
    
    img_ext=0 #Image extension
    wht_ext=0 #Weight extension
    
    fwhm0=0.13 #Seeing FWHM
    plate_scale=0.039 #Plate scale of images
    
    read_hsc_mask=0 #Instruct the routine whether to read the instrument mask
    
    
def default() :    

    """
    Description : This routine initialises the instance with default fitting
    parameters which are not specific to the instrument
    """
    global minsma0
    global maxsma0
    global sma0
    global eps0 
    global pa0 
    global agxy 
    global maxrad 
    global minrad 
    global maskin 
    global maskout 
    global nclip 
    global sclip 
    global maxit 
    global box 
    global fix_center 
    global fix_pa 
    global fix_eps 
    global mask_scale_ext 
    global mask_scale_cpt 
    global high_harmonics
    
    #Galaxy geometry and initialization parameters
    #0th order fit
    minsma0=100
    maxsma0=200
    sma0=150  #First annulus for ellipse fitting
    eps0=0.15 #Ellipticity
    pa0=45.*np.pi/180. #Position angle
    agxy=0.2             #Step for increasing SMA
    maxrad=250       #Max radius for the fitting (in pixel)
    minrad=1        #Min radius for the fitting (in pixel)
    maskin=5   #Inner radius of the mask selection (arcsec)
    maskout=200 #Outer radius 
    nclip=3
    sclip=2
    maxit=50
    box=500#Box size in pixels within which to fit galaxy
    high_harmonics=False #Higher harmonics
    
    #Fixing the center, pa and eps
    fix_center=False
    fix_pa=False
    fix_eps=False
    
    ##
    #masking radii for compact and extended sources
    # mask_scale_ext=3
    mask_scale_ext=5
    mask_scale_cpt=3




def vcc1025_i() :


    global minsma0
    global maxsma0
    global sma0
    global eps0 
    global pa0 
    global agxy 
    global maxrad 
    global minrad 
    global maskin 
    global maskout 
    global nclip 
    global sclip 
    global maxit 
    global box 
    global fix_center 
    global fix_pa 
    global fix_eps 
    global mask_scale_ext 
    global mask_scale_cpt     
    global fwhm0
        
    #Galaxy geometry and initialization parameters
    #
    #0th order fit
    #minsma0=250
    #maxsma0=300
    #sma0=270  #First annulus for ellipse fitting
    eps0=0.1 #Ellipticity
    pa0=15.*np.pi/180. #Position angle
    agxy=0.2           #Step for increasing SMA
    maxrad=450           #Max radius for the fitting
    minrad=1             #Min radius for the fitting
    maskin=5   #Inner radius of the mask selection (arcsec)
    maskout=200 #Outer radius 
    #fwhm0=0.7
    nclip=8
    sclip=2

    #Fixing the center, pa and eps
    #fix_center=False
    fix_pa=False
    #fix_eps=False
    
    #masking radii for compact and extended sources
    mask_scale_ext=5
    # #mask_scale_cpt=4



def vcc1025_g() :


    global minsma0
    global maxsma0
    global sma0
    global eps0 
    global pa0 
    global agxy 
    global maxrad 
    global minrad 
    global maskin 
    global maskout 
    global nclip 
    global sclip 
    global maxit 
    global box 
    global fix_center 
    global fix_pa 
    global fix_eps 
    global mask_scale_ext 
    global mask_scale_cpt     
    global fwhm0
        
        
    #Galaxy geometry and initialization parameters
    #
    #0th order fit
    # minsma0=250
    # maxsma0=300
    #sma0=250  #First annulus for ellipse fitting
    eps0=0.1 #Ellipticity
    pa0=15.*np.pi/180. #Position angle
    # agxy=0.2            #Step for increasing SMA
    # maxrad=550           #Max radius for the fitting
    #minrad=40             #Min radius for the fitting
    # maskin=5   #Inner radius of the mask selection (arcsec)
    # maskout=120 #Outer radius 
    nclip=8
    sclip=2
    # maxit=50
    #fwhm0=0.7
    #Fixing the center, pa and eps
    # fix_center=False
    # fix_pa=False
    # fix_eps=False

    #masking radii for compact and extended sources
    mask_scale_ext=5
    # #mask_scale_cpt=4


def vcc1355_i() :


    global minsma0
    global maxsma0
    global sma0
    global eps0 
    global pa0 
    global agxy 
    global maxrad 
    global minrad 
    global maskin 
    global maskout 
    global nclip 
    global sclip 
    global maxit 
    global box 
    global fix_center 
    global fix_pa 
    global fix_eps 
    global mask_scale_ext 
    global mask_scale_cpt     
    global fwhm0
        
    #Galaxy geometry and initialization parameters
    #
    #0th order fit
    #minsma0=250
    #maxsma0=300
    #sma0=270  #First annulus for ellipse fitting
    eps0=0.1 #Ellipticity
    pa0=15.*np.pi/180. #Position angle
    # agxy=0.2           #Step for increasing SMA
    # maxrad=450           #Max radius for the fitting
    # minrad=1             #Min radius for the fitting
    # maskin=5   #Inner radius of the mask selection (arcsec)
    # maskout=200 #Outer radius 
    #fwhm0=0.7
    nclip=8
    sclip=2

    #Fixing the center, pa and eps
    #fix_center=False
    fix_pa=False
    #fix_eps=False
    
    #masking radii for compact and extended sources
    mask_scale_ext=5
    # #mask_scale_cpt=4



def vcc1355_g() :


    global minsma0
    global maxsma0
    global sma0
    global eps0 
    global pa0 
    global agxy 
    global maxrad 
    global minrad 
    global maskin 
    global maskout 
    global nclip 
    global sclip 
    global maxit 
    global box 
    global fix_center 
    global fix_pa 
    global fix_eps 
    global mask_scale_ext 
    global mask_scale_cpt     
    global fwhm0
        
        
    #Galaxy geometry and initialization parameters
    #
    #0th order fit
    # minsma0=250
    # maxsma0=300
    #sma0=250  #First annulus for ellipse fitting
    eps0=0.1 #Ellipticity
    pa0=15.*np.pi/180. #Position angle
    # agxy=0.2            #Step for increasing SMA
    # maxrad=550           #Max radius for the fitting
    #minrad=40             #Min radius for the fitting
    # maskin=5   #Inner radius of the mask selection (arcsec)
    # maskout=120 #Outer radius 
    nclip=8
    sclip=2
    # maxit=50
    #fwhm0=0.7
    #Fixing the center, pa and eps
    # fix_center=False
    # fix_pa=False
    # fix_eps=False

    #masking radii for compact and extended sources
    mask_scale_ext=5
    # #mask_scale_cpt=4
    
    
def vcc0033_i() :


    global minsma0
    global maxsma0
    global sma0
    global eps0 
    global pa0 
    global agxy 
    global maxrad 
    global minrad 
    global maskin 
    global maskout 
    global nclip 
    global sclip 
    global maxit 
    global box 
    global fix_center 
    global fix_pa 
    global fix_eps 
    global mask_scale_ext 
    global mask_scale_cpt     
    global fwhm0
        
    #Galaxy geometry and initialization parameters
    #
    #0th order fit
    #minsma0=250
    #maxsma0=300
    #sma0=270  #First annulus for ellipse fitting
    eps0=0.1 #Ellipticity
    pa0=15.*np.pi/180. #Position angle
    agxy=0.2            #Step for increasing SMA
    maxrad=200           #Max radius for the fitting
    minrad=1             #Min radius for the fitting
    maskin=5   #Inner radius of the mask selection (arcsec)
    maskout=200 #Outer radius 
    #fwhm0=0.7
    nclip=8
    sclip=2

    #Fixing the center, pa and eps
    #fix_center=False
    fix_pa=False
    #fix_eps=False
    
    #masking radii for compact and extended sources
    mask_scale_ext=5
    # #mask_scale_cpt=4


def vcc0033_g() :


    global minsma0
    global maxsma0
    global sma0
    global eps0 
    global pa0 
    global agxy 
    global maxrad 
    global minrad 
    global maskin 
    global maskout 
    global nclip 
    global sclip 
    global maxit 
    global box 
    global fix_center 
    global fix_pa 
    global fix_eps 
    global mask_scale_ext 
    global mask_scale_cpt     
    global fwhm0
        
        
    #Galaxy geometry and initialization parameters
    #
    #0th order fit
    # minsma0=250
    # maxsma0=300
    #sma0=250  #First annulus for ellipse fitting
    eps0=0.1 #Ellipticity
    pa0=15.*np.pi/180. #Position angle
    agxy=0.2            #Step for increasing SMA
    maxrad=200           #Max radius for the fitting
    minrad=1             #Min radius for the fitting
    maskin=5   #Inner radius of the mask selection (arcsec)
    maskout=200 #Outer radius  
    nclip=8
    sclip=2
    # maxit=50
    #fwhm0=0.7
    #Fixing the center, pa and eps
    # fix_center=False
    # fix_pa=False
    # fix_eps=False

    #masking radii for compact and extended sources
    mask_scale_ext=5
    # #mask_scale_cpt=4
    
def vcc0032_i() :


    global minsma0
    global maxsma0
    global sma0
    global eps0 
    global pa0 
    global agxy 
    global maxrad 
    global minrad 
    global maskin 
    global maskout 
    global nclip 
    global sclip 
    global maxit 
    global box 
    global fix_center 
    global fix_pa 
    global fix_eps 
    global mask_scale_ext 
    global mask_scale_cpt     
    global fwhm0
        
    #Galaxy geometry and initialization parameters
    #
    #0th order fit
    #minsma0=250
    #maxsma0=300
    #sma0=270  #First annulus for ellipse fitting
    eps0=0.1 #Ellipticity
    pa0=15.*np.pi/180. #Position angle
    # agxy=0.2            #Step for increasing SMA
    # maxrad=200           #Max radius for the fitting
    # minrad=1             #Min radius for the fitting
    # maskin=5   #Inner radius of the mask selection (arcsec)
    # maskout=200 #Outer radius 
    #fwhm0=0.7
    nclip=8
    sclip=2

    #Fixing the center, pa and eps
    #fix_center=False
    fix_pa=False
    #fix_eps=False
    
    #masking radii for compact and extended sources
    mask_scale_ext=5
    # #mask_scale_cpt=4


def vcc0032_g() :


    global minsma0
    global maxsma0
    global sma0
    global eps0 
    global pa0 
    global agxy 
    global maxrad 
    global minrad 
    global maskin 
    global maskout 
    global nclip 
    global sclip 
    global maxit 
    global box 
    global fix_center 
    global fix_pa 
    global fix_eps 
    global mask_scale_ext 
    global mask_scale_cpt     
    global fwhm0
        
        
    #Galaxy geometry and initialization parameters
    #
    #0th order fit
    # minsma0=250
    # maxsma0=300
    #sma0=250  #First annulus for ellipse fitting
    eps0=0.1 #Ellipticity
    pa0=15.*np.pi/180. #Position angle
    # agxy=0.2            #Step for increasing SMA
    # maxrad=200           #Max radius for the fitting
    # minrad=1             #Min radius for the fitting
    # maskin=5   #Inner radius of the mask selection (arcsec)
    # maskout=200 #Outer radius  
    nclip=8
    sclip=2
    # maxit=50
    #fwhm0=0.7
    #Fixing the center, pa and eps
    # fix_center=False
    # fix_pa=False
    # fix_eps=False

    #masking radii for compact and extended sources
    mask_scale_ext=5
    # #mask_scale_cpt=4



def vcc0230_i() :


    global minsma0
    global maxsma0
    global sma0
    global eps0 
    global pa0 
    global agxy 
    global maxrad 
    global minrad 
    global maskin 
    global maskout 
    global nclip 
    global sclip 
    global maxit 
    global box 
    global fix_center 
    global fix_pa 
    global fix_eps 
    global mask_scale_ext 
    global mask_scale_cpt     
    global fwhm0
        
    #Galaxy geometry and initialization parameters
    #
    #0th order fit
    #minsma0=250
    #maxsma0=300
    #sma0=270  #First annulus for ellipse fitting
    eps0=0.1 #Ellipticity
    pa0=15.*np.pi/180. #Position angle
    agxy=0.2           #Step for increasing SMA
    maxrad=170           #Max radius for the fitting
    minrad=1             #Min radius for the fitting
    maskin=5   #Inner radius of the mask selection (arcsec)
    maskout=200 #Outer radius 
    #fwhm0=0.7
    nclip=8
    sclip=2

    #Fixing the center, pa and eps
    #fix_center=False
    fix_pa=False
    #fix_eps=False
    
    #masking radii for compact and extended sources
    mask_scale_ext=5
    # #mask_scale_cpt=4
    
def vcc0230_g() :


    global minsma0
    global maxsma0
    global sma0
    global eps0 
    global pa0 
    global agxy 
    global maxrad 
    global minrad 
    global maskin 
    global maskout 
    global nclip 
    global sclip 
    global maxit 
    global box 
    global fix_center 
    global fix_pa 
    global fix_eps 
    global mask_scale_ext 
    global mask_scale_cpt     
    global fwhm0
        
        
    #Galaxy geometry and initialization parameters
    #
    #0th order fit
    # minsma0=250
    # maxsma0=300
    #sma0=250  #First annulus for ellipse fitting
    eps0=0.1 #Ellipticity
    pa0=15.*np.pi/180. #Position angle
    agxy=0.2            #Step for increasing SMA
    # maxrad=550           #Max radius for the fitting
    #minrad=40             #Min radius for the fitting
    # maskin=5   #Inner radius of the mask selection (arcsec)
    # maskout=120 #Outer radius 
    nclip=8
    sclip=2
    # maxit=50
    #fwhm0=0.7
    #Fixing the center, pa and eps
    # fix_center=False
    # fix_pa=False
    # fix_eps=False

    #masking radii for compact and extended sources
    mask_scale_ext=5
    # #mask_scale_cpt=4    

    
    
def vcc0140_i() :


    global minsma0
    global maxsma0
    global sma0
    global eps0 
    global pa0 
    global agxy 
    global maxrad 
    global minrad 
    global maskin 
    global maskout 
    global nclip 
    global sclip 
    global maxit 
    global box 
    global fix_center 
    global fix_pa 
    global fix_eps 
    global mask_scale_ext 
    global mask_scale_cpt     
    global fwhm0
        
    #Galaxy geometry and initialization parameters
    #
    #0th order fit
    #minsma0=250
    #maxsma0=300
    #sma0=270  #First annulus for ellipse fitting
    eps0=0.1 #Ellipticity
    pa0=15.*np.pi/180. #Position angle
    agxy=0.2           #Step for increasing SMA
    #maxrad=550           #Max radius for the fitting
    #minrad=40             #Min radius for the fitting
    #maskin=5   #Inner radius of the mask selection (arcsec)
    #maskout=120 #Outer radius 
    #fwhm0=0.7
    nclip=8
    sclip=2

    #Fixing the center, pa and eps
    #fix_center=False
    fix_pa=False
    #fix_eps=False
    
    #masking radii for compact and extended sources
    mask_scale_ext=5
    # #mask_scale_cpt=4


def vcc0140_g() :


    global minsma0
    global maxsma0
    global sma0
    global eps0 
    global pa0 
    global agxy 
    global maxrad 
    global minrad 
    global maskin 
    global maskout 
    global nclip 
    global sclip 
    global maxit 
    global box 
    global fix_center 
    global fix_pa 
    global fix_eps 
    global mask_scale_ext 
    global mask_scale_cpt     
    global fwhm0
        
        
    #Galaxy geometry and initialization parameters
    #
    #0th order fit
    # minsma0=250
    # maxsma0=300
    #sma0=250  #First annulus for ellipse fitting
    eps0=0.1 #Ellipticity
    pa0=15.*np.pi/180. #Position angle
    # agxy=0.2            #Step for increasing SMA
    # maxrad=550           #Max radius for the fitting
    #minrad=40             #Min radius for the fitting
    # maskin=5   #Inner radius of the mask selection (arcsec)
    # maskout=120 #Outer radius 
    nclip=8
    sclip=2
    # maxit=50
    #fwhm0=0.7
    #Fixing the center, pa and eps
    # fix_center=False
    # fix_pa=False
    # fix_eps=False

    #masking radii for compact and extended sources
    mask_scale_ext=5
    # #mask_scale_cpt=4


def vcc0538_i() :


    global minsma0
    global maxsma0
    global sma0
    global eps0 
    global pa0 
    global agxy 
    global maxrad 
    global minrad 
    global maskin 
    global maskout 
    global nclip 
    global sclip 
    global maxit 
    global box 
    global fix_center 
    global fix_pa 
    global fix_eps 
    global mask_scale_ext 
    global mask_scale_cpt     
    global fwhm0
        
    #Galaxy geometry and initialization parameters
    #
    #0th order fit
    #minsma0=250
    #maxsma0=300
    #sma0=270  #First annulus for ellipse fitting
    eps0=0.1 #Ellipticity
    pa0=15.*np.pi/180. #Position angle
    agxy=0.1            #Step for increasing SMA
    #maxrad=550           #Max radius for the fitting
    #minrad=40             #Min radius for the fitting
    #maskin=5   #Inner radius of the mask selection (arcsec)
    #maskout=120 #Outer radius 
    #fwhm0=0.7
    nclip=8
    sclip=2

    #Fixing the center, pa and eps
    #fix_center=False
    fix_pa=False
    #fix_eps=False
    
    #masking radii for compact and extended sources
    mask_scale_ext=6
    # #mask_scale_cpt=4


def vcc0538_g() :


    global minsma0
    global maxsma0
    global sma0
    global eps0 
    global pa0 
    global agxy 
    global maxrad 
    global minrad 
    global maskin 
    global maskout 
    global nclip 
    global sclip 
    global maxit 
    global box 
    global fix_center 
    global fix_pa 
    global fix_eps 
    global mask_scale_ext 
    global mask_scale_cpt     
    global fwhm0
        
        
    #Galaxy geometry and initialization parameters
    #
    #0th order fit
    # minsma0=250
    # maxsma0=300
    #sma0=250  #First annulus for ellipse fitting
    eps0=0.1 #Ellipticity
    pa0=15.*np.pi/180. #Position angle
    agxy=0.1           #Step for increasing SMA
    # maxrad=550           #Max radius for the fitting
    #minrad=40             #Min radius for the fitting
    # maskin=5   #Inner radius of the mask selection (arcsec)
    # maskout=120 #Outer radius 
    nclip=8
    sclip=2
    # maxit=50
    #fwhm0=0.7
    #Fixing the center, pa and eps
    # fix_center=False
    # fix_pa=False
    # fix_eps=False

    #masking radii for compact and extended sources
    mask_scale_ext=6
    # #mask_scale_cpt=4


       
def ngc5813_i() :

    global minsma0
    global maxsma0
    global sma0
    global eps0 
    global pa0 
    global agxy 
    global maxrad 
    global minrad 
    global maskin 
    global maskout 
    global nclip 
    global sclip 
    global maxit 
    global box 
    global fix_center 
    global fix_pa 
    global fix_eps 
    global mask_scale_ext 
    global mask_scale_cpt 
    global fwhm0
    
    global read_hsc_mask
    global custom_maskpath
    global mask_ext
    global flags


    #read_hsc_mask=0

    #Galaxy geometry and initialization parameters
    #
    # minsma0=70
    # maxsma0=200
    # sma0=220  #First annulus for ellipse fitting
    eps0=0.7 #Ellipticity
    pa0=135.*np.pi/180. #Position angle
    #agxy=0.12             #Step for increasing SMA
    # maxrad=1700           #Max radius for the fitting
    maxrad=2250           #Max radius for the fitting
    minrad=40             #Min radius for the fitting
    maskin=5   #Inner radius of the mask selection (arcsec)
    maskout=200 #Outer radius 
    #nclip=3
    #sclip=2
    #maxit=50
  
    #read_hsc_mask=0
    #Fixing the center, pa and eps
    # fix_center=False
    # fix_pa=False
    # fix_eps=False

    #masking radii for compact and extended sources
    # mask_scale_ext=1
    # mask_scale_cpt=1
    read_hsc_mask=3
    custom_maskpath=f"../INPUT/IMAGES/{gxy}/mask_both.fits"
    mask_ext=0

    #Initialise flag values to create instrument mask
    flag_bad  = 0                   # 0                                                  
    flag_bright_object = 0 # 9                                                   
    flag_clipped =    0   # 14                                                        
    flag_cr   =    1               # 3                                                  
    flag_crosstalk =  0   # 10                                                      
    flag_detected =   0   # 5                                                        
    flag_detected_neg = 0    # 6                                               
    flag_edge =      1             #   4                                                  
    flag_inexact_psf =  0    # 16                                                    
    flag_intrp=       0            #   2                                                  
    flag_not_deblended = 0      # 11                                                  
    flag_no_data =   1      # 8                                                         
    flag_rejected =  0    # 13                                                       
    flag_sat  =        0           #  1                                                  
    flag_sensor_edge =  0 # 15                                                    
    flag_suspect =    0   # 7                                                         
    flag_unmasked_nan =  0 # 12     
    
    flags = [flag_bad,flag_bright_object,flag_clipped,flag_cr,flag_crosstalk,
             flag_detected,flag_detected_neg,flag_edge,flag_inexact_psf,flag_intrp,
             flag_not_deblended,flag_no_data,flag_rejected,flag_sat,
             flag_sensor_edge,flag_suspect,flag_unmasked_nan
        ]
    
def ngc1404_i() :


    global minsma0
    global maxsma0
    global sma0
    global eps0 
    global pa0 
    global agxy 
    global maxrad 
    global minrad 
    global maskin 
    global maskout 
    global nclip 
    global sclip 
    global maxit 
    global box 
    global fix_center 
    global fix_pa 
    global fix_eps 
    global mask_scale_ext 
    global mask_scale_cpt     
    global fwhm0
        
    #Galaxy geometry and initialization parameters
    #
    #0th order fit
    #minsma0=250
    #maxsma0=300
    #sma0=270  #First annulus for ellipse fitting
    eps0=0.1 #Ellipticity
    pa0=15.*np.pi/180. #Position angle
    agxy=0.2            #Step for increasing SMA
    #maxrad=550           #Max radius for the fitting
    #minrad=40             #Min radius for the fitting
    #maskin=5   #Inner radius of the mask selection (arcsec)
    #maskout=120 #Outer radius 
    #fwhm0=0.7
    nclip=8
    sclip=2

    #Fixing the center, pa and eps
    fix_center=False
    fix_pa=False
    #fix_eps=False
    
    #masking radii for compact and extended sources
    # mask_scale_ext=4
    # #mask_scale_cpt=4



def ngc1404_g() :


    global minsma0
    global maxsma0
    global sma0
    global eps0 
    global pa0 
    global agxy 
    global maxrad 
    global minrad 
    global maskin 
    global maskout 
    global nclip 
    global sclip 
    global maxit 
    global box 
    global fix_center 
    global fix_pa 
    global fix_eps 
    global mask_scale_ext 
    global mask_scale_cpt     
    global fwhm0
        
        
    #Galaxy geometry and initialization parameters
    #
    #0th order fit
    # minsma0=250
    # maxsma0=300
    #sma0=250  #First annulus for ellipse fitting
    eps0=0.1 #Ellipticity
    pa0=15.*np.pi/180. #Position angle
    agxy=0.2            #Step for increasing SMA
    # maxrad=550           #Max radius for the fitting
    #minrad=40             #Min radius for the fitting
    # maskin=5   #Inner radius of the mask selection (arcsec)
    # maskout=120 #Outer radius 
    nclip=8
    sclip=2
    # maxit=50
    #fwhm0=0.7
    #Fixing the center, pa and eps
    # fix_center=False
    # fix_pa=False
    # fix_eps=False

    #masking radii for compact and extended sources
    mask_scale_ext=5
    # #mask_scale_cpt=4    
    
    

       
def ngc5813_g() :

    global minsma0
    global maxsma0
    global sma0
    global eps0 
    global pa0 
    global agxy 
    global maxrad 
    global minrad 
    global maskin 
    global maskout 
    global nclip 
    global sclip 
    global maxit 
    global box 
    global fix_center 
    global fix_pa 
    global fix_eps 
    global mask_scale_ext 
    global mask_scale_cpt 
    global fwhm0
    global read_hsc_mask
    global custom_maskpath
    global mask_ext
        
    #read_hsc_mask=0
    #Galaxy geometry and initialization parameters
    #
    # minsma0=70
    # maxsma0=200
    # sma0=220  #First annulus for ellipse fitting
    eps0=0.7 #Ellipticity
    pa0=135.*np.pi/180. #Position angle
    #agxy=0.12             #Step for increasing SMA
    # maxrad=1700           #Max radius for the fitting
    maxrad=2250           #Max radius for the fitting
    minrad=60             #Min radius for the fitting
    maskin=5   #Inner radius of the mask selection (arcsec)
    maskout=200 #Outer radius 
    # nclip=3
    # sclip=2
    # maxit=50
 
    # #Fixing the center, pa and eps
    # fix_center=False
    # fix_pa=False
    # fix_eps=False
    
    # #masking radii for compact and extended sources
    # mask_scale_ext=4
    # mask_scale_cpt=4
    read_hsc_mask=1
    # custom_maskpath=f"../INPUT/IMAGES/{gxy}/{gxy}_gmask.fits"
    # mask_ext=0
    
    
        

       
def ngc5831_i() :

    global minsma0
    global maxsma0
    global sma0
    global eps0 
    global pa0 
    global agxy 
    global maxrad 
    global minrad 
    global maskin 
    global maskout 
    global nclip 
    global sclip 
    global maxit 
    global box 
    global fix_center 
    global fix_pa 
    global fix_eps 
    global mask_scale_ext 
    global mask_scale_cpt 
    global fwhm0
    global read_hsc_mask
    global custom_maskpath
    global mask_ext


    
    #Galaxy geometry and initialization parameters
    #0th order fit
    #minsma0=300
    #maxsma0=500
    #sma0=450  #First annulus for ellipse fitting
    eps0=0.15 #Ellipticity
    pa0=45.*np.pi/180. #Position angle
    #agxy=0.25             #Step for increasing SMA
    maxrad=1500           #Max radius for the fitting
    #minrad=30             #Min radius for the fitting
    #maskin=5   #Inner radius of the mask selection (arcsec)
    #maskout=120 #Outer radius 
    
    #Fixing the center, pa and eps
    # fix_center=True
    # fix_pa=True
    # fix_eps=False
  
    # #masking radii for compact and extended sources
    # mask_scale_ext=4
    # mask_scale_cpt=4
    read_hsc_mask=2
    custom_maskpath=f"../INPUT/IMAGES/{gxy}/{gxy}_imask.fits"
    mask_ext=0

    
    

def ngc5831_g() :
    
    global minsma0
    global maxsma0
    global sma0
    global eps0 
    global pa0 
    global agxy 
    global maxrad 
    global minrad 
    global maskin 
    global maskout 
    global nclip 
    global sclip 
    global maxit 
    global box 
    global fix_center 
    global fix_pa 
    global fix_eps 
    global mask_scale_ext 
    global mask_scale_cpt 
    global fwhm0
    global high_harmonics

    global read_hsc_mask
    global custom_maskpath
    global mask_ext


    #Galaxy geometry and initialization parameters
    #0th order fit
    #minsma0=300
    #maxsma0=500
    #sma0=450  #First annulus for ellipse fitting
    eps0=0.15 #Ellipticity
    pa0=45.*np.pi/180. #Position angle
    #agxy=0.25             #Step for increasing SMA
    maxrad=1500           #Max radius for the fitting
    # minrad=30             #Min radius for the fitting
    # maskin=5   #Inner radius of the mask selection (arcsec)
    # maskout=120 #Outer radius 
    # nclip=5
    # sclip=5
    
    # #Fixing the center, pa and eps
    # #fix_center=True
    # fix_pa=True
    # fix_eps=False
    
    
    # #masking radii for compact and extended sources
    # mask_scale_ext=4
    # mask_scale_cpt=4
    read_hsc_mask=2
    custom_maskpath=f"../INPUT/IMAGES/{gxy}/{gxy}_gmask_v2.fits"
    mask_ext=0

    

       
def ngc5839_i() :

    global minsma0
    global maxsma0
    global sma0
    global eps0 
    global pa0 
    global agxy 
    global maxrad 
    global minrad 
    global maskin 
    global maskout 
    global nclip 
    global sclip 
    global maxit 
    global box 
    global fix_center 
    global fix_pa 
    global fix_eps 
    global mask_scale_ext 
    global mask_scale_cpt 
    global fwhm0
    
    global read_hsc_mask
    global custom_maskpath
    global mask_ext
    
    #Galaxy geometry and initialization parameters
    #
    #0th order fit
    #minsma0=50
    #maxsma0=150
    #sma0=100  #First annulus for ellipse fitting
    eps0=0.1 #Ellipticity
    pa0=15.*np.pi/180. #Position angle
    #agxy=0.2            #Step for increasing SMA
   #  maxrad=550           #Max radius for the fitting
   # minrad=70             #Min radius for the fitting
   #  #maskin=5   #Inner radius of the mask selection (arcsec)
   #  #maskout=120 #Outer radius 
   #  #nclip=3
   #  #sclip=2
    
   #  #Fixing the center, pa and eps
   # fix_center=False
    fix_pa=True
   #  fix_eps=False
    
    
   # #masking radii for compact and extended sources
    mask_scale_ext=4
    mask_scale_cpt=4
    

    

       
def ngc5839_g() :

    global minsma0
    global maxsma0
    global sma0
    global eps0 
    global pa0 
    global agxy 
    global maxrad 
    global minrad 
    global maskin 
    global maskout 
    global nclip 
    global sclip 
    global maxit 
    global box 
    global fix_center 
    global fix_pa 
    global fix_eps 
    global mask_scale_ext 
    global mask_scale_cpt 
    global fwhm0
    
    global read_hsc_mask
    global custom_maskpath
    global mask_ext
    
    #Galaxy geometry and initialization parameters
    #
    #0th order fit
    #minsma0=80
    #maxsma0=150
    #sma0=100  #First annulus for ellipse fitting
    eps0=0.1 #Ellipticity
    pa0=15.*np.pi/180. #Position angle
    #agxy=0.2            #Step for increasing SMA
    # maxrad=550           #Max radius for the fitting
    #minrad=70             #Min radius for the fitting
    # #maskin=5   #Inner radius of the mask selection (arcsec)
    # #maskout=120 #Outer radius 
    # #nclip=3
    # #sclip=2
    
    # #Fixing the center, pa and eps
    # fix_center=False
    fix_pa=True
    # fix_eps=True
   
    # #masking radii for compact and extended sources
    mask_scale_ext=4
    mask_scale_cpt=4
    
       

