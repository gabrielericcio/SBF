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


#IMPORT GALAXY OBJECT
import init as gal

#IMPORT UTILS CLASS
import utils

gxy='../INPUT/IMAGES/VCC1025/VCC1025_g.fits'
img_ext=0

hdu = fits.open(gxy, do_not_scale_image_data=True)
hdu.info()

mask = 0

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
flag_sensor_edge,flag_suspect,flag_unmasked_nan]

flags_header=[0,9,14,3,10,5,6,4,16,2,11,8,13,1,15,7,12]

for i in range(len(flags)):
    mask += flags[i] << flags_header[i]
    print(flags_header[i],mask)