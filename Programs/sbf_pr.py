#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 17:47 2023

@author: hazra
"""


from astropy.table import Table
from astropy.io import fits
import numpy as np
from scipy.optimize import curve_fit 
import matplotlib.pyplot as plt
import time
import math

from scipy.special import erfc
from scipy.optimize import minimize
import copy

class GalaxyPhot:
    def __init__(self, table_path: str, mask_path: str, mask_ext: int, x0: int, y0: int, secperpixel: float, distance: float):
        
        #Initialization
        tab=Table.read(table_path, format='ascii')
        # self.idx=tab['col1'] 
        # self.x=tab['col2'] 
        # self.y=tab['col3'] 
        # self.r=tab['col4'] 
        # self.itype=tab['col5'] 
        # self.cxx=tab['col6'] 
        # self.cyy=tab['col7'] 
        # self.cxy=tab['col8'] 
        # self.kron=tab['col9']
        # self.aa=tab['col10'] 
        # # 11th column in mch file is FWHM
        # self.m=tab['col12'] 
        # self.dm=tab['col13'] 
        # self.cstar=tab['col14']
        self.idx=tab['NUMBER'] 
        self.x=tab['X_IMAGE'] 
        self.y=tab['Y_IMAGE'] 
        self.r=np.sqrt((self.x-x0)**2+(self.y-y0)**2)#tab['col4'] 
        # self.itype=tab['col5'] 
        self.cxx=tab['CXX_IMAGE'] 
        self.cyy=tab['CYY_IMAGE'] 
        self.cxy=tab['CXY_IMAGE'] 
        self.kron=tab['KRON_RADIUS']
        self.aa=tab['A_IMAGE'] 
        # 11th column in mch file is FWHM
        self.m=tab['MAG_AUTO'] 
        self.dm=tab['MAGERR_AUTO'] 
        self.cstar=tab['CLASS_STAR']


        self.tab=tab

        self.x0=x0 #Central pixel of galaxy in x
        self.y0=y0 #Central pixel of galaxy in y
        self.secperpixel=secperpixel
        self.distance=distance # Initial guess distance in Mpc
        #read the mask
        self.mask = fits.open(mask_path, do_not_scale_image_data=True)
        self.mask_ext=mask_ext
        # hdu.info()
        # mask_data = hdu[0].data
        # idx,idy=0.,0. # OFFSET BETWEEN CORNER PIXEL VALUE ACROSS SEXTRACTOR AND VISTA
        mask_data=self.mask[mask_ext].data
        self.nx,self.ny=mask_data.shape
        # print(nx)
        # print(ny)

        #########
        ##### check number of unexcised pixels in mask

        # print(tab)

        print('MAKING RADIAL PROFILE OF UNEXCISED PIXELS')
        print('OG code ignores a 3x3 border on the edges: we have not done this')

        # self.r_npix,self.npix=azimuthal_sum(mask_data)

        # plt.plot(r_npix,npix)
        # plt.show()
        # plt.clf()


        self.npixtot=np.sum(mask_data)
        # print("npixtot= ",npixtot)
        self.nobs=len(self.tab)

    @property
    def distmod(self):
        return 25. + 5.*np.log10(self.distance)




class FitParams:

    def __init__(self, galphot:GalaxyPhot, beta, alpha, cnorm, delta, icolor, snlim, rsky, drsky, npar):
        
        self.galphot=galphot
        # C Initial guess for gxy/total number
        self.beta = beta
        # C Initial guess for d lnN / d lnr of GC distribution
        self.alpha = alpha

        self.cnorm=cnorm   # ??????????
        

        ##PRESET PARAMETER
        brightcut=[21.0,20.5,20.0,19.5,19.0,19.0,17.5,19.5]
        #            B    V    R    I    J    H    K  z_AB
        # self.kscale = kscale
        self.snlim=snlim
        # icol=1 #dummy
        self.icolor=icolor #i-band

        
        rlim=[np.min(galphot.r),np.max(galphot.r)/np.sqrt(2.)]
        mlim=[0,0,0,0]
        mlim[3] = brightcut[icolor]
        if((mlim[3]<15) or (mlim[3]>25)): mlim[3] = 20

        # C We expect that:
        # C (1) This is FAINT object photometry so the errors are independent of m
        # C (2) The errors are a constant flux at a given radius
        # C (3) The errors are the square root of a mean galaxy flux + sky flux
        # C (4) The galaxy falls off as a power law, 1/r or 1/r^2; pick 1/r
        # C Therefore we expect that:
        # C
        # C        df = (constant error flux) = K * sqrt(gxy/r + sky)
        # C        dm = df / f
        # C           = df * 10**(+0.4(m-m1))
        # C    log dm = log df + 0.4 m - 0.4 m1
        # C           = 0.4 m + C1 + C2 log (1+RSKY/r)
        # C
        # C RSKY is a non-linear parameter, so let's guess at it and then iterate.
        # C fit all the errors which are reasonable in terms of this
        # C linear function of m and r, determining C1 and C2.  Then a choice of
        # C limiting signal to noise gives us a function of mcut(r):
        # C     mlim(1) = alog10(1/snlim) - C1
        # C     mlim(2) = C2
        # C     mlim(3) = rsky
        # C     mcut = 2.5*(mlim(1) - mlim(2)*alog10(1+mlim(3)/r(i)))

        m=galphot.m
        dm=galphot.dm
        r=galphot.r
        sel=(m>mlim[3]) & (dm>0.001) & (dm<2.5) & (r>rlim[0]) & (r<rlim[1])
        print('WARNING: NO FAINT MAGNITUDE CUT HAS BEEN APPLIED SO FAR')

        #initalize some params
        tabsel=galphot.tab[sel]
        nfit=len(tabsel)
        # abcissa=np.zeros([2,nfit])
        # abcissa[0,]=1

        ordinate=np.log10(dm[sel])-0.4*m[sel]
        # print(m,dm)
        # ordinate=

        rms=[]
        iter=1#500
        print('WARNING: THIS ALGORITHM NEEDS REFINEMENT, at its present status it takes a value of rsky that is ad hoc')
        for itry in range(iter):
            rsky=rsky+drsky
            # nfit=0
            
            abcissa=np.log10(1+rsky/r[sel])
            # print(abcissa)
            guess_a=0.
            guess_b=1.
            popt_lin,pcov_lin = curve_fit(func_linear,abcissa,ordinate,p0=[guess_a,guess_b])


            # plt.scatter(abcissa,ordinate)
            # plt.plot(abcissa,func_linear(abcissa, *popt_lin),color='C1')
            # plt.show()
            # plt.clf()
            # rmsval=np.sqrt(np.sum(np.diag(pcov_lin)))
            diff=ordinate-func_linear(abcissa, *popt_lin)
            rmsval=np.sqrt(np.mean(diff**2)-np.mean(diff)**2)
            # print(rms)

            rms.append(rmsval)
            # print(rsky,popt_lin, rms, rms2)

            # plt.scatter(rsky,rms)

        # plt.show()
        # plt.clf()

        rmsmin=np.min(rms)
        self.rmsmin=rmsmin
        self.rsky=rsky#+drsky*(np.argmin(rms)+1)
        # print(rmsmin,self.rsky)

        # Calculate values of the mlim array

        mlim[0]=np.log10(1./snlim)-popt_lin[1]
        mlim[1]=popt_lin[0]
        mlim[2]=self.rsky
        self.mlim=mlim
        self.rlim=rlim
        # print(rlim)
        # print(mlim)
        r1=round(rlim[0])
        r2=round(rlim[1])
        rsamp=np.arange(r1,r2+1,1)
        # print(r)
        # print(self.rlim)
        mask_data=self.galphot.mask[self.galphot.mask_ext].data
        self.r_npix,self.npix=azimuthal_sum(mask_data, rsamp)
        

        ##### getdata finishes here



        
        rmin=np.min(r)
        rmax=np.max(r)
        emmin=np.min(m)
        emmax=np.max(m)
        emcut=2.5*(mlim[0]-mlim[1]*np.log10(1.+mlim[2]/r))
        # plt.plot(r,emcut)
        # plt.show()
        # plt.clf()
        self.emcut=emcut
        # self.emenv=np.max(emcut)
        sel_lims = (r>=rlim[0]) & (r<=rlim[1]) & (m>=mlim[3]) & (m<=emcut)
        self.galphot.nok=len(self.galphot.tab[sel_lims])
        # print("nobs= ",nobs)
        # print("nok= ",nok)

        # C Assume icolor = 0/1/2/3/4/5/6 for B/V/R/I/J/H/K
        # C GC maximum brightness: (from Harris Extragalactic Distance Scale)
        # C Calibration on MW gives  B0 = -7.0 +/- 0.15
        # C Calibration on M31 gives  B0 = -6.85  with (m-M)V = 24.6 assumed
        # C Calibration on Virgo gives B0 = -7.0  with (m-M) = 31.7 assumed
        # C Observed Virgo B0 = 24.7, adopting (m-M) = 30.9 (15 Mpc) gives B0 = -6.2
        # C Since these results will be used in conjunction with fluctuation analysis
        # C   which finds a Virgo distance of 15 Mpc, let us abuse Prof. Harris a
        # C   bit and adjust his absolute magnitude from -7.0 to -6.2.
        # C M_V = -7.0, B-V = 0.80, V-R = 0.45, R-I = 0.50, 
        # C and averages from Table 2, Frogel et al 1978 ApJ 220 75,
        # C where V-J/V-H/V-K = 1.62,2.14,2.23

        # C      vabs = -7.0
        # C jpb, change 21/Apr/2004
        # C Note: z-band assumes AB calibration, while the rest are Vega.
        # C jpb, 31/May/2007:
        # C     update z-band number following Andres's results, Mz0=-8.4

        # TOM in V, in Vegamag
        vabs=-7.4
        emabs_arr=np.array([0.8,0.,-0.5,-1.,-1.62,-2.14,-2.23,-1.])

        self.emabs=emabs_arr[icolor]+vabs
        self.cmax=galphot.distmod+self.emabs

        # C d logN / d m for gxy, Assume icolor = 0/1/2/3 for B/V/R/I, 
        # C Tyson gives slope = 0.48/0.42/0.39/0.34
        # C Gardner ApJL 415 L9 and Cowie ApJ 434 114 find more or less that
        # C N = 10^4 * 10^{0.3(K-19)} gxy/deg^2/mag
        # C Just guess at the same for J and H

        gamma_arr=np.array([0.48,0.42,0.39,0.34,.35,.3,.3,.3])
        self.gamma=gamma_arr[icolor]

        # C Galaxy density normalization from Tyson
        tysonempersec_arr=[30.25,30.45,30.55,30.60,30.60,29.37,29.37,29.37]
        self.tysonempersec=tysonempersec_arr[icolor]
        # C And interpolate for J and H

        # C Width of GC distribution
        # C      delta = 1.35
        # C jpb, 31/May/2007:
        # C     update this to reflect ACS results in Virgo, fornax, etc:
        # Delta changed by Mik from JPB's delta=1.4
        self.delta = delta

        # Create the par array where all the fit(?) parameters are stored
        par=np.array([beta,alpha,self.cmax,cnorm,self.gamma,delta, *mlim])
        # self.par=np.concatenate((par,mlim))
        self.par=par
        self.fit_pars=par[:npar]
        self.fixed_pars=par[npar:]

    def likely(self, fit_pars, fixed_pars):

        # C N = npoints
        # C R = radius
        # C M = magnitude
        # C DM = magnitude error
        # C RLIM(2) = rmin, rmax
        # C MLIM(4) = A,B,C, mmin, where mmax = 2.5*(A-Blog(1+C/R))
        # C Data is guaranteed to all lie within these limits
        # C
        # C Parameters are stored in /params/ common and referred to as equivalenced
        # C #gxy/mag/pixel^2(m) = GNORM * 10**GAMMA * (m - mg)
        # C #GC/mag/pixel^2(m,r) = CNORM * r**-ALPHA * exp(-((m)-CMAX)^2/(2*DELTA^2))
        # C NFIT refers to how many of these parameters are being varied by the
        # C minimization routine. All are initially set, however.
        # C
        # C The integrated number density is forced to the number observed
        # C
        # C Beyond the basic 6, PAR also holds magnitude limit cofficients A,B,C, (7-9)
        # C and the lower magnitude limit (10)   

        pars=[*fit_pars,*fixed_pars]
        beta=pars[0]
        alpha=np.abs(pars[1])  # ????
        cmax=pars[2]
        # print("Cmax=",cmax)
        cnorm=pars[3]
        gamma=pars[4]
        delta=pars[5]
        mlim=pars[6:]

        # par=[*fit_pars,*fixed_pars]
        # print(fit_pars)
        galphot=self.galphot

        r1=round(self.rlim[0])
        r2=round(self.rlim[1])
        rlim=self.rlim

        
        # acc=1e-10

        # C log N (sec^-2 mag^-2) = GAMMA * (m - mg)
        # C Tyson's prediction of how many background galaxies we should count
        # C is GSUM = Int { 10**(GAMMA*m) } * SECPERPIXEL^2 * 10**(-GAMMA*MG)
        # C We will use JT's rule: in any color there is 1 gxy/mag/sec^2 at mg=30.5

        emgalsec=30.5
        tysonscale= galphot.secperpixel**2 * 10**(-1.*gamma*emgalsec)

        # ??????
        # C If GAMMA has changed we need to recalculate the ML normalization
        # print(len(npix))
        # print(len(grand(par,r)))
        r_npix=self.r_npix
        npix=self.npix
        # print(r_npix)
        # print(tysonscale, gamma)
        gsum=np.sum(np.multiply(npix,grand(pars,r_npix)))*tysonscale/(gamma*np.log(10))
        # print(np.multiply(npix,grand(self.par,r_npix)))
        # C If ALPHA, DELTA, or CMAX has changed, we need to redo more ML norm.
        # print(len(npix))
        # print(len(crand(par,r)))
        csum=np.sum(np.multiply(npix,crand(pars,r_npix)))*np.sqrt(np.pi/2.)*delta
        # print(gsum,csum)
        ### NH: I am not using the oldalpha, olbeta, oldgamma... variables
        #       because we can compute the sum each time without overheads
        #       and this minimizes the confusion of having too many variables

        # $$$$$$$$$$$$$$$$$$$$$$$$$
        # NOT IMPLEMENTED THIS BETA CLIPPING PART YET:
        # C Make sure that mini isn't giving us a bogus BETA
        #       if(beta.gt.0d0.and.beta.lt.1d0) then
        #          betaclip = beta
        #       else
        #          betaclip = dmin1(1d0,dmax1(0d0,beta))
        #          write(6,'(a,f12.4)') 'LIKELY: bad value for BETA',beta
        #       end if

        if ((beta>0.) & (beta<1.)): betaclip=beta
        else: betaclip=min(1.,max(0.,beta))
        # print("Betaclip=",betaclip)

        r=galphot.r#tab['col4'] 
        m=galphot.m#tab['col12'] 
        sel_lims = (r>=rlim[0]) & (r<=rlim[1]) & (m>=mlim[3]) & (m<=self.emcut)

        tablim=galphot.tab[(sel_lims)]
        nok=len(tablim)
        # print(nok)

        msel=galphot.m[(sel_lims)]
        rsel=galphot.r[(sel_lims)]
        gdens=tysonscale*2*np.pi*rsel*10**(gamma*msel)
        # print(gdens)
        carg=-0.5*((msel-cmax)/delta)**2
        cdens=2*np.pi*rsel**(1.-alpha)*np.exp(carg)
        prob=np.log(betaclip*gdens/gsum+(1-betaclip)*cdens/csum)
        psum=np.sum(prob)
        # print(sum)

        likely=-1.*psum/nok
        # print("Likely= ",likely)

        # C We constrain NOK = CNORM*CSUM + GNORM*GSUM, and set 
        # C BETA = GNORM*GSUM/(CNORM*CSUM + GNORM*GSUM) = GNORM*GSUM / NOK
        # C Let us keep gnorm from dropping below 0.5

        betamin=0.5*gsum/nok
        betamax=1.

        # C While we're at it we can set CNORM
        cnorm=nok*(1-betaclip)/csum
        # print(cnorm)
        # print("cnorm= ",cnorm)

        if ((beta>1.) or (beta<0.)):
            wall=(beta-0.5)**6
        else:
            dbeta=0.01
            wallmax = np.exp((beta-betamax)/dbeta)
            wallmin = np.exp((betamin-beta)/dbeta)
            wall = (wallmax + wallmin)/nok

        if (wall>1e4):
            # print("Beta wall hit very hard   ", beta)
            wall=np.log(wall)**2
        
        likely=likely+wall

        # amag0=19
        # amag1=25
        # r0=10
        # r1=500

        # amag=np.linspace(amag0,amag1,100)
        # rr=np.linspace(r0,r1,100)
        # emf=par[6]+rr*(par[7]+rr*par[8])

        # self.gdens=gdens
        # self.cdens=cdens
        # self.gsum=gsum
        # self.csum=csum

        # print(betaclip)

        # return gsum, csum, likely
        # fit_pars[0]=beta
        # fit_pars[1]=alpha
        # # fit_pars[2]=cmax
        self.cnorm=cnorm
        self.beta=beta
        self.alpha=alpha
        self.cmax=cmax
        self.csum=csum
        self.gsum=gsum

        return likely
    

    def post_fit(self, fit_pars):

        # Function to post-process theFitParams object after minimizing
        # likely. It uses the final parameter values supplied, and generates
        # physical measurables such as the gnorm, cnorm, num of galaxies,
        # number of GCs etc and prints them. Also stores them in the object

        
        finlikely=self.likely(fit_pars,self.fixed_pars)
        print(f"Likelihood = {finlikely}")    
        gsum=self.gsum
        csum=self.csum
        gnorm=self.beta*self.galphot.nok/gsum
        tnorm=gnorm*10**(self.gamma*(self.tysonempersec-30.5))
        gcdist=self.galphot.distance*10**(0.2*(self.cmax-(self.galphot.distmod+self.emabs)))

        self.gnorm=gnorm
        self.tnorm=tnorm
        self.gcdist=gcdist
        print(f"Beta : {self.beta}")
        print(f"Alpha : {self.alpha}")
        print(f"Cmax : {self.cmax}")
        print(f"Gsum : {gsum}")
        print(f"Csum : {csum}")
        # print(gnorm,tnorm)

        self.totgc=csum*self.cnorm
        self.totgal=gsum*gnorm
        print(f"Total # GC: {csum*self.cnorm}")
        print(f"Total # galaxies: {gsum*gnorm}")
        print(f"Total # objects (nok): {self.galphot.nok}")

        print(f"Initial galaxy distance: {self.galphot.distance} Mpc")   
        print(f"Final galaxy distance: {gcdist} Mpc")

    def calc_p_r(self,  yessoft=0, npred=100, embin=0.5):

        # START FUNCTION FOR CALCULATION OF P_r
        # C Compute residual variance as a function of radius and luminosity function
        # C We will write a file called xxx.lkr which has parameters
        # C observed and predicted N(m), N(r), res_var(r), etc

        # C NANNULI is the number of annuli in which results are computed
        # C NMARG  is the number of bins for the marginal distributions
        # C NPRED  is the number of points where predicted results are reported

        emmin=np.min(self.galphot.m)
        emmax=np.max(self.galphot.m)
        em0=embin*int(emmin/embin)
        em1=embin*int(emmax/embin+0.999)
        print("CALCULATING P_r")
        # print(em0,em1)
        nmarg=min(npred,round((em1-em0)/embin))
        # print(nmarg)

        emmarg=em0+embin*(np.arange(nmarg)+0.5)
        # print(emmarg)

        rlim=self.rlim
        # print(rlim)

        nannuli=5
        self.nannuli=nannuli
        self.npred=npred
        self.embin=embin

        # 
        # print(np.arange(int(rlim[0]),int(rlim[1])+1))
        # print(ir)
        fracann=np.zeros(nannuli)
        rann=np.zeros(nannuli)
        fracmarg=np.zeros(nmarg)
        rmarg=np.zeros(nmarg)

        rmin=np.min(self.galphot.r)
        rmax=np.max(self.galphot.r)
        # print(len(self.npix))
        # print(self.r_npix)
        # r_npix=
        # print(len(range(int(rmin), int(rmax))))
        # count=0
        rsamp=np.arange(round(rmin), round(rmax)+1)
        mask_data=self.galphot.mask[self.galphot.mask_ext].data
        r_npix,npix=azimuthal_sum(mask_data, rsamp)
        # print(len(r_npix),len(rsamp))
        for i in range(len(r_npix)):
            # print(j)
            j=i+round(rmin)
            if ((j>=round(rlim[0])) & (j<round(rlim[1]))):
                # print(j)
                ir=int(nannuli*(j-rlim[0])/(rlim[1]-rlim[0]))
                # print(ir)
                fracann[ir]+=npix[i]
                rann[ir]+=j*npix[i]
                # count=count+1
            ir = int(nmarg*(j-rmin)/(rmax-rmin))
            # print(ir)
            fracmarg[ir]+=npix[i]
            rmarg[ir]+=j*npix[i]
            # print(i,j,int(rmax))
        # print(fracann)        

        # plt.plot(fracmarg, linestyle='None', marker='o')
        # plt.show()
        # plt.clf()
        for i in range(len(rann)):
            rann[i]=rann[i]/max(1.,fracann[i])
            r0=rlim[0]+i/nannuli*(rlim[1]-rlim[0])
            r1=rlim[0]+(i+1)/nannuli*(rlim[1]-rlim[0])
            fracann[i]=fracann[i]/(np.pi*(r1**2-r0**2))
        # plt.plot(fracann, linestyle='None', marker='o')
        # plt.show()
        # plt.clf()
        for i in range(len(rmarg)):
            rmarg[i]=rmarg[i]/max(1.,fracmarg[i])
            r0=rmin+i/nmarg*(rmax-rmin)
            r1=rmin+(i+1)/nmarg*(rmax-rmin)
            fracmarg[i]=fracmarg[i]/(np.pi*(r1**2-r0**2))
        # plt.plot(fracmarg, linestyle='None', marker='o')
        # plt.show()
        # plt.clf()
        r=self.galphot.r
        m=self.galphot.m
        # print(self.galphot.nobs,len(r))
        count_r_dist=np.zeros(npred)
        count_m_dist=np.zeros(npred)
        count_rm_dist=np.zeros((npred,nannuli))
        for i in range(self.galphot.nobs):
            j=int(nmarg*(r[i]-rmin)/(rmax-rmin)-0.001)
            # print(j)
            if ((j>=0) & (j<npred)):
                count_r_dist[j]+=1
            k=int((m[i]-em0)/embin)
            # print(k)
            if ((k>=0) & (k<npred)):
                count_m_dist[k]+=1
                j = int(nannuli*(r[i]-rlim[0])/(rlim[1]-rlim[0])-0.001)
                if ((j>=0) & (j<nannuli)):
                    count_rm_dist[k,j]+=1

        # print(count_rm_dist)
        # print(count_m_dist)

        sqarcminperpix= (self.galphot.secperpixel/60.)**2
        rscale=(rlim[1]-rlim[0])/nannuli

        mlim=self.mlim
        dens_r=np.zeros(nmarg)
        dens_m=np.zeros(nmarg)
        dens_rm=np.zeros((nmarg,nannuli))
        for i in range(nmarg):
            r0=rmin+i/nmarg*(rmax-rmin)
            # print(r0)
            r1=rmin+(i+1)/nmarg*(rmax-rmin)
            # C   Bright end cutoff: global
            emc0=mlim[3]
            # C   Faint end cutoff: local, function of each annulus-- rmarg(j) is mean
            # C   radius of each annulus
            emc1=2.5*(mlim[0] - mlim[1]*np.log10(1+mlim[2]/rmarg[i]))
            # print(emc1)
            #C   Area: total unexcised area in each annulus, in arcmin^2
            area=fracmarg[i]*np.pi*(r1**2-r0**2)*sqarcminperpix
            # C   dens(j,1): total number of objects in each annulus, in units of
            # C   mag^(-1) arcmin^(-2)
            dens_r[i]=count_r_dist[i]/(emc1-emc0)/area


            # C   Area: total unexcised area in entire frame, in arcmin^2
            area=self.galphot.npixtot*sqarcminperpix
            # C   dens(j,2): total number of objects in each magnitude bin, divided by total
            # C   unexcised area, in units of mag^(-1) arcmin^(-2)
            dens_m[i]=count_m_dist[i]/embin/area

            for j in range(nannuli):
                r0=rlim[0]+j/nannuli*(rlim[1]-rlim[0])
                r1=rlim[0]+(j+1)/nannuli*(rlim[1]-rlim[0])
                area=fracann[j]*np.pi*(r1**2-r0**2)*sqarcminperpix
                dens_rm[i,j]=count_rm_dist[i,j]/embin/area
        self.dens_r=dens_r
        self.dens_m=dens_m
        self.dens_rm=dens_rm

        # C Compute magnitude marginal distributions for gxy, gc, gxy+gc
        # C Here is the normalization for the galaxy density in #/pixel^2
        emgalsec=30.5
        gscale = self.gnorm * self.galphot.secperpixel**2 * 10**(-1*self.gamma*emgalsec)
        emstep = 0.1
        emextra = 2
        dens_gxy=np.zeros(npred)
        dens_gc=np.zeros((npred,nannuli))
        dens_gc_tot=np.zeros(npred)
        dens_both=np.zeros((npred,nannuli))
        dens_both_tot=np.zeros(npred)
        # C Get gxy density, save as dens(1,1)=dens_gxy
        # C Get GC total and annuli averages, save as dens(1,*), * = 2, 3... dens(1,2)=dens_gc_tot, dens(1,3..)=dens_gc
        # C Get both total and annuli averages, save as dens(2,*), * = 2, 3...dens(2,2)=dens_both_tot, dens(2,3..)=dens_both
        # print(r_npix)
        # print(int(rmin))
        for i in range(npred):
            em=mlim[3]-emextra+i*emstep
            dens_gxy[i]=gscale*10**(self.gamma*em)/sqarcminperpix

            gaussian = np.exp(-0.5*((em-self.cmax)/self.delta)**2)
            for j in range(len(r_npix)):
                # print(j)
                k=j+round(rmin)
                gcdensity= self.cnorm*k**(-1.*self.alpha)*gaussian
                if ((k>=round(rlim[0])) & (k<=round(rlim[1]))):
                    ir=int(nannuli*(j-rlim[0])/(rlim[1]-rlim[0]))
                    dens_gc[i,ir]+=npix[j]*gcdensity
                dens_gc_tot[i]+=npix[j]*gcdensity
            dens_gc_tot[i]=dens_gc_tot[i]/self.galphot.npixtot/sqarcminperpix
            dens_both_tot[i]=dens_gc_tot[i]+dens_gxy[i]

            for j in range(nannuli):
                r0=rlim[0]+j/nannuli*(rlim[1]-rlim[0])
                r1=rlim[0]+(j+1)/nannuli*(rlim[1]-rlim[0])
                dens_gc[i,j]=dens_gc[i,j]/(fracann[j]*np.pi*(r1**2-r0**2))/sqarcminperpix
                dens_both[i,j]=dens_gc[i,j]+dens_gxy[i]

        self.dens_gxy=dens_gxy
        self.dens_gc=dens_gc
        self.dens_gc_tot=dens_gc_tot
        self.dens_both=dens_both
        self.dens_both_tot=dens_both_tot

        # C Compute the residual variances
        rstep=round((1.2*rmax-rmin/2)/(npred-1))
        # print(rstep)


        ind=np.arange(npred)
        rr=ind*rstep+round(rmin/2)
        # print(rstep)
        # C Bright and faint magnitude limits
        emb=mlim[3]
        emf=2.5*(mlim[0] - mlim[1]*np.log10(1+mlim[2]/rr))

        # C residual mbar from GC's
        offset = 0.4 * np.log(10.) * self.delta**2
        arg = (emf-self.cmax+2*offset)/(np.sqrt(2.)*self.delta)
        arglog = self.cnorm * rr**(-1*self.alpha) * np.sqrt(np.pi/2) * self.delta * erfc(arg)
        embargc = self.cmax - offset - 1.25 * np.log10(arglog)

        # C residual mbar from gxy's
        arglog = gscale * 10**(self.gamma*emf) / (np.log(10.)*(0.8-self.gamma))
        embargxy = emf - 1.25*np.log10(arglog)

        # C residual mbar from both
        if (yessoft):
            print("YESSOFT CORRECTION: ON")
            gxylim = gscale * 10.**(self.gamma*emf)
            arggc = (emf-self.cmax)/self.delta
            gclim = self.cnorm* rr**(-1.*self.alpha) * np.exp(-0.5*arggc**2)
            fermi = 4*self.snlim/np.sqrt(2*np.pi)
            slopelim = 1 + (self.gamma*np.log(10.)*gxylim-arggc*gclim)/(gxylim+gclim)
            rat = 1+5./3.*(3-slopelim)**2/(fermi**2-(3-slopelim)**2)
            embar = -1.25*np.log10(rat*(10**(-0.8*embargc)+10**(-0.8*embargxy)))
            
        else:    
            print("YESSOFT CORRECTION: OFF")
            embar = -1.25*np.log10(10**(-0.8*embargc)+10**(-0.8*embargxy))

        self.embargxy=embargxy
        self.embargc=embargc
        self.embar=embar
        self.rr=rr
        self.yessoft=yessoft

        # print(embargxy,embargc,embar)
        # print(dens_gxy)
        # print("Testing P_r final step")
        # print(self.galphot.x0,self.galphot.y0)

    def make_ann_mask(self,rstart,rend):

        mask_data=np.copy(self.galphot.mask[self.galphot.mask_ext].data)
        x0,y0=self.galphot.x0,self.galphot.y0

        nx,ny=mask_data.shape
        
        for i in range(ny):
            for j in range(nx):
                r=round(np.sqrt((j-x0)**2+(i-y0)**2))
                if (r<rstart) or (r>rend):
                    mask_data[i,j]=0
        
        # plt.imshow(mask_data)
        # plt.title('AFTER')
        # plt.show()
        # plt.clf()
        
        return mask_data





            


    
    def gclf(self):
        
        pass

    
    def init_mask(self, fwhm, kscale):
        
        self.galphot.fwhm=fwhm

        # C Compute a bitmap which is masked at the locations of the point sources
        # C Use a psf p(r) = A { exp(-1/2*r^2/s^2) + 0.4 * exp(-r/s) }
        # C For this psf the FWHM = 2.14*s and Total flux = 1.4 * 2 * pi * s^2 * A

        s=fwhm/2.14

        # C Excise down to this surface brightness (mag/pix^2) limit
        # C jpb: make fainter for ACS
        # C      surfcut = 30
        surfcut=32
        rmin=max(2.,fwhm/2.)
        r=self.galphot.r
        m=self.galphot.m
        mlim=self.mlim
        self.kscale=kscale
        emcut=self.emcut
        x=self.galphot.x
        y=self.galphot.y
        cstar=self.galphot.cstar
        surf=np.zeros(len(m))

        msel_cond=(m<emcut)

        msel=m[(msel_cond)]
        # print(m)

        # C The surface brightness A of the selected objects is
        surf=msel+2.5*np.log10(1.4*2*np.pi*s*s)

        # C Ignore the gaussian center, and just use the exponential skirt
        # C emcut = surf - 2.5*alog10(0.4*exp(-r/s))
        rc = s * np.log(0.4*10**(0.4*(surfcut-surf)))

        rcut= list(map(lambda x: np.exp(-1.*(1-x/fwhm))*(fwhm-rmin)+rmin if (x<fwhm) else x, rc))
        # print(rcut)

        self.surf=surf
        self.rcut=rcut
        # tabsel=self.tab[(msel_cond)]
        # finmask=stompout()
    
    def stompout(self):

        tstartm=time.time()
        tab=self.galphot.tab
        r=self.galphot.r
        m=self.galphot.m
        x=self.galphot.x
        y=self.galphot.y
        cstar=self.galphot.cstar
        cxx=self.galphot.cxx
        cyy=self.galphot.cyy
        cxy=self.galphot.cxy
        kron=self.galphot.kron
        aa=self.galphot.aa

        emcut=self.emcut
        kscale=self.kscale

        rcut=self.rcut

        msel_cond=(m<emcut)

        cstar=cstar[msel_cond]
        x=x[msel_cond]
        y=y[msel_cond]
        cxx=cxx[msel_cond]
        cyy=cyy[msel_cond]
        cxy=cxy[msel_cond]
        kron=kron[msel_cond]
        aa=aa[msel_cond]

        ext_flag=list(map(lambda x: 1 if x<0.6 else 0, cstar))
        # print(len(rcut),len(ext_flag),len(tab))
        bitmask=np.zeros((self.galphot.mask[self.galphot.mask_ext].data).shape)
        # print(bitmask.shape)
        # print(ext_flag)
        nx,ny=bitmask.shape
        aa=np.multiply(aa,kron)*self.kscale
        # test=[706]#[180,706]
        for i in range(len(ext_flag)):

            if (ext_flag[i]):
                # aa[i]=aa[i]*self.kscale*kron[i]
                lrad=self.kscale*kron[i]
                ircut=int(2*aa[i]+0.99)
                # print(self.galphot.idx[msel_cond][i],x[i],y[i],kron[i],aa[i],lrad)
                # print(ircut)

                # box=bitmask[max(0,round(x[i]-ircut)):min(nx-1,round(x[i]+ircut)),max(0,round(y[i]-ircut)):min(ny-1,round(y[i]+ircut))]

                x_range=np.arange(max(0,round(x[i]-ircut)),min(nx,round(x[i]+ircut)))
                # print(x[i],y[i])
                y_range=np.arange(max(0,round(y[i]-ircut)),min(ny,round(y[i]+ircut)))
                for j in x_range:
                    for k in y_range:
                        ir2=cxx[i]*(j-x[i])**2 +cyy[i]*(k-y[i])**2+cxy[i]*(j-x[i])*(k-y[i])
                        # print(cxx[i],cxy[i],cyy[i])
                        # bitmask[j+1,k+1]=1
                        if (ir2<=lrad**2):
                            bitmask[k,j]=1
                            # print(ir2,lrad**2)

            else:
                ircut=round(rcut[i])
                x_range=np.arange(max(0,round(x[i]-ircut)),min(nx,round(x[i]+ircut)))
                # print(x[i],y[i])
                y_range=np.arange(max(0,round(y[i]-ircut)),min(ny,round(y[i]+ircut)))
                for j in x_range:
                    for k in y_range:
                        ir2=(j-x[i])**2+(k-y[i])**2
                        
                        if(ir2<=rcut[i]**2):
                            bitmask[k,j]=1
        
        tstopm=time.time()
        print(f"Time spent in stompout: {tstopm-tstartm} s")
        # fits.writeto(f"../OUTPUT/stompout.fits",bitmask, header= (self.galphot.mask[self.galphot.mask_ext].header), overwrite=True)
        hdu=fits.PrimaryHDU(data=bitmask, header=(self.galphot.mask[self.galphot.mask_ext].header))
        self.bitmask_hdu=fits.HDUList([hdu])
        # plt.imshow(bitmask)
        # plt.show()
        # plt.clf()
    
    def spresid(self, magzp, ann_mask=0):
        
        xgxy,ygxy=self.galphot.x0,self.galphot.y0
        bitmask=1-self.bitmask_hdu[0].data #Converting to bitmask where masked=0
   
        if (np.any(ann_mask)):
            finmask=np.logical_and(ann_mask,bitmask).astype(np.int32)

        self.finmask=finmask
        #plt.imshow(bitmask)
        #plt.show()
        # plt.clf()

        rr=self.rr
        embar=self.embar
        embargc=self.embargc
        embargxy=self.embargxy
        # plt.imshow(bitmask)
        # plt.show()
        # plt.clf()

        nx,ny=finmask.shape
        rpix=np.zeros(finmask.shape)
        for i in range(ny):
            for j in range(nx):
                # pass
                rpix[i,j]=np.sqrt((j-xgxy)**2+(i-ygxy)**2)

        # plt.imshow(rpix)
        # plt.show()
        # plt.clf()
        rmax,rmin=np.max(rpix),np.min(rpix)
        # print(rmin,rmax)
        rtot=0.
        pres=0
        ndata=np.sum(finmask)
        # print(len(rr))
        amr_arr=[]
        #print("I am a slowpoke", ny,nx)

        for i in range(ny):
            for j in range(nx):
                # pass
                if (finmask[i,j]):
                    rtot+=rpix[i,j]
                    # rpix=sqrt((j-xgxy)**2+(i-ygxy)**2)

                    ann=np.argmax(rr>rpix[i,j])
                    # print(ann, rpix[i,j], rr[ann])
                    # print(rr[ann],rpix[i,j])
                    # diff.append(np.abs(rr[ann]-rpix[i,j]))

                    amr=embar[ann-1]+(embar[ann]-embar[ann-1])*(rpix[i,j]-rr[ann-1])/(rr[ann]-rr[ann-1])
                    pres+=10**(-0.8*(amr-magzp))
                    amr_arr.append(amr)
        # print("I am not a slowpoke")

                    # return
        # print(np.min(diff),np.max(diff))
        # print(np.argmax(diff))
        # print(len(diff))
        amrmin, amrmax=np.min(amr_arr), np.max(amr_arr)
        self.rav=rtot/ndata
        self.pres=pres



def func_linear(x,a,b):
        return a*x+b


def azimuthal_sum(arr_2D,rsamp) :
    start=time.time()
    
    X, Y=np.meshgrid(np.arange(arr_2D.shape[1]),np.arange(arr_2D.shape[0]))

    x0=arr_2D.shape[1]/2
    y0=arr_2D.shape[0]/2
    # binsize=binsize
    
    R=np.sqrt((X-x0)**2+(Y-y0)**2)
    # print("AZIMUTHAL AVERAGE R TESTING")
    # print(np.max(R))

    # rsamp=np.arange(0,R.max(),binsize)
    rmed=[np.mean([rsamp[n], rsamp[n+1]]) for n in np.arange(len(rsamp)-1)]
    
    
    
    flux=[]
    
    for n in np.arange(len(rsamp)-1):
        l= (R >=rsamp[n]) & (R <rsamp[n+1])
        flux.append(np.sum(arr_2D[l]))
        # print(rsamp[n],np.median(arr_2D[l]) )
    
    # print(rmed,flux)
    end=time.time()
    print(f"Total time in making azimuthal sum= {end-start} sec")
    return rmed,np.array(flux)

def grand(par,r):
    # C Integrand for the galaxy part of the ML normalization
    emf=2.5*(par[6]-par[7]*np.log10(1.+par[8]/r))
    gamma=par[4]
    # print(par[6],par[7],par[8])
    # print(emf)
    # print(r)
    grand=10.**(gamma*emf)-10.**(gamma*par[9])
    return grand

def crand(par,r):
    # C Integrand for the cluster part of the ML normalization
    alpha=par[1]
    cmax=par[2]
    delta=par[5]
    emf=2.5*(par[6]-par[7]*np.log10(1.+par[8]/r))
    emgral=erfc((cmax-emf)/(np.sqrt(2.)*delta))-erfc((cmax-par[9])/(np.sqrt(2.)*delta))
    # print(emgral)
    crand=r**(-1.*alpha)*emgral

    # print(alpha,cmax,delta)
    # print(emf)
    # print(emgral)
    # print(crand)
    return crand





def main(galname, in_rad, out_rad):
    
    tstart=time.time()
    # table_path=f'../OUTPUT/{galname.upper()}/{galname}i_v1.mch'
    # mask_path=f'../OUTPUT/{galname.upper()}/mask_resid.fits'
    table_path=f'../OUTPUT/{galname.upper()}/{galname.upper()}_i_p_r.cat'
    # print(table_path)
    mask_path=f'../OUTPUT/{galname.upper()}/{galname.upper()}_i_p_r_mask_cutout.fits'
    mask_ext=0
    x0=768
    y0=768
    secperpixel=0.187
    magzp=30.

    kscale = 1.
    distance=20
    snlim=4
    # icol=1 #dummy
    icolor=3 #i-band
    # C Initial guess for gxy/total number
    beta = 0.5 #0.7
    # C Initial guess for d lnN / d lnr of GC distribution
    alpha = 0.5 #0.6

    cnorm=1.   # ??????????
    # C Width of GC distribution
    delta = 1.2

    rsky=50
    drsky=10

    npar=3  # Fit first 3 parameters only


    galphot=GalaxyPhot(table_path, mask_path, mask_ext, x0, y0, secperpixel, distance)

    fitparams=FitParams(galphot, beta, alpha, cnorm, delta, icolor, snlim, rsky, drsky, npar)

    
    
    fit_pars=fitparams.fit_pars
    fixed_pars=fitparams.fixed_pars
    print(f'Initial likelihood={fitparams.likely(fit_pars,fixed_pars)}')

        
    print(f"Initial Beta : {fitparams.beta}")
    print(f"Initial Alpha : {fitparams.alpha}")
    print(f"Initial Cmax : {fitparams.cmax}")
    mini_like=minimize(fitparams.likely, x0=fitparams.fit_pars, args=(fitparams.fixed_pars), method="Nelder-Mead")

    # print(f"Final Beta : {fitparams.beta}")
    # print(f"Final Alpha : {fitparams.alpha}")
    # print(f"Final Cmax : {fitparams.cmax}")
    print(mini_like)

    # print(fitparams.__dict__)
    finparams=copy.deepcopy(fitparams)

    # print(finparams.alpha)
    print('DISPLAYING FITTED PARAMETERS AND MEASURABLES:')
    finparams.post_fit(mini_like.x)

    finparams.calc_p_r(yessoft=1)

    fwhm=3 #in pixels
    finparams.init_mask(fwhm,kscale)

    finparams.stompout()

    bitmask_hdu=finparams.bitmask_hdu
    bitmask_hdu.writeto(f"../OUTPUT/{galname.upper()}/{galname.upper()}_i_stompout.fits", overwrite=True)

    mod=fits.open(f'../OUTPUT/{galname.upper()}/{galname.upper()}_i_mod.fits', do_not_scale_image_data=True)

    p_r_ann=np.zeros(in_rad.shape)
    for ann in range(len(in_rad)):

        ann_mask=finparams.make_ann_mask(rstart=in_rad[ann], rend=out_rad[ann])
        finparams.spresid(magzp,ann_mask=ann_mask)
        # mod=fits.open('../../P_r/sbf_pr_module/mask_resid.fits', do_not_scale_image_data=True)
        finmask=finparams.finmask
        # print(finmask.shape)

        from astropy.nddata.utils import Cutout2D
        moddata=mod[0].data
        # print(np.max(moddata))
        gxycen=(round(moddata.shape[0]/2),round(moddata.shape[1]/2))
        modcutout=Cutout2D(moddata, position=gxycen, size=finmask.shape)
        smaskmod=(finmask*modcutout.data)
        gtot=np.sum(smaskmod)
        print(f"P_r in annulus# {ann+1}={finparams.pres/gtot}")
        p_r_ann[ann]=finparams.pres/gtot

    
    
    
    
    
    
    
    tstop=time.time()
    print(f"Total time spent inside P_r module: {tstop-tstart} s")
    # plt.imshow(smaskmod)
    # plt.show()
    # plt.clf()

    return p_r_ann

if __name__=='__main__':
    import sys
    main(sys.argv[1:])
