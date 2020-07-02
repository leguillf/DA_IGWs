#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 20:39:10 2020

@author: leguillou
"""

import numpy as np 
import scipy.interpolate
import scipy.fftpack
from scipy.fftpack import ifft, ifft2, fft, fft2
import scipy.interpolate

def gaspari_cohn(r,c):
    
    if type(r) is float or type(r) is int:
        ra = np.array([r])
    else:
        ra = r
    if c<=0:
        return np.zeros_like(ra)
    else:
        ra = 2*np.abs(ra)/c
        gp = np.zeros_like(ra)
        i= np.where(ra<=1.)[0]
        if len(i)>0:
            gp[i]=-0.25*ra[i]**5+0.5*ra[i]**4+0.625*ra[i]**3-5./3.*ra[i]**2+1.
        i =np.where((ra>1.)*(ra<=2.))[0]
        if len(i)>0:
            gp[i] = 1./12.*ra[i]**5-0.5*ra[i]**4+0.625*ra[i]**3+5./3.*ra[i]**2-5.*ra[i]+4.-2./3./ra[i]
        if type(r) is float:
            gp = gp[0]
    return gp


def gen_signal1d(fi, PSi, x, fmin=None, fmax=None, alpha=10, seed=None, lf_extpl=False, hf_extpl=False):
    
    # Make sure fi, PSi does not contain the zero frequency:
    PSi = PSi[fi>0]
    fi = fi[fi>0]
    
    # Interpolation function for the non-zero part of the spectrum
    finterp = scipy.interpolate.interp1d(np.log(fi[PSi>0]), np.log(PSi[PSi>0]), bounds_error=False, fill_value="extrapolate") 
    
    # Adjust fmin and fmax to fi bounds if not specified:
    if fmin==None: fmin=fi[0]
    if fmax==None: fmax=fi[-1]
    fmaxr = alpha*fmax # We need to go alpha times further in frequency to avoid interpolation aliasing. Recomm: alpha=10

    f = np.arange(fmin,fmaxr+fmin, fmin)

    PS = np.exp(finterp(np.log(f))) 

    # lf_extpl=True prolongates the PSi as a plateau below min(fi). Otherwise, we consider zeros values. same for hf
    if lf_extpl: PS[f<fi[0]] = PSi[0]
    else: PS[f<fi[0]] = 0.    
    if hf_extpl: PS[f>fi[-1]] = PSi[-1]
    else: PS[f>fi[-1]] = 0.
        
    PS[f>fmax]=0.
    
    # Detect the sections (if any) where PSi==0 and apply it to PS
    finterp_mask = scipy.interpolate.interp1d(fi, PSi, bounds_error=False, fill_value="extrapolate")
    PSmask = finterp_mask(f)
    PS[PSmask==0.] = 0.
    

    phase=np.empty((2*len(f)+1))
    np.random.seed(seed=seed)
    phase[1:len(f)+1] = np.random.random(len(f))*2*np.pi
    phase[0] = 0.
    phase[-len(f):]=-phase[1:len(f)+1][::-1]
    
    FFT1A = np.concatenate(([0],0.5* PS,0.5* PS[::-1]),axis=0)**0.5 * np.exp(1j*phase)/ fmin**0.5

    yg = 2*fmaxr*np.real(scipy.fftpack.ifft(FFT1A))
    xg=np.linspace(0,0.5/fmaxr*yg.shape[0],yg.shape[0])

    finterp = scipy.interpolate.interp1d(xg, yg)
    y=finterp(np.mod(x,xg.max()))

    return y


def gen_signal2d_rectangle(fi, PSi, x,y, fminx=None,fminy=None, fmax=None, alpha=10, seed=None, lf_extpl=False, hf_extpl=False):
    
    # dx=x[1]-x[0]
    revert=False
    if fminy<fminx:
        revert = True
        fmin = +fminy
        fminy = +fminx
        x,y = y,x
    else:
        fmin = +fminx
    
    
    # Make sure fi, PSi does not contain the zero frequency:
    PSi = PSi[fi>0]
    fi = fi[fi>0]
    
    
    # Interpolation function for the non-zero part of the spectrum
    finterp = scipy.interpolate.interp1d(np.log(fi[PSi>0]), np.log(PSi[PSi>0]), bounds_error=False, fill_value="extrapolate") 

    fmaxr = alpha*fmax # We need to go alpha times further in frequency to avoid interpolation aliasing. Recomm: alpha=10

    f = np.arange(fmin,fmaxr+fmin, fmin)
    ###df = +fmin

    PS = np.exp(finterp(np.log(f))) 

    # lf_extpl=True prolongates the PSi as a plateau below min(fi). Otherwise, we consider zeros values. same for hf
    if lf_extpl: PS[f<fi[0]] = PSi[0]
    else: PS[f<fi[0]] = 0.    
    if hf_extpl: PS[f>fi[-1]] = PSi[-1]
    else: PS[f>fi[-1]] = 0.
        
    PS[f>fmax]=0.
    
    # Detect the sections (if any) where PSi==0 and apply it to PS
    finterp_mask = scipy.interpolate.interp1d(fi, PSi, bounds_error=False, fill_value="extrapolate")
    PSmask = finterp_mask(f)
    PS[PSmask==0.] = 0.
    PS1D = PS
    
    
    # Build the 2D PSD following the given 1D PSD
    fx = np.concatenate(([0],f))
    fy = np.concatenate(([0],np.arange(fminy,fmaxr+fminy, fminy)))
    fx2,fy2=np.meshgrid(fx,fy)
    f2=(fx2**2+fy2**2)**0.5
    dfx=fmin
    dfy=fminy

    PS2D = f2*0.

    for iff in range(len(f)):
        fc = f2[:,-iff-1]
        ind1 = np.where((fc>=f[-iff-1]-dfx/2)&(fc<f[-iff-1]+dfx/2))[0]
        S = np.sum(PS2D[:,-iff-1]) * dfx*dfy
        MISS = PS1D[-iff-1]*dfx - S
        if MISS<=0:
            PS2D[np.where((f2>=f[-iff-1]-dfx/2)&(f2<f[-iff-1]+dfx/2))] = 0.
        else:
            PS2D[np.where((f2>=f[-iff-1]-dfx/2)&(f2<f[-iff-1]+dfx/2))] = MISS / len(ind1) / (dfx*dfy)

    PS2D[f2>fmax]=0
    
    np.random.seed(seed=seed)
    phase = (np.random.random((2*len(fy)-1,len(fx)))*2*np.pi)
    phase[0,0]=0.
    phase[-len(fy)+1:,0]=-phase[1:len(fy),0][::-1]

    FFT2A = np.concatenate((0.25*PS2D,0.25*PS2D[1:,:][::-1,:]),axis=0)**0.5 * np.exp(1j*phase)/(dfx*dfy)**0.5

    FFT2 = np.zeros((2*len(fy)-1,2*len(fx)-1),dtype=complex)
    FFT2[:,:len(fx)]=FFT2A
    FFT2[1:,-len(fx)+1:]=FFT2A[1:,1:].conj()[::-1,::-1]
    FFT2[0,-len(fx)+1:]=FFT2A[0,1:].conj()[::-1]

    sg = (4*fy[-1]*fx[-1]) * np.real(scipy.fftpack.ifft2(FFT2))
    xg = np.linspace(0,1./fmin,sg.shape[1])
    yg = np.linspace(0,1./fminy,sg.shape[0])   
    

    finterp = scipy.interpolate.interp2d(xg, yg, sg)
    
    yl = y-y[0]
    yl = yl[yl<yg.max()]
    xl = x-x[0]
    xl = xl[xl<xg.max()]
    rectangle = finterp(xl,yl)
    
    x_n, x_r = np.divmod(x.max()-x[0], xg.max())
    y_n, y_r = np.divmod(y.max()-y[0], yg.max())
    
    signal = np.zeros((len(y),len(x)))
    
    for i_x_n in range(int(x_n+1)):
        ix0=np.where(((x-x[0])>=i_x_n*xg.max())&((x-x[0])<(i_x_n+1)*xg.max()))[0]
        for i_y_n in range(int(y_n+1)):
            iy0=np.where(((y-y[0])>=i_y_n*yg.max())&((y-y[0])<(i_y_n+1)*yg.max()))[0]
            signal[iy0[0]:iy0[-1]+1, ix0[0]:ix0[-1]+1] = rectangle[:len(iy0),:len(ix0)]

    
    
    if revert == True:
        return signal.transpose()
    else:
        return signal
