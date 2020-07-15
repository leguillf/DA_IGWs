#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 10:23:46 2020

@author: leguillou
"""

import numpy as np 
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

params = {
        'font.size'           : 30      ,
        'axes.labelsize': 20,
        'axes.titlesize': 25,
        'xtick.labelsize'     : 15      ,
        'ytick.labelsize'     : 15      ,
        'legend.fontsize': 20,
        'legend.handlelength': 2,
        'lines.linewidth':4,
        'legend.title_fontsize':20}

plt.rcParams.update(params)

def compute_score(true,ana):
    score = []
    nt = true.shape[0]
    if nt>0:
        true = true.reshape(nt,int(true.size//nt))
        ana = ana.reshape(nt,int(ana.size//nt)) 
    else:
        true = true.ravel()
        ana = ana.ravel()
    for t in range(nt):
        rmse = np.sqrt(np.sum(np.square(ana[t]-true[t]))/int(true.size//nt))
        score.append(1-rmse/np.std(true[t]))
    
    return score

def get_ana_traj(Xopt,M,times_true,time_assim,time_spinup=None):
    
    u = Xopt[M.sliceu].reshape(M.shapeu)
    v = Xopt[M.slicev].reshape(M.shapev)
    h = Xopt[M.sliceh].reshape(M.shapeh)
    He = Xopt[M.sliceHe].reshape(M.shapeHe)
    hbcx = Xopt[M.slicehbcx].reshape(M.shapehbcx)
    hbcy = Xopt[M.slicehbcy].reshape(M.shapehbcy)
    
    # Spin-up
    t0 = 0
    if time_spinup is not None:
        traj_spinup = M.run(t0,time_spinup,u,v,h,He=He,hbcx=hbcx,hbcy=hbcy)
        u,v,h = traj_spinup[-1]
        t0 += time_spinup//M.dt*M.dt 
            
    u_ana = [u]
    v_ana = [v]
    h_ana = [h]
    He_ana = [M.get_He2d(t=t0,He=He)]
    hbcx0,hbcy0 = M.get_hbc1d(t=t0,hbcx=hbcx,hbcy=hbcy)
    hbcx_ana = [hbcx0]
    hbcy_ana = [hbcy0]
    
    t = t0
    
    tt = [t-t0]
    
    while t<t0+time_assim:
        
        u,v,h = M.step(t,u,v,h,He=He,hbcx=hbcx,hbcy=hbcy)
        t += M.dt
        
        u_ana.append(u)
        v_ana.append(v)
        h_ana.append(h)
        He_ana.append(M.get_He2d(t=t,He=He))
        hbcx0,hbcy0 = M.get_hbc1d(t=t,hbcx=hbcx,hbcy=hbcy)
        hbcx_ana.append(hbcx0)
        hbcy_ana.append(hbcy0)
        
        tt.append(t-t0)
    
    u_ana = np.asarray(u_ana)
    v_ana = np.asarray(v_ana)
    h_ana = np.asarray(h_ana)
    He_ana = np.asarray(He_ana)
    hbcx_ana = np.asarray(hbcx_ana)
    hbcy_ana = np.asarray(hbcy_ana)
    tt = np.asarray(tt)

    # time interpolation
    f = interpolate.interp1d(tt,u_ana,axis=0)
    u_ana = f(times_true)
    f = interpolate.interp1d(tt,v_ana,axis=0)
    v_ana = f(times_true)
    f = interpolate.interp1d(tt,h_ana,axis=0)
    h_ana = f(times_true)
    f = interpolate.interp1d(tt,He_ana,axis=0)
    He_ana = f(times_true)
    f = interpolate.interp1d(tt,hbcx_ana,axis=0)
    hbcx_ana = f(times_true)
    f = interpolate.interp1d(tt,hbcy_ana,axis=0)
    hbcy_ana = f(times_true)
    
    return u_ana,v_ana,h_ana,He_ana,hbcx_ana,hbcy_ana



def plot_diags_scores(
        u_igws_true,v_igws_true,h_igws_true,h_true,
        u_ana,v_ana,h_ana,He_ana,hbcx_ana,hbcy_ana,
        times,tobs,dir_out,ind,title,plot=False):

    fig,axs = plt.subplots(4,3,figsize=(30, 30)) 
    
    fig.suptitle(title)
    
    # Scores 
    score_u = compute_score(u_igws_true,u_ana)
    score_v = compute_score(v_igws_true,v_ana)
    score_h = compute_score(h_igws_true,h_ana)
    
    # Ranges
    range_u = 0.9*np.max(np.abs(u_igws_true))
    range_v = 0.9*np.max(np.abs(v_igws_true))
    range_h = 0.9*np.max(np.abs(h_igws_true))
    
    # U
    im1 = axs[0,0].pcolormesh(u_igws_true[ind],vmin=-range_u,vmax=range_u,cmap='RdBu_r')
    cbar = plt.colorbar(im1,ax=axs[0,0])
    cbar.ax.ticklabel_format(style='sci', axis='y',scilimits=(0,0))  
    axs[0,0].set_title('u true')
    im2 = axs[0,1].pcolormesh(u_ana[ind],vmin=-range_u,vmax=range_u,cmap='RdBu_r')
    cbar = plt.colorbar(im2,ax=axs[0,1])
    cbar.ax.ticklabel_format(style='sci', axis='y',scilimits=(0,0))  
    axs[0,1].set_title('u ana')
    axs[0,2].set_title('RMSE score (u)')
    axs[0,2].plot(times[:ind]/3600/24,score_u[:ind])
    axs[0,2].set_xlim(0,times[-1]/3600/24)
    axs[0,2].set_ylim(0,1)
    axs[0,2].set_xlabel('time (days)')
    
    # V
    im1 = axs[1,0].pcolormesh(v_igws_true[ind],vmin=-range_v,vmax=range_v,cmap='RdBu_r')
    cbar = plt.colorbar(im1,ax=axs[1,0])
    cbar.ax.ticklabel_format(style='sci', axis='y',scilimits=(0,0))  
    axs[1,0].set_title('v true')
    im2 = axs[1,1].pcolormesh(v_ana[ind],vmin=-range_v,vmax=range_v,cmap='RdBu_r')
    cbar = plt.colorbar(im2,ax=axs[1,1])
    cbar.ax.ticklabel_format(style='sci', axis='y',scilimits=(0,0))  
    axs[1,1].set_title('v ana')
    axs[1,2].set_title('RMSE score (v)')
    axs[1,2].plot(times[:ind]/3600/24,score_v[:ind])
    axs[1,2].set_xlim(0,times[-1]/3600/24)
    axs[1,2].set_ylim(0,1)
    axs[1,2].set_xlabel('time (days)')
    
    # SSH
    im1 = axs[2,0].pcolormesh(h_igws_true[ind],vmin=-range_h,vmax=range_h,cmap='RdBu_r')
    cbar = plt.colorbar(im1,ax=axs[2,0])
    cbar.ax.ticklabel_format(style='sci', axis='y',scilimits=(0,0))  
    axs[2,0].set_title('ssh true')
    im2 = axs[2,1].pcolormesh(h_ana[ind],vmin=-range_h,vmax=range_h,cmap='RdBu_r')
    cbar = plt.colorbar(im2,ax=axs[2,1])
    cbar.ax.ticklabel_format(style='sci', axis='y',scilimits=(0,0))  
    axs[2,1].set_title('ssh ana')
    axs[2,2].plot(times[:ind]/3600/24,score_h[:ind])
    axs[2,2].set_title('RMSE score (ssh)')
    axs[2,2].plot(tobs/3600/24,0.5*np.ones((len(tobs))),'xr',markersize=20)
    axs[2,2].set_xlim(0,times[-1]/3600/24)
    axs[2,2].set_xlabel('time (days)')
    axs[2,2].set_ylim(0,1)
    
    # He & Boundary conditions 
    im1 = axs[3,0].pcolormesh(h_true[ind])
    cbar = plt.colorbar(im1,ax=axs[3,0])
    cbar.ax.ticklabel_format(style='sci', axis='y',scilimits=(0,0))  
    axs[3,0].set_title('ssh (full) true')
    
    im2 = axs[3,1].pcolormesh(He_ana[ind])
    cbar = plt.colorbar(im2,ax=axs[3,1])
    cbar.ax.ticklabel_format(style='sci', axis='y',scilimits=(0,0))  
    axs[3,1].set_title('He ana')
    
    axs[3,2].plot(hbcx_ana[ind,0,0],label='South (cos)',c='b')
    axs[3,2].plot(hbcx_ana[ind,0,1],label='South (sin)',c='b',linestyle='--')
    axs[3,2].plot(hbcx_ana[ind,1,0],label='North (cos)',c='r')
    axs[3,2].plot(hbcx_ana[ind,1,1],label='North (sin)',c='r',linestyle='--')
    axs[3,2].plot(hbcy_ana[ind,0,0],label='West (cos)',c='c')
    axs[3,2].plot(hbcy_ana[ind,0,1],label='West (sin)',c='c',linestyle='--')
    axs[3,2].plot(hbcy_ana[ind,1,0],label='East (cos)',c='g')
    axs[3,2].plot(hbcy_ana[ind,1,1],label='East (sin)',c='g',linestyle='--')
    axs[3,2].set_ylabel('ssh (m)')
    axs[3,2].legend()
    
    if plot:
        plt.show()
 
    return fig
        
    
    

def plot_traj_scores(u_igws_true,v_igws_true,h_igws_true,h_true,
        u_ana,v_ana,ssh_ana,He_ana,hbcx_ana,hbcy_ana,
        times,tobs,dir_out,dt,plot=False):
    
    time = 0
    nt = len(h_true)
    
    
    
    for ind in range(nt):
        
        fig = plot_diags_scores(
            u_igws_true,v_igws_true,h_igws_true,h_true,
            u_ana,v_ana,ssh_ana,He_ana,hbcx_ana,hbcy_ana,
            times,tobs,dir_out,ind,str(round(time/3600,1)) + ' hrs',plot)
        fig.savefig(dir_out + '/snapshot_iter' + str(ind).zfill(6),bbox_inches='tight')
        
        fig.clf()
        plt.close()
        
        time += dt
