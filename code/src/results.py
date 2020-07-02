#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 10:23:46 2020

@author: leguillou
"""

import numpy as np 
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
    true = true.reshape(nt,int(true.size//nt))
    ana = ana.reshape(nt,int(ana.size//nt)) 
    for t in range(nt):
        rmse = np.sqrt(np.sum(np.square(ana[t]-true[t]))/int(true.size//nt))
        score.append(1-rmse/np.std(true[t]))
    
    return score

def get_ana_traj(Xopt,M,time_assim,time_spinup=None):
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
    
    while t<t0+time_assim:
        
        u,v,h = M.step(t,u,v,h,He=He,hbcx=hbcx,hbcy=hbcy)
        
        u_ana.append(u)
        v_ana.append(v)
        h_ana.append(h)
        He_ana.append(M.get_He2d(t=t,He=He))
        hbcx0,hbcy0 = M.get_hbc1d(t=t,hbcx=hbcx,hbcy=hbcy)
        hbcx_ana.append(hbcx0)
        hbcy_ana.append(hbcy0)
        
        t += M.dt
    
    u_ana = np.asarray(u_ana)
    v_ana = np.asarray(v_ana)
    h_ana = np.asarray(h_ana)
    He_ana = np.asarray(He_ana)
    hbcx_ana = np.asarray(hbcx_ana)
    hbcy_ana = np.asarray(hbcy_ana)
    
    return u_ana,v_ana,h_ana,He_ana,hbcx_ana,hbcy_ana



def plot_result(
        u_true,v_true,h_true,He_true,hbcx_true,hbcy_true,
        u_ana,v_ana,h_ana,He_ana,hbcx_ana,hbcy_ana,
        tobs,niter,dir_out):

    gs = gridspec.GridSpec(6, 4,height_ratios=[1,1,1,1,0.5,0.5])
    fig = plt.figure(figsize=(30, 30)) 
    fig.suptitle(str(niter) + ' iterations')
    
    
    # Scores
    score_u  = compute_score(u_true,u_ana)
    score_v  = compute_score(v_true,v_ana)
    score_h  = compute_score(h_true,h_ana)
    score_He  = compute_score(He_true,He_ana)
    hbc_true_all = np.concatenate((
        hbcx_true[:,0,0,:],hbcx_true[:,0,1,:],
        hbcx_true[:,1,0,:],hbcx_true[:,1,1,:],
        hbcy_true[:,0,0,:],hbcy_true[:,0,1,:],
        hbcy_true[:,1,0,:],hbcy_true[:,1,1,:]),axis=1)
    hbc_ana_all = np.concatenate((
        hbcx_ana[:,0,0,:],hbcx_ana[:,0,1,:],
        hbcx_ana[:,1,0,:],hbcx_ana[:,1,1,:],
        hbcy_ana[:,0,0,:],hbcy_ana[:,0,1,:],
        hbcy_ana[:,1,0,:],hbcy_ana[:,1,1,:]),axis=1)
    score_hbc  = compute_score(hbc_true_all,hbc_ana_all)
    
    # Ranges
    range_u  = np.max(np.abs(u_true[0]))
    range_v  = np.max(np.abs(v_true[0]))
    range_h  = np.max(np.abs(h_true[0]))
    min_He = np.min(He_true[0])
    max_He = np.max(He_true[0])
    
    # U
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])  
    ax3 = plt.subplot(gs[0, 2:])  
    im = ax1.pcolormesh(u_true[0], vmin=-range_u,vmax=range_u,cmap='RdBu_r')
    ax2.pcolormesh(u_ana[0], vmin=-range_u,vmax=range_u,cmap='RdBu_r')
    cbar = plt.colorbar(im,ax=(ax1,ax2))
    cbar.ax.set_ylabel('U (m/S)')
    cbar.ax.ticklabel_format(style='sci', axis='y',scilimits=(0,0))  
    ax1.set_title('Truth')
    ax2.set_title('Analysis')
    ax3.plot(score_u)
    ax3.set_ylim(0.,1)

    
    # V
    ax1 = plt.subplot(gs[1, 0])
    ax2 = plt.subplot(gs[1, 1])  
    ax3 = plt.subplot(gs[1, 2:])  
    im = ax1.pcolormesh(v_true[0], vmin=-range_v,vmax=range_v,cmap='RdBu_r')
    ax2.pcolormesh(v_ana[0], vmin=-range_v,vmax=range_v,cmap='RdBu_r')
    cbar = plt.colorbar(im,ax=(ax1,ax2))
    cbar.ax.set_ylabel('V (m/S)')
    cbar.ax.ticklabel_format(style='sci', axis='y',scilimits=(0,0))  
    ax1.set_title('Truth')
    ax2.set_title('Analysis')
    ax3.plot(score_v)
    ax3.set_ylim(0.,1)
    
    # SLA
    ax1 = plt.subplot(gs[2, 0])
    ax2 = plt.subplot(gs[2, 1])  
    ax3 = plt.subplot(gs[2, 2:])  
    im = ax1.pcolormesh(h_true[0], vmin=-range_h,vmax=range_h,cmap='RdBu_r')
    ax2.pcolormesh(h_ana[0], vmin=-range_h,vmax=range_h,cmap='RdBu_r')
    cbar = plt.colorbar(im,ax=(ax1,ax2))
    cbar.ax.set_ylabel('SLA (m)')
    cbar.ax.ticklabel_format(style='sci', axis='y',scilimits=(0,0))  
    ax1.set_title('Truth')
    ax2.set_title('Analysis')
    ax3.plot(score_h)
    ax3.plot(tobs,0.5*np.ones((len(tobs))),'xr',markersize=20)
    ax3.set_ylim(0.,1)
    
    # He 
    ax1 = plt.subplot(gs[3, 0])
    ax2 = plt.subplot(gs[3, 1])  
    ax3 = plt.subplot(gs[3, 2:])  
    im = ax1.pcolormesh(He_true[0],vmin=min_He,vmax=max_He)
    ax2.pcolormesh(He_ana[0],vmin=min_He,vmax=max_He)
    cbar = plt.colorbar(im,ax=(ax1,ax2))
    cbar.ax.set_ylabel('He (m)')
    cbar.ax.ticklabel_format(style='sci', axis='y',scilimits=(0,0))  
    ax1.set_title('Truth')
    ax2.set_title('Analysis')
    ax3.plot(score_He)
    ax3.set_ylim(0.,1)
    
    # Boundary conditions 
    ax1 = plt.subplot(gs[4, 0])
    ax2 = plt.subplot(gs[4, 1])
    ax3 = plt.subplot(gs[5, 0])  
    ax4 = plt.subplot(gs[5, 1])
    ax5 = plt.subplot(gs[4:, 2:])   

    
    ax1.plot(hbcx_true[0,1,0],label='cos (truth)',c='b')
    ax1.plot(hbcx_ana[0,1,0],label='cos (ana)',c='b',linestyle='--')
    ax1.plot(hbcx_true[0,1,1],label='sin (truth)',c='r')
    ax1.plot(hbcx_ana[0,1,1],label='sin (ana)',c='r',linestyle='--')
    ax1.legend(loc=3)
    ax1.set_title('Northern boundary')        
    ax1.set_ylabel('SLA (m)')
    ax1.ticklabel_format(style='sci', axis='y',scilimits=(0,0))  
    ax1.set_ylim(-range_h,range_h)
    
    ax2.plot(hbcx_true[0,0,0],label='cos (true)',c='b')
    ax2.plot(hbcx_ana[0,0,0],label='cos (ana)',c='b',linestyle='--')
    ax2.plot(hbcx_true[0,0,1],label='sin (true)',c='r')
    ax2.plot(hbcx_ana[0,0,1],label='sin (ana)',c='r',linestyle='--')
    ax2.set_title('Southern boundary')
    ax2.set_ylabel('SLA (m)')
    ax2.set_xlabel('nx')
    ax2.ticklabel_format(style='sci', axis='y',scilimits=(0,0))  
    ax2.set_ylim(-range_h,range_h)
    
    ax3.plot(hbcy_true[0,0,0],label='cos (true)',c='b')
    ax3.plot(hbcy_ana[0,0,0],label='cos (ana)',c='b',linestyle='--')
    ax3.plot(hbcy_true[0,0,1],label='sin (true)',c='r')
    ax3.plot(hbcy_ana[0,0,1],label='sin (ana)',c='r',linestyle='--')
    ax3.set_title('Western boundary')
    ax3.ticklabel_format(style='sci', axis='y',scilimits=(0,0))  
    ax3.set_ylim(-range_h,range_h)
    
    ax4.plot(hbcy_true[0,1,0],label='cos (true)',c='b')
    ax4.plot(hbcy_ana[0,1,0],label='cos (ana)',c='b',linestyle='--')
    ax4.plot(hbcy_true[0,1,1],label='sin (true)',c='r')
    ax4.plot(hbcy_ana[0,1,1],label='sin (ana)',c='r',linestyle='--')
    ax4.set_title('Eastern boundary')
    ax4.ticklabel_format(style='sci', axis='y',scilimits=(0,0))  
    ax4.set_xlabel('ny')
    ax4.set_ylim(-range_h,range_h)
    
    ax5.plot(score_hbc)
    ax5.set_ylim(0.,1)
    
    fig.savefig( dir_out + '/results_iter' + str(niter).zfill(6),bbox_inches='tight' )
    
    plt.show()
    

def plot_traj(u_true,v_true,h_true,He_true,hbcS_true,hbcN_true,hbcW_true,hbcE_true,
        u_ana,v_ana,h_ana,He_ana,hbcS_ana,hbcN_ana,hbcW_ana,hbcE_ana,
        tobs,dir_out,dt):
    
    time = 0
    nt = len(u_true)
    
    
    
    # Ranges
    range_u = np.max(np.abs(u_true))
    range_v = np.max(np.abs(v_true))
    range_h = np.max(np.abs(h_true))
    min_He = np.min(He_true)
    max_He = np.max(He_true)
    
    for ind in range(nt):
        
        # Scores
        score_u = compute_score(np.asarray(u_true)[:ind],np.asarray(u_ana)[:ind])
        score_v = compute_score(np.asarray(v_true)[:ind],np.asarray(v_ana)[:ind])
        score_h = compute_score(np.asarray(h_true)[:ind],np.asarray(h_ana)[:ind])
        score_He = compute_score(np.asarray(He_true)[:ind],np.asarray(He_ana)[:ind])
        
        gs = gridspec.GridSpec(6, 4,height_ratios=[1,1,1,1,0.5,0.5])
        fig = plt.figure(figsize=(30, 30)) 
        fig.suptitle(str(round(ind*dt/3600,2)) + ' hrs')
    
        # U
        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[0, 1])  
        ax3 = plt.subplot(gs[0, 2:])  
        im = ax1.pcolormesh(u_true[ind], vmin=-range_u,vmax=range_u,cmap='RdBu_r')
        ax2.pcolormesh(u_ana[ind], vmin=-range_u,vmax=range_u,cmap='RdBu_r')
        cbar = plt.colorbar(im,ax=(ax1,ax2))
        cbar.ax.set_ylabel('U (m/S)')
        cbar.ax.ticklabel_format(style='sci', axis='y',scilimits=(0,0))  
        ax1.set_title('Truth')
        ax2.set_title('Analysis')
        ax3.plot(score_u)
        ax3.set_ylim(0.,1)
        ax3.set_xlim(0,nt)
    
        
        # V
        ax1 = plt.subplot(gs[1, 0])
        ax2 = plt.subplot(gs[1, 1])  
        ax3 = plt.subplot(gs[1, 2:])  
        im = ax1.pcolormesh(v_true[ind], vmin=-range_v,vmax=range_v,cmap='RdBu_r')
        ax2.pcolormesh(v_ana[ind], vmin=-range_v,vmax=range_v,cmap='RdBu_r')
        cbar = plt.colorbar(im,ax=(ax1,ax2))
        cbar.ax.set_ylabel('V (m/S)')
        cbar.ax.ticklabel_format(style='sci', axis='y',scilimits=(0,0))  
        ax1.set_title('Truth')
        ax2.set_title('Analysis')
        ax3.plot(score_v)
        ax3.set_ylim(0.,1)
        ax3.set_xlim(0,nt)
        
        # SLA
        ax1 = plt.subplot(gs[2, 0])
        ax2 = plt.subplot(gs[2, 1])  
        ax3 = plt.subplot(gs[2, 2:])  
        im = ax1.pcolormesh(h_true[ind], vmin=-range_h,vmax=range_h,cmap='RdBu_r')
        ax2.pcolormesh(h_ana[ind], vmin=-range_h,vmax=range_h,cmap='RdBu_r')
        cbar = plt.colorbar(im,ax=(ax1,ax2))
        cbar.ax.set_ylabel('SLA (m)')
        cbar.ax.ticklabel_format(style='sci', axis='y',scilimits=(0,0))  
        ax1.set_title('Truth')
        ax2.set_title('Analysis')
        ax3.plot(score_h)
        ax3.plot(tobs,0.5*np.ones((len(tobs))),'xr',markersize=20)
        ax3.set_ylim(0.,1)
        ax3.set_xlim(0,nt)
        
        # He 
        ax1 = plt.subplot(gs[3, 0])
        ax2 = plt.subplot(gs[3, 1])  
        ax3 = plt.subplot(gs[3, 2:])  
        im = ax1.pcolormesh(He_true[ind],vmin=min_He,vmax=max_He)
        ax2.pcolormesh(He_ana[ind],vmin=min_He,vmax=max_He)
        cbar = plt.colorbar(im,ax=(ax1,ax2))
        cbar.ax.set_ylabel('He (m)')
        cbar.ax.ticklabel_format(style='sci', axis='y',scilimits=(0,0))  
        ax1.set_title('Truth')
        ax2.set_title('Analysis')
        ax3.plot(score_He)
        ax3.set_ylim(0.,1)
        ax3.set_xlim(0,nt)
        
        # Boundary conditions 
        ax1 = plt.subplot(gs[4, 0])
        ax2 = plt.subplot(gs[4, 1])
        ax3 = plt.subplot(gs[5, 0])  
        ax4 = plt.subplot(gs[5, 1])
        ax5 = plt.subplot(gs[4:, 2:])   
    
        
        ax1.plot(hbcN_true[0],label='cos (truth)',c='b')
        ax1.plot(hbcN_ana[0],label='cos (ana)',c='b',linestyle='--')
        ax1.plot(hbcN_true[1],label='sin (truth)',c='r')
        ax1.plot(hbcN_ana[1],label='sin (ana)',c='r',linestyle='--')
        ax1.legend(loc=3)
        ax1.set_title('Northern boundary')        
        ax1.set_ylabel('SLA (m)')
        ax1.ticklabel_format(style='sci', axis='y',scilimits=(0,0))  
        ax1.set_ylim(-range_h,range_h)
        
        ax2.plot(hbcS_true[0],label='cos (true)',c='b')
        ax2.plot(hbcS_ana[0],label='cos (ana)',c='b',linestyle='--')
        ax2.plot(hbcS_true[1],label='sin (true)',c='r')
        ax2.plot(hbcS_ana[1],label='sin (ana)',c='r',linestyle='--')
        ax2.set_title('Southern boundary')
        ax2.set_ylabel('SLA (m)')
        ax2.set_xlabel('nx')
        ax2.ticklabel_format(style='sci', axis='y',scilimits=(0,0))  
        ax2.set_ylim(-range_h,range_h)
        
        ax3.plot(hbcW_true[0],label='cos (true)',c='b')
        ax3.plot(hbcW_ana[0],label='cos (ana)',c='b',linestyle='--')
        ax3.plot(hbcW_true[1],label='sin (true)',c='r')
        ax3.plot(hbcW_ana[1],label='sin (ana)',c='r',linestyle='--')
        ax3.set_title('Western boundary')
        ax3.ticklabel_format(style='sci', axis='y',scilimits=(0,0))  
        ax3.set_ylim(-range_h,range_h)
        
        ax4.plot(hbcE_true[0],label='cos (true)',c='b')
        ax4.plot(hbcE_ana[0],label='cos (ana)',c='b',linestyle='--')
        ax4.plot(hbcE_true[1],label='sin (true)',c='r')
        ax4.plot(hbcE_ana[1],label='sin (ana)',c='r',linestyle='--')
        ax4.set_title('Eastern boundary')
        ax4.ticklabel_format(style='sci', axis='y',scilimits=(0,0))  
        ax4.set_xlabel('ny')
        ax4.set_ylim(-range_h,range_h)
    
        fig.savefig( dir_out + '/snapshot_' + str(time).zfill(6),bbox_inches='tight' )
        if ind<nt:
            plt.close('all')
            
        time += dt
