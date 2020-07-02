#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 16:47:46 2020

@author: leguillou
"""

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




def plot_result(u_true,v_true_h_trie,u_ana,v_ana,h_ana,score_U,score_V,score_H,range_U,range_V,range_H,niter):

    gs = gridspec.GridSpec(5, 4,height_ratios=[1,1,1,0.5,0.5])
    fig = plt.figure(figsize=(30, 25)) 
    fig.suptitle(str(niter) + ' iterations')
    
    # U
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])  
    ax3 = plt.subplot(gs[0, 2:])  
    im = ax1.pcolormesh(u_true[ind], vmin=-range_U,vmax=range_U,cmap='RdBu_r')
    ax2.pcolormesh(u_ana[ind], vmin=-range_U,vmax=range_U,cmap='RdBu_r')
    cbar = plt.colorbar(im,ax=(ax1,ax2))
    cbar.ax.set_ylabel('U (m/S)')
    cbar.ax.ticklabel_format(style='sci', axis='y',scilimits=(0,0))  
    ax1.set_title('Truth')
    ax2.set_title('Analysis')
    ax3.plot(score_U[:ind])
    ax3.set_ylim(0.,1)
    ax3.set_xlim(0,nt)
    
    # V
    ax1 = plt.subplot(gs[1, 0])
    ax2 = plt.subplot(gs[1, 1])  
    ax3 = plt.subplot(gs[1, 2:])  
    im = ax1.pcolormesh(v_true[ind], vmin=-range_V,vmax=range_V,cmap='RdBu_r')
    ax2.pcolormesh(v_ana[ind], vmin=-range_V,vmax=range_V,cmap='RdBu_r')
    cbar = plt.colorbar(im,ax=(ax1,ax2))
    cbar.ax.set_ylabel('V (m/S)')
    cbar.ax.ticklabel_format(style='sci', axis='y',scilimits=(0,0))  
    ax1.set_title('Truth')
    ax2.set_title('Analysis')
    ax3.plot(score_V[:ind])
    ax3.set_ylim(0.,1)
    ax3.set_xlim(0,nt)
    
    # H
    ax1 = plt.subplot(gs[2, 0])
    ax2 = plt.subplot(gs[2, 1])  
    ax3 = plt.subplot(gs[2, 2:])  
    im = ax1.pcolormesh(h_true[ind], vmin=-range_H,vmax=range_H,cmap='RdBu_r')
    ax2.pcolormesh(h_ana[ind], vmin=-range_H,vmax=range_H,cmap='RdBu_r')
    cbar = plt.colorbar(im,ax=(ax1,ax2))
    cbar.ax.set_ylabel('SLA (m)')
    cbar.ax.ticklabel_format(style='sci', axis='y',scilimits=(0,0))  
    ax1.set_title('Truth')
    ax2.set_title('Analysis')
    ax3.plot(score_H[:ind])
    ax3.plot(tobs,0.5*np.ones((len(tobs))),'xr',markersize=20)
    ax3.set_ylim(0.,1)
    ax3.set_xlim(0,nt)
    
    # He & Boundary conditions
    ax1 = plt.subplot(gs[3:, 0])
    ax2 = plt.subplot(gs[3:, 1])  
    ax3 = plt.subplot(gs[3, 2])  
    ax4 = plt.subplot(gs[3, 3])  
    ax5 = plt.subplot(gs[4, 2])  
    ax6 = plt.subplot(gs[4, 3])  
    
    im = ax1.pcolormesh(He_true,vmin=np.min(He_ana),vmax=np.max(He_ana))
    ax2.pcolormesh(He_ana,vmin=np.min(He_ana),vmax=np.max(He_ana))
    cbar = plt.colorbar(im,ax=(ax1,ax2))
    cbar.ax.set_ylabel('He (m)')
    cbar.ax.ticklabel_format(style='sci', axis='y',scilimits=(0,0))  
    ax1.set_title('Truth')
    ax2.set_title('Analysis')
    
    ax3.plot(hN_cos_true,label='cos (truth)',c='b',linestyle='--')
    ax3.plot(hN_cos,label='cos (ana)',c='r',linestyle='--')
    ax3.plot(hN_sin_true,label='sin (truth)',c='b')
    ax3.plot(hN_sin,label='sin (ana)',c='r')
    ax3.legend(loc=3)
    ax3.set_title('Northern boundary')
    ax3.set_ylabel('SLA (m)')
    ax3.ticklabel_format(style='sci', axis='y',scilimits=(0,0))  
    
    ax4.plot(hW_cos_true,label='cos (true)',c='b',linestyle='--')
    ax4.plot(hW_cos,label='cos (ana)',c='r',linestyle='--')
    ax4.plot(hW_sin_true,label='sin (true)',c='b')
    ax4.plot(hW_sin,label='sin (ana)',c='r')
    ax4.set_title('Western boundary')
    ax4.ticklabel_format(style='sci', axis='y',scilimits=(0,0))  
    
    ax5.plot(hS_cos_true,label='cos (true)',c='b',linestyle='--')
    ax5.plot(hS_cos,label='cos (ana)',c='r',linestyle='--')
    ax5.plot(hS_sin_true,label='sin (true)',c='b')
    ax5.plot(hS_sin,label='sin (ana)',c='r')
    ax5.set_title('Southern boundary')
    ax5.set_ylabel('SLA (m)')
    ax5.set_xlabel('nx')
    ax5.ticklabel_format(style='sci', axis='y',scilimits=(0,0))  
    
    ax6.plot(hE_cos_true,label='cos (true)',c='b',linestyle='--')
    ax6.plot(hE_cos,label='cos (ana)',c='r',linestyle='--')
    ax6.plot(hE_sin_true,label='sin (true)',c='b')
    ax6.plot(hE_sin,label='sin (ana)',c='b')
    ax6.set_title('Eastern boundary')
    ax6.ticklabel_format(style='sci', axis='y',scilimits=(0,0))  
    ax6.set_xlabel('ny')