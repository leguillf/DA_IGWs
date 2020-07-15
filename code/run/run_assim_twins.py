#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 16:32:00 2020

@author: leguillou
"""

###############################################################################
# Packages, libraries
###############################################################################
import sys,os
import numpy as np 
import matplotlib.pyplot as plt
sys.path.insert(0,'../src')
from swm_rk import Swm
from H import *
from B import *
from var import *
from tools import *
from results import *
import scipy.optimize as opt
import shutil
from datetime import datetime
from scipy import interpolate
import xarray as xr


###############################################################################
# Config
###############################################################################
if len(sys.argv)==2:
    file_config = str(sys.argv[1])
else:
    file_config = 'config_example'
path_config = '../config/' + file_config + '.py'
shutil.copyfile(path_config, 'config_specific.py')
from config_specific import *

###############################################################################
# NR
###############################################################################
print('*** Nature run ***')
ds = xr.open_dataset(path_nr)
print(ds)

times_true = ds[name_nr_time][t1:t2].values 
times_true -= times_true[0]
u_true = ds[name_nr_u][t1:t2,i1:i2,j1:j2].values
v_true = ds[name_nr_v][t1:t2,i1:i2,j1:j2].values
ssh_true = ds[name_nr_ssh][t1:t2,i1:i2,j1:j2].values
u_igws_true = ds[name_nr_u_igws][t1:t2,i1:i2,j1:j2].values
v_igws_true = ds[name_nr_v_igws][t1:t2,i1:i2,j1:j2].values
ssh_igws_true = ds[name_nr_ssh_igws][t1:t2,i1:i2,j1:j2].values

if plot:
    fig, (ax1,ax2,ax3,ax4) = plt.subplots(1,4, figsize=(20,5))
    im1 = ax1.pcolormesh(u_true[0],cmap='RdBu_r')
    im2 = ax2.pcolormesh(v_true[0],cmap='RdBu_r')
    im3 = ax3.pcolormesh(ssh_true[0])
    cbar = plt.colorbar(im1,ax=ax1)
    cbar.ax.set_title('m/s')
    cbar = plt.colorbar(im2,ax=ax2)
    cbar.ax.set_title('m/s')
    cbar = plt.colorbar(im3,ax=ax3)
    cbar.ax.set_title('m')
    ax1.set_title('U',fontsize=15)
    ax2.set_title('V',fontsize=15)
    ax3.set_title('SSH',fontsize=15)
    plt.show()

dx = (ds[name_nr_x][1]-ds[name_nr_x][0]).values
dy = (ds[name_nr_y][1]-ds[name_nr_y][0]).values
dt_true = times_true[1] - times_true[0] # 3hrs
nt,ny,nx = ssh_true.shape

Lt = nt*dt_true
Lx = nx*dx
Ly = ny*dy

##############################################################################
# Propagator
##############################################################################
print('*** Model ***')
swm = Swm(
    Lt=Lt+time_spinup,nx=nx,ny=ny,Lx=Lx,Ly=Ly,dt=dt,
    He=He,omegas=w_igws,D_He=D_He,D_bc=D_bc,T_He=T_He,T_bc=T_bc)

##############################################################################
# Observations 
##############################################################################
print('*** Observations ***')
nobs = n_obs_per_day*int(Lt/3600/24)
ind_obs = [np.random.randint(0,t2-t1) for _ in range(nobs)]#[0,9,15,26,30,44,48]
yo = {}
for ind in ind_obs:
    if kind_obs=='igws':
        yo[times_true[ind]] = ssh_igws_true[ind]
    elif kind_obs=='full':
        yo[times_true[ind]] = ssh_true[ind]
    if plot:
        plt.figure()
        plt.pcolormesh(yo[times_true[ind]])
        plt.title(str(round(times_true[ind]/3600,1))+' hrs')
        plt.show()
    yo[times_true[ind]] = yo[times_true[ind]].ravel()
        
tobs = np.array(list(yo.keys()))
y_list = np.array(list(yo.values()))

ntobs =  tobs/dt_true

H = Obsopt(nx*ny,iobsxsub,Lt,tobs,dt)

H.yo = yo

if sigmao ==  0:
    _sigmao = 1
else:
    _sigmao = sigmao
R = _sigmao*_sigmao*np.eye(H.nobs,H.nobs)
Rinv = 1/(_sigmao*_sigmao)*np.eye(H.nobs,H.nobs)


##############################################################################
# Background
##############################################################################
print('*** Background ***')
B = None
Xb = None

if reg:
    B = Bopt(swm,sigma=sigmab,alpha=alpha)
    Xb = np.zeros((swm.ntot+swm.nHe+swm.nbc))

##############################################################################
# Variational
##############################################################################
print('*** Variational ***')
var = Variational(
    Xb=Xb, M=swm, H=H, R=R, Rinv=Rinv, B=B,time_assim=Lt,time_spinup=time_spinup)


# Gradient check
def grad_test(alpha, J, G, X):
    h = G(X)/np.linalg.norm(G(X))
    test = (J(X+alpha*h) - J(X))/(alpha*np.inner(h,G(X)))
    return test

Xopt = np.zeros((swm.ntot+swm.nHe+swm.nbc))

print('grad test:',grad_test(1e-7,var.cost,var.grad,Xopt))
    
Jini = var.cost(Xopt)
Gini = var.grad(Xopt)
print(Jini)

##############################################################################
# DA
##############################################################################
print('*** DA experiment ... ***')
res_list = []

niter = 0

now = datetime.now()
current_time = now.strftime("%Y-%m-%d_%H%M")
print(current_time)
dir_out = '../../Outputs/' + name_exp + '_' + current_time
if not os.path.exists(dir_out):
    os.makedirs(dir_out)
shutil.copyfile(path_config, dir_out + '/config_specific.py')

def callback(XX):

    res_list.append(XX)
    
    global niter
    if niter % 10 == 0:
        
        u_ana,v_ana,ssh_ana,He_ana,hbcx_ana,hbcy_ana \
            = get_ana_traj(XX,swm,times_true,Lt,time_spinup)
        fig = plot_diags_scores(
            u_igws_true,v_igws_true,ssh_igws_true,ssh_true,
            u_ana,v_ana,ssh_ana,He_ana,hbcx_ana,hbcy_ana,
            times_true,tobs,dir_out,-1,str(niter) + ' iterations',plot)
        fig.savefig(dir_out + '/result_iter' + str(niter).zfill(6),bbox_inches='tight' )
        
    niter+=1

res = opt.minimize(var.cost,Xopt,
                   method='L-BFGS-B',
                   jac=var.grad,
                   options={'disp': None, 'gtol': gtol, 'maxiter': maxiter, 'iprint':iprint},
                   callback=callback)
print()
print ('Is the minimization successful? {}'.format(res.success))
print()
print ('Initial cost function value: {}'.format(Jini))
print()
print ('Final cost function value: {}'.format(res.fun))
print()
print ('Number of iterations: {}'.format(res.nit))
print()


print('*** Plot results ***')

u_ana,v_ana,h_ana,He_ana,hbcx_ana,hbcy_ana \
            = get_ana_traj(res_list[-1],swm,times_true,Lt,time_spinup)
plot_traj_scores(u_igws_true,v_igws_true,ssh_igws_true,ssh_true,
          u_ana,v_ana,h_ana,He_ana,hbcx_ana,hbcy_ana,times_true,tobs,dir_out,dt_true,plot)






