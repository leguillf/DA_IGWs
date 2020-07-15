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


plot = True

###############################################################################
# NR
###############################################################################
path_nr = '/Users/leguillou/WORK/Developpement/Studies/Jet/4DVAR/DATA/uvh_igws_degraded.nc'
#ds = xr.open_zarr(path_nr)
ds = xr.open_dataset(path_nr)
print(ds)


slice_t = slice(3000,3200)
i1,i2 = 70,-60
j1,j2 = 5,-5

times_true = ds.time[slice_t].values 
times_true -= times_true[0]
ssh_true = ds.ssh_igws[slice_t,i1:i2,j1:j2].values
u_true = ds.u_igws[slice_t,i1:i2,j1:j2].values
v_true = ds.v_igws[slice_t,i1:i2,j1:j2].values

ssh_obs = ds.ssh[slice_t,i1:i2,j1:j2].values

if plot:
    fig, (ax1,ax2,ax3,ax4) = plt.subplots(1,4, figsize=(20,5))
    im1 = ax1.pcolormesh(u_true[0],cmap='RdBu_r')
    im2 = ax2.pcolormesh(v_true[0],cmap='RdBu_r')
    im3 = ax3.pcolormesh(ssh_true[0])
    im4 = ax4.pcolormesh(ssh_obs[0])
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

dx = (ds.x_rho[1]-ds.x_rho[0]).values
dy = (ds.y_rho[1]-ds.y_rho[0]).values
dt_true = times_true[1] - times_true[0] # 3hrs
nt,ny,nx = ssh_true.shape

Lt = nt*dt_true
Lx = nx*dx
Ly = ny*dy

##############################################################################
# Propagator
##############################################################################
w_igws = [2*np.pi/12/3600]
swm = Swm(
    Lt=Lt,nx=nx,ny=ny,Lx=Lx,Ly=Ly,dt=3600,
    He=He_true,omegas=w_igws,D_He=D_He,D_bc=D_bc,T_He=T_He,T_bc=T_bc)
dt = swm.dt

##############################################################################
# Observations 
##############################################################################
ind_obs = [np.random.randint(100,200) for _ in range(10)]#[0,9,15,26,30,44,48]
yo = {}
for ind in ind_obs:
    yo[times_true[ind]] = ssh_obs[ind].ravel()
    if plot:
        plt.figure()
        plt.pcolormesh(ssh_obs[ind])
        plt.title(str(round(times_true[ind]/3600,1))+' hrs')
        plt.show()
        
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
B = None
Xb = None

if reg:
    B = Bopt(swm,sigma=sigmab,alpha=alpha)
    Xb = np.zeros((swm.ntot+swm.nHe+swm.nbc))

##############################################################################
# Variational
##############################################################################
var = Variational(
    Xb=Xb, M=swm, H=H, R=R, Rinv=Rinv, B=B,time_assim=Lt,time_spinup=None)


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
            u_true,v_true,ssh_true,
            u_ana,v_ana,ssh_ana,He_ana,hbcx_ana,hbcy_ana,
            times_true,tobs,dir_out,-1,str(niter) + ' iterations')
        fig.savefig(dir_out + '/result_iter' + str(niter).zfill(6),bbox_inches='tight' )
        
    niter+=1

res = opt.minimize(var.cost,Xopt,
                   method='L-BFGS-B',
                   jac=var.grad,
                   options={'disp': None, 'gtol': 1e-4, 'maxiter': 10000, 'iprint':10},
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


u_ana,v_ana,h_ana,He_ana,hbcx_ana,hbcy_ana \
            = get_ana_traj(res_list[-1],swm,times_true,Lt,None)
plot_traj_scores(u_true,v_true,ssh_true,
          u_ana,v_ana,h_ana,He_ana,hbcx_ana,hbcy_ana,times_true,tobs,dir_out,dt_true)






