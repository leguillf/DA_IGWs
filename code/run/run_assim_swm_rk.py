#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 16:32:00 2020

@author: leguillou
"""

# Import
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
import numpy as np 
import matplotlib.pyplot as plt
from scipy.linalg import inv 
from math import exp
import scipy.linalg as lin
import scipy.optimize as opt
import matplotlib.gridspec as gridspec
import shutil
from datetime import datetime
from scipy import interpolate



# Config

if len(sys.argv)==2:
    file_config = str(sys.argv[1])
else:
    file_config = 'config_example'
    

path_config = '../config/' + file_config + '.py'
shutil.copyfile(path_config, 'config_specific.py')
from config_specific import *

##############################################################################
# NR
##############################################################################
w_igws = [2*np.pi/12/3600]
nr = Swm(nx=nx,ny=ny,Lx=Lx,Ly=Ly,He=He_true,omegas=w_igws,dt=dt)

##############################################################################
# Propagator
swm = Swm(
    Lt=time_assim+time_spinup,nx=nx,ny=ny,Lx=Lx,Ly=Ly,
    He=He_true,omegas=w_igws,dt=dt,D_He=D_He,D_bc=D_bc,T_He=T_He,T_bc=T_bc)

##############################################################################

##############################################################################
# Experimental setup
##############################################################################
# Steady state
u = np.zeros(swm.shapeu)
v = np.zeros(swm.shapev)
h = np.zeros(swm.shapeh)

# He
if T_He is None:
    He0 = nr.random_He(hc=500e3,seed=10)
    He_t = np.repeat(He0[np.newaxis,:,:],len(swm.t),axis=0)
else:
    He = []
    tt = []
    t = 0
    while t<time_assim+T_He:
        tt.append(t)
        He_tmp = nr.random_He(hc=500e3)
        He.append(He_tmp)
        # plt.figure()
        # plt.pcolormesh(He_tmp)
        # plt.title('He')
        # plt.colorbar()
        # plt.show()
        t += T_He
    He0 = He[0]
    try:
        f = interpolate.interp1d(np.asarray(tt), np.asarray(He),axis=0,kind='cubic')
    except:
        f = interpolate.interp1d(np.asarray(tt), np.asarray(He),axis=0)
    He_t = f(np.arange(1 + int(time_assim/dt)) * dt)


# Boundary conditions 
if T_bc is None:
    acos = np.random.random()*1e-2
    asin = np.random.random()*1e-2
    hbcx,hbcy = nr.random_bc(hc=500e3,West=[acos,asin])
    hbcx_t = np.repeat(hbcx[np.newaxis,:,:,:],len(swm.t),axis=0)
    hbcy_t = np.repeat(hbcy[np.newaxis,:,:,:],len(swm.t),axis=0)
else:
    hbcx = []
    hbcy = []
    tt = []
    t = 0
    while t<time_assim+T_bc:
        tt.append(t)
        acos = np.random.random()*1e-2
        asin = np.random.random()*1e-2
        hbcx_tmp,hbcy_tmp = nr.random_bc(hc=500e3,West=[acos,asin])
        hbcx.append(hbcx_tmp)
        hbcy.append(hbcy_tmp)
        # plt.figure()
        # plt.plot(hbcy_tmp[0,0],label='cos')
        # plt.plot(hbcy_tmp[0,1],label='sin')
        # plt.title('West BC')
        # plt.legend()
        # plt.show()
        t += T_bc
    hbcx0 = hbcx[0]
    hbcy0 = hbcy[0]

    try:
        fx = interpolate.interp1d(np.asarray(tt), np.asarray(hbcx),axis=0,kind='cubic')
        fy = interpolate.interp1d(np.asarray(tt), np.asarray(hbcy),axis=0,kind='cubic')
    except:
        fx = interpolate.interp1d(np.asarray(tt), np.asarray(hbcx),axis=0)
        fy = interpolate.interp1d(np.asarray(tt), np.asarray(hbcy),axis=0)
    hbcx_t = fx(np.arange(1 + int(time_assim/dt)) * dt)
    hbcy_t = fy(np.arange(1 + int(time_assim/dt)) * dt)



##############################################################################
# Spin-up
##############################################################################
if plot_spinup:
    dir_spinup = '../../Spin-up/' + name_exp 
    if not os.path.exists(dir_spinup):
        os.makedirs(dir_spinup)
    else:
        cmd = 'rm ' + dir_spinup + '/*'
        print(cmd)

# Integration to reach permanent state
t = 0
while t<=time_init:
    
    if plot_spinup:        
            
        time_str = str(round(t/3600,2))
        fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(15,5))
        fig.suptitle('t = '+time_str + 'h',fontsize=20)
        im1 = ax1.pcolormesh(u, vmin=-0.15,vmax=0.15,cmap='RdBu_r')
        im2 = ax2.pcolormesh(v, vmin=-0.15,vmax=0.15,cmap='RdBu_r')
        im3 = ax3.pcolormesh(h, vmin=-0.05,vmax=0.05,cmap='RdBu_r')
        cbar = plt.colorbar(im1,ax=ax1)
        cbar.ax.set_title('m/s')
        cbar = plt.colorbar(im2,ax=ax2)
        cbar.ax.set_title('m/s')
        cbar = plt.colorbar(im3,ax=ax3)
        cbar.ax.set_title('m')
        ax1.set_title('U',fontsize=15)
        ax2.set_title('V',fontsize=15)
        ax3.set_title('SLA',fontsize=15)
        
        fig.savefig( dir_spinup + '/spinup_' +\
                    str(int(t)).zfill(6),bbox_inches='tight' )
            
        plt.close('all')
        
    u,v,h = nr.step(t,u,v,h,He=He0,hbcx=hbcx0,hbcy=hbcy0)
    t += nr.dt
    
##############################################################################
# Init
##############################################################################
u_init = +u
v_init = +v
h_init = +h

##############################################################################
# Observations 
##############################################################################
H_nr = Obsopt(nr.nx*nr.ny,iobsxsub,time_assim,time_obs,nr.dt)
# generate observations and true fields
u_true,v_true,h_true = H_nr.gen_obs(nr,u_init,v_init,h_init,He_t,hbcx_t,hbcy_t,0)

tobs = list(H_nr.yo.keys())
y_list = list(H_nr.yo.values())

ntobs = np.asarray(tobs)//nr.dt

# for t,y in zip(tobs,y_list):
#     fig, (ax1) = plt.subplots(1,1, figsize=(5,5))

#     im1 = ax1.pcolormesh(y.reshape(nr.ny,nr.nx))

#     cbar.ax.set_title('m')
#     ax1.set_title(str(t//3600) + 'hrs')
#     plt.show()

##############################################################################
# Observation operator  
##############################################################################
H = Obsopt(nr.nx*nr.ny,iobsxsub,time_assim,time_obs,nr.dt)

H.yo = H_nr.yo

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
    Xb=Xb, M=swm, H=H, R=R, Rinv=Rinv, B=B,time_assim=time_assim,time_spinup=time_spinup)


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

# fig, (ax1,ax2,ax3,ax4) = plt.subplots(1,4, figsize=(25,5))
# im1 = ax1.pcolormesh(Gini[swm.sliceu].reshape(swm.shapeu),cmap='RdBu_r')
# im2 = ax2.pcolormesh(Gini[swm.slicev].reshape(swm.shapev),cmap='RdBu_r')
# im3 = ax3.pcolormesh(Gini[swm.sliceh].reshape(swm.shapeh))
# im4 = ax4.pcolormesh(swm.get_He2d(Gini[swm.sliceHe].reshape(swm.shapeHe)))
# plt.colorbar(im1,ax=ax1)
# plt.colorbar(im2,ax=ax2)
# plt.colorbar(im3,ax=ax3)
# plt.colorbar(im4,ax=ax4)
# ax1.set_title('gradJ_U',fontsize=15)
# ax2.set_title('gradJ_V',fontsize=15)
# ax3.set_title('gradJ_h',fontsize=15)       
# ax4.set_title('gradJ_He',fontsize=15)       
# plt.show()





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
        
        u_ana,v_ana,h_ana,He_ana,hbcx_ana,hbcy_ana \
            = get_ana_traj(XX,swm,time_assim,time_spinup)
        plot_result(
            u_true,v_true,h_true,He_t,hbcx_t,hbcy_t, 
            u_ana,v_ana,h_ana,He_ana,hbcx_ana,hbcy_ana,ntobs,niter,dir_out)
        
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
            = get_ana_traj(res_list[-1],swm,time_assim,None)
plot_traj(u_true,v_true,h_true,He_t,hbcx_t,hbcy_t, 
          u_ana,v_ana,h_ana,He_ana,hbcx_ana,hbcy_ana,ntobs,dir_out,swm.dt)






