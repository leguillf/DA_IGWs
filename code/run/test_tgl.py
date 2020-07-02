#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 13:23:31 2020

@author: leguillou
"""


import sys, shutil
import numpy as np 
import matplotlib.pyplot as plt
sys.path.insert(0,'../src')
from swm_rk import Swm


# Config
file_config = 'config_rk'
path_config = '../config/' + file_config + '.py'
shutil.copyfile(path_config, 'config_specific.py')
from config_specific import *



swm = Swm(Lt=Lt,nx=nx,ny=ny,Lx=Lx,Ly=Ly,He=He_true,omegas=w_igws,dt=dt,D_He=500e3,D_bc=500e3,T_He=Lt/2,T_bc=Lt/2)
swm.random_bc(North=[0.01,0.02],West=[0.02,0.01])

u  = np.random.random(swm.shapeu)
v  = np.random.random(swm.shapev)
h  = np.random.random(swm.shapeh)
He = np.random.random(swm.shapeHe)
hbcx = np.random.random(swm.shapehbcx)
hbcy = np.random.random(swm.shapehbcy)

du  = np.random.random(swm.shapeu)
dv  = np.random.random(swm.shapev)
dh  = np.random.random(swm.shapeh)
dHe = np.random.random(swm.shapeHe)
dhbcx = np.random.random(swm.shapehbcx)
dhbcy = np.random.random(swm.shapehbcy)

lambd = 1e-7

t = 500
u1,v1,h1 = swm.step(t,u+lambd*du,v+lambd*dv,h+lambd*dh,He=He+lambd*dHe,hbcx=hbcx+lambd*dhbcx,hbcy=hbcy+lambd*dhbcy)
u2,v2,h2 = swm.step(t,u,v,h,He=He,hbcx=hbcx,hbcy=hbcy)
du1,dv1,dh1 = swm.step_tgl(t,du,dv,dh,u,v,h,dHe=dHe,He=He,dhbcx=dhbcx,hbcx=hbcx,dhbcy=dhbcy,hbcy=hbcy)

psu = np.linalg.norm((u1-u2)-lambd*du1)/np.linalg.norm(lambd*du1)
psv = np.linalg.norm((v1-v2)-lambd*dv1)/np.linalg.norm(lambd*dv1)
psh = np.linalg.norm((h1-h2)-lambd*dh1)/np.linalg.norm(lambd*dh1)

print(psu,psv,psh)

tint = Lt
traj1 = swm.run(t,tint,u+lambd*du,v+lambd*dv,h+lambd*dh,He=He+lambd*dHe,hbcx=hbcx+lambd*dhbcx,hbcy=hbcy+lambd*dhbcy)
u1,v1,h1 = traj1[-1]
traj2 = swm.run(t,tint,u,v,h,He=He,hbcx=hbcx,hbcy=hbcy)
u2,v2,h2 = traj2[-1]
du,dv,dh = swm.run_tgl(t,tint,du,dv,dh,u,v,h,dHe=dHe,He=He,dhbcx=dhbcx,hbcx=hbcx,dhbcy=dhbcy,hbcy=hbcy)

psu = np.linalg.norm((u1-u2)-lambd*du)/np.linalg.norm(lambd*du)
psv = np.linalg.norm((v1-v2)-lambd*dv)/np.linalg.norm(lambd*dv)
psh = np.linalg.norm((h1-h2)-lambd*dh)/np.linalg.norm(lambd*dh)

print(psu,psv,psh)


