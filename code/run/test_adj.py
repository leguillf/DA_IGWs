#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 16:18:57 2020

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



swm = Swm(Lt=Lt,nx=nx,ny=ny,Lx=Lx,Ly=Ly,He=He_true,omegas=w_igws,dt=dt,D_He=500e3,D_bc=500e3,T_He=Lt/2)
swm.random_bc(North=[0.01,0.02],West=[0.02,0.01])

u = np.random.random(swm.shapeu)
v = np.random.random(swm.shapev)
h = np.random.random(swm.shapeh)
He = np.random.random(swm.shapeHe)
hbcx = np.random.random(swm.shapehbcx)
hbcy = np.random.random(swm.shapehbcy)


X = np.concatenate((u.ravel(),v.ravel(),h.ravel(),He.ravel(),hbcx.ravel(),hbcy.ravel()))

du1 = np.random.random(swm.shapeu)
dv1 = np.random.random(swm.shapev)
dh1 = np.random.random(swm.shapeh)
dHe1 = np.random.random(swm.shapeHe)
dhbcx1 = np.random.random(swm.shapehbcx)
dhbcy1 = np.random.random(swm.shapehbcy)


dX1 = np.concatenate((du1.ravel(),dv1.ravel(),dh1.ravel(),dHe1.ravel(),dhbcx1.ravel(),dhbcy1.ravel()))

du2 = np.random.random(swm.shapeu)
dv2 = np.random.random(swm.shapev)
dh2 = np.random.random(swm.shapeh)
dHe2 = np.random.random(swm.shapeHe)
dhbcx2 = np.random.random(swm.shapehbcx)
dhbcy2 = np.random.random(swm.shapehbcy)

dX2 = np.concatenate((du2.ravel(),dv2.ravel(),dh2.ravel(),dHe2.ravel(),dhbcx2.ravel(),dhbcy2.ravel()))

t = 0
Mdu1,Mdv1,Mdh1 = swm.step_tgl(t,du1,dv1,dh1,u,v,h,dHe=dHe1,He=He,hbcx=hbcx,dhbcx=dhbcx1,hbcy=hbcy,dhbcy=dhbcy1)
MdX1 = np.concatenate((Mdu1.ravel(),Mdv1.ravel(),Mdh1.ravel(),dHe1.ravel(),dhbcx1.ravel(),dhbcy1.ravel()))

Adu2,Adv2,Adh2,AdHe2,Adhbcx2,Adhbcy2 = swm.step_adj(t,du2,dv2,dh2,u,v,h,adHe0=dHe2,He=He,hbcx=hbcx,adhbcx0=dhbcx2,hbcy=hbcy,adhbcy0=dhbcy2)
AdX2 = np.concatenate((Adu2.ravel(),Adv2.ravel(),Adh2.ravel(),AdHe2.ravel(),Adhbcx2.ravel(),Adhbcy2.ravel()))

ps1 = np.inner(MdX1,dX2)
ps2 = np.inner(dX1,AdX2)

#print(np.mean(AdhbcW2))
print(ps1/ps2)

tint = Lt
Mdu1,Mdv1,Mdh1 = swm.run_tgl(t,tint,du1,dv1,dh1,u,v,h,dHe=dHe1,He=He,hbcx=hbcx,dhbcx=dhbcx1,hbcy=hbcy,dhbcy=dhbcy1)
MdX1 = np.concatenate((Mdu1.ravel(),Mdv1.ravel(),Mdh1.ravel(),dHe1.ravel(),dhbcx1.ravel(),dhbcy1.ravel()))

Adu2,Adv2,Adh2,AdHe2,Adhbcx2,Adhbcy2 = swm.run_adj(t,tint,du2,dv2,dh2,u,v,h,adHe0=dHe2,He=He,hbcx=hbcx,adhbcx0=dhbcx2,hbcy=hbcy,adhbcy0=dhbcy2)
AdX2 = np.concatenate((Adu2.ravel(),Adv2.ravel(),Adh2.ravel(),AdHe2.ravel(),Adhbcx2.ravel(),Adhbcy2.ravel()))

ps1 = np.inner(MdX1,dX2)
ps2 = np.inner(dX1,AdX2)

print(ps1/ps2)


