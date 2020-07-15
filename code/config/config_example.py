#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 17:22:09 2020

@author: leguillou
"""
from math import pi
from datetime import timedelta

# Experiment
name_exp = 'twin_exp'
plot = False

# Nature run 
path_nr = '/Users/leguillou/WORK/Developpement/Studies/Jet/4DVAR/DATA/uvh_igws_degraded.nc'
t1,t2 = 500,600
i1,i2 = 70,-60
j1,j2 = 5,-5
name_nr_x = 'x_rho'
name_nr_y = 'y_rho'
name_nr_time = 'time'
name_nr_u_igws = 'u_igws'
name_nr_v_igws = 'v_igws'
name_nr_ssh_igws = 'ssh_igws'
name_nr_u = 'u'
name_nr_v = 'v'
name_nr_ssh = 'ssh'

# Model
w_igws = [2*pi/12/3600]
He = 1.
dt = 3600
D_He = 500e3
T_He = timedelta(days=10).total_seconds()
D_bc = 500e3
T_bc = timedelta(days=10).total_seconds()

# Observations
kind_obs = 'igws' # or 'full'
n_obs_per_day = 1

# Regularisation
reg = False
alpha = {'V':100}
sigmab = {'h':0.2,'He':0.2,'bc':0.2}

# Time parameters
time_init = timedelta(days=20).total_seconds()
time_spinup = timedelta(days=10).total_seconds()
time_assim =  timedelta(days=10).total_seconds()
              
# Assimilation Parameters
sigmao = 1              # Observation error std
noise = False
iobsxsub = 1    

# Minimization
gtol = 1e-4
maxiter = 11
iprint = 10                


# Save parameters
path_plot = '/scratch/4Dvar_IGWs/Outputs/'
plot_spinup = False
path_save = '/home/leguilfl/git/4Dvar_IGWs/movies/'
path_images2mp4 = '/home/leguilfl/git/climporn/ffmpeg/images2mp4.sh'