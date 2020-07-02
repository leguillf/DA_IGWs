#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 17:22:09 2020

@author: leguillou
"""
from math import pi
from datetime import timedelta

# Experiment
name_exp = 'swm_long_window_2obs_3h_nospinup'

# Model
Lt = 10*24*3600
nx = 50
ny = 50
Lx = 1000e3
Ly = 1000e3
He_true = 1.
dt = 3600

w_igws = [2*pi/12/3600]

# param estimation
He_b = He_true

# Regularisation
reg = False
alpha = {'V':100}
sigmab = {'h':0.2,'He':0.2,'bc':0.2}



# Time parameters
time_init = timedelta(days=20).total_seconds()
time_spinup = timedelta(days=20).total_seconds()
time_assim =  timedelta(days=10).total_seconds()

# Error staristics
sigmao = 1              # Observation error std
noise = False
              
# Assimilation Parameters
time_obs = [
            timedelta(days=0,hours=6).total_seconds(),
            timedelta(days=1,hours=3).total_seconds(),
            timedelta(days=2,hours=3).total_seconds(),
            timedelta(days=3,hours=3).total_seconds(),
            timedelta(days=4,hours=3).total_seconds(),
            timedelta(days=5,hours=3).total_seconds(),
            timedelta(days=6,hours=3).total_seconds(),
            timedelta(days=7,hours=3).total_seconds(),
            timedelta(days=8,hours=3).total_seconds(),
            timedelta(days=9,hours=3).total_seconds(),
            ]   
iobsxsub = 1                     # Frequency of spatial subsampling of observations, [1:nx], 1=every space step
D_He = 500e3
T_He = timedelta(days=2).total_seconds()
D_bc = 500e3
T_bc = timedelta(days=2).total_seconds()

# Save parameters
path_plot = '/scratch/4Dvar_IGWs/Outputs/'
plot_spinup = False
path_save = '/home/leguilfl/git/4Dvar_IGWs/movies/'
path_images2mp4 = '/home/leguilfl/git/climporn/ffmpeg/images2mp4.sh'