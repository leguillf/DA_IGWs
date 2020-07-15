from tools import gaspari_cohn,gen_signal1d, gen_signal2d_rectangle

import numpy as np 
from math import cos,sin,sqrt

import copy

import matplotlib.pyplot as plt

class Swm: 
    
    ###########################################################################
    #                             Initialization                              #
    ###########################################################################
    
    def __init__(self,Lt=24*3600,nx=100,Lx=1e6,ny=101,Ly=1e6,dt=None,
                 f=1e-4,He=100,g=9.81,bc_kind='Flather',omegas=[],
                 D_He=None,D_bc=None,time_scheme='rk4',T_He=None,T_bc=None):
        
        # Set parameters
        self.nx = nx
        self.Lx = Lx
        self.dx = Lx / nx
        self.ny = ny
        self.Ly = Ly
        self.dy = Ly / ny
        self.f = f
        self.g = g
        if dt is None:
            if time_scheme=='rk4':
                self.dt = 0.9 * min(self.dx, self.dy) / np.sqrt(g * He)
            elif time_scheme=='rk2':
                self.dt = 0.1 * min(self.dx, self.dy) / np.sqrt(g * He)
        else:
            self.dt = dt
        self.rossby_radius = np.sqrt(g * He) / f
        print('dt:','{:.2f}'.format(self.dt),'s')
        print('dx:','{:.2f}'.format(self.dx*1e-3),'km')
        print('dy:','{:.2f}'.format(self.dy*1e-3),'km')
        print('rossby radius:','{:.2f}'.format(self.rossby_radius*1e-3),'km')
        
        # grid setup
        self.y, self.x = (
            np.arange(ny) * self.dy,
            np.arange(nx) * self.dx
        )
        self.Y, self.X = np.meshgrid(self.y, self.x, indexing='ij')
        
        # Dimensions
        self.shapeu = [self.ny,self.nx-1]
        self.shapev = [self.ny-1,self.nx]
        self.shapeh = [self.ny,self.nx]           
        self.shapeout = [self.ny-2,self.nx-2]   
        self.nu = np.prod(self.shapeu)
        self.nv = np.prod(self.shapev)
        self.nh = np.prod(self.shapeh)
        self.ntot = self.nu+self.nv+self.nh
        self.nout = np.prod(self.shapeout)
        self.sliceu = slice(0,self.nu)
        self.slicev = slice(self.nu,self.nu+self.nv)
        self.sliceh = slice(self.nu+self.nv,self.ntot)
    
        # Time parameters
        self.nt = 1 + int(Lt/self.dt)
        self.t = np.arange(self.nt) * self.dt
        self.time_scheme = time_scheme
        
               
        # He gaussian components  
        self.He_mean = He * np.ones(self.shapeh)
        self.shapeHe = self.shapeh
        self.nHe = self.nh
        self.He_gauss = 0
        ## In Space
        if D_He is not None:
            self.He_gauss = 1
            He_xy_gauss = []
            isub_He = int((D_He/4)/self.dy)  
            jsub_He = int((D_He/4)/self.dx)  
            for i in range(-2*isub_He,ny+3*isub_He,isub_He):
                y = i*self.dy
                for j in range(-2*jsub_He,nx+3*jsub_He,jsub_He):
                    x = j*self.dx
                    mat = np.ones((ny,nx))
                    for ii,yy in enumerate(self.y):
                        for jj,xx in enumerate(self.x):
                            dist = sqrt((yy-y)**2+(xx-x)**2)
                            mat[ii,jj] = gaspari_cohn(dist,D_He/2)
                    He_xy_gauss.append(mat)
            self.He_xy_gauss = np.asarray(He_xy_gauss)
            self.nHe = len(He_xy_gauss)        
            self.shapeHe = [self.nHe]
            ## In time 
            if T_He is not None:
                self.He_gauss = 2
                He_t_gauss = []
                ksub_He = int((T_He/4)/self.dt)  
                for k in range(-2*ksub_He,self.nt+3*ksub_He,ksub_He):
                    He_t_gauss.append(gaspari_cohn(self.t-k*self.dt,T_He/2))
                self.He_t_gauss = np.asarray(He_t_gauss)
                self.shapeHe = [len(self.He_t_gauss),self.nHe]
                self.nHe = np.prod(self.shapeHe)
                
            print('Gaussian He:',self.shapeHe)
        self.sliceHe = slice(self.ntot,self.ntot+self.nHe)
        
        
        # Boundary conditions 
        self.omegas = omegas
        self.bc_kind = bc_kind
        if self.bc_kind=='Flather':
            # gaussian components
            self.shapehbcx = [2,2,self.nx]
            self.shapehbcy = [2,2,self.ny]
            self.bc_gauss = 0
            ## In Space
            if D_bc is not None:
                self.bc_gauss = 1
                bc_x_gauss = []
                bc_y_gauss = []
                isub_He = int((D_bc/4)/self.dy)  
                jsub_He = int((D_bc/4)/self.dx)  
                for j in range(-2*jsub_He,nx+3*jsub_He,jsub_He):
                    bc_x_gauss.append(gaspari_cohn(self.x-j*self.dx,D_bc/2))
                for i in range(-2*isub_He,ny+3*isub_He,isub_He):
                    bc_y_gauss.append(gaspari_cohn(self.y-i*self.dy,D_bc/2))   
                self.bc_x_gauss = np.asarray(bc_x_gauss)
                self.bc_y_gauss = np.asarray(bc_y_gauss)
                self.shapehbcx = [2,2,len(bc_x_gauss)]
                self.shapehbcy = [2,2,len(bc_y_gauss)]
                ## In time 
                if T_bc is not None:
                    self.bc_gauss = 2
                    bc_t_gauss = []
                    ksub_bc = int((T_bc/4)/self.dt)  
                    for k in range(-2*ksub_bc,self.nt+3*ksub_bc,ksub_bc):
                        bc_t_gauss.append(gaspari_cohn(self.t-k*self.dt,T_bc/2))
                    self.bc_t_gauss = np.asarray(bc_t_gauss)
                    self.shapehbcx = [2,2,len(self.He_t_gauss),len(bc_x_gauss)]
                    self.shapehbcy = [2,2,len(self.He_t_gauss),len(bc_y_gauss)]
                print('Gaussian BC x:',self.shapehbcx[2:])
                print('Gaussian BC y:',self.shapehbcy[2:])
                
            self.nbcx = np.prod(self.shapehbcx)
            self.nbcy = np.prod(self.shapehbcy)
            self.nbc = self.nbcx + self.nbcy
            self.slicehbcx = slice(
                self.ntot+self.nHe,self.ntot+self.nHe+self.nbcx)
            self.slicehbcy = slice(
                self.ntot+self.nHe+self.nbcx,self.ntot+self.nHe+self.nbcx+self.nbcy)
        
            
        
    
    def initial_condition_gauss(self):
        
        h0 = 0.05* np.exp(
            - (self.X - self.x[self.nx // 4]) ** 2 / (5*self.rossby_radius) ** 2
            - (self.Y - self.y[self.ny // 3]) ** 2 / (5*self.rossby_radius) ** 2
        )       
        
        self.h[self.lold,:,:] = +h0
        self.h[self.lnew,:,:] = +h0
        
        X0 = np.zeros(self.ntot)
        X0[self.sliceh] = h0.ravel()
        
        return X0
    
    def initial_condition_steady(self):
        
        X0 = np.zeros(self.ntot)
     
        return X0
    
    def random_bc(self,hc=300e3,West=None,East=None,South=None,North=None,seed=None):
        
        dl = np.sqrt(self.dx**2+self.dy**2)
        # Construct spectrum
        L = np.mean([self.Ly,self.Lx])
        fi = np.arange(1./L, 1./float(2. * dl), 1./L)
        
        slope=-4
        PSDi = fi**slope
        Lcut=500e3
        PSDi[fi<1./Lcut]=(1./Lcut)**slope
        
        hbcx = np.zeros((2,2,self.nx))
        hbcy = np.zeros((2,2,self.ny))
        
        if South is not None:
            hbcx[0,0,:] = gen_signal1d(
                fi,PSDi, self.x, fmin=1./self.Lx, fmax=1./(hc), alpha=10, seed=seed)
            hbcx[0,1,:] = gen_signal1d(
                fi,PSDi, self.x, fmin=1./self.Lx, fmax=1./(hc), alpha=10, seed=seed)
            hbcx[0,0,:] *= South[0]/np.max(np.abs(hbcx[0,0,:]))
            hbcx[0,1,:] *= South[1]/np.max(np.abs(hbcx[0,1,:]))
            
        if North is not None:
            hbcx[1,0,:] = gen_signal1d(
                fi,PSDi, self.x, fmin=1./self.Lx, fmax=1./(hc), alpha=10, seed=seed)
            hbcx[1,1,:] = gen_signal1d(
                fi,PSDi, self.x, fmin=1./self.Lx, fmax=1./(hc), alpha=10, seed=seed)
            hbcx[1,0,:] *= North[0]/np.max(np.abs(hbcx[1,0,:]))
            hbcx[1,1,:] *= North[1]/np.max(np.abs(hbcx[1,1,:]))
        
        if West is not None:
            hbcy[0,0,:] = gen_signal1d(
                fi,PSDi, self.y, fmin=1./self.Ly, fmax=1./(hc), alpha=10, seed=seed)
            hbcy[0,1,:] = gen_signal1d(
                fi,PSDi, self.y, fmin=1./self.Ly, fmax=1./(hc), alpha=10, seed=seed)
            hbcy[0,0,:] *= West[0]/np.max(np.abs(hbcy[0,0,:]))
            hbcy[0,1,:] *= West[1]/np.max(np.abs(hbcy[0,1,:]))
            
        if East is not None:
            hbcy[1,0,:] = gen_signal1d(
                fi,PSDi, self.y, fmin=1./self.Ly, fmax=1./(hc), alpha=10, seed=seed)
            hbcy[1,1,:] = gen_signal1d(
                fi,PSDi, self.y, fmin=1./self.Ly, fmax=1./(hc), alpha=10, seed=seed)
            hbcy[1,0,:] *= East[0]/np.max(np.abs(hbcy[1,0,:]))
            hbcy[1,1,:] *= East[1]/np.max(np.abs(hbcy[1,1,:]))
            
        return hbcx,hbcy
        
    
    def random_He(self,hc=300e3,ampl=0.2,seed=None):
        dl = np.sqrt(self.dx**2+self.dy**2)
        # Construct spectrum
        L = np.mean([self.Ly,self.Lx])
        fi = np.arange(1./L, 1./float(2. * dl), 1./L)
        
        slope=-4
        PSDi = fi**slope
        Lcut=hc
        PSDi[fi<1./Lcut]=(1./Lcut)**slope
    
        He_ano = gen_signal2d_rectangle(fi,PSDi, self.x, self.y, 
                                        fminx=1./self.Lx,fminy=1./self.Ly, 
                                        fmax=1./hc, alpha=10, seed=seed, 
                                        lf_extpl=True)
        He_ano /= np.abs(He_ano).max()
        
        He = self.He_mean +  0.2*He_ano
        
        return He
        
        
    def get_He2d(self,t=None,He=None,tgl=False):
        
        if tgl:
            He_mean = np.zeros((self.shapeh))
        else:
            He_mean = self.He_mean
        
        if He is not None:
            if self.He_gauss==2:
                if len(np.shape(He))!=2:
                    raise SystemExit('Error: He need to be 2D in space-time \
                                     gaussian mode')
                He3d = np.tensordot(He,self.He_xy_gauss,(1,0))
                indt = int(t/self.dt)
                He2d = He_mean + \
                    np.tensordot(He3d,self.He_t_gauss[:,indt],(0,0))
                    
            elif self.He_gauss==1:
                if len(np.shape(He))!=1:
                    raise SystemExit('Error: He need to be 1D in space \
                                     gaussian mode')
                He2d = He_mean +\
                    np.sum(He[:,np.newaxis,np.newaxis]*self.He_xy_gauss,axis=0)
            else:
                He2d = He
                
        else:
            He2d = He_mean
        
            
        return He2d
    

    def get_hbc1d(self,t=None,hbcx=None,hbcy=None):
        
        # South/North
        if hbcx is not None:
            if self.bc_gauss==2:
                if len(np.shape(hbcx))!=4:
                    raise SystemExit('Error: hbcx need to be 4D in space-time \
                                     gaussian mode')
                hbcx_2d = np.tensordot(hbcx,self.bc_x_gauss,(3,0))
                indt = int(t/self.dt)
                hbcx_1d = np.tensordot(hbcx_2d,self.bc_t_gauss[:,indt],(2,0))
            
            elif self.bc_gauss==1:
                hbcx_1d = np.tensordot(hbcx,self.bc_x_gauss,(2,0))
            else:
                hbcx_1d = hbcx
        else:
            hbcx_1d = np.zeros([2,2,self.nx])
        
        # West/East
        if hbcy is not None:
            if self.bc_gauss==2:
                if len(np.shape(hbcy))!=4:
                    raise SystemExit('Error: hbcy need to be 4D in space-time \
                                     gaussian mode')
                hbcy_2d = np.tensordot(hbcy,self.bc_y_gauss,(3,0))
                indt = int(t/self.dt)
                hbcy_1d = np.tensordot(hbcy_2d,self.bc_t_gauss[:,indt],(2,0))
            
            elif self.bc_gauss==1:
                hbcy_1d = np.tensordot(hbcy,self.bc_y_gauss,(2,0))
            else:
                hbcy_1d = hbcy
        else:
            hbcy_1d = np.zeros([2,2,self.ny])
        
        return hbcx_1d,hbcy_1d
    
                            
    ###########################################################################
    #                           Spatial scheme                                #
    ###########################################################################
    
    def mean_u(self,u):
        
        um = 0.25 * (u[2:-1,:-1] + u[2:-1,1:] + u[1:-2,:-1] + u[1:-2,1:])
        
        return um
    
    def mean_u_adj(self,adum):
        
        adu = np.zeros(self.shapeu)
        adu[2:-1,:-1] += 0.25 * adum
        adu[2:-1,1:]  += 0.25 * adum
        adu[1:-2,:-1] += 0.25 * adum
        adu[1:-2,1:]  += 0.25 * adum
        
        return adu
    
    def mean_v(self,v):
        
        vm = 0.25 * (v[:-1,2:-1] + v[:-1,1:-2] + v[1:,2:-1] + v[1:,1:-2])
        
        return vm
    
    def mean_v_adj(self,advm):
        
        adv = np.zeros(self.shapev)
        adv[:-1,2:-1] += 0.25 * advm
        adv[:-1,1:-2] += 0.25 * advm
        adv[1:,2:-1]  += 0.25 * advm
        adv[1:,1:-2]  += 0.25 * advm
        
        return adv
    
    
    ###########################################################################
    #                  Right hand side for u equation                         #
    ###########################################################################
    
    def rhs_u(self,vm,h):
        
        rhs_u = np.zeros(self.shapeu)
        rhs_u[1:-1,1:-1] = self.f * vm - self.g * (h[1:-1,2:-1] - h[1:-1,1:-2]) / self.dx
        
        return rhs_u
    
    def rhs_u_adj(self,adrhs_u):
        
      
        adh = np.zeros(self.shapeh)
        
        advm = self.f * adrhs_u[1:-1,1:-1]
        
        adh[1:-1,2:-1] += - self.g * adrhs_u[1:-1,1:-1] / self.dx
        adh[1:-1,1:-2] += + self.g * adrhs_u[1:-1,1:-1] / self.dx
        
        return advm,adh
    
    ###########################################################################
    #                  Right hand side for v equation                         #
    ###########################################################################
    
    def rhs_v(self,um,h):
        
        rhs_v = np.zeros(self.shapev)
        
        rhs_v[1:-1,1:-1] = - self.f * um - self.g * (h[2:-1,1:-1] - h[1:-2,1:-1]) / self.dy
        
        return rhs_v
    
    def rhs_v_adj(self,advrhs):
        
        
        adh = np.zeros(self.shapeh)
        
        adum = - self.f * advrhs[1:-1,1:-1]
        
        adh[2:-1,1:-1] += - self.g * advrhs[1:-1,1:-1] / self.dy
        adh[1:-2,1:-1] += + self.g * advrhs[1:-1,1:-1] / self.dy
        
        return adum,adh
    
    ###########################################################################
    #                  Right hand side for h equation                         #
    ###########################################################################
    
    def rhs_h(self,u,v,He):
        rhs_h = np.zeros(self.shapeh)
        rhs_h[1:-1,1:-1] = - He[1:-1,1:-1] * (\
                (u[1:-1,1:] - u[1:-1,:-1]) / self.dx + \
                (v[1:,1:-1] - v[:-1,1:-1]) / self.dy)
          
        return rhs_h
    
    def rhs_h_tgl(self,du,dv,dHe,u,v,He):
        
        drhs_h = np.zeros(self.shapeh)
        drhs_h[1:-1,1:-1] += \
            - dHe[1:-1,1:-1] * \
                ((u[1:-1,1:] - u[1:-1,:-1]) / self.dx +   \
                 (v[1:,1:-1] - v[:-1,1:-1]) / self.dy)    \
            - He[1:-1,1:-1] * \
                ((du[1:-1,1:] - du[1:-1,:-1]) / self.dx +   \
                 (dv[1:,1:-1] - dv[:-1,1:-1]) / self.dy)    
    
        return drhs_h
       

    def rhs_h_adj(self,t,adrhs_h,u,v,He):
        
        adu = np.zeros(self.shapeu)
        adv = np.zeros(self.shapev)
        adHe = np.zeros(self.shapeHe)
        
        
        adu[1:-1,1:] += - He[1:-1, 1:-1] * adrhs_h[1:-1, 1:-1] / self.dx
        adu[1:-1,:-1]  += + He[1:-1, 1:-1] * adrhs_h[1:-1, 1:-1] / self.dx
        
        adv[1:,1:-1] += - He[1:-1, 1:-1] * adrhs_h[1:-1, 1:-1] / self.dy
        adv[:-1,1:-1]  += + He[1:-1, 1:-1] * adrhs_h[1:-1, 1:-1] / self.dy
        
        adHe2d = - adrhs_h[1:-1, 1:-1] * \
                ((u[1:-1,1:] - u[1:-1,:-1]) / self.dx +   \
                 (v[1:,1:-1] - v[:-1,1:-1]) / self.dy)  
        
        if self.He_gauss==2:
            indt = int(t/self.dt)
            adHe3d = adHe2d[:,:,np.newaxis]*self.He_t_gauss[:,indt]
            adHe += np.tensordot(adHe3d,
                                 self.He_xy_gauss[:,1:-1, 1:-1],([0,1],[1,2]))


        elif self.He_gauss==1:
            for p in range(self.nHe):
                adHe_p= np.sum(np.sum(
                    adHe2d*self.He_xy_gauss[p,1:-1, 1:-1],axis=1),axis=0)
                adHe[p] += adHe_p
        else:
            adHe[1:-1, 1:-1] = adHe2d
            
        return adu,adv,adHe
    
    

    ###########################################################################
    #                  Boundary conditions functions                          #
    ###########################################################################
    
    
    def bc_closed(self,u,v,h):
        # in y direction
        u[0,1:-1] = 0
        u[-1,1:-1] = 0
        v[0,1:-1] = v[1,1:-1]
        v[-1,1:-1] = v[-2,1:-1]
        h[0,1:-1] = h[1,1:-1]
        h[-1,1:-1] = h[-2,1:-1]
        
        # in x direction
        u[:,0] = u[:,1]
        u[:,-1] = u[:,-2]
        v[:,0] = 0
        v[:,-1] = 0
        h[:,0] = h[:,1]
        h[:,-1] = h[:,-2]
        
        return u,v,h
    
    
    def bcs(self,t,u,v,h,He,hbcx,hbcy):
        
        if self.bc_kind=='closed':
        
            u,v,h = self.bc_closed(u,v,h)  
        
        elif self.bc_kind=='Flather':
            
            # Init boundaries
            u[:,0]  = 0
            u[:,-1] = 0
            v[0,:]  = 0
            v[-1,:] = 0
            
            for w in self.omegas:
            
                COS = (w/np.sqrt(w**2-self.f**2)+1)* cos(w*t)
                SIN = (w/np.sqrt(w**2-self.f**2)+1)* sin(w*t)
                
                # South
                v[0,:] -= np.sqrt(2*self.g/(He[0,:]+He[1,:]))*\
                    (h[0,:] + h[1,:] + 2*hbcx[0,0]*COS  + 2*hbcx[0,1]*SIN)/2
                
                # North
                v[-1,:] += np.sqrt(2*self.g/(He[-2,:]+He[-1,:]))*\
                    (h[-2,:] + h[-1,:] + 2*hbcx[1,0]*COS  + 2*hbcx[1,1]*SIN)/2
                
                # West
                u[:,0] -= np.sqrt(2*self.g/(He[:,0]+He[:,1]))*\
                    (h[:,0] + h[:,1] + 2*hbcy[0,0]*COS  + 2*hbcy[0,1]*SIN)/2
                
                # East
                u[:,-1] += np.sqrt(2*self.g/(He[:,-2]+He[:,-1]))*\
                    (h[:,-2] + h[:,-1] + 2*hbcy[1,0]*COS  + 2*hbcy[1,1]*SIN)/2
            
        return u,v,h
    

    
    def bcs_tgl(self,t,du,dv,dh,u,v,h,
                dHe,He,
                dhbcx,dhbcy,hbcx,hbcy):
        
        if self.bc_kind=='closed':
        
            du,dv,dh = self.bc_closed(du,dv,dh)
        
        elif self.bc_kind=='Flather':
            # Init boundaries
            du[:,0]  = 0
            du[:,-1] = 0
            dv[0,:]  = 0
            dv[-1,:] = 0
            # Loop over tidal frequencies
            for w in self.omegas:          
                COS = (w/np.sqrt(w**2-self.f**2)+1)* cos(w*t)
                SIN = (w/np.sqrt(w**2-self.f**2)+1)* sin(w*t)
    
                # South
                K1 = -sqrt(self.g/2)*(He[0,:]+He[1,:])**(-3/2)*(h[0,:]+h[1,:]+\
                                       2*hbcx[0,0]*COS+2*hbcx[0,1]*SIN)/2
                K2 = np.sqrt(2*self.g/(He[0,:]+He[1,:]))/2
                
                dv[0,:] -= K1*(dHe[0,:]+dHe[1,:]) +\
                    K2*(dh[0,:]+dh[1,:]+2*(dhbcx[0,0]*COS+dhbcx[0,1]*SIN))
                
                # North
                K1 = \
                -sqrt(self.g/2)*(He[-2,:]+He[-1,:])**(-3/2)*(h[-2,:]+h[-1,:]+\
                                        2*hbcx[1,0]*COS+2*hbcx[1,1]*SIN)/2
                K2 = np.sqrt(2*self.g/(He[-2,:]+He[-1,:]))/2
                
                dv[-1,:] += K1*(dHe[-2,:]+dHe[-1,:]) +\
                    K2*(dh[-2,:]+dh[-1,:]+2*(dhbcx[1,0]*COS+dhbcx[1,1]*SIN))
            
                # West
                K1 = \
                -sqrt(self.g/2)*(He[:,0]+He[:,1])**(-3/2)*(h[:,0]+h[:,1]+\
                                        2*hbcy[0,0]*COS+2*hbcy[0,1]*SIN)/2
                K2 = np.sqrt(2*self.g/(He[:,0]+He[:,1]))/2
                
                du[:,0] -= K1*(dHe[:,0]+dHe[:,1]) +\
                    K2*(dh[:,0]+dh[:,1]+2*(dhbcy[0,0]*COS+dhbcy[0,1]*SIN))
                
                # East
                K1 = \
                -sqrt(self.g/2)*(He[:,-2]+He[:,-1])**(-3/2)*(h[:,-2]+h[:,-1]+\
                                        2*hbcy[1,0]*COS+2*hbcy[1,1]*SIN)/2
                K2 = np.sqrt(2*self.g/(He[:,-2]+He[:,-1]))/2
                
                du[:,-1] += K1*(dHe[:,-2]+dHe[:,-1]) +\
                    K2*(dh[:,-2]+dh[:,-1]+2*(dhbcy[1,0]*COS+dhbcy[1,1]*SIN))
        
        return du,dv,dh
    
    
    def _bcs_adj(self,t,adhbc,adUbc,COS,SIN,K,_dir,sign):
        
        if self.bc_gauss>0:
            if _dir=='x':
                bc_dir_gauss = self.bc_x_gauss
            else:
                bc_dir_gauss = self.bc_y_gauss
                
        if self.bc_gauss==2:
            indt = int(t/self.dt)
            # cos
            adhbc2d_cos = (2*K*COS*adUbc)[:,np.newaxis]*\
                self.bc_t_gauss[:,indt]
            adhbc[0] += sign* np.tensordot(adhbc2d_cos,
                      bc_dir_gauss,(0,1))
            # sin
            adhbc2d_sin = (2*K*SIN*adUbc)[:,np.newaxis]*\
                self.bc_t_gauss[:,indt]
            adhbc[1] += sign* np.tensordot(adhbc2d_sin,
                      bc_dir_gauss,(0,1))
            
        elif self.bc_gauss==1:
            for p in range(len(bc_dir_gauss)):
                adhbc[0][p] += sign* np.sum(
                    2*K*COS*adUbc*bc_dir_gauss[p],axis=0)
                adhbc[1][p] += sign* np.sum(
                    2*K*SIN*adUbc*bc_dir_gauss[p],axis=0)
        else:
            adhbc[0] += sign* 2*K*COS*adUbc
            adhbc[1] += sign* 2*K*SIN*adUbc
            
        return adhbc
    
                        
    def bcs_adj(self,t,adu,adv,adh,u,v,h,
                adHe,He,
                adhbcx,adhbcy,hbcx,hbcy):
        
        if self.bc_kind=='closed':
        
            adu,adv,adh = self.bc_closed(adu,adv,adh)
        
        elif self.bc_kind=='Flather':
            
            for w in self.omegas:
            
                COS = (w/np.sqrt(w**2-self.f**2)+1)* cos(w*t)
                SIN = (w/np.sqrt(w**2-self.f**2)+1)* sin(w*t)
                
                ##########
                #  South #
                ##########
                K1 = -sqrt(self.g/2)*(He[0,:]+He[1,:])**(-3/2)*(h[0,:]+h[1,:]+\
                                       2*hbcx[0,0]*COS+2*hbcx[0,1]*SIN)/2
                K2 = np.sqrt(2*self.g/(He[0,:]+He[1,:]))/2
                # h
                adh[0,:]  -= K2*adv[0,:]
                adh[1,:]  -= K2*adv[0,:]
                # He
                if adHe is not None:
                    if self.He_gauss==2:
                        indt = int(t/self.dt)
                        adHe3d = (K1*adv[0,:])[:,np.newaxis]*\
                            self.He_t_gauss[:,indt]
                        adHe -= np.tensordot(adHe3d,
                                  self.He_xy_gauss[:,0,:]+
                                  self.He_xy_gauss[:,1,:],(0,1))
                    
                    elif self.He_gauss==1:
                        for p in range(self.nHe):
                            adHe[p] -= np.sum(
                                K1*adv[0,:]*self.He_xy_gauss[p,0,:],axis=0)
                            adHe[p] -= np.sum(
                                K1*adv[0,:]*self.He_xy_gauss[p,1,:],axis=0)
                    else:
                        adHe[0,:] -= K1*adv[0,:]
                        adHe[1,:] -= K1*adv[0,:]
                # BC
                if adhbcx is not None:
                    adhbcx[0] = self._bcs_adj(
                        t,adhbcx[0],adv[0,:],COS,SIN,K2,'x',-1)
                    
                ##########
                #  North #
                ##########
                K1 = \
                -sqrt(self.g/2)*(He[-2,:]+He[-1,:])**(-3/2)*(h[-2,:]+h[-1,:]+\
                                        2*hbcx[1,0]*COS+2*hbcx[1,1]*SIN)/2
                K2 = np.sqrt(2*self.g/(He[-2,:]+He[-1,:]))/2
                # h
                adh[-1,:]  += K2*adv[-1,:]
                adh[-2,:]  += K2*adv[-1,:]
                # He
                if adHe is not None:
                    if self.He_gauss==2:
                        indt = int(t/self.dt)
                        adHe3d = (K1*adv[-1,:])[:,np.newaxis]*\
                            self.He_t_gauss[:,indt]
                        adHe += np.tensordot(adHe3d,
                                  self.He_xy_gauss[:,-1,:]+
                                  self.He_xy_gauss[:,-2,:],(0,1))
                    
                    
                    elif self.He_gauss==1:
                        for p in range(self.nHe):
                            adHe[p] += np.sum(
                                K1*adv[-1,:]*self.He_xy_gauss[p,-1,:],axis=0)
                            adHe[p] += np.sum(
                                K1*adv[-1,:]*self.He_xy_gauss[p,-2,:],axis=0)
                    else:
                        adHe[-1,:] += K1*adv[-1,:]
                        adHe[-2,:] += K1*adv[-1,:]
                # BC
                if adhbcx is not None:
                    adhbcx[1] = self._bcs_adj(
                        t,adhbcx[1],adv[-1,:],COS,SIN,K2,'x',1)
                    
                ##########
                #  West  #
                ##########
                K1 = -sqrt(self.g/2)*(He[:,0]+He[:,1])**(-3/2)*(h[:,0]+h[:,1]+\
                                        2*hbcy[0,0]*COS+2*hbcy[0,1]*SIN)/2
                K2 = np.sqrt(2*self.g/(He[:,0]+He[:,1]))/2
                # h
                adh[:,0]  -= K2*adu[:,0]
                adh[:,1]  -= K2*adu[:,0]
                # He
                if adHe is not None:
                    if self.He_gauss==2:
                        indt = int(t/self.dt)
                        adHe3d = (K1*adu[:,0])[:,np.newaxis]*\
                            self.He_t_gauss[:,indt]
                        adHe -= np.tensordot(adHe3d,
                                  self.He_xy_gauss[:,:,0]+
                                  self.He_xy_gauss[:,:,1],(0,1))
                    
                    
                    elif self.He_gauss==1:
                        for p in range(self.nHe):
                            adHe[p] -= np.sum(
                                K1*adu[:,0]*self.He_xy_gauss[p,:,0],axis=0)
                            adHe[p] -= np.sum(
                                K1*adu[:,0]*self.He_xy_gauss[p,:,1],axis=0)
                    else:
                        adHe[:,0] -= K1*adu[:,0]
                        adHe[:,1] -= K1*adu[:,0]
                # BC
                if adhbcy is not None:
                    adhbcy[0] = self._bcs_adj(
                        t,adhbcy[0],adu[:,0],COS,SIN,K2,'y',-1)
                    
                ##########
                #  East  #
                ##########
                K1 = -sqrt(self.g/2)*(He[:,-2]+He[:,-1])**(-3/2)*(h[:,-2]+h[:,-1]+\
                                        2*hbcy[1,0]*COS+2*hbcy[1,1]*SIN)/2
                K2 = +np.sqrt(2*self.g/(He[:,-2]+He[:,-1]))/2
                # h
                adh[:,-1]  += K2*adu[:,-1]
                adh[:,-2]  += K2*adu[:,-1]
                # He
                if adHe is not None:
                    if self.He_gauss==2:
                        indt = int(t/self.dt)
                        adHe3d = (K1*adu[:,-1])[:,np.newaxis]*\
                            self.He_t_gauss[:,indt]
                        adHe += np.tensordot(adHe3d,
                                  self.He_xy_gauss[:,:,-1]+
                                  self.He_xy_gauss[:,:,-2],(0,1))
                    
                    elif self.He_gauss==1:
                        for p in range(self.nHe):
                            adHe[p] += np.sum(
                                K1*adu[:,-1]*self.He_xy_gauss[p,:,-1],axis=0)
                            adHe[p] += np.sum(
                                K1*adu[:,-1]*self.He_xy_gauss[p,:,-2],axis=0)
                    else:
                        adHe[:,-1] += K1*adu[:,-1]
                        adHe[:,-2] += K1*adu[:,-1]
                # BC
                if adhbcy is not None:
                    adhbcy[1] = self._bcs_adj(
                        t,adhbcy[1],adu[:,-1],COS,SIN,K2,'y',1)
                
            adu[:,0]  = 0
            adu[:,-1] = 0
            adv[0,:]  = 0
            adv[-1,:] = 0
        
        return adu,adv,adh,adHe,adhbcx,adhbcy
    
    
    ###########################################################################
    #                 Auxillary functions for step_adj_rk*                    #
    ###########################################################################
    
    def kv_adj(self,kv2,adu,adh,ku1=None,kh1=None,c=1):
    
        adu_tmp,adh_tmp = self.rhs_v_adj(kv2)
        adu_tmp = self.mean_u_adj(adu_tmp)
        adu += adu_tmp*self.dt
        adh += adh_tmp*self.dt
        kv2 = 0.
        res = [kv2,adu,adh]
        if ku1 is not None:
            ku1 += c*adu_tmp*self.dt
            res.append(ku1)
        if kh1 is not None:
            kh1 += c*adh_tmp*self.dt
            res.append(kh1)
        
        return res
    
    def ku_adj(self,ku2,adv,adh,kv1=None,kh1=None,c=1):
    
        adv_tmp,adh_tmp = self.rhs_u_adj(ku2)
        adv_tmp = self.mean_v_adj(adv_tmp)
        adv += adv_tmp*self.dt
        adh += adh_tmp*self.dt
        ku2 = 0.
        res = [ku2,adv,adh]
        if kv1 is not None:
            kv1 += c*adv_tmp*self.dt
            res.append(kv1)
        if kh1 is not None:
            kh1 += c*adh_tmp*self.dt
            res.append(kh1)
        
        return res
    
    def kh_adj(self,t,kh2,adu,adv,adHe,u,v,He,ku1=None,kv1=None,c=1):
        adu_tmp,adv_tmp,adHe_tmp = self.rhs_h_adj(t,kh2,u,v,He)
        adu += adu_tmp*self.dt
        adv += adv_tmp*self.dt
        adHe += adHe_tmp*self.dt
        kh2 = 0
        res = [kh2,adu,adv,adHe]
        if ku1 is not None:
            ku1 += c*adu_tmp*self.dt
            res.append(ku1)
        if kv1 is not None:
            kv1 += c*adv_tmp*self.dt
            res.append(kv1)
        
        return res

    
    
    
    ###########################################################################
    #                            One time step                                #
    ###########################################################################

    def step(self,t,u0,v0,h0,He=None,hbcx=None,hbcy=None):
        
        if self.time_scheme=='rk2':
            return self.step_rk2(t,u0,v0,h0,He,hbcx,hbcy)
        if self.time_scheme=='rk4':
            return self.step_rk4(t,u0,v0,h0,He,hbcx,hbcy)
        return
            
    def step_tgl(self,t,du0,dv0,dh0, u0,v0,h0,
                 dHe=None, He=None,dhbcx=None,dhbcy=None,hbcx=None,hbcy=None):
        
        if self.time_scheme=='rk2':
            return self.step_tgl_rk2(
                t,du0,dv0,dh0,u0,v0,h0,dHe,He,dhbcx,dhbcy,hbcx,hbcy)
        if self.time_scheme=='rk4':
            return self.step_tgl_rk4(
                t,du0,dv0,dh0,u0,v0,h0,dHe,He,dhbcx,dhbcy,hbcx,hbcy)
        return
            
    def step_adj(self,t,adu0,adv0,adh0, u0,v0,h0, adHe0=None, He=None,
                 adhbcx0=None,adhbcy0=None,hbcx=None,hbcy=None):
        
        if self.time_scheme=='rk2':
            return self.step_adj_rk2(t,adu0,adv0,adh0,u0,v0,h0,adHe0,He,
                adhbcx0,adhbcy0,hbcx,hbcy)
        if self.time_scheme=='rk4':
            return self.step_adj_rk4(t,adu0,adv0,adh0,u0,v0,h0,adHe0,He,
                adhbcx0,adhbcy0,hbcx,hbcy)
        return
            
        
            
    
    def step_rk4(self,t,u0,v0,h0,He=None,hbcx=None,hbcy=None):
        
        He = self.get_He2d(t,He)
        
        #######################
        #       Init          #
        #######################
        u = +u0
        v = +v0
        h = +h0
        
        #######################
        # Current trajectory  #
        #######################
        # k1
        ku1 = self.rhs_u(self.mean_v(v),h)*self.dt
        kv1 = self.rhs_v(self.mean_u(u),h)*self.dt
        kh1 = self.rhs_h(u,v,He)*self.dt
        # k2
        ku2 = self.rhs_u(self.mean_v(v+0.5*kv1),h+0.5*kh1)*self.dt
        kv2 = self.rhs_v(self.mean_u(u+0.5*ku1),h+0.5*kh1)*self.dt
        kh2 = self.rhs_h(u+0.5*ku1,v+0.5*kv1,He)*self.dt
        # k3
        ku3 = self.rhs_u(self.mean_v(v+0.5*kv2),h+0.5*kh2)*self.dt
        kv3 = self.rhs_v(self.mean_u(u+0.5*ku2),h+0.5*kh2)*self.dt
        kh3 = self.rhs_h(u+0.5*ku2,v+0.5*kv2,He)*self.dt
        # k4
        ku4 = self.rhs_u(self.mean_v(v+kv3),h+kh3)*self.dt
        kv4 = self.rhs_v(self.mean_u(u+ku3),h+kh3)*self.dt
        kh4 = self.rhs_h(u+ku3,v+kv3,He)*self.dt
        # Update
        u += 1/6*(ku1+2*ku2+2*ku3+ku4)
        v += 1/6*(kv1+2*kv2+2*kv3+kv4)
        h += 1/6*(kh1+2*kh2+2*kh3+kh4)

        #######################
        # Boundary conditions #
        #######################
        hbcx,hbcy = self.get_hbc1d(t,hbcx,hbcy)
        u,v,h = self.bcs(t,u,v,h,He,hbcx,hbcy)

        return u,v,h
    
        
    def step_tgl_rk4(self,t,du0,dv0,dh0,u0,v0,h0,
                 dHe=None,He=None,dhbcx=None,dhbcy=None,hbcx=None,hbcy=None):
        
        He = self.get_He2d(t,He)
        dHe = self.get_He2d(t,dHe,tgl=True)
        
        #######################
        #       Init          #
        #######################
        u = +u0
        v = +v0
        h = +h0
        du = +du0
        dv = +dv0
        dh = +dh0
            
        #######################
        # Current trajectory  #
        #######################
        # k1
        ku1 = self.rhs_u(self.mean_v(v),h)*self.dt
        kv1 = self.rhs_v(self.mean_u(u),h)*self.dt
        kh1 = self.rhs_h(u,v,He)*self.dt
        # k2
        ku2 = self.rhs_u(self.mean_v(v+0.5*kv1),h+0.5*kh1)*self.dt
        kv2 = self.rhs_v(self.mean_u(u+0.5*ku1),h+0.5*kh1)*self.dt
        kh2 = self.rhs_h(u+0.5*ku1,v+0.5*kv1,He)*self.dt
        # k3
        ku3 = self.rhs_u(self.mean_v(v+0.5*kv2),h+0.5*kh2)*self.dt
        kv3 = self.rhs_v(self.mean_u(u+0.5*ku2),h+0.5*kh2)*self.dt
        kh3 = self.rhs_h(u+0.5*ku2,v+0.5*kv2,He)*self.dt
        # k4
        ku4 = self.rhs_u(self.mean_v(v+kv3),h+kh3)*self.dt
        kv4 = self.rhs_v(self.mean_u(u+ku3),h+kh3)*self.dt
        kh4 = self.rhs_h(u+ku3,v+kv3,He)*self.dt
        # Update
        u += 1/6*(ku1+2*ku2+2*ku3+ku4)
        v += 1/6*(kv1+2*kv2+2*kv3+kv4)
        h += 1/6*(kh1+2*kh2+2*kh3+kh4)
        
        #######################
        # Perturbations       #
        #######################
        # k1_p
        ku1_p = self.rhs_u(self.mean_v(dv),dh)*self.dt
        kv1_p = self.rhs_v(self.mean_u(du),dh)*self.dt
        kh1_p = self.rhs_h_tgl(du,dv,dHe,u0,v0,He)*self.dt
        # k2_p
        ku2_p = self.rhs_u(self.mean_v(dv+0.5*kv1_p),dh+0.5*kh1_p)*self.dt
        kv2_p = self.rhs_v(self.mean_u(du+0.5*ku1_p),dh+0.5*kh1_p)*self.dt
        kh2_p = self.rhs_h_tgl(
            du+0.5*ku1_p,dv+0.5*kv1_p,dHe,u0+0.5*ku1,v0+0.5*kv1,He)*self.dt
        # k3_p
        ku3_p = self.rhs_u(self.mean_v(dv+0.5*kv2_p),dh+0.5*kh2_p)*self.dt
        kv3_p = self.rhs_v(self.mean_u(du+0.5*ku2_p),dh+0.5*kh2_p)*self.dt
        kh3_p = self.rhs_h_tgl(
            du+0.5*ku2_p,dv+0.5*kv2_p,dHe,u0+0.5*ku2,v0+0.5*kv2,He)*self.dt
        # k4_p
        ku4_p = self.rhs_u(self.mean_v(dv+kv3_p),dh0+kh3_p)*self.dt
        kv4_p = self.rhs_v(self.mean_u(du+ku3_p),dh0+kh3_p)*self.dt
        kh4_p = self.rhs_h_tgl(
            du+ku3_p,dv+kv3_p,dHe,u0+ku3,v0+kv3,He)*self.dt
        # Update
        du += 1/6*(ku1_p+2*ku2_p+2*ku3_p+ku4_p)
        dv += 1/6*(kv1_p+2*kv2_p+2*kv3_p+kv4_p)
        dh += 1/6*(kh1_p+2*kh2_p+2*kh3_p+kh4_p)

        #######################
        # Boundary conditions #
        #######################
        hbcx,hbcy = self.get_hbc1d(t,hbcx,hbcy)
        dhbcx,dhbcy = self.get_hbc1d(t,dhbcx,dhbcy)
        du,dv,dh = self.bcs_tgl(t,du,dv,dh,u,v,h,dHe,He,dhbcx,dhbcy,hbcx,hbcy)
        
        return du,dv,dh
    
    
    def step_adj_rk4(self,t,adu0,adv0,adh0, u0,v0,h0, adHe0=None,He=None,
                 adhbcx0=None,adhbcy0=None,hbcx=None,hbcy=None):
        
        He = self.get_He2d(t,He)
            
        #######################
        #       Init          #
        #######################
        u = +u0
        v = +v0
        h = +h0
        adu = +adu0
        adv = +adv0
        adh = +adh0
        adHe = None
        adhbcx = None
        adhbcy = None 
        if adHe0 is not None:
            adHe = +adHe0
        if adhbcx0 is not None:
            adhbcx = +adhbcx0
        if adhbcy0 is not None:
            adhbcy = +adhbcy0
    
        #######################
        # Current trajectory  #
        #######################
        # k1
        ku1 = self.rhs_u(self.mean_v(v),h)*self.dt
        kv1 = self.rhs_v(self.mean_u(u),h)*self.dt
        kh1 = self.rhs_h(u,v,He)*self.dt
        # k2
        ku2 = self.rhs_u(self.mean_v(v+0.5*kv1),h+0.5*kh1)*self.dt
        kv2 = self.rhs_v(self.mean_u(u+0.5*ku1),h+0.5*kh1)*self.dt
        kh2 = self.rhs_h(u+0.5*ku1,v+0.5*kv1,He)*self.dt
        # k3
        ku3 = self.rhs_u(self.mean_v(v+0.5*kv2),h+0.5*kh2)*self.dt
        kv3 = self.rhs_v(self.mean_u(u+0.5*ku2),h+0.5*kh2)*self.dt
        kh3 = self.rhs_h(u+0.5*ku2,v+0.5*kv2,He)*self.dt
        # k4
        ku4 = self.rhs_u(self.mean_v(v+kv3),h+kh3)*self.dt
        kv4 = self.rhs_v(self.mean_u(u+ku3),h+kh3)*self.dt
        kh4 = self.rhs_h(u+ku3,v+kv3,He)*self.dt
        # Update
        u += 1/6*(ku1+2*ku2+2*ku3+ku4)
        v += 1/6*(kv1+2*kv2+2*kv3+kv4)
        h += 1/6*(kh1+2*kh2+2*kh3+kh4)
    
        
        #######################
        # Boundary conditions #
        #######################
        hbcx,hbcy = self.get_hbc1d(t,hbcx,hbcy)
        adu,adv,adh,adHe,adhbcx,adhbcy = self.bcs_adj(
            t,adu,adv,adh,u,v,h,adHe,He,adhbcx,adhbcy,hbcx,hbcy)
        
        #######################
        # Perturbations       #
        #######################
        # Update
        kh1_ad = 1/6 * adh
        kh2_ad = 1/3 * adh
        kh3_ad = 1/3 * adh
        kh4_ad = 1/6 * adh
        adh0 = 0
        kv1_ad = 1/6 * adv
        kv2_ad = 1/3 * adv
        kv3_ad = 1/3 * adv
        kv4_ad = 1/6 * adv
        adv0 = 0
        ku1_ad = 1/6 * adu0
        ku2_ad = 1/3 * adu0
        ku3_ad = 1/3 * adu0
        ku4_ad = 1/6 * adu0
        adu0 = 0
        
        # kh4_ad
        kh4_ad,adu,adv,adHe,ku3_ad,kv3_ad = self.kh_adj(
            t,kh4_ad,adu,adv,adHe,u0+ku3,v0+kv3,He,ku3_ad,kv3_ad)
        
        # kv4_ad
        kv4_ad,adu,adh,ku3_ad,kh3_ad = self.kv_adj(
            kv4_ad,adu,adh,ku3_ad,kh3_ad)
        
        # ku4_ad
        ku4_ad,adv,adh,kv3_ad,kh3_ad = self.ku_adj(
            ku4_ad,adv,adh,kv3_ad,kh3_ad)
        
        # kh3_ad
        kh3_ad,adu,adv,adHe,ku2_ad,kv2_ad = self.kh_adj(
            t,kh3_ad,adu,adv,adHe,u0+0.5*ku2,v0+0.5*kv2,He,ku2_ad,kv2_ad,1/2)
        
        # kv3_ad
        kv3_ad,adu,adh,ku2_ad,kh2_ad = self.kv_adj(
            kv3_ad,adu,adh,ku2_ad,kh2_ad,1/2)
        
        # ku3_ad
        ku3_ad,adv,adh,kv2_ad,kh2_ad = self.ku_adj(
            ku3_ad,adv,adh,kv2_ad,kh2_ad,1/2)
        
        # kh2_ad
        kh2_ad,adu,adv,adHe,ku1_ad,kv1_ad = self.kh_adj(
            t,kh2_ad,adu,adv,adHe,u0+0.5*ku1,v0+0.5*kv1,He,ku1_ad,kv1_ad,1/2)
        
        # kv2_ad
        kv2_ad,adu,adh,ku1_ad,kh1_ad = self.kv_adj(
            kv2_ad,adu,adh,ku1_ad,kh1_ad,1/2)
        
        # ku2_ad
        ku2_ad,adv,adh,kv1_ad,kh1_ad = self.ku_adj(
            ku2_ad,adv,adh,kv1_ad,kh1_ad,1/2)
        
        # kh1_ad
        kh1_ad,adu,adv,adHe = self.kh_adj(
            t,kh1_ad,adu,adv,adHe,u0,v0,He)
        
        # kv1_ad
        kv1_ad,adu,adh = self.kv_adj(
            kv1_ad,adu,adh,None,None)
        
        # ku1_ad
        ku1_ad,adv,adh = self.ku_adj(
            ku1_ad,adv,adh,None,None)
        
        # Only return no-None values
        res = [adu,adv,adh]
    
        if adHe is not None:
            res.append(adHe)
        if adhbcx is not None:
            res.append(adhbcx)
        if adhbcy is not None:
            res.append(adhbcy)
        
        return res
    
    
    ###########################################################################
    #                           Time integration                              #
    ###########################################################################
        
    def run(self,t0,tint,u0,v0,h0,He=None,hbcx=None,hbcy=None):
        
        if self.time_scheme=='rk2':
            return self.run_rk2(t0,tint,u0,v0,h0,He,hbcx,hbcy)
        if self.time_scheme=='rk4':
            return self.run_rk4(t0,tint,u0,v0,h0,He,hbcx,hbcy)
        return
    
    def run_tgl(self,t0,tint,du0,dv0,dh0, u0,v0,h0,
                 dHe=None, He=None, dhbcx=None,dhbcy=None,hbcx=None,hbcy=None):
        if self.time_scheme=='rk2':
            return self.run_tgl_rk2(t0,tint,du0,dv0,dh0,u0,v0,h0,
                dHe,He,dhbcx,dhbcy,hbcx,hbcy)
        if self.time_scheme=='rk4':
            return self.run_tgl_rk4(t0,tint,du0,dv0,dh0,u0,v0,h0,
                dHe,He,dhbcx,dhbcy,hbcx,hbcy)
        return
    
    def run_adj(self,t0,tint,adu0,adv0,adh0, u0,v0,h0,adHe0=None, He=None,
                adhbcx0=None,adhbcy0=None,hbcx=None,hbcy=None):
        if self.time_scheme=='rk2':
            return self.run_adj_rk2(t0,tint,adu0,adv0,adh0,u0,v0,h0,adHe0,He,
                adhbcx0,adhbcy0,hbcx,hbcy)
        if self.time_scheme=='rk4':
            return self.run_adj_rk4(t0,tint,adu0,adv0,adh0,u0,v0,h0,adHe0,He,
                adhbcx0,adhbcy0,hbcx,hbcy)
        return
    
    def run_rk4(self,t0,tint,u0,v0,h0,
             He=None,hbcx=None,hbcy=None):
        
        # Init
        t = t0
        tf = t0 + tint//self.dt*self.dt 
        u = +u0
        v = +v0
        h = +h0
        
        # Time loop 
        traj = [[u,v,h]]
        while t<tf:
            
            
            u,v,h = self.step(t,u,v,h,He=He,hbcx=hbcx,hbcy=hbcy)
            
            traj.append([u,v,h])
            
            t += self.dt
        
        return traj
    
    
    def run_tgl_rk4(self,t0,tint,du0,dv0,dh0,u0,v0,h0,
                 dHe=None,He=None,dhbcx=None,dhbcy=None,hbcx=None,hbcy=None):
        
        # Init
        t = t0
        tf = t0 + tint//self.dt*self.dt 
        du = +du0
        dv = +dv0
        dh = +dh0
        
        # Current trajectory
        traj = self.run_rk4(t0,tint,u0,v0,h0,He=He,hbcx=hbcx,hbcy=hbcy)
        
        # Time loop
        it = 0
        while t<tf:
            u,v,h = traj[it]
            du,dv,dh = self.step_tgl_rk4(
                t,du,dv,dh,u,v,h,dHe,He,dhbcx,dhbcy,hbcx,hbcy)
            
            t += self.dt
            it += 1
            
        return du,dv,dh
    
    
    def run_adj_rk4(self,t0,tint,adu0,adv0,adh0, u0,v0,h0,
                 adHe0=None, He=None,
                 adhbcx0=None,adhbcy0=None,hbcx=None,hbcy=None):
        
        # Init
        tf = t0 + tint//self.dt*self.dt 
        adu = +adu0
        adv = +adv0
        adh = +adh0
        adHe= +adHe0
        adhbcx = +adhbcx0
        adhbcy = +adhbcy0

        traj = self.run_rk4(t0,tint,u0,v0,h0,He=He,hbcx=hbcx,hbcy=hbcy)
        
        # Time loop 
        t = tf
        it = -1
        while t>t0:

            u,v,h = traj[it-1]
            it -= 1
            t -= self.dt
            
            adu,adv,adh,adHe,adhbcx,adhbcy = self.step_adj(
                t,adu,adv,adh,u,v,h,
                adHe0=adHe,He=He,
                adhbcx0=adhbcx,adhbcy0=adhbcy,hbcx=hbcx,hbcy=hbcy)
        
        return adu,adv,adh,adHe,adhbcx,adhbcy

    
    
        
        