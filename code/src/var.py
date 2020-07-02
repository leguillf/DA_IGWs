import numpy as np
import matplotlib.pyplot as plt
import os 

class Variational:
    def __init__(self, Xb=None, B=None, M=None, H=None, R=None, Rinv=None, 
                 prec=False, time_spinup=None,time_assim=None):
        self.Xb = Xb
        self.B = B
        self.M = M
        self.H = H
        self.Rinv = Rinv
        self.prec = prec
        self.time_assim = time_assim
        self.time_spinup = time_spinup


    def cost(self,X0):
        
        X = +X0
        
        # Reshape
        u = +X[self.M.sliceu].reshape(self.M.shapeu)
        v = +X[self.M.slicev].reshape(self.M.shapev)
        h = +X[self.M.sliceh].reshape(self.M.shapeh)
        He = +X[self.M.sliceHe].reshape(self.M.shapeHe)
        hbcx = +X[self.M.slicehbcx].reshape(self.M.shapehbcx)
        hbcy = +X[self.M.slicehbcy].reshape(self.M.shapehbcy)
                
        if self.B is not None:
            Jb = (X-self.Xb).dot(self.B.inv(X-self.Xb))
        else:
            Jb = 0
            
        # Spin-up
        t0 = 0
        
        if self.time_spinup is not None:
            traj_spinup = self.M.run(t0,self.time_spinup,u,v,h,
                               He=He,hbcx=hbcx,hbcy=hbcy)
            u,v,h = traj_spinup[-1]
            t0 += self.time_spinup//self.M.dt*self.M.dt 
  
        # Time Loop. Cost function evaluation
        t = t0
        tf = t0 + self.time_assim//self.M.dt*self.M.dt
        Jo = 0.
        while t<tf:
            tobs = t-t0
            if self.H.isobserved(tobs):
                misfit = self.H.misfit(tobs,h.ravel()) # d=Hx-xobs``
                Jo = Jo + misfit.dot(self.Rinv.dot(misfit))
            u,v,h = self.M.step(t,u,v,h,He=He,hbcx=hbcx,hbcy=hbcy)
            t += self.M.dt

        if self.H.isobserved(tf-t0):
            misfit = self.H.misfit(tf-t0,h.ravel()) # d=Hx-xobsx
            Jo = Jo + misfit.dot(self.Rinv.dot(misfit))  
            
            
        J = 0.5*(Jb+Jo) # Total cost function
            
        return J
    
    
        
    def grad(self,X0): 
        
        X = +X0 
        
        # Reshape
        u = +X[self.M.sliceu].reshape(self.M.shapeu)
        v = +X[self.M.slicev].reshape(self.M.shapev)
        h = +X[self.M.sliceh].reshape(self.M.shapeh)
        He = +X[self.M.sliceHe].reshape(self.M.shapeHe)
        hbcx = +X[self.M.slicehbcx].reshape(self.M.shapehbcx)
        hbcy = +X[self.M.slicehbcy].reshape(self.M.shapehbcy)
        
        if self.B is not None:
            gb = self.B.inv(X-self.Xb)
        else:            
            gb = 0

        # Spin-up
        t0 = 0
        if self.time_spinup is not None:
            traj_spinup = self.M.run(t0,self.time_spinup,
                                     u,v,h,He=He,hbcx=hbcx,hbcy=hbcy)
            u,v,h = traj_spinup[-1]
            t0 += self.time_spinup//self.M.dt*self.M.dt 
            
        # Run the forward model to get the state trajectory
        
        traj = self.M.run(t0,self.time_assim,u,v,h,He=He,hbcx=hbcx,hbcy=hbcy)
        
        tf = t0 + self.time_assim//self.M.dt*self.M.dt
    
        # Ajoint computation   
        adX = np.zeros_like(X0)
        # Reshape
        adu = +adX[self.M.sliceu].reshape(self.M.shapeu)
        adv =+ adX[self.M.slicev].reshape(self.M.shapev)
        adh = +adX[self.M.sliceh].reshape(self.M.shapeh)
        adHe = +adX[self.M.sliceHe].reshape(self.M.shapeHe)
        adhbcx =+ adX[self.M.slicehbcx].reshape(self.M.shapehbcx)
        adhbcy = +adX[self.M.slicehbcy].reshape(self.M.shapehbcy)
        
        
        if self.H.isobserved(tf-t0):
            misfit = self.H.misfit(tf-t0,h.ravel()) # d=Hx-xobs
            adh += self.H.adj(tf-t0,self.Rinv.dot(misfit)).reshape(self.M.shapeh)
        
        t = tf
        it = -1
        while t>=t0:
            
            # Retreive chekpoints
            u,v,h = traj[it]

            # One backward step
            adu,adv,adh,adHe,adhbcx,adhbcy = self.M.step_adj(
                t,adu,adv,adh,u,v,h,adHe0=adHe,He=He,
                adhbcx0=adhbcx,adhbcy0=adhbcy,hbcx=hbcx,hbcy=hbcy)
            
            # Calculation of adjoint forcing
            if self.H.isobserved(t-t0):
                misfit = self.H.misfit(t-t0,h.ravel()) # d=Hx-xobs                
                incr = self.H.adj(t-t0,self.Rinv.dot(misfit)).reshape(
                                                                self.M.shapeh)
                adh += incr  
        
            t -= self.M.dt
            it -= 1
            
        
        # Spin-up
        if self.time_spinup is not None:
            adu,adv,adh,adHe,adhbcx,adhbcy = self.M.run_adj(
                0,self.time_spinup,adu,adv,adh,u,v,h,adHe0=adHe,He=He,
                adhbcx0=adhbcx,adhbcy0=adhbcy,hbcx=hbcx,hbcy=hbcy)
    
        
        # Reshape
        adX[self.M.sliceu] = adu.ravel()
        adX[self.M.slicev] = adv.ravel()
        adX[self.M.sliceh] = adh.ravel()
        adX[self.M.sliceHe] = adHe.ravel()
        adX[self.M.slicehbcx] = adhbcx.ravel()
        adX[self.M.slicehbcy] = adhbcy.ravel()
        
        g = (adX + gb) 
        
        return g 
        