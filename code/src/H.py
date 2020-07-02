import numpy as np 

class Obsopt:

    def __init__(self,npix,xsub,time_assim,time_obs,dt):
        self.npix = npix
        self.time_assim = time_assim  # time and
        self.time_obs = np.asarray(time_obs)  # space subsampling
        self.dt = dt
        self.yo = {}      # observation vectors
        
        obsxmask = np.array([(i%xsub==0)  for i in range(1,npix+1)],dtype=float)
        obs_location = np.flatnonzero(obsxmask)
        self.nobs = np.size(obs_location)
            
        self.mat=np.zeros((self.nobs,npix)) # Only observations on h
            
        for i in range(self.nobs):
            self.mat[i,obs_location[i]] = 1. 
        

    def isobserved(self,t):
        
        is_obs = np.min(np.abs(t-self.time_obs))<=self.dt/2
        
        return is_obs

        
    
    def gen_obs(self,M,u,v,h,He,hbcx,hbcy,sigmao,noise=True):
                
        # true trajectory
        u_true = [u] 
        v_true = [v] 
        h_true = [h] 

        
        t = 0
        i = 0
        while t<self.time_assim:
            
            if self.isobserved(t):
                if noise:
                    err = np.random.normal(0.,sigmao,self.npix)
                else:
                    err = 0
                self.yo[t] = np.dot(self.mat,(h.ravel() + err))
                
            u,v,h = M.step(t,u,v,h,He=He[i],hbcx=hbcx[i],hbcy=hbcy[i])
            
            u_true.append(u)
            v_true.append(v)
            h_true.append(h)
            
            t += M.dt
            i += 1
        
                                                                                                                                                                                                                                    
        if self.isobserved(t):
            if noise:
                err = np.random.normal(0.,sigmao,self.npix)
            else:
                err = 0
            self.yo[t] = np.dot(self.mat,(h.ravel() + err))
        
            
        u_true = np.asarray(u_true)
        v_true = np.asarray(v_true)
        h_true = np.asarray(h_true)

        return u_true,v_true,h_true
    
    
    def dir(self,t,X):
        if self.isobserved(t):
            return np.dot(self.mat,X)

    def tan(self,t,X):
        if self.isobserved(t):
            return np.dot(self.mat,X)

    def adj(self,t,y):
        if self.isobserved(t):
            return np.dot(self.mat.T,y)

    def misfit(self,t,X):
         if self.isobserved(t): 
            return np.dot(self.mat,X) - self.yo[t]
       
    