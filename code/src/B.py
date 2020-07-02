import numpy as np
from scipy import ndimage

class Bopt:

    def __init__(self,M,sigma={},alpha={},beta={}):
        
        self.M = M
        self.np = M.ny*M.nx
        self.dx = 1/(M.nx-1)
        self.dy = 1/(M.ny-1)
        self.sigma = sigma
        self.alpha = alpha
        self.beta = beta
      
    def compute_kernel(self,alpha,beta):
        #######################################################################
        # Compute kernels 
        #######################################################################
        # Init
        BRi_11 = np.zeros((3,3))
        BRi_22 = np.zeros((3,3))
        BRi_12 = np.zeros((3,3))
        BRi_21 = np.zeros((3,3))
        # BRi_11        
        BRi_11[0,1] = -alpha
        BRi_11[1,0] = -alpha -beta
        BRi_11[1,1] = 2*((alpha+beta)/self.dx**2 + beta/self.dy**2)
        BRi_11[1,2] = -alpha -beta
        BRi_11[2,1] = -alpha
        # BRi_22
        BRi_22[0,1] = -alpha -beta
        BRi_22[1,0] = -alpha 
        BRi_22[1,1] = 2*(beta/self.dx**2 + (alpha+beta)/self.dy**2)
        BRi_22[1,2] = -alpha 
        BRi_22[2,1] = -alpha -beta
        # BRi_12
        BRi_12[0,0] = beta/(self.dx*self.dy)
        BRi_12[0,1] = -beta/(self.dx*self.dy)
        BRi_12[1,0] = -beta/(self.dx*self.dy)
        BRi_12[1,1] = beta/(self.dx*self.dy)
        # BRi_21
        BRi_21[1,1] = beta/(self.dx*self.dy)
        BRi_21[1,2] = -beta/(self.dx*self.dy)
        BRi_21[2,1] = -beta/(self.dx*self.dy)
        BRi_21[2,2] = beta/(self.dx*self.dy)
        
        return BRi_11,BRi_22,BRi_12,BRi_21
    
    
    def inv(self,X):
        # Reshaping
        u2d = X[self.M.sliceu].reshape(self.M.shapeu)
        v2d = X[self.M.slicev].reshape(self.M.shapev)
        
        # Init
        res = np.zeros_like(X)

        # u,v
        if 'V' in [self.alpha,self.beta]:
            alphaV = 0
            if 'V' in self.alpha:
                alphaV = self.alpha['V']
            betaV = 0
            if 'V' in self.beta:
                betaV = self.beta['V']
            
            BRi_11,BRi_22,BRi_12,BRi_21 = self.compute_kernel(alphaV,betaV)
            res[self.M.sliceu] = ( \
                ndimage.convolve(u2d, BRi_11, mode='constant', cval=0.) +\
                ndimage.convolve(v2d, BRi_12, mode='constant', cval=0.)
                ).ravel()
            res[self.M.slicev] = (
                ndimage.convolve(u2d, BRi_21, mode='constant', cval=0.) +\
                ndimage.convolve(v2d, BRi_22, mode='constant', cval=0.)
                ).ravel()
        elif 'V' in self.sigma:
            res[self.M.sliceu] = 1/self.sigma['V']**2 * u2d.ravel()
            res[self.M.slicev] = 1/self.sigma['V']**2 * v2d.ravel()
        
        if 'h' in self.sigma:
            res[self.M.sliceh] = 1/self.sigma['h']**2 * X[self.M.sliceh]
        if 'He' in self.sigma:
            res[self.M.sliceHe] = 1/self.sigma['He']**2 * X[self.M.sliceHe]
        if 'bc' in self.sigma:
            res[self.M.slicehbcS] = 1/self.sigma['bc']**2 * X[self.M.slicehbcS]
            res[self.M.slicehbcN] = 1/self.sigma['bc']**2 * X[self.M.slicehbcN]
            res[self.M.slicehbcW] = 1/self.sigma['bc']**2 * X[self.M.slicehbcW]
            res[self.M.slicehbcE] = 1/self.sigma['bc']**2 * X[self.M.slicehbcE]
        
        return res
        
        
        
    