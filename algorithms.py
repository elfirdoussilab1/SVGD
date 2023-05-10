# In this file, we will implement the algorithms subject to this project

import numpy as np
from utilities import gradient
from kernels import gauss_kernel, gr_gauss_kernel, band_kernel, gr_band_kernel

# SVGD with Gaussian (RBF) kernel
def SVGD_gauss(F, X, h, lam, sigma, T): 
    """
    F : Potential function
    X : matrix of shape Nxd (N rows et d columns) 
    h : gradient step (argument of the function gradient)
    lam : learning rate
    sigma: parameter in the RBF kernel
    T : The maximum number of iterations
    """
    # Initial state
    Xf = X.copy() 
    
    # Length of the data set
    N = len(X)
    try:
        d = len(X[0])
    except:
        d=1
    
    # SVGD Algorithm
    for t in range(T):
        for i in range(N): # update data
            s = 0
            for j in range(N):
                s += gradient(F, Xf[j], h) * gauss_kernel(Xf[i], Xf[j], sigma) - gr_gauss_kernel(Xf[j],Xf[i],sigma)
            Xf[i] = Xf[i] - (lam / N) * s
    
    return Xf

# Defining the SVGD algorithm with the Band limited kernel (so bad performances !)
def SVGD_band(F, X, h, lam, a, T):
    # Initial state
    Xf = X.copy() 
    
    # Length of the data set
    N = len(X)
    try:
        d = len(X[0])
    except:
        d=1
    
    # SVGD Algorithm
    for t in range(T):
        for i in range(N): # update data
            s = 0
            for j in range(N):
                s += gradient(F, Xf[j], h) * band_kernel(Xf[i], Xf[j], a) - gr_band_kernel(Xf[j],Xf[i],a)
            Xf[i] = Xf[i] - (lam / N) * s
        
    return Xf

# Langevin algorithm
def LA(F, N, d, h, ht): # ht est dans l'algo de Langevin, et h pour le gradient, N is the number of particles that we want at the end
    """
    input :
    - F : the function in the gradient
    - N : the total number of samples desired
    - d : the dimension of samples
    - h : the gradient parameter ($\sim 10^{-6}$)
    - ht : the gradient descent step: learning rate
    
    return :
        the samples
    """
    T = 1000
    # list of realisations of Langevin
    X = np.zeros((N, d)) 

    for i in range(N):
        xf = np.random.uniform(low = -5, high = 5, size = d)
        t = 0
        while t < T: # Gradient descent
            xi = np.random.normal(loc = 0, scale = 1, size = d)
            xf = xf - ht * gradient(F, xf, h) + np.sqrt(2 * ht) * xi
            t+=1
        X[i] = xf

    return X

# Stochastic SVGD
def SSVGD_gauss(F, X, h, lam, sigma, T):
    """
    input :
    - F : potential
    - X : data
    - h : gradient step
    - lam : learning rate
    - sigma : RBF kernel
    - T : total number of iterations
    """
    # Initial state
    Xf = X.copy() 
    
    # Length of the data set
    N = len(X)
    try:
        d = len(X[0])
    except:
        d=1
    
    # SVGD Algorithm
    for t in range(T):
        for i in range(N): # update data
            k = np.random.randint(N)
            s = lam * (gradient(F, Xf[k], h) * gauss_kernel(Xf[i], Xf[k], sigma) - gr_gauss_kernel(Xf[k],Xf[i],sigma))
            Xf[i] = Xf[i] - s
    
    return Xf

# Noisy SVGD
def SVGD_noise(F, X, h, lam, sigma, T) : 
   
    Xf = X.copy()
    N = len(X)
    try:
        d = len(X[0])
    except:
        d=1
    for t in range(T):
        for i in range(N): # update data
            s = 0
            xi = np.random.normal(loc = 0, scale = 1)
            for j in range(N):
                s += gradient(F, Xf[j], h) * gauss_kernel(Xf[i], Xf[j], sigma) - gr_gauss_kernel(Xf[j], Xf[i], sigma) +  np.sqrt(2 * lam) * xi
            Xf[i] = Xf[i] -(lam / N) * s 

    return Xf

# Kernelized Wasserstein Gradient Descent
# Kernelized langevin
def KWGD(F, X, h, eps, hk, T):
    """
    input : 
    - X : initialization
    - h : for the gradient of F
    - eps : learning rate
    - hk : la fenêtre du kernel: paramètre très important !
    - T : total number of iterations
    return :
    Xf : the final configuration of points
    """
    
    Xf = X.copy()
    N = len(X)
    
    for t in range(T):
        for i in range(N):
            num = 0
            den = 0
            for j in range(N):
                num += gr_gauss_kernel(Xf[i], Xf[j], hk)
                den += gauss_kernel(Xf[i], Xf[j], hk)
                
            Xf[i] = Xf[i] - eps * (gradient(F, Xf[i], h) + num / (hk * den))
    return Xf


#################################### Stochastic Wasserstin Gradient Descent ####################################
# defining the algorithm
def SWGD(X, F, T, eps, h, hk):
    """
    input : 
    - X : initialization
    - F : target potential
    - T : the total number of iterations
    - eps : epsilon: learning rate
    - h : for the gradient of F
    - hk : la fenêtre du kernel: paramètre très important !
    
    return :
    Xf : the final configuration of points
    """
    Xf = X.copy()
    N = len(X)
    for t in range(T):
        for i in range(N):
            j = np.random.randint(N)
            num = gr_gauss_kernel(Xf[i], Xf[j], 1)
            den = hk * gauss_kernel(Xf[i], Xf[j], 1)
            Xf[i] = Xf[i] - eps * (gradient(F, Xf[i], h) +  (num / den))
            
    return Xf

#################################### Mixing Lagevin and SVGD ####################################
def LASVGD(F, X, T, eps, h, sigma):
    """
    input :
    - F : target potential
    - X : data
    - T : total number of  iterations
    - eps: learning rate
    - h : for gradient
    - sigma : gaussian kernel

    """
    Xf = X.copy()
    N = len(X)
    for t in range(T):
        for i in range(N):
            s = 0
            xi = np.random.normal(loc = 0, scale= 1)
            for j in range(N):
                s += gradient(F, Xf[j], h) * gauss_kernel(Xf[i], Xf[j], sigma) - gr_gauss_kernel(Xf[j],Xf[i],sigma)
            # Langevin update
            Xf[i] = Xf[i] - (eps / 2) * gradient(F, Xf[i], h) + np.sqrt(2 * eps) * xi

            # SVGD update
            Xf[i] = Xf[i] - eps * s / (2 * N)
    return Xf
