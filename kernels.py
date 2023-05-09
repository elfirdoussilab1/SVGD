# In this file, we will define the kernels and their gradients 
# Gaussian kernel and its gradient (RBF)

import numpy as np

def gauss_kernel(x, y, sigma): 
    norm = np.sum((x - y) ** 2)
    return np.exp(- norm / ( 2 * sigma**2 )) 

def gr_gauss_kernel(x, y, sigma): # Gradient du RBF par rapport a x
    return - gauss_kernel(x, y, sigma) * (x - y) / sigma**2

# Kernel of band limited continuous functions
def band_kernel(x, y, a):
    if abs(x - y) < 1e-15:
        return a / np.pi
    else:
        return np.sin(a * (x - y)) / (np.pi * (x - y))

def gr_band_kernel(x, y, a):
    num1 = a * (x - y) * np.cos(a * (x - y))
    num2 = np.sin(a * (x - y))
    if abs(x - y) < 1e-15:
        return 1e10 # l'infini
    
    return num1 - num2 / (np.pi * (x - y)**2)
