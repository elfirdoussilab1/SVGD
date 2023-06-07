# In this file, we will define the basic functions that we will need in every computation (gradient, gauss, ...)
import numpy as np


############################### Preliminaries ######################################
def gradient_1d(f ,x ,h): # dérivée par la méthode des différences finies
    return (f(x+h) - f(x)) / h

def gradient(F, x ,h): # Gradient en multidimension
    # Get the dimension of the vector x
    try:
        d = len(x)
    except:
        d = 1 
    
    # the gradient vector (Jacobian)
    G = np.zeros(d)
    
    for i in range(d):
        H =  np.zeros(d)
        H[i] = h
        g = (F(x + H) - F(x)) / h # la ieme dérivée partielle
        G[i] = g
        
    return np.array(G)
   
# Generating uniform random vectors
def gen_unif(N, d, low, high):
    """
    N : the number of samples
    d : dimension
    low : the lower bound of the interval
    high : the upper bound
    """
    x = []
    for i in range(N):
        x.append(np.random.uniform(low , high, size=d))
    return np.array(x)


###################################### Estimators ######################################
# Estimator of mean
def empirical_mean(x):
    N = len(x)
    return np.sum(x) / N

# Unbiased estimator of the variance
def empirical_variance(x):
    N = len(x)
    if N == 1:
        return 0
    else:
        return np.sum((x - empirical_mean(x))**2) / (N - 1)
    

############################## Functions to test algorithms #####################
# Potential of a normal distribution
def potential_N21(x): 
    num = np.sum((x - 2.0)**2)
    return num / 2

def gauss_1d(x, mean, std):
    return np.exp(- (x - mean)**2 / (2 * std**2)) / (std * np.sqrt(2 * np.pi))

# Gaussian multidimensional density
def gauss(x, mean, sigma):
    d = x.shape[1]
    norm = np.sum((x - mean) ** 2)
    return np.exp(- norm / ( 2 * sigma **2 )) / np.sqrt((2 * np.pi * sigma**2) ** d)

# Gaussian mixture function
def mix_gauss(x, mean1, mean2, sigma1,sigma2):
    return (1/2) * gauss_1d(x, mean1, sigma1) + (1/2) * gauss_1d(x, mean2, sigma2)

# Generating from a Gaussian mixture
def gen_gauss_mix(N, mean1, sigma1, mean2, sigma2, p):
    """
    p : is the probability of the first distribution
    """
    samples = np.zeros(N)
    for i in range(N):
        # Bernoulli 
        u = np.random.uniform(low = 0, high = 1)
        if u <= p: # distribution 1
            samples[i] = np.random.normal(loc = mean1 , scale = sigma1)
        else:
            samples[i] = np.random.normal(loc = mean2 , scale = sigma2)
    return samples

# A Gaussian mixture target potential
def tar_mix(x): 
    return -np.log(mix_gauss(x, 0, 7, 1, 1))

