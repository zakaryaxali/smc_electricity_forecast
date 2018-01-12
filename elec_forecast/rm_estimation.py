from math import inf, sqrt
from scipy.stats import truncnorm, invgamma
import numpy as np

class RmEstimation():
    """
    Parameter estimation using Robbins Monroe algorithm
    Estimate theta_star = (sigma_s, sigma_g, u_heat, sigma)
    theta(n+1) = theta(n) - gamma(n)(f(theta(n))-epsilon(n))
    """
    def __init__(self, sigma_s_init, sigma_g_init, u_heat_init, sigma_init, variance_epsilon, iterations, f, gamma):
        self.sigma_s = sigma_s_init + np.zeros((1,iterations))
        self.cm_sigma_s = np.zeros((1,iterations))
        self.sigma_g = sigma_g_init + np.zeros((1,iterations))
        self.cm_sigma_g = np.zeros((1,iterations))
        self.u_heat = u_heat_init + np.zeros((1,iterations))
        self.cm_u_heat = np.zeros((1,iterations))
        self.sigma = sigma_init + np.zeros((1,iterations))
        self.cm_sigma = np.zeros((1,iterations))
        self.variance_epsilon = variance_epsilon
        self.iterations = iterations
        self.f = f       
        self.gamma = gamma       
    
    def epsilon(self, n):
        return np.random.multivariate_normal(np.zeros(self.variance_epsilon.shape[0]), self.variance_epsilon, 1)
        
    def new_theta(self, n):
        prev_sigma_s = self.sigma_s[0,n-1]
        prev_sigma_g = self.sigma_g[0,n-1]
        prev_u_heat = self.u_heat[0,n-1]
        prev_sigma = self.sigma[0,n-1]
        
        f_sigma_s, f_sigma_g, f_u_heat, f_sigma = self.f(prev_sigma_s, prev_sigma_g, prev_u_heat, prev_sigma)
        
        gamma_n = self.gamma(n)
        
        epsilon_n = self.epsilon(n)
        
        self.sigma_s[0,n] = prev_sigma_s - gamma_n * (f_sigma_s + epsilon_n[0,0])
        self.sigma_g[0,n] = prev_sigma_g - gamma_n * (f_sigma_g + epsilon_n[0,1])
        self.u_heat[0,n] = prev_u_heat - gamma_n * (f_u_heat + epsilon_n[0,2])
        self.sigma[0,n] = prev_sigma - gamma_n * (f_sigma + epsilon_n[0,3])
        
        #return new_sigma_s, new_sigma_g, new_u_heat, new_sigma
    
    def cesaro_mean(self, n):        
        self.cm_sigma_s[0,n] = ((n-1)/n) * self.cm_sigma_s[0,n-1] + (1/n) * self.sigma_s[0,n]
        self.cm_sigma_g[0,n] = ((n-1)/n) * self.cm_sigma_g[0,n-1] + (1/n) * self.sigma_g[0,n]
        self.cm_u_heat[0,n] = ((n-1)/n) * self.cm_u_heat[0,n-1] + (1/n) * self.u_heat[0,n]
        self.cm_sigma[0,n] = ((n-1)/n) * self.cm_sigma[0,n-1] + (1/n) * self.sigma[0,n]
        
        #return cm_sigma_s, cm_sigma_g, cm_u_heat, cm_sigma
    
    def estimation(self):
        for i in range(1, self.iterations):
            self.new_theta(i)
            self.cesaro_mean(i)
        