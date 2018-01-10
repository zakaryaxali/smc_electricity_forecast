from math import inf, sqrt
from scipy.stats import truncnorm, invgamma
import numpy as np

class BootstrapFilterModel():
    def __init__(self, temperatures, daytypes, consumptions, nb_days, nb_particles, sigma2, kappa, u_heat):
        """Method to initialize the bootstrap filter model.
        Like in the article, we suppose sigma_s_star and sigma_g_star fixed
        Parameters :
        temperatures: list of temperatures (per day)
        daytypes: list of daytypes (per day)
        consumptions: list of electricity consumption (per day)
        nb_days: number of iterations for the initialization step
        nb_particles: number of particles
        sigma2: initial sigma squared (fixed)
        kappa: 
        u_heat: 		"""
        self.temperatures = temperatures
        self.daytypes = daytypes
        self.consumptions = consumptions
        self.nb_days = nb_days
        self.nb_particles = nb_particles
        self.sigma2 = sigma2
        self.kappa = kappa
        self.u_heat = u_heat
        #Var init
        self.s = np.zeros((nb_days, nb_particles)) 
        self.g_heat = np.zeros((nb_days, nb_particles))
        #sigma_s and sigma_g are fixed
        self.sigma_s_star_2 = np.zeros((1, nb_particles)) 
        self.sigma_g_star_2 = np.zeros((1, nb_particles))
        self.x_season = np.zeros((1, nb_particles))
        self.x_heat = np.zeros((1, nb_particles))
        self.x = np.zeros((1, nb_particles))
        self.w = np.zeros((1, nb_particles))


    def truncated_norm(self, lower_bound, upper_bound, mean, variance):
        std = sqrt(variance)
        a, b = (lower_bound - mean) / std, (upper_bound - mean) / std
        result = truncnorm.rvs(a = a,b = b, loc= mean, scale = std, size=1)[0]
        if(result==inf):
            print ("Warning : truncnorm returned inf !")
        return  result


    def bf_initialization_gibbs(self, sigma2_s_param=None, sigma2_g_param=None):
        """	Method to initialize the bootstrap filter using a Gibbs sampler.
        Like in the article, we suppose sigma_s_star and sigma_g_star fixed
        Parameters :
        temperatures: list of temperatures (per day)
        daytypes: list of daytypes (per day)
        consumptions: list of electricity consumption (per day)
        nb_days: number of iterations for the initialization step
        nb_particles: number of particles
        sigma2: initial sigma squared (fixed)
        kappa: 
        u_heat: """
        #Gibbs : Initialization step
        self.gibbs_init_step(self.nb_days, self.nb_particles, sigma2_s_param, sigma2_g_param)

        #Gibbs : step t > 0
        for j in range(1, self.nb_particles):
            if(j%(self.nb_particles/10)==0 or j==1):
                print("Gibbs sampling for particle " + str(j) + "/" + str(self.nb_particles))


            self.s[:,j] = self.s[:,j-1]
            self.g_heat[:,j] = self.g_heat[:,j-1]
            self.sigma_s_star_2[:,j] = self.sigma_s_star_2[:,j-1]
            self.sigma_g_star_2[:,j] = self.sigma_g_star_2[:,j-1]

            # Compute s[0] for particle j (j>0)
            self.compute_s_0(j)

            # Compute s[n] for particle j (n>0 and j>0)
            for i in range(1, self.nb_days):
                self.compute_s(i,j)

            # Compute g_heat[O] for particle j (and j>0)
            self.compute_g_0(j)

            # Compute g_heat[n] for particle j (n>0 and j>0)
            for i in range(1, self.nb_days):
                self.compute_g(i,j)

            shape = 0.01 + ((self.nb_days - 1)/2)
            # Compute the new sigma_s_star2 for particle j (j>0) (follow Inverse Gamma)
            self.sigma_s_star_2[0, j] = self.compute_sigma_star_2(shape, self.s, j)

            # Compute the new sigma_g_star2 for particle j (j>0) (follow Inverse Gamma)
            self.sigma_s_star_2[0, j] = self.compute_sigma_star_2(shape, self.g_heat, j)

        #Compute x
        self.compute_x()
        #Compute w
        self.compute_w()

    def gibbs_init_step(self, nb_days, nb_particles, sigma2_s_param=None, sigma2_g_param=None):
        """	Initialize all the variables required for the bootstrap filter init using gibbs sampling
        And perform the gibbs initialization step """

        #Gibbs step 0
        if sigma2_s_param is None:
            self.sigma_s_star_2[0, 0] = invgamma.rvs(a=10**(-2), scale=10**(2), size = 1)[0]
        else :
            self.sigma_s_star_2[0, 0] = sigma2_s_param
        if sigma2_g_param is None:
            self.sigma_g_star_2[0, 0] = invgamma.rvs(a=10**(-2), scale=10**(2), size = 1)[0]
        else :
            self.sigma_g_star_2[0, 0] = sigma2_g_param

        self.s[0,0] = self.truncated_norm(0, inf, 0, 10**8)
        self.g_heat[0,0] =  self.truncated_norm(-inf, 0, 0, 10**8)
        for i in range(1, nb_days):
            self.s[i,0] = self.s[i-1,0] + self.truncated_norm(-self.s[i-1,0], inf, 0, self.sigma_s_star_2[0,0])
            self.g_heat[i,0] = self.g_heat[i-1,0] + self.truncated_norm(-inf, - self.g_heat[i-1,0], 0, self.sigma_g_star_2[0,0])


    def compute_s_0(self, j):
        """	Draw s_0 from its full conditional (page 12/13 summary project) """
        #Compute variance and mean denominator (same denominator for both)
        denominator = (10**8) * self.sigma_s_star_2[0, j] * (self.kappa[self.daytypes[0]]**2) + self.sigma2 * self.sigma_s_star_2[0, j] + (10**8) * self.sigma2 

        #Compute mean numerator
        numerator_mean = (10**8) * (self.sigma2 * self.s[1,j] + self.sigma_s_star_2[0, j] * self.kappa[self.daytypes[0]] * (self.consumptions[0] - self.g_heat[0,j] * min(self.temperatures[0] - self.u_heat, 0)))
        #Mean
        mean = numerator_mean / denominator

        #Compute variance numerator
        variance_numerator = (10**8) * self.sigma2 * self.sigma_s_star_2[0, j]
        #Variance
        variance = variance_numerator / denominator

        self.s[0,j] = self.truncated_norm(0, inf, mean, variance)


    def compute_s(self, i, j):
        """ Draw s from its full conditional (page 12/13 summary project) """
        #Compute variance and mean denominator (same denominator for both)
        denominator = 2*self.sigma2 + self.sigma_s_star_2[0, j]*(self.kappa[self.daytypes[i]]**2)

        #Compute mean numerator
        s_next = 0
        if (i+1 < self.nb_days-1):
            s_next = self.s[i+1,j]

        numerator_mean = self.sigma2*(self.s[i-1,j] + s_next) + self.sigma_s_star_2[0, j]*self.kappa[self.daytypes[i]]*(self.consumptions[i] - min(0,self.g_heat[i,j]*(self.temperatures[i] - self.u_heat)))
        #Mean
        mean = numerator_mean / denominator

        #Compute variance numerator
        variance_numerator = self.sigma2*self.sigma_s_star_2[0, j]
        #Variance
        variance = variance_numerator / denominator

        self.s[i,j] = self.truncated_norm(0,inf,mean,variance)


    def compute_g_0(self, j):
        """ Draw g_0 from its full conditional (page 12/13 summary project)
        WARNING : Not clearly written in the pdf """
        #Compute variance and mean denominator (same denominator for both)
        denominator = self.sigma2 * self.sigma_g_star_2[0, j] + (10**8) * self.sigma2

        numerator_mean = (10**8) * self.sigma2 * self.g_heat[1,j]
        if (self.u_heat > self.temperatures[0]):
            denominator = denominator + (10**8) * self.sigma_g_star_2[0, j] * ((self.temperatures[0] - self.u_heat )**2)
            numerator_mean = numerator_mean + \
                                 (10**8) * self.sigma_g_star_2[0, j] * (self.temperatures[0] - self.u_heat) * (self.consumptions[0] - self.s[0,j] * self.kappa[self.daytypes[0]])

        #Mean
        mean = numerator_mean / denominator

        #Compute variance numerator
        variance_numerator = ((10**8) * self.sigma2 * self.sigma_g_star_2[0, j])
        #Variance
        variance = variance_numerator / denominator

        self.g_heat[0,j] = self.truncated_norm(-inf, 0, mean, variance)


    def compute_g(self, i, j):
        """ Draw g from its full conditional (page 12/13 summary project) """
        #Compute variance and mean denominator (same denominator for both)
        g_next = 0
        if (i+1 < self.nb_days-1):
            g_next = self.g_heat[i+1,j]

        denominator = 2 * self.sigma2
        numerator_mean = self.sigma2 * (self.g_heat[i-1,j] + g_next)
        if (self.u_heat > self.temperatures[i]):
            denominator = denominator + self.sigma_g_star_2[0, j] * ((self.temperatures[i] - self.u_heat)**2)
            numerator_mean = numerator_mean + \
                                 self.sigma_g_star_2[0, j] * (self.temperatures[i] - self.u_heat) * (self.consumptions[i] - self.s[i,j] * self.kappa[self.daytypes[i]])

        #Mean
        mean = numerator_mean / denominator

        #Compute variance numerator
        variance_numerator = (self.sigma2 * self.sigma_g_star_2[0, j])
        #Variance
        variance = variance_numerator / denominator

        self.g_heat[i,j] =  self.truncated_norm(-inf, 0, mean, variance)


    def compute_sigma_star_2(self, shape, array, j):
        """ Draw sigma_star_2 from its full conditional (page 14 summary project) """
        array_lag = np.roll(array[:,j], 1)
        array_lag[0] = array[0,j]
        scale = (0.01 + sum((array[:,j] - array_lag)**2))**(-1)
        return invgamma.rvs(a = shape, scale = scale, size = 1)[0]


    def compute_x(self):
        """ Simulate X """
        self.x_season = self.kappa[self.daytypes[self.nb_days-1]] * self.s[self.nb_days-1,:]
        self.x_heat = np.maximum((self.temperatures[self.nb_days-1] - self.u_heat) * self.g_heat[self.nb_days-1,:], 0)
        self.x = self.x_season + self.x_heat


    def compute_w(self):
        """ Compute w """
        self.w = np.exp(-(np.square(self.consumptions[self.nb_days-1]-self.x))/(2*self.sigma2))