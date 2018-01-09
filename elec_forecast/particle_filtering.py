import pandas as pd
import numpy as np
from scipy.stats import truncnorm, invgamma
import pickle
from math import inf, sqrt
import time

class BootstrapFilterModel():
    def __init__(self):	
        self.g_heat =
        self.sigma_s = 0
        self.sigma_g = 0
        self.x_init =
        self.s =
        self.g_heat =
		self.sigma_s_star_2 =
		self.sigma_g_star_2 =
		self.w =

	def truncated_norm(lower_bound, upper_bound, mean, std, size):
		a, b = (lower_bound - mean) / std, (upper_bound - mean) / std
		if size = 1:
			return  truncnorm.rvs(a = a,b = b, loc= mean, scale = std, size=size)[0]
		else :
			return truncnorm.rvs(a = a,b = b, loc= mean, scale = std, size=size)
		
	def gibbs_initialisation(temperatures, daytype_list, consumptions, nb_days, nb_particles, sigma2, kappa, u_heat, log):
		"""
		We initialize the bootstrap filter using a Gibbs sampler.
		Like in the article, we suppose sigma_s_star and sigma_g_star fixed
		Parameters :
		temperatures: list of temperatures (per day)
		daytype_list: list of daytypes (per day)
		consumptions: list of electricity consumption (per day)
		nb_days: number of iterations for the initialization step
		nb_particles: number of particles
		sigma2: initial sigma squared (fixed)
		kappa: 
		u_heat: 
		log: if True returns value of the computed parameters during the sampling
		"""
		# Variables initialization
		s = np.zeros((nb_days, nb_particles)) 
		g_heat = np.zeros((nb_days, nb_particles))
		#sigma_s and sigma_g are fixed
		sigma_s_star_2 = np.zeros((1, nb_particles)) 
		sigma_g_star_2 = np.zeros((1, nb_particles))

		#Gibbs : Initialization
		s[0,0] = truncated_norm(0, inf, 0, 10**4, 1)
		g_heat[0,0] =  truncated_norm(-inf, 0, 0, 10**4, 1)
		sigma_s_star_2[0, 0] = invgamma.rvs(a=10**(-2), scale=10**(2), size = 1)[0]
		sigma_g_star_2[0, 0] = invgamma.rvs(a=10**(-2), scale=10**(2), size = 1)[0]
		
		#Gibbs : Step 0
		sigma_s_star0 = sqrt(sigma_s_star_2[0,0])
		sigma_g_star0 = sqrt(sigma_g_star_2[0,0])
		for i in range(1, nb_days):
			s[i,0] = s[i-1,0] + \
					 truncnorm.rvs(a = -s[i-1,0]/sigma_s_star0,b = inf, loc= 0, scale = sigma_s_star0, size=1)[0] #page 18
			g_heat[i,0] = g_heat[i-1,0] + \
						  truncnorm.rvs(a = -inf, b=- g_heat[i-1,0]/sigma_g_star0, loc= 0, scale = sigma_g_star0, size=1)[0]
		
		#Gibbs : step t > 0
		for j in range(1, nb_particles):
			if(j%(nb_particles/10)==0):
				print("Gibbs sampling for particle " + str(j) + "/" + str(nb_particles))
			s[:,j] = s[:,j-1]
			g_heat[:,j] = g_heat[:,j-1]
			sigma_s_star_2[:,j] = sigma_s_star_2[:,j-1]
			sigma_g_star_2[:,j] = sigma_g_star_2[:,j-1]
			
			# Simulate s0
			#Compute variance and mean denominator
			denom_s_0 = (10**4)*sigma_s_star_2[0, j]*(kappa[daytype_list[0]]**2) + sigma2*sigma_s_star_2[0, j] + (10**8)*sigma2 
			#Compute mean numerator
			numerator_mean_s_0 = (10**4)*sigma2* s[1,j] + (10**8)*sigma_s_star_2[0, j]*kappa[daytype_list[0]]*consumptions[0]
			if (u_heat > temperatures[0]):
				numerator_mean_s_0 -= (10**4)*sigma_s_star_2[0, j]*kappa[daytype_list[0]]*g_heat[0,j]*(temperatures[0] - u_heat)
			#Compute the final parameters of the truncated normal that simulates from the full conditional of s_0
			#Mean
			mean_s_0 = numerator_mean_s_0 / denom_s_0
			#Variance
			var_s_0 = ((10**8) *sigma2*sigma_s_star_2[0, j]) / denom_s_0
			std_s_0 = sqrt(var_s_0)

			a=-mean_s_0/std_s_0
			while True:
				s[0,j] = truncnorm.rvs(a = a, b = inf, loc= mean_s_0, scale = std_s_0, size=1)[0]
				if(s[0,j]!=inf):
					break
			if log:
				print("s[0,"+str(j)+"]")
				print(s[0,j])
			
			# Simulate s(i), i>0
			for i in range(1, nb_days):
				denom_s_i = 2*sigma2 + sigma_s_star_2[0, j]*(kappa[daytype_list[i]]**2)
				dependence_next_s = 0
				if (i+1 < nb_days-1):
					dependence_next_s = s[i+1,j]
				#Compute mean numerator
				numerator_mean_s_i = sigma2*(s[i-1,j] + dependence_next_s) + \
									 sigma_s_star_2[0, j]*kappa[daytype_list[i]]*(consumptions[i])
				if (u_heat > temperatures[i]):
					numerator_mean_s_i = numerator_mean_s_i - \
										 sigma_s_star_2[0, j]*kappa[daytype_list[i]]*g_heat[i,j]*(temperatures[i] - u_heat)
				mean_s_i = numerator_mean_s_i / denom_s_i
				var_s_i = (sigma2*sigma_s_star_2[0, j]) / denom_s_i
				std_s_i = sqrt(var_s_i)
				
				a=-mean_s_i/std_s_i
				temp=0
				while True:
					s[i,j] = truncnorm.rvs(a = a, b = inf, loc= mean_s_i, scale = std_s_i, size=1)[0]
					if(s[i,j]!=inf and s[i,j]!=-inf):
						break
					
					temp+=1
					if(temp%5==0):
						print(temp)
				if log:
					print("s["+str(i)+","+str(j)+"]")
					print(s[i,j])
			
			# Simulate g_heat0
			denom_g_0 = sigma2*sigma_g_star_2[0, j] + (10**8)*sigma2
			numerator_mean_g_0 = (10**8)*sigma2* g_heat[1,j]
			if (u_heat > temperatures[0]):
				denom_g_0 = denom_g_0 + (10**8)*sigma_g_star_2[0, j]*((temperatures[0] - u_heat )**2)
				numerator_mean_g_0 = numerator_mean_g_0 + \
									 (10**8)*sigma_g_star_2[0, j]*(temperatures[0] - u_heat)*(consumptions[0] - s[0,j]*kappa[daytype_list[0]])
			#Compute the final parameters of the truncated normal that simulates from the full conditional of g_0
			mean_g_0 = numerator_mean_g_0 / denom_g_0
			var_g_0 = ((10**8) *sigma2*sigma_g_star_2[0, j]) / denom_g_0
			std_g_0 = sqrt(var_g_0)

			b=-mean_g_0/std_g_0
			while True:
				g_heat[0,j] =  truncnorm.rvs(a = -inf, b = b, loc= mean_g_0, scale = std_g_0, size=1)[0]
				if(g_heat[0,j]!=-inf):
					break
			if log:
				print("g_heat["+str(0)+","+str(j)+"]")
				print(g_heat[0,j])

			# Simulate g_heat(i), i>0
			for i in range(1, nb_days):
				dependence_next_g = 0
				if (i+1 < nb_days-1):
					dependence_next_g = g_heat[i+1,j]

				denom_g_i = 2*sigma2
				numerator_mean_g_i = sigma2*(g_heat[i-1,j] + dependence_next_g)
				if (u_heat > temperatures[i]):
					denom_g_i = denom_g_i + sigma_g_star_2[0, j]*((temperatures[i] - u_heat )**2)
					numerator_mean_g_i = numerator_mean_g_i + \
										 sigma_g_star_2[0, j]*(temperatures[i] - u_heat )*(consumptions[i] - s[i,j]*kappa[daytype_list[i]])

				mean_g_i = numerator_mean_g_i / denom_g_i
				var_g_i = (sigma2*sigma_g_star_2[0, j]) / denom_g_i
				std_g_i = sqrt(var_g_i)

				b=-mean_g_i/std_g_i
				while True:
					g_heat[i,j] =  truncnorm.rvs(a = -inf, b = b, loc= mean_g_i, scale = std_g_i, size=1)[0]
					if(g_heat[i,j]!=-inf):
						break
				if log:
					print("g_heat["+str(i)+","+str(j)+"]")
					print(g_heat[i,j])

			# Simulate the variances
			shape_variances = 0.01 + ((nb_days - 1)/2)
			s_lag = np.roll(s[:,j], 1)
			s_lag[0] = s[0,j]
			scale_s = (0.01 + sum((s[:,j] - s_lag)**2))**(-1)
			sigma_s_star_2[0, j] = invgamma.rvs(a=shape_variances, scale=scale_s, size = 1)[0]
			g_lag = np.roll(g_heat[:,j], 1)
			g_lag[0] = g_heat[0,j]
			scale_g = (0.01 + sum((g_heat[:,j] - g_lag)**2))**(-1)
			sigma_g_star_2[0, j] = invgamma.rvs(a=shape_variances, scale=scale_g, size = 1)[0]

			if log:
				print(s[:,j])
				print(g_heat[:,j])

		# Return the initialization of the Particle Filter at date (nb_days - 1)
		s_init = s[nb_days-1,]
		g_heat_init = g_heat[nb_days-1,:]
		sigma_s_init = np.sqrt(sigma_s_star_2[0,nb_particles-1])
		sigma_g_init = np.sqrt(sigma_g_star_2[0,nb_particles-1])

		x_season = kappa[daytype_list[nb_days-1]]*s_init
		x_heat = np.maximum((temperatures[nb_days-1]-u_heat)*g_heat_init,0)
		print("x_heat")
		print(x_heat)
		print("x_season")
		print(x_season)
		x_init = x_season + x_heat
		
		return s_init, g_heat_init, sigma_s_init, sigma_g_init, x_init, s, g_heat, sigma_s_star_2, sigma_g_star_2