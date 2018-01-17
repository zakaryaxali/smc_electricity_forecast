# ENSAE : Hidden Markov Models & Sequential Monte Carlo Methods
## On particle filters applied to electricity load forecasting
Based on the article "On particle filters applied to electricity load forecasting"  
Code : Antoine Grelety, Samir Tanfous, Zakarya Ali  

## Code Structure :
* Data treatment
  * [Import Temperatures.ipynb](https://github.com/zakaryaxali/smc_electricity_forecast/blob/master/Import%20Temperatures.ipynb) : take care of temperatures data
  * [utils.ipynb](https://github.com/zakaryaxali/smc_electricity_forecast/blob/master/utils.ipynb) : add daytypes columns to RTE electricity load data
  * [/data](https://github.com/zakaryaxali/smc_electricity_forecast/tree/master/data) : folder containing all the data
* Parameters initialization
  * [gibbs-parameters_init_v*.ipynb](https://github.com/zakaryaxali/smc_electricity_forecast/blob/master/gibbs-parameters_init_v4.ipynb) : Application of Gibbs Sampling to perform parameters initialization
  * [elec_forecast/gibbs_sampling_model.py](https://github.com/zakaryaxali/smc_electricity_forecast/blob/master/elec_forecast/gibbs_sampling_model.py) : Python class with Gibbs sampling methods
* Models evaluation
  * [particle_filtering_*.ipynb](https://github.com/zakaryaxali/smc_electricity_forecast/blob/master/particle_filter_3pm_final-V2.ipynb) : We use particle filter to estimate the model parameter and perform predictions
  * [elec_forecast/particle_filter.py](https://github.com/zakaryaxali/smc_electricity_forecast/blob/master/elec_forecast/particle_filter.py) : Python class with Particle Filter methods (PMMH, Prediction, Resampling)
  * Robbins-Monro
    * [parameter_estimation_robbins_monro.ipynb](https://github.com/zakaryaxali/smc_electricity_forecast/blob/master/parameter_estimation_robbins_monro.ipynb) : notebook dedicated to parameter estimation via Robbins-Monro algorithm
    * [elec_forecast/rm_estimation.py](https://github.com/zakaryaxali/smc_electricity_forecast/blob/master/elec_forecast/rm_estimation.py) : Python class with Robbins-Monro algorithm methods
