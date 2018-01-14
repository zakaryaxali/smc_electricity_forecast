# ENSAE : Hidden Markov Models & Sequential Monte Carlo Methods
## On particle filters applied to electricity load forecasting
Based on the article "On particle filters applied to electricity load forecasting"  
Code : Antoine Grelety, Samir Tanfous, Zakarya Ali  

## Code Structure :
* Data treatment
  * Import Temperatures.ipynb : take care of temperatures data
  * utils.ipynb : add daytypes columns to RTE electricity load data
* Parameters initialization
  * gibbs-parameters_init_v*.ipynb : Application of Gibbs Sampling to perform parameters initialization
* Model evaluation
  * particle_filtering_*.ipynb : We use particle filter to estimate the model parameter and perform predictions
