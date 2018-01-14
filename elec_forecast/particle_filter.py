import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm, norm

class ParticleFilter():
    def __init__(self, consumption_day_ahead, T_h, daytype, n_pred, M, x_init, w_init, kappa, u_h, sigma_s_init, sigma_g_init, s_init, g_heat_init, sigma2):
        self.consumption_day_ahead = consumption_day_ahead
        self.T_h = T_h
        self.daytype = daytype
        self.n_pred = n_pred
        self.M = M
        self.x_init = x_init
        self.w_init = w_init
        self.kappa = kappa
        self.u_h = u_h
        self.sigma_s_init = sigma_s_init
        self.sigma_g_init = sigma_g_init
        self.s_init = s_init
        self.g_heat_init = g_heat_init
        self.sigma2 = sigma2
        #initialize matrix of x_heat, x_season
        self.ESS = np.array(np.ones(n_pred+1))
        self.x = np.zeros([n_pred+1,M])
        self.w = np.zeros([n_pred+1,M])
        self.lh_y_n = np.zeros(n_pred+1)
        self.x_season = np.zeros([n_pred+1,M])
        self.x_heat = np.zeros([n_pred+1,M])
        #store the accepted value from pmmh in lists
        self.accept_rate = 0
        self.u_h_list=[]
        self.sigma_list=[]
        self.sigma_g_list=[]
        self.sigma_s_list=[]

        self.x[0,:], self.w[0,:], self.ESS[0], self.lh_y, self.sigma_s_init, self.sigma_g_init, self.g_heat_init, self.s_init = self.resample(x_init,
                                                                                                                                            w_init,
                                                                                                                                            2,
                                                                                                                                            15,
                                                                                                                                            0,
                                                                                                                                            sigma_s_init,
                                                                                                                                            sigma_g_init,
                                                                                                                                            g_heat_init,
                                                                                                                                            s_init,
                                                                                                                                            sigma2**0.5)


    def resample(self, x_pred, w_prev, nbdays_pred_today, len_init, n, sigma_s, sigma_g, g_h, s, sigma):
        #STEP 2 OF PARTICLE FILTER
        M = self.M
        #compute y_n
        delta_cons_gaus=-np.square(self.consumption_day_ahead[len_init+n+nbdays_pred_today]-x_pred)/(2*sigma**2)
        y_n=np.exp(delta_cons_gaus)
        #compute new weights
        if n>0:
            w_ = w_prev*y_n
        else:
            w_ = w_prev
        #likelihood of y_n
        lh_y_n = np.sum(delta_cons_gaus)
        #normalize weights
        w_h = w_/sum(w_)
        #calculate ESS
        ESS = 1/sum(np.square(w_h))
        x = np.zeros(M)
        w = np.zeros(M)
        print("ESS of normalized weights=",round(ESS,6))
        if ESS <0.001*M: #reset the weights, keep x predicted as such
            print("ESS <0.001*M")
            x = x_pred
            if n==0:
                w = np.ones(M)*(1/M)
            w=w_prev
        elif (ESS >= 0.001*M and ESS < 0.5*M):  #reset all the weights, add some noise to a fraction of the x's
            print("ESS>=0.001*M and ESS_0<0.5*M")
            x, w, sigma_s, sigma_g, g_h, s = self.resample_multinomial(x_pred, w_h, sigma_s, sigma_g, g_h, s, M)
        elif ESS>=0.5*M:  #No degeneracy
            print("ESS>=0.5*M")
            x = x_pred
            w = w_h
        else:
            print("ESS critically low")
            x = x_pred
            if n==0:
                w=np.ones(M)*(1/M)
            w = w_prev

        print("new ESS=",round(1/sum(np.square(w)),6))
        return x,w,ESS,lh_y_n,sigma_s,sigma_g,g_h,s

    def resample_multinomial(self,x_temp, w_temp, sigma_s, sigma_g, g_h, s, M):
        multinomial = np.random.multinomial(1,w_temp,M)
        new_x = np.zeros(M)
        new_s = np.zeros(M)
        new_g_heat = np.zeros(M)
        new_sigma_s = np.zeros(M)
        new_sigma_g = np.zeros(M)

        for i in range(M):
            new_x[i] = x_temp[np.argmax(multinomial[i,])]
            new_s[i] = s[np.argmax(multinomial[i,])]
            new_g_heat[i] = g_h[np.argmax(multinomial[i,])]
            new_sigma_s[i] = sigma_s[np.argmax(multinomial[i,])]
            new_sigma_g[i] = sigma_g[np.argmax(multinomial[i,])]

        new_w = (1/M)*np.ones(M)

        return new_x,new_w,new_sigma_s,new_sigma_g,new_g_heat,new_s

    def particle_filter(self, nbdays_pred_today, len_init, len_filtering, s, g_h, sigma_s, sigma_g, sigma):
        lh_y_n = np.zeros(len_filtering)
        x_pred = np.zeros([len_filtering,self.M])
        x_pred_mean = np.zeros(len_filtering)
        ESS = np.zeros(len_filtering)
        relative_entropy = np.zeros(len_filtering)
        #calcul of the MAPE for 1 day forecast
        mape_one_day_forecast = np.zeros(len_filtering)
        

        sigma_s = self.sigma_s_init
        sigma_g = self.sigma_g_init
        s_prev = s
        g_h_prev= g_h
        for n in range(1,len_filtering):
            print("n=",n)
            #prediction X[n] one day ahead, hourly forecast
            x_s, s_current, sigma_s = self.compute_x_season(int(self.daytype[len_init + n + nbdays_pred_today]), self.kappa, s_prev, sigma_s)
            x_h, g_h_current, sigma_g = self.compute_x_heat(g_h_prev, n + len_init + nbdays_pred_today, sigma_g)
            s_prev=s_current
            g_h_prev=g_h_current
            x_pred[n,:] = x_s + x_h
            #print("number of negative values:",len(x_pred[x_pred<0]))
            print("x_pred_mean =","{:.2e}".format(np.mean(x_pred[n,:])),
                  "real consumption=","{:.2e}".format(self.consumption_day_ahead[n]))
            print("x_pred min=","{:.2e}".format(np.min(x_pred[n,:])),"x_pred max","{:.2e}".format(np.max(x_pred[n,:])))
            #take new values of parameters to feed x_season and x_heat in the next step
            #regularization
            x_pred[n,:], self.w[n,:], ESS[n], lh_y_n[n], sigma_s, sigma_g, g_h, s = self.resample(x_pred[n,:],
                                                                                                      self.w[n-1,:],
                                                                                                      nbdays_pred_today,
                                                                                                      len_init,
                                                                                                      n,
                                                                                                      sigma_s,
                                                                                                      sigma_g,
                                                                                                      g_h,
                                                                                                      s,
                                                                                                      sigma)

            print("------------------------")
            x_pred_mean[n]=np.mean(x_pred[n,:])
            mape_one_day_forecast[n] = abs((x_pred_mean[n]-self.consumption_day_ahead[n])/self.consumption_day_ahead[n])
            relative_entropy[n] = -sum(self.w[n,:]*np.log(self.w[n,:]))
            
        mape_value = (1/(len_filtering-nbdays_pred_today))*sum(mape_one_day_forecast)           
        print("MAPE value =","{:.2e}".format(mape_value))
        
        #relative entropy
        fig=plt.figure(figsize=(12,6))
        plt.plot(range(len_filtering),relative_entropy/np.log(self.M))
        plt.title("Evolution of relative entropy",fontweight='bold')
        plt.xlabel('forecasted day')
        plt.xlim(0,len_filtering)
        plt.show()

        return lh_y_n,x_pred_mean,ESS

    def predict(self, pred_forward, n_pred, len_init):
        x_predict=np.zeros([len(pred_forward),n_pred])
        ESS_calc=np.zeros([len(pred_forward),n_pred])
        for i in range(len(pred_forward)):
            log_lh_init, x_predict[i,:], ESS_calc[i,:] = self.particle_filter(pred_forward[i],
                                                                              len_init,
                                                                              n_pred,
                                                                              self.s_init,
                                                                              self.g_heat_init,
                                                                              self.sigma_s_init,
                                                                              self.sigma_g_init,
                                                                              self.sigma2**0.5)

        return x_predict, ESS_calc

    def log_joint_prior(self, u_h, sigma, sigma_g, sigma_s):
    #joint prior density of parameters
        res = (-(u_h-14)**2)/2
        res = res+(-0.01-1)*np.log(sigma**2) - (0.01/sigma**2)
        res = res+(-0.01-1)*np.log(sigma_g**2) - (0.01/sigma_g**2)
        res = res+(-0.01-1)*np.log(sigma_s**2) - (0.01/sigma_s**2)
        return res

    def pmmh(self, len_filter_pmmh, len_iter_mha, pred_forward, len_init):
        ## PMMH
        u_h_current = 13
        sigma_s_current = 10**4
        sigma_g_current = 10**4
        sigma_current = 10**4

        accept_log_proba = np.zeros(len_filter_pmmh)
        log_lh_init = np.zeros(len_filter_pmmh)
        lh_y_prop = np.zeros(len_filter_pmmh)

        #### Initialization of hyperparameters

        #standard deviation of normal/trunc normal proposals on parameters
        std_hyp_sigma_g = 5*10**3
        std_hyp_sigma_s = 5*10**3
        std_hyp_sigma = 5*10**3
        std_hyp_u=1

        #joint log prior density initialize
        log_prior_init = self.log_joint_prior(u_h_current,sigma_current,sigma_g_current,sigma_s_current)
        print("log_prior_init=",log_prior_init)
        #initial parameters otbained from Gibbs. These initialized parameters will not change through iterations

        #### Run initial particle filter and get the log likelihood

        log_lh_init = self.particle_filter(pred_forward[0],
                                          len_init,
                                          len_filter_pmmh,
                                          self.s_init,
                                          self.g_heat_init,
                                          sigma_s_current,
                                          sigma_g_current,
                                          sigma_current,
                                          )[0]

        print("log_lh_init[len_filter_pmmh-1]=",log_lh_init[len_filter_pmmh-1])

        #### PMMH Algorithm

        for step in range(len_iter_mha):
            print("___________________________________________________________")
            print("Metropolis Hastings step:",step)
            #we need 6 inputs to compute the (log) acceptance probability log(r):
            #log_likelihood, joint prior density, log proposal density for both current parameters and proposed parameters
            #sample proposal for u_h, sigma, sigma_g, sigma_s
            u_h_prop = np.random.normal(u_h_current, std_hyp_u, size=1)
            sigma_prop = truncnorm.rvs(a=(0-sigma_current)/std_hyp_sigma, b=np.inf,scale=std_hyp_sigma,loc=sigma_current, size=1)
            sigma_g_prop = truncnorm.rvs(a=(0-sigma_g_current)/std_hyp_sigma_g, b=np.inf,scale=std_hyp_sigma_g,loc=sigma_g_current, size=1)
            sigma_s_prop = truncnorm.rvs(a=(0-sigma_s_current)/std_hyp_sigma_s, b=np.inf,scale=std_hyp_sigma_s,loc=sigma_s_current, size=1)
            print("proposed parameters:","u_heat:",u_h_prop,"sigma:",sigma_prop,"sigma_g:",sigma_g_prop,"sigma_s:",sigma_s_prop)

            #1/run a particle filter with the proposed parameters to obtain a an estimation of likelihood proposed
            #  consider the likelihood of the last day of the fitering
            lh_y_prop = self.particle_filter(pred_forward[0],
                                             len_init,
                                             len_filter_pmmh,
                                             self.s_init,
                                             self.g_heat_init,
                                             sigma_s_prop,
                                             sigma_g_prop,
                                             sigma_prop,
                                             )[0]
            print("log likelihood proposal of y:",np.sum(lh_y_prop))

            #2/generate prior proposals and compute joint log density of them
            log_prior_prop = self.log_joint_prior(u_h_prop,sigma_prop,sigma_g_prop,sigma_s_prop)
            print("proposed log prior:",log_prior_prop)

            #3/compute log proposal density h(current parameter|proposed parameter)
            current_log_density=np.log(norm.cdf(sigma_current/std_hyp_sigma,loc=0,scale=1)) + \
                                np.log(norm.cdf(sigma_s_current/std_hyp_sigma_s,loc=0,scale=1)) + \
                                np.log(norm.cdf(sigma_g_current/std_hyp_sigma_g,loc=0,scale=1))
            print("proposal log density initial parameters given proposed param:",current_log_density)

            #4/log likelihood from initial parameters --> already done: log_lh_init
            #5/joint prior of the initial parameters: we already have them
            #6/compute log proposal density h(proposed parameter|current parameter)
            prop_log_density=np.log(norm.cdf(sigma_prop/std_hyp_sigma,loc=0,scale=1)) + \
                                np.log(norm.cdf(sigma_s_prop/std_hyp_sigma_s,loc=0,scale=1)) + \
                                np.log(norm.cdf(sigma_g_prop/std_hyp_sigma_g,loc=0,scale=1))
            print("proposal log density proposed parameters given current param:",prop_log_density)

            #we add up these elements to construct the log acceptance probability
            #numerator
            accept_log_proba[step]=np.sum(lh_y_prop)+log_prior_prop+current_log_density
            #denominator
            accept_log_proba[step]=accept_log_proba[step]-np.sum(log_lh_init)-log_prior_init-prop_log_density
            print("acceptance log probability:",accept_log_proba[step])
            u = np.random.random()
            #to get an acceptance rate > 5%, we need log_proba to be at least -3
            if(np.log(u)<min(0,accept_log_proba[step])):
                print("ACCEPT")
                log_lh_init = lh_y_prop
                sigma_current = sigma_prop
                sigma_g_current = sigma_g_prop
                sigma_s_current = sigma_s_prop
                u_h_current = u_h_prop
                #store the accepted values
                self.accept_rate += 1
                self.u_h_list.append(u_h_current)
                self.sigma_list.append(sigma_current)
                self.sigma_g_list.append(sigma_g_current)
                self.sigma_s_list.append(sigma_s_current)
            else:
                print("REJECT")

        print("accept_rate/len_iter_mha=",self.accept_rate/len_iter_mha)

        print("sigma_current=",sigma_current)
        print("sigma_g_current=",sigma_g_current)
        print("sigma_s_current=",sigma_s_current)
        print("u_h_current=",u_h_current)

    def compute_x_season(self, day_type, k_day, s_prev, sigma_s_prev):
        nu=truncnorm.rvs(a = (-sigma_s_prev-0) / self.sigma_s_init , b = np.inf, loc= 0, scale = self.sigma_s_init, size=self.M)
        sigma_s=sigma_s_prev+nu
        err=truncnorm.rvs(a = -s_prev / sigma_s , b = np.inf, loc= 0, scale = sigma_s, size=self.M)
        s=s_prev+err
        x_s=s*k_day[day_type]

        return x_s, s, sigma_s

    def compute_x_heat(self, g_h_prev, day, sigma_g_prev):
        nu=truncnorm.rvs(a = -sigma_g_prev / self.sigma_g_init , b = np.inf, loc= 0, scale = self.sigma_g_init, size=self.M)
        sigma_g=sigma_g_prev+nu
        err=truncnorm.rvs(a = -np.inf , b =(-g_h_prev-0) / sigma_g, loc= 0, scale = sigma_g, size=self.M)
        g_h=g_h_prev+err

        if(self.u_h-self.T_h[day]<0):
            print("No heating effect")

        x_h=g_h*(self.T_h[day]-self.u_h)*max(np.sign(self.u_h-self.T_h[day]),0)

        return x_h, g_h, sigma_g
