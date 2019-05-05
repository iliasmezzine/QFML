#!/usr/bin/env python
# coding: utf-8

# In[414]:


import numpy as np
import pandas as pd
import os
from scipy.optimize import minimize
import matplotlib.pyplot as plt 

path = "C://Users//ilias//OneDrive//Bureau//ML"
os.chdir(path)

df = pd.read_excel("preds.xlsx")
dfpred = df.drop("SXXP",axis=1)
dfrets = df["SXXP"]
cons = {'type':'eq', 'fun': lambda u:np.sum(u)-1} #Optimization constraints on weights
vInd = np.vectorize(lambda u:u/abs(u))

##### Misc Functions #####

def window_split(df,size): #splits the dataframe into overlapping rolling windows of given size
    return [df.iloc[i:size+i] for i in range(len(df)-size)]

def ann_sharpe(rets):
    return np.sqrt(252) * rets.mean() / rets.std()

def MDD(rets): #Metric 3 : Max DD
    maxrets = np.maximum.accumulate(rets)
    draws = 1 - rets/ maxrets
    return np.max(draws)

##### Metric Functions for weights optimization #####

def cpd_return(weights,*window): #Metric 1 : Compound Return
    
    preds = np.array(window[0].drop("SXXP",axis=1))
    rets = np.array(window[0]["SXXP"])
    #pos = np.tanh(np.dot(preds,weights))  #tanh activation
    pos = np.dot(preds,weights)          #identity activation
    return -np.cumprod(1+rets*pos)[-1]

def sharpe_ratio(weights,*window): #Metric 2 : Sharpe Ratio
    
    preds = np.array(window[0].drop("SXXP",axis=1))
    rets = np.array(window[0]["SXXP"])
    
    pos = np.tanh(np.dot(preds,weights))  #tanh activation
    #pos = np.dot(preds,weights)          #identity activation
    return -ann_sharpe(pos)

def max_drawdown(weights,*window): #Metric 3 : Max Drawdown
    
    preds = np.array(window[0].drop("SXXP",axis=1))
    rets = np.array(window[0]["SXXP"])
    
    pos = np.tanh(np.dot(preds,weights))  #tanh activation
    #pos = np.dot(preds,weights)          #identity activation
    return MDD(pos)

#,constraints=cons
##### Optimization Functions #####

def optimize_weights(metric,window): #Optimize a given metric in a given window 
    return minimize(metric, [0,1,0,0],method='SLSQP',args=(window,),bounds =((0, 1),) *4).x

def mass_optim(metric,size): #Return the full list of weights for a given function and a given window size
    global df
    spl = window_split(df,size)
    return [np.array(optimize_weights(metric,spl[i])) for i in range(len(spl))]

#### Optimize for what metric and what window size ? ---> Change the metric to any function applied to the window ####

wsize = 30
current_metric = cpd_return

#### Final Operations #####

current_optim = mass_optim(current_metric,wsize)
positions = np.array([np.dot(dfpred.iloc[wsize+i],current_optim[i]) for i in range(len(current_optim))])
rets_sxxp = dfrets.iloc[wsize:] #returns of SXXP in the remaining frames
rets_longshort = dfrets.iloc[wsize:]*positions #returns of LS strategy using cpd ret optimization

#### Plot compound returns ####

plt.plot(np.cumprod(1+rets_sxxp),dashes=[1,1],label="Cpd. Returns SXXP")
plt.plot(np.cumprod(1+rets_longshort),label="Cpd. Returns opt. LS")

plt.legend()
plt.show()

current_optim
positions


# In[438]:


#### Final Operations #####

wsize = 40
current_metric = cpd_return

current_optim = mass_optim(current_metric,wsize)
positions = np.array([np.dot(dfpred.iloc[wsize+i],current_optim[i]) for i in range(len(current_optim))])
rets_sxxp = dfrets.iloc[wsize:] #returns of SXXP in the remaining frames
rets_longshort = dfrets.iloc[wsize:]*positions #returns of LS strategy using cpd ret optimization

#### Plot compound returns ####

plt.plot(np.cumprod(1+rets_sxxp),dashes=[1,1],label="Cpd. Returns SXXP")
plt.plot(np.cumprod(1+rets_longshort),label="Cpd. Returns opt. LS")

plt.legend()
plt.show()


# In[418]:


plt.plot(current_optim)

