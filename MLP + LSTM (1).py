#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Rolling PCA + all the preprocessing functions

import os
import pandas as pd
import numpy as np
from numpy import concatenate
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import pywt as wt
import matplotlib.pyplot as plt 


def denoise_wt(data,c): #signal denoising with wavelet transform, sig = nb of sig deviation
    
    (u, d) = wt.dwt(data, "haar") #first we apply discrete wavelet transform (using the Haar basis)
    up_thresh = wt.threshold(u, np.std(u)/c, mode="soft")                
    down_thresh = wt.threshold(d, np.std(d)/c, mode="soft")   #remove all the wavelets that have component deviating from more than 1 sigma
    
    return wt.idwt(up_thresh, down_thresh, "haar") #then apply inverse discrete wavelet transform on this result

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True): #convert pb to slp
    n_vars = 1 if type(data) is list else data.shape[1]
    df = data
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg

os.chdir("C://Users//ilias//OneDrive//Bureau//ML")
df = pd.read_excel("sxxp_returns.xlsx")
idx = df.index


def compute_lstm_results(c):
    
    df = pd.DataFrame(denoise_wt(df,c)).set_index(idx)
    df_comp = pd.read_excel("sxxp_comp_returns.xlsx")
    pca = PCA(n_components = 13)

    c_change = [np.sum((df_comp.iloc[i+1]-df_comp.iloc[i])**2) != 0 for i in range(len(df_comp) -1)]
    c_change.insert(0,0)

    df_comp['indic_change'] = np.cumsum(c_change) #indicator function for a composition change
    grp_comp = df_comp.groupby(df_comp['indic_change'],axis=0).groups #time dfs with stable composition
    time_index = [pd.to_datetime(grp_comp[i]) for i in range(len(grp_comp))] #list of time indices for each subset

    pca_ready_dfs = [] #list holding the dfs for rolling PCA

    for u in range(len(time_index)):
        curr_df = df.loc[time_index[u]] #current dataframe
        curr_comp = df_comp.loc[time_index[0]].iloc[0] #current index comp.
        for name in curr_df.columns:
            if curr_comp[name] == 0: #dropping the stocks that are not in the index for that time period
                curr_df.drop(name, axis=1, inplace=True)

        pca_ready_dfs.append(curr_df) #array storing the ready-to-use PCA inputs (last if we have to transform them)


    all_inputs = [] #list holding the inputs

    for x in pca_ready_dfs: #run the PCA for each df, store all the components (in value)
        pca.fit(x)
        #here the .dot is used to retrieve the component in value (<returns_(t), pc_1>, ... <returns_(t), pc_max> )
        curr_input = [[np.dot(x.iloc[i],pca.components_[j]) for i in range(len(x))] for j in range(len(pca.components_))]
        all_inputs = all_inputs + [curr_input]

    final_inputs = [] #list merging the inputs
    for j in range(13):
        cp_0 = [] 
        for i in range(len(all_inputs)):
            cp_0 += all_inputs[i][j]
        final_inputs +=[cp_0]

    #final result as xlsx

    final_inputs = pd.DataFrame(final_inputs).transpose()
    sxxp = pd.read_excel("sxxp_returns_solo.xlsx")
    final_inputs.set_index(sxxp.index,drop=True,inplace=True)
    final_inputs.rename(index=str, columns={i:"pc_{}".format(i) for i in range(13)},inplace=True)

    #concatenate the sxxp and principal components df

    train_set = pd.concat([sxxp,final_inputs], axis=1, join='outer')

    #Preprocessing the data
    #Reframing the problem into a SLP.
    #Splitting the ds into test, train
    #Reshaping everything into LSTM dimension
    #Rescale the data using MinMaxScaler

    values = train_set.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = pd.DataFrame(scaler.fit_transform(values))
    reframed = series_to_supervised(scaled, 1, 1)

    #Drop the unnecessary columns (the PCs[t])
    ldrop = [i for i in range(15,28)]
    reframed.drop(reframed.columns[ldrop], axis=1, inplace=True)

    # Split the data into training and test sets. Taking 80% as training set and 20% as test set

    n_samples = int(len(reframed)*0.8) 
    values = reframed.values
    train = values[:n_samples, :]
    test = values[n_samples:, :]

    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]

    # Reshape everything into LSTM dimension

    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

    # Model Specification (LSTM)
    # Model Training

    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=0, shuffle=False)

    #plt.plot(history.history['loss'], label='in-sample MSE')
    #plt.plot(history.history['val_loss'], label='out-of-sample MSE')
    #plt.legend()
    #plt.show()

    # Model Predictions
    # Predict the test set

    yhat = model.predict(test_X)
    #Reshape test set in initial shape
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2])) 
    #Concatenate the prediction with the test X, rescale all back then keep only the first column

    #Rescale the y_predicted

    inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]

    #Rescale the y_actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]

    #Compute Model RMSE, plot actual vs predicted

    plt.plot(inv_y, dashes=[1,1])
    plt.plot(inv_yhat)
    
rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))

