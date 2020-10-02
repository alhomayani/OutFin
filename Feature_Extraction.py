import numpy as np
import os
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import random


mpl.rc('font', family='serif')
plt.rc('text', usetex=True)

os.chdir('C:/Users/uf11/Desktop/OutFin/Measurements/')
df_all = pd.DataFrame()

# concatenate all WiFi measurements to ge unique BSSIDs observed in the entire data collection environment
for i in range(2):
    for j in range(122):
        df_temp = pd.read_csv('Phone'+str(i+1)+'_WiFi_'+str(j+1)+'.csv')
        df_all = df_all.append(df_temp, ignore_index=True)

BSSID = df_all.BSSID.unique()
df_BSSID = pd.DataFrame({'BSSID': BSSID})
df_all = pd.DataFrame()

# pre-process the files to be suitable for dimensionality reduction
for i in range(2):
    for j in range(122):
        df_temp = pd.read_csv('Phone'+str(i+1)+'_WiFi_'+str(j+1)+'.csv')
        df_temp = df_temp.drop_duplicates(subset='BSSID', keep="first")
        result = pd.merge(df_BSSID, df_temp, on='BSSID', how='left')
        result = result[['BSSID', 'RSS_0', 'RSS_1', 'RSS_2', 'RSS_3', 'RSS_4', 'RSS_5', 'RSS_6', 'RSS_7', 'RSS_8']]
        result = result.T
        new_header = result.iloc[0]
        result = result[1:]
        result.columns = new_header
        result = result[BSSID]
        result['RP'] = np.ones(len(result))*(j+1)
        df_all = pd.concat([df_all, result], ignore_index=True)

data = shuffle(df_all)
data = data.values
X_train = data[:,0:len(BSSID)]
Y_train = data[:,len(BSSID)]
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_train = np.nan_to_num(X_train)

# apply PCA
mu = X_train.mean(axis=0)
U,s,V = np.linalg.svd(X_train - mu, full_matrices=False)
PC_PCA = np.dot(X_train - mu, V.transpose())
Reconst_PCA = np.dot(PC_PCA[:, :3], V[:3, :]) + mu
err_PCA = np.sum((X_train - Reconst_PCA) ** 2) / Reconst_PCA.shape[0] / Reconst_PCA.shape[1]

# apply Autoencoder
activation = 'relu'
m = Sequential()
m.add(Dense(600,  activation=activation, input_shape=(len(BSSID),)))
m.add(Dense(250,  activation=activation))
m.add(Dense(3,    activation='linear', name="bottleneck"))
m.add(Dense(250,  activation=activation))
m.add(Dense(600,  activation=activation))
m.add(Dense(len(BSSID),  activation=activation))
m.compile(loss='mean_squared_error', optimizer = Adam())
history = m.fit(X_train, X_train, batch_size=100, epochs=400, verbose=1, validation_data=(X_train, X_train))
encoder = Model(m.input, m.get_layer('bottleneck').output)
LV_AE = encoder.predict(X_train)
Reconst_AE = m.predict(X_train)
err_AE = np.sum((X_train - Reconst_AE) ** 2) / Reconst_AE.shape[0] / Reconst_AE.shape[1]

# randomly select 10 RPs to perform analysis and plot results
randomRPs = random.sample(range(1, 122), 10)

colors = ('tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan')
RPs = tuple(randomRPs)

fig = plt.figure()
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

for j, color, RP in zip(randomRPs, colors, RPs):
    df_P1P2 = pd.DataFrame()
    for i in range(2):
        df_temp = pd.read_csv('Phone' + str(i + 1) + '_WiFi_' + str(j) + '.csv')
        df_temp = df_temp.drop_duplicates(subset='BSSID', keep="first")
        result = pd.merge(df_BSSID, df_temp, on='BSSID', how='left')
        result = result[['BSSID', 'RSS_0', 'RSS_1', 'RSS_2', 'RSS_3', 'RSS_4', 'RSS_5', 'RSS_6', 'RSS_7', 'RSS_8']]
        result = result.T
        new_header = result.iloc[0]
        result = result[1:]
        result.columns = new_header
        result = result[BSSID]
        result['RP'] = np.ones(len(result)) * j
        df_P1P2 = pd.concat([df_P1P2, result], ignore_index=True)

    df_P1P2 = df_P1P2.values
    Xtrain = df_P1P2[:,0:len(BSSID)]
    Ytrain = df_P1P2[:,len(BSSID)]
    Xtrain = scaler.transform(Xtrain)
    Xtrain = np.nan_to_num(Xtrain)
    PC_PCA = np.dot(Xtrain - mu, V.transpose())
    LV_AE = encoder.predict(Xtrain)
    ax1.scatter(LV_AE[:, 0], LV_AE[:, 1], LV_AE[:, 2], c=color, s=20, label=RP)
    ax2.scatter(PC_PCA[:, 0], PC_PCA[:, 1], PC_PCA[:, 2], c=color, s=20, label=RP)

ax1.legend(title='RP', loc=6, fontsize='small', fancybox=True)
ax2.legend(title='RP', loc=6, fontsize='small', fancybox=True)

labelpad=-5
ax2.grid(True)
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.set_zticklabels([])
ax2.set_xlabel('PC 1', labelpad=labelpad)
ax2.set_ylabel('PC 2', labelpad=labelpad)
ax2.set_zlabel('PC 3', labelpad=labelpad)
ax2.set_title(r'{PCA}' +'\n' + r'{\small Reconstruction Cost (MSE) = ' + str(round(err_PCA, 4)) + '}')

ax1.grid(True)
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1.set_zticklabels([])
ax1.set_xlabel('LV 1', labelpad=labelpad)
ax1.set_ylabel('LV 2', labelpad=labelpad)
ax1.set_zlabel('LV 3', labelpad=labelpad)
ax1.set_title(r'{Autoencoder}' +'\n' + r'{\small Reconstruction Cost (MSE) = ' + str(round(err_AE, 4)) + '}')

plt.show()

