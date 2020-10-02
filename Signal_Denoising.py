import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from statistics import mean


mpl.rc('font', family='serif')
plt.rc('xtick', labelsize=9)
plt.rc('ytick', labelsize=9)

# create a function that will contaminate the data with a predefined probability
def contaminate(probability, DF):
    DF = DF.values
    mask = np.random.choice([0,1], (len(DF),9), p=[probability, 1-probability])
    DF[:, 0:9] = np.multiply(DF[:, 0:9], mask)
    return DF

# build the denosing autoencoder
def create_model():
    input_sample = Input(shape=(9,))
    activation = 'relu'
    dropout_rate = 0.1
    encoded = Dense(256, activation=activation)(input_sample)
    encoded = Dropout(dropout_rate)(encoded)
    encoded = Dense(64, activation=activation)(encoded)
    encoded = Dropout(dropout_rate)(encoded)
    bottleneck = Dense(16, activation=activation)(encoded)
    decoded = Dropout(dropout_rate)(bottleneck)
    decoded = Dense(64, activation=activation)(decoded)
    decoded = Dropout(dropout_rate)(decoded)
    decoded = Dense(256, activation=activation)(decoded)
    decoded = Dropout(dropout_rate)(decoded)
    output = Dense(9, activation='sigmoid')(decoded)
    autoencoder = Model(input_sample, output)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    return autoencoder

os.chdir('C:/Users/uf11/Desktop/OutFin/Coordinates/')  # NOTE: change directory to where OutFin dataset resides
points_mapping = pd.read_csv('Site2_Local.csv') # load Site 2 local coordinates
RP = points_mapping['RP_ID'].to_numpy()
df_all = pd.DataFrame()

# change directory to get cellular measurements
os.chdir('C:/Users/uf11/Desktop/OutFin/Measurements/')

# concatenate all cellular measurements form both phones to create the training/testing dataset
for i in range(2):
    for j in RP:
        df_temp = pd.read_csv('Phone'+str(i+1)+'_Cellular_'+str(j)+'.csv')
        df_temp = df_temp[['UMTS_neighbors', 'LTE_neighbors', 'RSRP_strongest', 'ECI', 'Frequency', 'EARFCN', 'TA', 'RSRP', 'RSRQ']]
        df_temp['RP'] = np.ones(len(df_temp))*(j)
        df_all = df_all.append(df_temp, ignore_index=True)

# shuffle the data and split into training and testing
data = shuffle(df_all, random_state=100)
print(len(data))

# create a list of contamination probabilities to iterate over
cont_probs = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.1]

knn_acc_noisy = []
knn_acc_cleaned =[]
knn_err_noisy = []
knn_err_cleaned = []

for prob in cont_probs:
    data_noisy = contaminate(prob, data)
    data_clean = data.values
    X_clean = data_clean[:,0:len(df_all.columns)-1]
    X_noisy = data_noisy[:,0:len(df_all.columns)-1]
    y = data_noisy[:,len(df_all.columns)-1]
    y = y.astype('int')
    X_train_clean, X_test_clean, y_train, y_test = train_test_split(X_clean, y, test_size=0.4, random_state=100)
    X_train_noisy, X_test_noisy, y_train, y_test = train_test_split(X_noisy, y, test_size=0.4, random_state=100)

    # perform preprocessing
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data_clean[:,0:len(df_all.columns)-1])
    X_train_clean = scaler.transform(X_train_clean)
    X_train_noisy = scaler.transform(X_train_noisy)
    X_test_clean = scaler.transform(X_test_clean)
    X_test_noisy = scaler.transform(X_test_noisy)
    X_train_clean = np.nan_to_num(X_train_clean)
    X_train_noisy = np.nan_to_num(X_train_noisy)
    X_test_noisy = np.nan_to_num(X_test_noisy)
    X_test_clean = np.nan_to_num(X_test_clean)

    model = create_model()
    model.fit(X_train_noisy, X_train_clean,
              epochs=800,
              batch_size=100,
              shuffle=True,
              validation_data=(X_test_noisy, X_test_clean))

    X_train_cleaned = model.predict(X_train_noisy)
    X_test_cleaned = model.predict(X_test_noisy)

# perform positioning using noisy data and then using their denoised version to see the difference in performance
    clf = KNeighborsClassifier(1)
    clf.fit(X_train_noisy, y_train)
    y_pred = clf.predict(X_test_noisy)
    all_distances = []
    for i in range(len(y_pred)):
        distance = 0
        for j in range(len(points_mapping)):
            if y_pred[i] == points_mapping.RP_ID[j]:
                x1 = points_mapping.X[j]
                y1 = points_mapping.Y[j]
            if y_test[i] == points_mapping.RP_ID[j]:
                x2 = points_mapping.X[j]
                y2 = points_mapping.Y[j]
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        all_distances.append(distance)

    print("===================",prob,"==========================")
    knn_acc_noisy.append(accuracy_score(y_test, y_pred))
    print("Accuracy: ",knn_acc_noisy)
    print("---------------------------------------------")
    knn_err_noisy.append(np.mean(all_distances))
    print('Mean distance is:', knn_err_noisy)

    clf = KNeighborsClassifier(1)
    clf.fit(X_train_cleaned, y_train)
    y_pred = clf.predict(X_test_cleaned)
    all_distances = []
    for i in range(len(y_pred)):
        distance = 0
        for j in range(len(points_mapping)):
            if y_pred[i] == points_mapping.RP_ID[j]:
                x1 = points_mapping.X[j]
                y1 = points_mapping.Y[j]
            if y_test[i] == points_mapping.RP_ID[j]:
                x2 = points_mapping.X[j]
                y2 = points_mapping.Y[j]
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        all_distances.append(distance)

    print("============================================")
    knn_acc_cleaned.append(accuracy_score(y_test, y_pred))
    print("Accuracy: ",knn_acc_cleaned)
    print("---------------------------------------------")
    knn_err_cleaned.append(np.mean(all_distances))
    print('Mean distance is:', knn_err_cleaned)


# plot the results
fig, ax1 = plt.subplots()
size= 8
linewidth = 1
color_1 = 'r'
color_2 = 'b'
ax1.set_xlabel('$p_{loss}$', size=size)
ax1.set_ylabel('Accuracy', size=size)
l1 = ax1.plot(cont_probs, knn_acc_noisy, color=color_1, linestyle='-', marker='x', linewidth=linewidth, label='Accuracy (noisy features)')
l2 = ax1.plot(cont_probs, knn_acc_cleaned, color=color_2, linestyle='-', marker='x', linewidth=linewidth, label='Accuracy (denoised features)')
ax1.tick_params(axis='y', size=size, labelsize=size)
ax1.tick_params(axis='x', size=size, labelsize=size)

ax2 = ax1.twinx()
ax2.set_ylabel('Mean Positioning Error ($\it{cm}$)', size=size)
l3 = ax2.plot(cont_probs, knn_err_noisy, color=color_1, linestyle='-', marker='.', linewidth=linewidth, label='Mean Positioning Error (noisy features)')
l4 = ax2.plot(cont_probs, knn_err_cleaned, color=color_2, linestyle='-', marker='.', linewidth=linewidth, label='Mean Positioning Error (denoised features)')
ax2.tick_params(axis='y', size=size, labelsize=size)

fig.tight_layout()

leg = l1 + l2 + l3 + l4
labs = [l.get_label() for l in leg]
ax1.legend(leg, labs, loc=9, prop={'size': size-1})
ax1.xaxis.grid(which='both')
ax1.set_xticks(cont_probs)
fig.autofmt_xdate(rotation=45)

plt.show()

mean_acc_gain = [a - b for a, b in zip(knn_acc_cleaned, knn_acc_noisy)]
mean_err_reduction = [a - b for a, b in zip(knn_err_noisy, knn_err_cleaned)]
print('Mean gain in accuracy: ', mean(mean_acc_gain), 'Mean reduction in positioning error', mean(mean_err_reduction))