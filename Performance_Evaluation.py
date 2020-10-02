import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import math


os.chdir('C:/Users/uf11/Desktop/OutFin/Coordinates/')  # NOTE: change directory to where OutFin dataset resides
points_mapping = pd.read_csv('Site4_Local.csv') # load Site 4 local coordinates
RP = points_mapping['RP_ID'].to_numpy()
df_all = pd.DataFrame()

os.chdir('C:/Users/uf11/Desktop/OutFin/Measurements/')

# concatenate all Bluetooth measurements to ge unique MAC addresses observed on Site 4

for i in range(2):
    for j in RP:
        df_temp = pd.read_csv('Phone'+str(i+1)+'_Bluetooth_'+str(j)+'.csv')
        df_all = df_all.append(df_temp, ignore_index=True)

MAC_address = df_all.MAC_address.unique()
df_MAC_address = pd.DataFrame({'MAC_address': MAC_address})
df_all = pd.DataFrame()

for i in range(2):
    for j in RP:
        df_temp = pd.read_csv('Phone'+str(i+1)+'_Bluetooth_'+str(j)+'.csv')
        df1 = df_temp.groupby('MAC_address')['RSS'].apply(list).reset_index(name='RSS_ALL')
        df2 = pd.DataFrame(df1['RSS_ALL'].to_list())
        df3 = pd.concat([df1[['MAC_address']], df2], axis=1)
        result = pd.merge(df_MAC_address, df3, on='MAC_address', how='left')
        result = result.T
        new_header = result.iloc[0]
        result = result[1:]
        result.columns = new_header
        result = result[MAC_address]
        result['RP'] = np.ones(len(result))*(j)
        df_all = pd.concat([df_all, result], ignore_index=True)


# shuffle the data and split into training and testing
data = shuffle(df_all, random_state=100)
data = data.values
X = data[:,0:len(MAC_address)]
y = data[:,len(MAC_address)]
y = y.astype('int')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=100)

# perform preprocessing
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_train = np.nan_to_num(X_train)
X_test = scaler.transform(X_test)
X_test = np.nan_to_num(X_test)

# specify the classifiers under comparision and perform the comparision analysis
names = ["Nearest Neighbors", "RBF SVM", "Decision Tree", "Naive Bayes"]
classifiers = [KNeighborsClassifier(3), SVC(gamma='auto', C=100000), DecisionTreeClassifier(), GaussianNB()]
for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
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

    print("================",name,"================")
    print("Accuracy: ",accuracy_score(y_test, y_pred))
    print("Precision: ",precision_score(y_test, y_pred, average='weighted'))
    print("Recall", recall_score(y_test, y_pred, average='weighted'))
    print("F1", f1_score(y_test, y_pred, average='weighted'))
    print("---------------------------------------------")
    print("Min. distance is: ", min(all_distances))
    print("Max. distance is: ", max(all_distances))
    print('Mean distance is:', np.mean(all_distances))
    print('STD is:', np.std(all_distances))
