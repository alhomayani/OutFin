import os
import pandas as pd
from scipy import stats
import random

os.chdir('C:/Users/uf11/Desktop/OutFin') # NOTE: change the path to where OutFin dataset resides
os.chdir('Code/temporal_data')

def Validity1_WiFi():
    DF1 = {} # creating a dictionary of dataframes where each will hold the WiFi data for a given day from Phone 1
    DF2 = {} # creating a dictionary of dataframes where each will hold the WiFi data for a given day from Phone 2
    # iterate over the three days, read the WiFi data for each phone, calculate the mean RSS of APs, and store the data
    # in the respective dataframe:
    for i in range(3):
        DF1[i] = pd.read_csv('Phone1_day' + str(i+1) + '_WiFi.csv')
        df_temp = DF1[i].iloc[:, 8:]
        DF1[i]['RSS'] = df_temp.mean(axis=1)
        DF2[i] = pd.read_csv('Phone2_day' + str(i+1) + '_WiFi.csv')
        df_temp = DF2[i].iloc[:, 8:]
        DF2[i]['RSS'] = df_temp.mean(axis=1)
    # iterate over all possible day combinations between the two phones:
    for i in range(3):
        for j in range(3):
            temp_list1 = [] # a temporarily list that will hold the averaged RSS readings of intersecting APs between
            # the phones form Phone 1
            temp_list2 = [] # a temporarily list that will hold the averaged RSS readings of intersecting APs between
            # the phones form Phone 2
            common_APs = pd.merge(DF1[i]['BSSID'], DF2[j]['BSSID'], how='inner') # find intersecting APs between the
            # two phones
            randomAPs = random.sample(range(0, len(common_APs)), 50) # randomly select 50 of those APs, calculate their
            # mean RSS value and store them in their corresponding list:
            for k in range(len(randomAPs)):
                temp_RSS = DF1[i].loc[DF1[i]['BSSID'] == common_APs.BSSID[randomAPs[k]]].RSS
                temp_RSS = temp_RSS.values
                temp_RSS = temp_RSS.mean()
                temp_list1.append(temp_RSS.tolist())
                temp_RSS = DF2[j].loc[DF2[j]['BSSID'] == common_APs.BSSID[randomAPs[k]]].RSS
                temp_RSS = temp_RSS.values
                temp_RSS = temp_RSS.mean()
                temp_list2.append(temp_RSS.tolist())
            # calculate the correlation between the RSS values of the two phones for different day combinations:
            print('Phone 1 Day',i+1,'and Phone 2 Day', j+1,'(WiFi):')
            rho, p_value = stats.spearmanr(temp_list1, temp_list2)
            print('Spearman rho:', rho, '(p value:', p_value, ')')
            tau, p_value = stats.kendalltau(temp_list1, temp_list2)
            print('Kendall tau: ', tau, '(p value:', p_value, ')')


def Validity1_Bluetooth():
    DF1 = {} # creating a dictionary of dataframes where each will hold the Bluetooth data for a given day from Phone 1
    DF2 = {} # creating a dictionary of dataframes where each will hold the Bluetooth data for a given day from Phone 2
    # iterate over the three days, read the Bluetooth data for each phone, calculate the mean RSS of Bluetooth devices,
    # and store the data in the respective dataframe:
    for i in range(3):
        DF1[i] = pd.read_csv('Phone1_day' + str(i+1) + '_Bluetooth.csv', usecols=['MAC_address', 'RSS'])
        DF1[i] = DF1[i].groupby(['MAC_address'], as_index=False).mean()
        DF2[i] = pd.read_csv('Phone2_day' + str(i+1) + '_Bluetooth.csv', usecols=['MAC_address', 'RSS'])
        DF2[i] = DF2[i].groupby(['MAC_address'], as_index=False).mean()
    # iterate over all possible day combinations between the two phones:
    for i in range(3):
        for j in range(3):
            temp_list1 = [] # a temporarily list that will hold the averaged RSS readings of intersecting Bluetooth
            # devices between the phones form Phone 1
            temp_list2 = [] # a temporarily list that will hold the averaged RSS readings of intersecting Bluetooth
            # devices between the phones form Phone 2
            common_devices = pd.merge(DF1[i]['MAC_address'], DF2[j]['MAC_address'], how='inner') # find intersecting
            # devices between the two phones
            random_devices = random.sample(range(0, len(common_devices)), 15) # randomly select 15 of those devices,
            # calculate their mean RSS value and store them in their corresponding list:
            for k in range(len(random_devices)):
                temp_RSS = DF1[i].loc[DF1[i]['MAC_address'] == common_devices.MAC_address[random_devices[k]]].RSS
                temp_RSS = temp_RSS.values
                temp_RSS = temp_RSS.mean()
                temp_list1.append(temp_RSS.tolist())
                temp_RSS = DF2[j].loc[DF2[j]['MAC_address'] == common_devices.MAC_address[random_devices[k]]].RSS
                temp_RSS = temp_RSS.values
                temp_RSS = temp_RSS.mean()
                temp_list2.append(temp_RSS.tolist())
            # calculate the correlation between the RSS values of the two phones for different day combinations:
            print('Phone 1 Day',i+1,'and Phone 2 Day', j+1,'(Bluetooth):')
            rho, p_value = stats.spearmanr(temp_list1, temp_list2)
            print('Spearman rho:', rho, '(p value:', p_value, ')')
            tau, p_value = stats.kendalltau(temp_list1, temp_list2)
            print('Kendall tau: ', tau, '(p value:', p_value, ')')


def Validity1_Cellular():
    DF1 = {} # creating a dictionary of dataframes where each will hold the cellular data for a given day from Phone 1
    DF2 = {} # creating a dictionary of dataframes where each will hold the cellular data for a given day from Phone 2
    # iterate over the three days, read the cellular data for each phone, select only the features of interest,
    # calculate the mean value of those features, and store the data in the respective dataframe:
    for i in range(3):
        DF1[i] = pd.read_csv('Phone1_day'+str(i+1)+'_Cellular.csv')
        DF1[i] = DF1[i][['UMTS_neighbors','LTE_neighbors','RSRP_strongest','Frequency','EARFCN','RSRP','RSRQ']]
        DF1[i] = DF1[i].mean()
        DF2[i] = pd.read_csv('Phone2_day'+str(i+1)+'_Cellular.csv')
        DF2[i] = DF2[i][['UMTS_neighbors','LTE_neighbors','RSRP_strongest','Frequency','EARFCN','RSRP','RSRQ']]
        DF2[i] = DF2[i].mean()
    # iterate over all possible day combinations between the two phones:
    for i in range(3):
        for j in range(3):
            # calculate the correlation between the values of the features of interest of the two phones for different
            # day combinations:
            print('Phone 1 Day',i+1,'and Phone 2 Day (Cellular)', j+1,':')
            rho, p_value = stats.spearmanr(DF1[i], DF2[j])
            print('Spearman rho:', rho, '(p value:', p_value, ')')
            tau, p_value = stats.kendalltau(DF1[i], DF2[j])
            print('Kendall tau: ', tau, '(p value:', p_value, ')')


def Validity1_Sensors():
    DF1 = {} # creating a dictionary of dataframes where each will hold the sensors data for a given day from Phone 1
    DF2 = {} # creating a dictionary of dataframes where each will hold the sensors data for a given day from Phone 2
    # iterate over the three days, read the sensors data for each phone, select only the features of interest,
    # calculate the mean value of those features, and store the data in the respective dataframe:
    for i in range(3):
        DF1[i] = pd.read_csv('Phone1_day'+str(i+1)+'_Sensors.csv')
        DF1[i] = DF1[i][['ax', 'ay', 'az', 'wx', 'wy', 'wz', 'Bx', 'By', 'Bz', 'gFx', 'gFy', 'gFz', 'Yaw', 'Pitch', 'Roll',
                 'Pressure', 'Illuminance']]
        DF1[i] = DF1[i].mean()
        DF2[i] = pd.read_csv('Phone2_day'+str(i+1)+'_Sensors.csv')
        DF2[i] = DF2[i][['ax', 'ay', 'az', 'wx', 'wy', 'wz', 'Bx', 'By', 'Bz', 'gFx', 'gFy', 'gFz', 'Yaw', 'Pitch', 'Roll',
                 'Pressure', 'Illuminance']]
        DF2[i] = DF2[i].mean()
    # iterate over all possible day combinations between the two phones:
    for i in range(3):
        for j in range(3):
            # calculate the correlation between the values of the features of interest of the two phones for different
            # day combinations:
            print('Phone 1 Day',i+1,'and Phone 2 Day (Sensors)', j+1,':')
            rho, p_value = stats.spearmanr(DF1[i], DF2[j])
            print('Spearman rho:', rho, '(p value:', p_value, ')')
            tau, p_value = stats.kendalltau(DF1[i], DF2[j])
            print('Kendall tau:', tau, '(p value:', p_value, ')')


# uncomment any of the functions below to activate

# Validity1_WiFi()
# Validity1_Bluetooth()
# Validity1_Cellular()
# Validity1_Sensors()
