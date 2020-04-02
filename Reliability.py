import os
import pandas as pd
from scipy import stats
import random

os.chdir('C:/Users/uf11/Desktop/OutFin') # NOTE: change the path to where OutFin dataset resides
os.chdir('Code/temporal_data')

# indices that will be used to create all unordered day pairs (e.g., {day1, day2} = {x1[0]+1, x2[0]+1}, {day3, day1} =
# {x1[3]+1, x2[3]+1}):
x1 = [0,1,2]
x2 = [1,2,0]

def Reliability_WiFi(phone):
    DF = {} # creating a dictionary of dataframes where each will hold the WiFi data for a given day
    L = {} # creating a dictionary of lists where each will hold the RSS value of APs observed over the three days
    for i in range(3): # iterate over the three days
        L[i] = [] # initialize the list for a given day
        DF[i] = pd.read_csv(phone+'_day' + str(i+1) + '_WiFi.csv') # read the WiFi data for a given day
        df_temp = DF[i].iloc[:, 8:] # select only columns of interest i.e., RSS readings
        DF[i]['RSS'] = df_temp.mean(axis=1) # calculate the mean RSS value from APs
    # merge the dataframes of all days to obtain the APs that were observed over the three days:
    common_APs = pd.merge(DF[0]['BSSID'], DF[1]['BSSID'], how='inner')
    common_APs = pd.merge(DF[2]['BSSID'], common_APs['BSSID'], how='inner')
    randomAPs = random.sample(range(0, len(common_APs)), 50) # randomly select 50 of those APs
    # iterate over the three days to store the RSS value of the randomly selected APs to their corresponding lists:
    for i in range(3):
        for j in range(len(randomAPs)):
            temp_RSS = DF[i].loc[DF[i]['BSSID'] == common_APs.BSSID[randomAPs[j]]].RSS
            temp_RSS = temp_RSS.values
            temp_RSS = temp_RSS.mean()
            L[i].append(temp_RSS.tolist())
    # calculate the correlation between the RSS values of pairs of days:
    print('===================',phone,'(WiFi) ===================')
    for i, j in zip(x1, x2):
        rho, p_value = stats.spearmanr(L[i], L[j])
        print('Day(',i+1,',',j+1,'):')
        print('Spearman rho:', rho,'(p value:', p_value,')')
        tau, p_value = stats.kendalltau(L[i], L[j])
        print('Kendall tau: ', tau,'(p value:', p_value,')')
    print('====================================================')


def Reliability_Bluetooth(phone):
    DF = {} # creating a dictionary of dataframes where each will hold the Bluetooth data for a given day
    L = {} # creating a dictionary of lists where each will hold the RSS value of Bluetooth devices observed over the
    # three days
    for i in range(3): # iterate over the three days
        L[i] = [] # initialize the list for a given day
        DF[i] = pd.read_csv(phone+'_day' + str(i+1) + '_Bluetooth.csv', usecols=['MAC_address', 'RSS']) # read the
        # Bluetooth data for a given day
        DF[i] = DF[i].groupby(['MAC_address'], as_index=False).mean() # calculate the mean RSS value from Bluetooth
        # devices
    # merge the dataframes of all days to obtain the Bluetooth devices that were observed over the three days:
    common_devices = pd.merge(DF[0]['MAC_address'], DF[1]['MAC_address'], how='inner')
    common_devices = pd.merge(DF[2]['MAC_address'], common_devices['MAC_address'], how='inner')
    random_devices = random.sample(range(0, len(common_devices)), 15) # randomly select 15 of those Bluetooth devices
    # iterate over the three days to store the RSS value of the randomly selected Bluetooth devices to their
    # corresponding lists:
    for i in range(3):
        for j in range(len(random_devices)):
            temp_RSS = DF[i].loc[DF[i]['MAC_address'] == common_devices.MAC_address[random_devices[j]]].RSS
            temp_RSS = temp_RSS.values
            temp_RSS = temp_RSS.mean()
            L[i].append(temp_RSS.tolist())
    # calculate the correlation between the RSS values of pairs of days:
    print('===================',phone,'(Bluetooth) ===================')
    for i, j in zip(x1, x2):
        rho, p_value = stats.spearmanr(L[i], L[j])
        print('Day(',i+1,',',j+1,'):')
        print('Spearman rho:', rho,'(p value:', p_value,')')
        tau, p_value = stats.kendalltau(L[i], L[j])
        print('Kendall tau: ', tau,'(p value:', p_value,')')
    print('====================================================')


def Reliability_Cellular(phone):
    df_all = pd.DataFrame(columns=['UMTS_neighbors','LTE_neighbors','RSRP_strongest','Frequency','EARFCN','RSRP',
                                   'RSRQ']) # initialize a dataframe that will hold the averaged cellular data (i.e.,
    # one row for each day)
    for i in range(3): # iterate over the three days
        df = pd.read_csv(phone+'_day'+str(i+1)+'_Cellular.csv') # read the cellular data for a given day
        df = df[['UMTS_neighbors','LTE_neighbors','RSRP_strongest','Frequency','EARFCN','RSRP','RSRQ']] # select only
        # columns of interest
        # calculate the mean and store the result in a row of the df_all dataframe:
        df.loc['day'+str(i+1)+'mean'] = df.mean()
        df = df.loc[['day'+str(i+1)+'mean'],:]
        df_all = df_all.append(df)
    # calculate the correlation between the columns of interest of pairs of days:
    print('================', phone, '(Cellular) ================')
    for i, j in zip(x1, x2):
        rho, p_value = stats.spearmanr(df_all.iloc[i, :], df_all.iloc[j, :])
        print('Day(',i+1,',',j+1,'):')
        print('Spearman rho:', rho,'(p value:', p_value,')')
        tau, p_value = stats.kendalltau(df_all.iloc[i, :], df_all.iloc[j, :])
        print('Kendall tau: ', tau, '(p value:', p_value, ')')
    print('===================================================')


def Reliability_Sensors(phone):
    df_all = pd.DataFrame(columns=['ax', 'ay', 'az', 'wx', 'wy', 'wz', 'Bx', 'By', 'Bz', 'gFx', 'gFy', 'gFz', 'Yaw',
                                    'Pitch', 'Roll', 'Pressure', 'Illuminance']) # initialize a dataframe that will
    # hold the averaged sensors data (i.e., one row for each day)
    for i in range(3): # iterate over the three days
        df = pd.read_csv(phone+'_day'+str(i+1)+'_Sensors.csv') # read the sensors data for a given day
        df = df[['ax', 'ay', 'az', 'wx', 'wy', 'wz', 'Bx', 'By', 'Bz', 'gFx', 'gFy', 'gFz', 'Yaw', 'Pitch', 'Roll',
                 'Pressure', 'Illuminance']] # select only columns of interest
        # calculate the mean and store the result in a row of the df_all dataframe:
        df.loc['day'+str(i+1)+'mean'] = df.mean()
        df = df.loc[['day'+str(i+1)+'mean'],:]
        df_all = df_all.append(df)
    # calculate the correlation between the columns of interest of pairs of days:
    print('================', phone, '(Sensors) ================')
    for i, j in zip(x1, x2):
        rho, p_value = stats.spearmanr(df_all.iloc[i, :], df_all.iloc[j, :])
        print('Day(',i+1,',',j+1,'):')
        print('Spearman rho:', rho,'(p value:', p_value,')')
        tau, p_value = stats.kendalltau(df_all.iloc[i, :], df_all.iloc[j, :])
        print('Kendall tau: ', tau, '(p value:', p_value, ')')
    print('===================================================')


Phone1 = 'Phone1'
Phone2 = 'Phone2'

# uncomment any of the functions below to activate

# Reliability_WiFi(Phone2)
# Reliability_Bluetooth(Phone2)
# Reliability_Cellular(Phone2)
# Reliability_Sensors(Phone2)