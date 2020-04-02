import os
import pandas as pd
import statistics
import numpy as np

os.chdir('C:/Users/uf11/Desktop/OutFin') # NOTE: change the path to where OutFin dataset resides
os.chdir('Measurements')

# RPs visited for a given day:
day1 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
day2 = [21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42]
day3 = [43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60]
day4 = [61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77]
day5 = [78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99]
day6 = [100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122]

def Stats_WiFi(phone):
    df_all = pd.DataFrame() # initializing a dataframe to aggregate all WiFi .csv files for a given phone
    l1 = [] # initializing a list to hold the number of detected SSIDs for each RP
    l2 = [] # initializing a list to hold the number of detected BSSIDs for each RP
    for i in range(122): # iterate over RPs
        df_temp = pd.read_csv(phone+'_WiFi_' + str(i + 1) + '.csv') # read the WiFi .csv file
        temp = df_temp.iloc[:, 8:] # select all RSS readings of the APs
        df_temp['RSS'] = temp.mean(axis=1) # calculate the mean RSS value of the APs
        df_temp = df_temp[['SSID', 'BSSID', 'RSS']] # leave only the columns of interest
        SSIDs = df_temp.SSID.unique() # obtain unique SSIDs
        l1.append(len(SSIDs)) # get number of unique SSIDs and append to the l1 list
        BSSIDs = df_temp.BSSID.unique() # obtain unique BSSIDs
        l2.append(len(BSSIDs)) # get number of unique BSSIDs and append to the l2 list
        df_all = df_all.append(df_temp, ignore_index=True) # append to the df_all dataframe
    print('========= Descriptive Statistics for',phone,'(WiFi) =========')
    print('Detected SSID (min): ', min(l1))
    print('Detected SSID (max): ', max(l1))
    print('Detected SSID (mean):', statistics.mean(l1))
    print('Detected SSID (SD):  ', statistics.pstdev(l1))
    print('Detected BSSID (min): ', min(l2))
    print('Detected BSSID (max): ', max(l2))
    print('Detected BSSID (mean):', statistics.mean(l2))
    print('Detected BSSID (SD):  ', statistics.pstdev(l2))
    print('RSS (min): ', df_all.RSS.min(skipna=True))
    print('RSS (max): ',df_all.RSS.max(skipna=True))
    print('RSS (mean):',df_all.RSS.mean(skipna=True))
    print('RSS (SD):  ',df_all.RSS.std(skipna=True))
    print('==========================================================')


def Stats_Bluetooth(phone):
    df_all = pd.DataFrame() # initializing a dataframe to aggregate all Bluetooth .csv files for a given phone
    l = [] # initializing a list to hold the number of detected Bluetooth devices for each RP
    for i in range(122): # iterate over RPs
        df_temp = pd.read_csv(phone+'_Bluetooth_' + str(i + 1) + '.csv') # read the Bluetooth .csv file
        df_temp = df_temp[['MAC_address', 'RSS']] # leave only the columns of interest
        MAC_addresses = df_temp.MAC_address.unique() # obtain unique MAC addresses
        l.append(len(MAC_addresses)) # get number of unique MAC addresses and append to the l list
        df_temp = df_temp.groupby(['MAC_address'], as_index=False).mean() # calculate the mean RSS value of the MAC
        # addresses
        df_all = df_all.append(df_temp, ignore_index=True) # append to the df_all dataframe
    print('========= Descriptive Statistics for',phone,'(Bluetooth) =========')
    print('Detected MAC addresses (min): ', min(l))
    print('Detected MAC addresses (max): ', max(l))
    print('Detected MAC addresses (mean):', statistics.mean(l))
    print('Detected MAC addresses (SD):  ', statistics.pstdev(l))
    print('RSS (min): ', df_all.RSS.min(skipna=True))
    print('RSS (max): ',df_all.RSS.max(skipna=True))
    print('RSS (mean):',df_all.RSS.mean(skipna=True))
    print('RSS (SD):  ',df_all.RSS.std(skipna=True))
    print('==========================================================')


def Stats_Cellular(phone):
    df_all = pd.DataFrame() # initializing a dataframe to aggregate all cellular .csv files for a given phone
    l = [] # initializing a list to hold the number of detected ECIs devices for each RP
    for i in range(122): # iterate over RPs
        df_temp = pd.read_csv(phone+'_Cellular_' + str(i + 1) + '.csv') # read the cellular .csv file
        df_temp = df_temp[['LTE_neighbors', 'RSRP_strongest', 'ECI', 'RSRP', 'RSRQ']] # leave only the columns of
        # interest
        ECIs = df_temp.ECI.unique() # obtain unique ECIs
        l.append(len(ECIs)) # get number of unique ECIs and append to the l list
        df_all = df_all.append(df_temp, ignore_index=True) # append to the df_all dataframe
    print('========= Descriptive Statistics for',phone,'(Cellular) =========')
    print('Detected ECIs (min): ', min(l))
    print('Detected ECIs (max): ', max(l))
    print('Detected ECIs (mean):', statistics.mean(l))
    print('Detected ECIs (SD):  ', statistics.pstdev(l))
    print('LTE neighbors (min): ', df_all.LTE_neighbors.min(skipna=True))
    print('LTE neighbors (max): ', df_all.LTE_neighbors.max(skipna=True))
    print('LTE neighbors (mean):', df_all.LTE_neighbors.mean(skipna=True))
    print('LTE neighbors (SD):  ', df_all.LTE_neighbors.std(skipna=True))
    print('RSRP strongest (min): ', df_all.RSRP_strongest.min(skipna=True))
    print('RSRP strongest (max): ', df_all.RSRP_strongest.max(skipna=True))
    print('RSRP strongest (mean):', df_all.RSRP_strongest.mean(skipna=True))
    print('RSRP strongest (SD):  ', df_all.RSRP_strongest.std(skipna=True))
    print('RSRP (min): ', df_all.RSRP.min(skipna=True))
    print('RSRP (max): ', df_all.RSRP.max(skipna=True))
    print('RSRP (mean):', df_all.RSRP.mean(skipna=True))
    print('RSRP (SD):  ', df_all.RSRP.std(skipna=True))
    print('RSRQ (min): ', df_all.RSRQ.min(skipna=True))
    print('RSRQ (max): ', df_all.RSRQ.max(skipna=True))
    print('RSRQ (mean):', df_all.RSRQ.mean(skipna=True))
    print('RSRQ (SD):  ', df_all.RSRQ.std(skipna=True))
    print('==========================================================')


def Stats_Sensors(phone):
    df_all = pd.DataFrame() # initializing a dataframe to aggregate all sensors .csv files for a given phone
    if phone == 'Phone1':
        offset = [0.9908,-1.8639,0.4391,-0.8586,1.7032,-0.4984,-1.1204,-0.3366,1.1165,-2.0947,23.6993,-13.5657,-0.3352,
                  8.5964,-1.2695,-1.5723,2.1944,0.7699] # calibration offsets as obtained by Calibration.py (three
        # components for each day (offset_x, offset_y, offset_z))
        scale = [0.9951,1.0562,0.9538,1.0092,0.9945,0.9963,1.0227,0.9899,0.9880,1.0003,1.0129,0.9870,1.0882,0.8722,
                 1.0698,1.0209,0.9821,0.9976] # calibration scales as obtained by Calibration.py (three components for
        # each day (scale_x, scale_y, scale_z))
    elif phone == 'Phone2':
        offset = [0.0700,0.6768,0.0131,1.0061,0.1598,1.2743,2.3444,0.3117,-0.5986,0.2895,-0.8330,-0.2518,-0.7629,0.6813,
                  0.1127,0.6256,0.2184,-0.2729]
        scale = [1.0428,0.9899,0.9700,1.0219,0.9891,0.9896,1.0904,0.9693,0.9512,1.0295,0.9684,1.0039,1.0756,0.9763,
                 0.9559,1.0406,0.9838,0.9778]
    for i in range(122): # iterate over RPs
        df_temp = pd.read_csv(phone+'_Sensors_' + str(i + 1) + '.csv') # read the sensors .csv file
        df_temp = df_temp[['Bx','By','Bz','Pressure','Illuminance']] # leave only the columns of interest
        # calibrate magnetic filed readings:
        if i + 1 in day1:
            df_temp['Bx'] = (df_temp['Bx'] - offset[0]) * scale[0]
            df_temp['By'] = (df_temp['By'] - offset[1]) * scale[1]
            df_temp['Bz'] = (df_temp['Bz'] - offset[2]) * scale[2]
        elif i + 1 in day2:
            df_temp['Bx'] = (df_temp['Bx'] - offset[3]) * scale[3]
            df_temp['By'] = (df_temp['By'] - offset[4]) * scale[4]
            df_temp['Bz'] = (df_temp['Bz'] - offset[5]) * scale[5]
        elif i + 1 in day3:
            df_temp['Bx'] = (df_temp['Bx'] - offset[6]) * scale[6]
            df_temp['By'] = (df_temp['By'] - offset[7]) * scale[7]
            df_temp['Bz'] = (df_temp['Bz'] - offset[8]) * scale[8]
        elif i + 1 in day4:
            df_temp['Bx'] = (df_temp['Bx'] - offset[9]) * scale[9]
            df_temp['By'] = (df_temp['By'] - offset[10]) * scale[10]
            df_temp['Bz'] = (df_temp['Bz'] - offset[11]) * scale[11]
        elif i + 1 in day5:
            df_temp['Bx'] = (df_temp['Bx'] - offset[12]) * scale[12]
            df_temp['By'] = (df_temp['By'] - offset[13]) * scale[13]
            df_temp['Bz'] = (df_temp['Bz'] - offset[14]) * scale[14]
        elif i + 1 in day6:
            df_temp['Bx'] = (df_temp['Bx'] - offset[15]) * scale[15]
            df_temp['By'] = (df_temp['By'] - offset[16]) * scale[16]
            df_temp['Bz'] = (df_temp['Bz'] - offset[17]) * scale[17]
        df_all = df_all.append(df_temp, ignore_index=True) # append to the df_all dataframe
    df_all['Magnitude'] = (df_all['Bx']**2 + df_temp['By']**2 + df_temp['Bz']**2).apply(np.sqrt) # calculate magnitude
    # of magnetic filed
    print('========= Descriptive Statistics for' ,phone, '(Sensors) =========')
    print('Magnitude of magnetic filed (min): ', df_all.Magnitude.min(skipna=True))
    print('Magnitude of magnetic filed (max): ', df_all.Magnitude.max(skipna=True))
    print('Magnitude of magnetic filed (mean):', df_all.Magnitude.mean(skipna=True))
    print('Magnitude of magnetic filed (SD):  ', df_all.Magnitude.std(skipna=True))
    print('Pressure (min): ', df_all.Pressure.min(skipna=True))
    print('Pressure (max): ', df_all.Pressure.max(skipna=True))
    print('Pressure (mean):', df_all.Pressure.mean(skipna=True))
    print('Pressure (SD):  ', df_all.Pressure.std(skipna=True))
    print('Illuminance (min): ', df_all.Illuminance.min(skipna=True)*10**-6)
    print('Illuminance (max): ', df_all.Illuminance.max(skipna=True)*10**-6)
    print('Illuminance (mean):', df_all.Illuminance.mean(skipna=True)*10**-6)
    print('Illuminance (SD):  ', df_all.Illuminance.std(skipna=True)*10**-6)
    print('==========================================================')



Phone1 = 'Phone1'
Phone2 = 'Phone2'

# uncomment any of the functions below to activate

# Stats_WiFi(Phone2)
# Stats_Bluetooth(Phone2)
# Stats_Cellular(Phone2)
# Stats_Sensors(Phone2)