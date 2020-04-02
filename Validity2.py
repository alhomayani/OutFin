import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib as mpl
import numpy as np
import random

os.chdir('C:/Users/uf11/Desktop/OutFin')  # NOTE: change the path to where OutFin dataset resides
os.chdir('Measurements')

def Validity2_WiFi():
    # ---------------------------------- Setting up the layout of the plot ---------------------------------- #
    mpl.rc('font', family='serif')
    plt.rc('xtick', labelsize=7)
    plt.rc('ytick', labelsize=7)
    plt.rc('legend', fontsize=5)
    plt.rc('axes', titlesize=7)
    plt.rc('figure', titlesize=9)
    fig, axs = plt.subplots(2, 4, figsize=(10,5))
    fig.suptitle('WiFi')
    axs[0,0].set_ylabel('Phone1', rotation=0, fontsize=7, labelpad=20)
    axs[1,0].set_ylabel('Phone2', rotation=0, fontsize=7, labelpad=20)
    # ------------------------------------------------------------------------------------------------------- #
    df_all = pd.DataFrame() # initializing a dataframe to aggregate all WiFi .csv files for a given phone
    scaler_list = [] # initializing a list to hold the Min Max estimators (i.e., one for each phone)
    for i in range(2): # iterate over phones
        for j in range(122): # iterate over RPs
            df_temp = pd.read_csv('Phone' + str(i + 1) + '_WiFi_' + str(j + 1) + '.csv') # read the WiFi .csv file
            df_all = df_all.append(df_temp, ignore_index=True) # append it to the df_all dataframe
        df_all = df_all[['SSID', 'BSSID', 'Channel', 'Width', 'Center_Frequency_0', 'Center_Frequency_1', 'Band',
                     'Capabilities', 'RSS_0', 'RSS_1', 'RSS_2', 'RSS_3', 'RSS_4', 'RSS_5', 'RSS_6', 'RSS_7', 'RSS_8',
                     'RSS_9', 'RSS_10']] # rearrange the columns for processing
        df_temp = df_all.iloc[:, 8:] # select all RSS readings of the APs
        df_all['RSS'] = df_temp.mean(axis = 1) # calculate the mean RSS value of the APs
        df_all = df_all[['BSSID', 'Channel', 'Width', 'Center_Frequency_0', 'Center_Frequency_1', 'Band', 'RSS']] # leave
        # only the columns of interest
        scaler = MinMaxScaler(feature_range=(0, 1)) # use Min Max estimator to scale features between 0 and 1
        scaler.fit(df_all[['Channel', 'Width', 'Center_Frequency_0', 'Center_Frequency_1', 'Band', 'RSS']]) # fit the
        # estimator to the columns of interest
        scaler_list.append(scaler)  # append the estimator to the list to be used for the respective phone
    randomRP = random.sample(range(1, 122), 4) # randomly select 4 RPs
    for i in range(len(randomRP)): # iterate over the selected RPs
        df1 = pd.read_csv('Phone1_WiFi_' + str(randomRP[i]) + '.csv') # read the WiFi .csv file corresponding to the RP for
        # Phone 1
        df_temp = df1.iloc[:, 8:] # select all RSS readings of the APs
        df1['RSS'] = df_temp.mean(axis = 1) # calculate the mean RSS value of the APs
        df1 = df1[['BSSID', 'Channel', 'Width', 'Center_Frequency_0', 'Center_Frequency_1', 'Band', 'RSS']] # leave only
        #  the columns of interest
        df1[['Channel', 'Width', 'Center_Frequency_0', 'Center_Frequency_1', 'Band', 'RSS']] = \
            scaler_list[0].transform(df1[['Channel', 'Width', 'Center_Frequency_0', 'Center_Frequency_1', 'Band', 'RSS']]) #
        #  scale the values according to the estimator of Phone 1
        df2 = pd.read_csv('Phone2_WiFi_' + str(randomRP[i]) + '.csv') # read the WiFi .csv file corresponding to the RP for
        # Phone 2
        df_temp = df2.iloc[:, 8:] # select all RSS readings of the APs
        df2['RSS'] = df_temp.mean(axis = 1) # calculate the mean RSS value of the APs
        df2 = df2[['BSSID', 'Channel', 'Width', 'Center_Frequency_0', 'Center_Frequency_1', 'Band', 'RSS']] # leave only
        #  the columns of interest
        df2[['Channel', 'Width', 'Center_Frequency_0', 'Center_Frequency_1', 'Band', 'RSS']] = \
            scaler_list[1].transform(df2[['Channel', 'Width', 'Center_Frequency_0', 'Center_Frequency_1', 'Band', 'RSS']]) #
        #   scale the values according to the estimator of Phone 2
        common_APs = pd.merge(df1['BSSID'], df2['BSSID'], how='inner') # merge the dataframes
        common_APs = common_APs.BSSID.unique()  # obtain intersecting BSSIDs
        randomAPs = random.sample(range(0, len(common_APs)), 10) # randomly select ten intersecting BSSIDs
        for j in range(len(df1)): # iterate over all rows of the dataframe of Phone 1
            for k in range(len(randomAPs)): # iterate over all randomly selected BSSIDs
                if df1.BSSID[j] == common_APs[k]: # if a match is found assign a unique color to it and
                    # plot the data associated with it (i.e., Channel, Width, Center_Frequency_0, Center_Frequency_1,
                    # Band, and RSS)
                    if df1.BSSID[j] == common_APs[0]:
                            pcolor = 'r'
                            lstyle = (0, (5, 19))
                    elif df1.BSSID[j] == common_APs[1]:
                            pcolor = 'b'
                            lstyle = (0, (5, 17))
                    elif df1.BSSID[j] == common_APs[2]:
                            pcolor = 'g'
                            lstyle = (0, (5, 15))
                    elif df1.BSSID[j] == common_APs[3]:
                            pcolor = 'c'
                            lstyle = (0, (5, 13))
                    elif df1.BSSID[j] == common_APs[4]:
                            pcolor = 'm'
                            lstyle = (0, (5, 11))
                    elif df1.BSSID[j] == common_APs[5]:
                            pcolor = 'y'
                            lstyle = (0, (5, 9))
                    elif df1.BSSID[j] == common_APs[6]:
                            pcolor = 'k'
                            lstyle = (0, (5, 7))
                    elif df1.BSSID[j] == common_APs[7]:
                            pcolor = 'tab:brown'
                            lstyle = (0, (5, 5))
                    elif df1.BSSID[j] == common_APs[8]:
                            pcolor = 'tab:orange'
                            lstyle = (0, (5, 3))
                    elif df1.BSSID[j] == common_APs[9]:
                            pcolor = 'tab:gray'
                            lstyle = (0, (5, 1))
                    dataRow = df1.iloc[j, 1:]
                    dataRow.plot(color=pcolor, ax=axs[0, i], grid=True, alpha=1, legend=True, linestyle=lstyle, lw=1)
                    legend_elements = [Line2D([0], [0], color='r', lw=1, label=str(common_APs[0])),
                                       Line2D([0], [0], color='b', lw=1, label=str(common_APs[1])),
                                       Line2D([0], [0], color='g', lw=1, label=str(common_APs[2])),
                                       Line2D([0], [0], color='c', lw=1, label=str(common_APs[3])),
                                       Line2D([0], [0], color='m', lw=1, label=str(common_APs[4])),
                                       Line2D([0], [0], color='y', lw=1, label=str(common_APs[5])),
                                       Line2D([0], [0], color='k', lw=1, label=str(common_APs[6])),
                                       Line2D([0], [0], color='tab:brown', lw=1, label=str(common_APs[7])),
                                       Line2D([0], [0], color='tab:orange', lw=1, label=str(common_APs[8])),
                                       Line2D([0], [0], color='tab:gray', lw=1, label=str(common_APs[9]))]
                    axs[0, i].legend(handles=legend_elements, title='BSSID:', title_fontsize=5)
                    axs[0, i].set_title('RP_' + str(randomRP[i]))
        for j in range(len(df2)): # iterate over all rows of the dataframe of Phone 2
            for k in range(len(randomAPs)): # iterate over all randomly selected BSSIDs
                if df2.BSSID[j] == common_APs[k]: # if a match is found assign a unique color to it and
                    # plot the data associated with it (i.e., Channel, Width, Center_Frequency_0, Center_Frequency_1,
                    # Band, and RSS)
                    if df2.BSSID[j] == common_APs[0]:
                            pcolor = 'r'
                            lstyle = (0, (5, 19))
                    elif df2.BSSID[j] == common_APs[1]:
                            pcolor = 'b'
                            lstyle = (0, (5, 17))
                    elif df2.BSSID[j] == common_APs[2]:
                            pcolor = 'g'
                            lstyle = (0, (5, 15))
                    elif df2.BSSID[j] == common_APs[3]:
                            pcolor = 'c'
                            lstyle = (0, (5, 13))
                    elif df2.BSSID[j] == common_APs[4]:
                            pcolor = 'm'
                            lstyle = (0, (5, 11))
                    elif df2.BSSID[j] == common_APs[5]:
                            pcolor = 'y'
                            lstyle = (0, (5, 9))
                    elif df2.BSSID[j] == common_APs[6]:
                            pcolor = 'k'
                            lstyle = (0, (5, 7))
                    elif df2.BSSID[j] == common_APs[7]:
                            pcolor = 'tab:brown'
                            lstyle = (0, (5, 5))
                    elif df2.BSSID[j] == common_APs[8]:
                            pcolor = 'tab:orange'
                            lstyle = (0, (5, 3))
                    elif df2.BSSID[j] == common_APs[9]:
                            pcolor = 'tab:gray'
                            lstyle = (0, (5, 1))
                    dataRow = df2.iloc[j, 1:]
                    dataRow.plot(color=pcolor, ax=axs[1, i], grid=True, alpha=1, linestyle=lstyle, lw=1)
    for ax in fig.axes:
        plt.sca(ax)
        plt.xticks(rotation=20, ha='right')
    x = np.arange(6)
    for i in range(2):
        for j in range(4):
            axs[i, j].set_xticks(x)
            axs[0, j].set_xticklabels(('', '', '', '', '', ''))
            axs[1, j].set_xticklabels(('Channel', 'Width', 'Center_Frequency_0', 'Center_Frequency_1', 'Band', 'RSS'))
    plt.tight_layout()
    plt.show()


def Validity2_Bluetooth():
    # ---------------------- Setting up the layout of the plot ---------------------- #
    mpl.rc('font', family='serif')
    plt.rc('xtick', labelsize=7)
    plt.rc('ytick', labelsize=7)
    plt.rc('legend', fontsize=5)
    plt.rc('axes', titlesize=7)
    plt.rc('figure', titlesize=9)
    fig, axs = plt.subplots(2, 4, figsize=(10, 5))
    fig.suptitle('Bluetooth')
    axs[0, 0].set_ylabel('Phone1', rotation=0, fontsize=7, labelpad=20)
    axs[1, 0].set_ylabel('Phone2', rotation=0, fontsize=7, labelpad=20)
    y = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    x = np.arange(2)
    twinax = {}
    for i in range(2):
        for j in range(4):
            axs[i, j].set_yticks(y)
            axs[i, j].set_ylim(0, 1)
            axs[i, j].set_xticks(x)
            axs[1, j].set_xticklabels(('RSS', 'Protocol'))
            twinax[i, j] = axs[i, j].twinx()
            twinax[i, j].set_yticks([0, 0.5, 1])
            twinax[i, j].set_yticklabels(('', '', ''))
        twinax[i, 3].set_yticklabels(('CLASSIC', 'DUAL', 'BLE'))
    # ------------------------------------------------------------------------------- #
    df_all = pd.DataFrame()  # initializing a dataframe to aggregate all Bluetooth .csv files for a given phone
    scaler_list = []  # initializing a list to hold the Min Max estimators (i.e., one for each phone)
    for i in range(2):  # iterate over phones
        for j in range(122):  # iterate over PRs
            df_temp = pd.read_csv(
                'Phone' + str(i + 1) + '_Bluetooth_' + str(j + 1) + '.csv')  # read the Bluetooth .csv file
            df_all = df_all.append(df_temp, ignore_index=True)  # append it to the df_all dataframe
        df_all = df_all[['MAC_address', 'RSS', 'Protocol']]  # after aggregation, leave only the columns of interest
        df_all.replace({'Protocol': {'CLASSIC': 0, 'DUAL': 1, 'BLE': 2}}, inplace=True)  # replace string with integer
        # values for processing
        scaler = MinMaxScaler(feature_range=(0, 1))  # use Min Max estimator to scale features between 0 and 1
        scaler.fit(df_all[['RSS', 'Protocol']])  # fit the estimator to the columns of interest
        scaler_list.append(scaler)  # append the estimator to the list to be used for the respective phone
    randomRP = random.sample(range(1, 122), 4)  # randomly select 4 RPs
    for i in range(len(randomRP)):  # iterate over the selected RPs
        df1 = pd.read_csv(
            'Phone1_Bluetooth_' + str(randomRP[i]) + '.csv')  # read the Bluetooth .csv file corresponding to
        # the RP for Phone 1
        df1 = df1[['MAC_address', 'RSS', 'Protocol']]  # leave only the columns of interest
        df1.replace({'Protocol': {'CLASSIC': 0, 'DUAL': 1, 'BLE': 2}},
                    inplace=True)  # replace string with integer values
        # for processing
        df1[['RSS', 'Protocol']] = scaler_list[0].transform(
            df1[['RSS', 'Protocol']])  # scale the values according to the
        # estimator of Phone 1
        df2 = pd.read_csv(
            'Phone2_Bluetooth_' + str(randomRP[i]) + '.csv')  # read the Bluetooth .csv file corresponding to the
        # RP for Phone 2
        df2 = df2[['MAC_address', 'RSS', 'Protocol']]  # leave only the columns of interest
        df2.replace({'Protocol': {'CLASSIC': 0, 'DUAL': 1, 'BLE': 2}},
                    inplace=True)  # replace string with integer values
        # for processing
        df2[['RSS', 'Protocol']] = scaler_list[1].transform(
            df2[['RSS', 'Protocol']])  # scale the values according to the
        # estimator of Phone 2
        common_devices = pd.merge(df1['MAC_address'], df2['MAC_address'], how='inner')  # merge the dataframes
        common_devices = common_devices.MAC_address.unique() # obtain intersecting MAC addresses
        if len(common_devices) < 10: # print an error message in case intersecting MAC addresses is less than ten
            print('common devices for RP',randomRP[i],'is less than ten, please execute again!')
        random_devices = random.sample(range(0, len(common_devices)),
                                       10)  # randomly select ten intersecting MAC addresses
        for j in range(len(df1)):  # iterate over all rows of the dataframe of Phone 1
            for k in range(len(random_devices)):  # iterate over all randomly selected MAC addresses
                if df1.MAC_address[j] == common_devices[k]:  # if a match is found assign a
                    # unique color to it and plot the data associated with it (i.e., RSS and Protocol type)
                    if df1.MAC_address[j] == common_devices[0]:
                        pcolor = 'r'
                        lstyle = (0, (5, 19))
                    elif df1.MAC_address[j] == common_devices[1]:
                        pcolor = 'b'
                        lstyle = (0, (5, 17))
                    elif df1.MAC_address[j] == common_devices[2]:
                        pcolor = 'g'
                        lstyle = (0, (5, 15))
                    elif df1.MAC_address[j] == common_devices[3]:
                        pcolor = 'c'
                        lstyle = (0, (5, 13))
                    elif df1.MAC_address[j] == common_devices[4]:
                        pcolor = 'm'
                        lstyle = (0, (5, 11))
                    elif df1.MAC_address[j] == common_devices[5]:
                        pcolor = 'y'
                        lstyle = (0, (5, 9))
                    elif df1.MAC_address[j] == common_devices[6]:
                        pcolor = 'k'
                        lstyle = (0, (5, 7))
                    elif df1.MAC_address[j] == common_devices[7]:
                        pcolor = 'tab:brown'
                        lstyle = (0, (5, 5))
                    elif df1.MAC_address[j] == common_devices[8]:
                        pcolor = 'tab:orange'
                        lstyle = (0, (5, 3))
                    elif df1.MAC_address[j] == common_devices[9]:
                        pcolor = 'tab:gray'
                        lstyle = (0, (5, 1))
                    dataRow = df1.iloc[j, 1:]
                    dataRow.plot(color=pcolor, linestyle=lstyle, ax=axs[0, i], grid=True, alpha=1, legend=True, lw=1)
                    legend_elements = [
                        Line2D([0], [0], color='r', lw=1, label=str(common_devices[0])),
                        Line2D([0], [0], color='b', lw=1, label=str(common_devices[1])),
                        Line2D([0], [0], color='g', lw=1, label=str(common_devices[2])),
                        Line2D([0], [0], color='c', lw=1, label=str(common_devices[3])),
                        Line2D([0], [0], color='m', lw=1, label=str(common_devices[4])),
                        Line2D([0], [0], color='y', lw=1, label=str(common_devices[5])),
                        Line2D([0], [0], color='k', lw=1, label=str(common_devices[6])),
                        Line2D([0], [0], color='tab:brown', lw=1, label=str(common_devices[7])),
                        Line2D([0], [0], color='tab:orange', lw=1,
                               label=str(common_devices[8])),
                        Line2D([0], [0], color='tab:gray', lw=1, label=str(common_devices[9]))]
                    axs[0, i].legend(handles=legend_elements, title='MAC_address:', title_fontsize=5)
                    axs[0, i].set_title('RP_' + str(randomRP[i]))
        for j in range(len(df2)):  # iterate over all rows of the dataframe of Phone 2
            for k in range(len(random_devices)):  # iterate over all randomly selected MAC addresses
                if df2.MAC_address[j] == common_devices[k]:  # if a match is found assign a
                    # unique color to it and plot the data associated with it (i.e., RSS and Protocol type)
                    if df2.MAC_address[j] == common_devices[0]:
                        pcolor = 'r'
                        lstyle = (0, (5, 19))
                    elif df2.MAC_address[j] == common_devices[1]:
                        pcolor = 'b'
                        lstyle = (0, (5, 17))
                    elif df2.MAC_address[j] == common_devices[2]:
                        pcolor = 'g'
                        lstyle = (0, (5, 15))
                    elif df2.MAC_address[j] == common_devices[3]:
                        pcolor = 'c'
                        lstyle = (0, (5, 13))
                    elif df2.MAC_address[j] == common_devices[4]:
                        pcolor = 'm'
                        lstyle = (0, (5, 11))
                    elif df2.MAC_address[j] == common_devices[5]:
                        pcolor = 'y'
                        lstyle = (0, (5, 9))
                    elif df2.MAC_address[j] == common_devices[6]:
                        pcolor = 'k'
                        lstyle = (0, (5, 7))
                    elif df2.MAC_address[j] == common_devices[7]:
                        pcolor = 'tab:brown'
                        lstyle = (0, (5, 5))
                    elif df2.MAC_address[j] == common_devices[8]:
                        pcolor = 'tab:orange'
                        lstyle = (0, (5, 3))
                    elif df2.MAC_address[j] == common_devices[9]:
                        pcolor = 'tab:gray'
                        lstyle = (0, (5, 1))
                    dataRow = df2.iloc[j, 1:]
                    dataRow.plot(color=pcolor, linestyle=lstyle, ax=axs[1, i], grid=True, alpha=1, lw=1)
    for ax in fig.axes:
        plt.sca(ax)
        plt.xticks(rotation=20, ha='right')
    for ax in axs.flat:
        ax.label_outer()
    plt.tight_layout()
    plt.show()


def Validity2_Cellular():
    # ---------------------------------- Setting up the layout of the plot ---------------------------------- #
    mpl.rc('font', family='serif')
    plt.rc('xtick', labelsize=7)
    plt.rc('ytick', labelsize=7)
    plt.rc('legend', fontsize=7)
    plt.rc('axes', titlesize=7)
    plt.rc('figure', titlesize=9)
    fig, axs = plt.subplots(2, 4, figsize=(10, 5))
    fig.suptitle('Cellular')
    axs[0, 0].set_ylabel('Phone1', rotation=0, fontsize=7, labelpad=20)
    axs[1, 0].set_ylabel('Phone2', rotation=0, fontsize=7, labelpad=20)
    # ------------------------------------------------------------------------------------------------------- #
    df_all = pd.DataFrame()  # initializing a dataframe to aggregate all cellular .csv files for a given phone
    scaler_list = []  # initializing a list to hold the Min Max estimators (i.e., one for each phone)
    for i in range(2):  # iterate over phones
        for j in range(122):  # iterate over RPs
            df_temp = pd.read_csv(
                'Phone' + str(i + 1) + '_Cellular_' + str(j + 1) + '.csv')  # read the cellular .csv file
            df_all = df_all.append(df_temp, ignore_index=True)  # append it to the df_all dataframe
        scaler = MinMaxScaler(feature_range=(0, 1))  # use Min Max estimator to scale features between 0 and 1
        scaler.fit(df_all[['LTE_neighbors', 'RSRP_strongest', 'Frequency', 'EARFCN', 'TA', 'RSRP', 'RSRQ']])  # fit the
        # estimator to the columns of interest
        scaler_list.append(scaler)  # append the estimator to the list to be used for the respective phone
    randomRP = random.sample(range(1, 122), 4)  # randomly select 4 RPs
    for i in range(len(randomRP)):  # iterate over the selected RPs
        df12 = pd.DataFrame(
            columns=['ECI', 'TA', 'RSRP', 'RSRQ', 'LTE_neighbors', 'RSRP_strongest', 'Frequency', 'EARFCN'])  #
        # initialize a dataframe to aggregate the .csv files obtained from Phone 1 and 2 for a given RP
        df1 = pd.read_csv(
            'Phone1_Cellular_' + str(randomRP[i]) + '.csv')  # read the cellular .csv file corresponding to
        # the RP  for Phone 1
        df1[['LTE_neighbors', 'RSRP_strongest', 'Frequency', 'EARFCN', 'TA', 'RSRP', 'RSRQ']] = \
            scaler_list[0].transform(
                df1[['LTE_neighbors', 'RSRP_strongest', 'Frequency', 'EARFCN', 'TA', 'RSRP', 'RSRQ']])
        # scale the values according to the estimator of Phone 1
        df1 = df1[
            ['ECI', 'TA', 'RSRP', 'RSRQ', 'LTE_neighbors', 'RSRP_strongest', 'Frequency', 'EARFCN']]  # leave only the
        # columns of interest
        df12 = df12.append(df1, ignore_index=True)  # append it to the df12 dataframe
        df2 = pd.read_csv(
            'Phone2_Cellular_' + str(randomRP[i]) + '.csv')  # read the cellular .csv file corresponding to
        # the RP for Phone 2
        df2[['LTE_neighbors', 'RSRP_strongest', 'Frequency', 'EARFCN', 'TA', 'RSRP', 'RSRQ']] = \
            scaler_list[1].transform(
                df2[['LTE_neighbors', 'RSRP_strongest', 'Frequency', 'EARFCN', 'TA', 'RSRP', 'RSRQ']])
        # scale the values according to the estimator of Phone 2
        df2 = df2[
            ['ECI', 'TA', 'RSRP', 'RSRQ', 'LTE_neighbors', 'RSRP_strongest', 'Frequency', 'EARFCN']]  # leave only the
        # columns of interest
        df12 = df12.append(df2, ignore_index=True)  # append it to the df12 dataframe
        ECIs = df12.ECI.unique()  # obtain all unique ECIs for the df12 dataframe
        for j in range(
                len(df1)):  # iterate over all rows of the dataframe of Phone 1, and if a ECI match is found assign
            # a color to it and plot it
            if df1.ECI[j] == ECIs[0]:
                pcolor = 'r'
            elif df1.ECI[j] == ECIs[1]:
                pcolor = 'b'
            elif df1.ECI[j] == ECIs[2]:
                pcolor = 'g'
            elif df1.ECI[j] == ECIs[3]:
                pcolor = 'c'
            elif df1.ECI[j] == ECIs[4]:
                pcolor = 'm'
            elif df1.ECI[j] == ECIs[5]:
                pcolor = 'y'
            dataRow = df1.iloc[j, 1:]
            dataRow.plot(color=pcolor, ax=axs[0, i], grid=True, alpha=0.1, legend=True)
            if len(ECIs) == 1:
                legend_elements = [Line2D([0], [0], color='r', lw=1, label=str(ECIs[0]))]
            elif len(ECIs) == 2:
                legend_elements = [Line2D([0], [0], color='r', lw=1, label=str(ECIs[0])),
                                   Line2D([0], [0], color='b', lw=1, label=str(ECIs[1]))]
            elif len(ECIs) == 3:
                legend_elements = [Line2D([0], [0], color='r', lw=1, label=str(ECIs[0])),
                                   Line2D([0], [0], color='b', lw=1, label=str(ECIs[1])),
                                   Line2D([0], [0], color='g', lw=1, label=str(ECIs[2]))]
            elif len(ECIs) == 4:
                legend_elements = [Line2D([0], [0], color='r', lw=1, label=str(ECIs[0])),
                                   Line2D([0], [0], color='b', lw=1, label=str(ECIs[1])),
                                   Line2D([0], [0], color='g', lw=1, label=str(ECIs[2])),
                                   Line2D([0], [0], color='c', lw=1, label=str(ECIs[3]))]
            elif len(ECIs) == 5:
                legend_elements = [Line2D([0], [0], color='r', lw=1, label=str(ECIs[0])),
                                   Line2D([0], [0], color='b', lw=1, label=str(ECIs[1])),
                                   Line2D([0], [0], color='g', lw=1, label=str(ECIs[2])),
                                   Line2D([0], [0], color='c', lw=1, label=str(ECIs[3])),
                                   Line2D([0], [0], color='m', lw=1, label=str(ECIs[4]))]
            elif len(ECIs) == 6:
                legend_elements = [Line2D([0], [0], color='r', lw=1, label=str(ECIs[0])),
                                   Line2D([0], [0], color='b', lw=1, label=str(ECIs[1])),
                                   Line2D([0], [0], color='g', lw=1, label=str(ECIs[2])),
                                   Line2D([0], [0], color='c', lw=1, label=str(ECIs[3])),
                                   Line2D([0], [0], color='m', lw=1, label=str(ECIs[4])),
                                   Line2D([0], [0], color='y', lw=1, label=str(ECIs[5]))]
            axs[0, i].legend(handles=legend_elements, title='ECI:', title_fontsize=7)
            axs[0, i].set_title('RP_' + str(randomRP[i]))
        for j in range(
                len(df2)):  # iterate over all rows of the dataframe of Phone 2, and if a ECI match is found assign
            # a color to it and plot it
            if df2.ECI[j] == ECIs[0]:
                pcolor = 'r'
            elif df2.ECI[j] == ECIs[1]:
                pcolor = 'b'
            elif df2.ECI[j] == ECIs[2]:
                pcolor = 'g'
            elif df2.ECI[j] == ECIs[3]:
                pcolor = 'c'
            elif df2.ECI[j] == ECIs[4]:
                pcolor = 'm'
            elif df2.ECI[j] == ECIs[5]:
                pcolor = 'y'
            dataRow = df2.iloc[j, 1:]
            dataRow.plot(color=pcolor, ax=axs[1, i], grid=True, alpha=0.1)
    for ax in fig.axes:
        plt.sca(ax)
        plt.xticks(rotation=20, ha='right')
    x = np.arange(7)
    for i in range(2):
        for j in range(4):
            axs[i, j].set_xticks(x)
            axs[0, j].set_xticklabels(('', '', '', '', '', '', ''))
            axs[1, j].set_xticklabels(('TA', 'RSRP', 'RSRQ', 'LTE_neighbors', 'RSRP_strongest', 'Frequency', 'EARFCN'))
    plt.tight_layout()
    plt.show()


def Validity2_Sensors():
    # --------------------------- Setting up the layout of the plot --------------------------- #
    mpl.rc('font', family='serif')
    plt.rc('xtick', labelsize=7)
    plt.rc('ytick', labelsize=7)
    plt.rc('legend', fontsize=6)
    plt.rc('axes', titlesize=7)
    plt.rc('figure', titlesize=9)
    fig, axs = plt.subplots(2, 4, figsize=(10, 5))
    fig.suptitle('Sensors')
    fig.text(0.5, 0.01, 'Time', ha='center', fontsize=7)
    axs[0, 0].set_ylabel('Phone1', rotation=0, fontsize=7, labelpad=20)
    axs[1, 0].set_ylabel('Phone2', rotation=0, fontsize=7, labelpad=20)
    # ----------------------------------------------------------------------------------------- #
    df_all = pd.DataFrame()  # initializing a dataframe to aggregate all sensors .csv files for a given phone
    scaler_list = []  # initializing a list to hold the Min Max estimators (i.e., one for each phone)
    for i in range(2):  # iterate over phones
        for j in range(122):  # iterate over RPs
            df_temp = pd.read_csv(
                'Phone' + str(i + 1) + '_Sensors_' + str(j + 1) + '.csv')  # read the sensors .csv file
            df_all = df_all.append(df_temp, ignore_index=True)  # append it to the df_all dataframe
        scaler = MinMaxScaler(feature_range=(0, 1))  # use Min Max estimator to scale features between 0 and 1
        scaler.fit(
            df_all[['Bx', 'By', 'Bz', 'Yaw', 'Pitch', 'Roll', 'Pressure', 'Illuminance']])  # fit the estimator to the
        # columns of interest
        scaler_list.append(scaler)  # append the estimator to the list to be used for the respective phone
    randomRP = random.sample(range(1, 122), 4)  # randomly select 4 RPs
    for i in range(2):  # iterate over phones
        for j in range(len(randomRP)):  # iterate over the selected RPs
            df = pd.read_csv(
                'Phone' + str(i + 1) + '_Sensors_' + str(randomRP[j]) + '.csv')  # read the sensors .csv file
            # corresponding to the RP for a given phone
            df[['Bx', 'By', 'Bz', 'Yaw', 'Pitch', 'Roll', 'Pressure', 'Illuminance']] = \
                scaler_list[i].transform(
                    df[['Bx', 'By', 'Bz', 'Yaw', 'Pitch', 'Roll', 'Pressure', 'Illuminance']])  # scale
            # the values according to the estimator of the given phone
            # ---------------------- plot the features of interest over time ---------------------- #
            df.plot(kind='line', x='Time', y='Bx', color='r', ax=axs[i, j], legend=False, alpha=0.7)
            df.plot(kind='line', x='Time', y='By', color='b', ax=axs[i, j], legend=False, alpha=0.7)
            df.plot(kind='line', x='Time', y='Bz', color='g', ax=axs[i, j], legend=False, alpha=0.7)
            df.plot(kind='line', x='Time', y='Yaw', color='c', ax=axs[i, j], legend=False, alpha=0.7)
            df.plot(kind='line', x='Time', y='Pitch', color='m', ax=axs[i, j], legend=False, alpha=0.7)
            df.plot(kind='line', x='Time', y='Roll', color='y', ax=axs[i, j], legend=False, alpha=0.7)
            df.plot(kind='line', x='Time', y='Pressure', color='k', ax=axs[i, j], legend=False, alpha=0.7)
            df.plot(kind='line', x='Time', y='Illuminance', color='tab:brown', ax=axs[i, j], legend=False, alpha=0.7)
            axs[0, j].set_title('RP_' + str(randomRP[j]))
            axs[1, j].set_xlabel('')
    for ax in fig.axes:
        plt.sca(ax)
        plt.grid(True)
        plt.xticks(rotation=20, ha='right')
    axs[0, 3].legend(loc='center left', bbox_to_anchor=(1, 0.7))
    for i in range(4):
        axs[0, i].set_xlabel('')
    plt.tight_layout()
    plt.show()


# uncomment any of the functions below to activate

# Validity2_WiFi()
# Validity2_Bluetooth()
# Validity2_Cellular()
# Validity2_Sensors()
