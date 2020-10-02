import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pandas as pd

mpl.rc('font', family='serif')
plt.rc('xtick', labelsize=9)
plt.rc('ytick', labelsize=9)

# setting the x and y limits for the grid
x0 = -61
x1 = 427
y0 = -61
y1 = 305

os.chdir('C:/Users/uf11/Desktop/OutFin/Coordinates/')  # NOTE: change directory to where OutFin dataset resides

grid_x, grid_y = np.mgrid[x0:x1:427j, y0:y1:305j] # create the grid

df = pd.read_csv('Site3_Local.csv') # load Site 3 local coordinates
x = df['X'].to_numpy()
y = df['Y'].to_numpy()
RP = df['RP_ID'].to_numpy()
Day3_RP = [43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60]
Day4_RP = [61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77]
os.chdir('C:/Users/uf11/Desktop/OutFin/Measurements/')

# obtain calibration results for Phone1/Phone2 Day3/Day4 using Calibration.py
o_1_3_x = -1.1204500000000017
o_1_3_y = -0.3366000000000007
o_1_3_z = 1.116500000000002
s_1_3_x = 1.0227674901419832
s_1_3_y = 0.9899185440921173
s_1_3_z = 0.9880675588361983
o_1_4_x = -2.0947500000000012
o_1_4_y = 23.6993
o_1_4_z = -13.565700000000001
s_1_4_x = 1.0003702502260423
s_1_4_y = 1.0129396688419525
s_1_4_z = 0.9870260500210885
o_2_3_x = 2.3444499999999984
o_2_3_y = 0.31174999999999997
o_2_3_z = -0.5986499999999992
s_2_3_x = 1.0904367823075802
s_2_3_y = 0.9693091154889374
s_2_3_z = 0.9512271119853873
o_2_4_x = 0.2895000000000003
o_2_4_y = -0.8330999999999982
o_2_4_z = -0.2518999999999991
s_2_4_x = 1.029515555659627
s_2_4_y = 0.9684695048028366
s_2_4_z = 1.0039028441289468

# calibrate the magnetic field readings and save the calibrated readings as new files
for i in Day3_RP:
    df = pd.read_csv('Phone1_Sensors_' + str(i) + '.csv')
    df['Bx'] = df['Bx'] - o_1_3_x
    df['By'] = df['By'] - o_1_3_y
    df['Bz'] = df['Bz'] - o_1_3_z
    df['Bx'] = df['Bx'] * s_1_3_x
    df['By'] = df['By'] * s_1_3_y
    df['Bz'] = df['Bz'] * s_1_3_z
    df.to_csv('P1_S_' + str(i) + '.csv', index=False)
    df = pd.read_csv('Phone2_Sensors_' + str(i) + '.csv')
    df['Bx'] = df['Bx'] - o_2_3_x
    df['By'] = df['By'] - o_2_3_y
    df['Bz'] = df['Bz'] - o_2_3_z
    df['Bx'] = df['Bx'] * s_2_3_x
    df['By'] = df['By'] * s_2_3_y
    df['Bz'] = df['Bz'] * s_2_3_z
    df.to_csv('P2_S_' + str(i) + '.csv', index=False)

for i in Day4_RP:
    df = pd.read_csv('Phone1_Sensors_' + str(i) + '.csv')
    df['Bx'] = df['Bx'] - o_1_4_x
    df['By'] = df['By'] - o_1_4_y
    df['Bz'] = df['Bz'] - o_1_4_z
    df['Bx'] = df['Bx'] * s_1_4_x
    df['By'] = df['By'] * s_1_4_y
    df['Bz'] = df['Bz'] * s_1_4_z
    df.to_csv('P1_S_' + str(i) + '.csv', index=False)
    df = pd.read_csv('Phone2_Sensors_' + str(i) + '.csv')
    df['Bx'] = df['Bx'] - o_2_4_x
    df['By'] = df['By'] - o_2_4_y
    df['Bz'] = df['Bz'] - o_2_4_z
    df['Bx'] = df['Bx'] * s_2_4_x
    df['By'] = df['By'] * s_2_4_y
    df['Bz'] = df['Bz'] * s_2_4_z
    df.to_csv('P2_S_' + str(i) + '.csv', index=False)

v_temp = {}
v = [] # create a list that will hold the averaged magnitude
for i in range(2):
    v_temp[i] = []
    for j in RP: # compute the magnitude of the magnetic field vector
        df = pd.read_csv('P'+str(i+1)+'_S_'+str(j)+'.csv')
        df = df[['Bx','By','Bz']]
        df = df * df
        df = df.apply(np.sum, axis=1)
        df = df.apply(np.sqrt)
        df = df.mean()
        v_temp[i].append(df)

for i in range(len(RP)):
    v.append((v_temp[0][i] + v_temp[1][i])/2)
v = np.asarray(v)

xy = np.vstack((x,y)).T

grid_z1 = griddata(xy, v, (grid_x, grid_y), method='linear') # interpolate on the grid using linear interpolation
grid_z2 = griddata(xy, v, (grid_x, grid_y), method='cubic') # interpolate on the grid using cubic interpolation

ms = 3
# plot the interpolation results
plt.subplot(121)
plt.imshow(grid_z1.T, extent=(x0,x1,y0,y1), origin='lower', cmap= 'jet')
plt.plot(xy[:,0], xy[:,1], 'k.', ms=ms)
plt.title('Linear interpolation')
plt.ylabel('$\it{cm}$')
plt.xlabel('$\it{cm}$')
plt.colorbar(label='$\it{\u03BCT}$')
plt.subplot(122)
plt.imshow(grid_z2.T, extent=(x0,x1,y0,y1), origin='lower', cmap= 'jet')
plt.plot(xy[:,0], xy[:,1], 'k.', ms=ms)
plt.title('Cubic interpolation')
plt.ylabel('$\it{cm}$')
plt.xlabel('$\it{cm}$')
plt.colorbar(label='$\it{\u03BCT}$')
plt.show()
