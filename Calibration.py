import os
import pandas as pd

os.chdir('C:/Users/uf11/Desktop/OutFin')  # NOTE: change the path to where OutFin dataset resides
os.chdir('Calibration')

# the function below calculates the Hard iron offsets and Soft iron scales as described by Kris Winer in:
# https://github.com/kriswiner/MPU6050/wiki/Simple-and-Effective-Magnetometer-Calibration
def HardSoftIronCalibration(phone, day):
    df = pd.read_csv(phone + '_' + day + '.csv')
    max_x = df['Bx'].max()
    max_y = df['By'].max()
    max_z = df['Bz'].max()
    min_x = df['Bx'].min()
    min_y = df['By'].min()
    min_z = df['Bz'].min()
    offset_x = (max_x + min_x)/2
    offset_y = (max_y + min_y)/2
    offset_z = (max_z + min_z)/2
    avg_delta_x = (max_x - min_x) / 2
    avg_delta_y = (max_y - min_y) / 2
    avg_delta_z = (max_z - min_z) / 2
    avg_delta = (avg_delta_x + avg_delta_y + avg_delta_z) / 3
    scale_x = avg_delta / avg_delta_x
    scale_y = avg_delta / avg_delta_y
    scale_z = avg_delta / avg_delta_z
    print('Calibration values for ' + phone + ' on ' + day + ' are:')
    print('offsets:', offset_x,'(x)', offset_y,'(y)', offset_z,'(z)')
    print('scales: ', scale_x,'(x)', scale_y,'(y)', scale_z,'(z)')

# calibrated readings are then calculated as follows:
# calibrated_x = (measured_x - offset_x) * scale_x
# calibrated_y = (measured_y - offset_y) * scale_y
# calibrated_z = (measured_z - offset_z) * scale_z

Phone1 = 'Phone1'
Phone2 = 'Phone2'

Day1 = '031119'
Day2 = '041119'
Day3 = '051119'
Day4 = '071119'
Day5 = '081119'
Day6 = '091119'

# uncomment the function below to activate; specify the phone and day for which the Hard and Soft iron calibration
# values are to be calculated

# HardSoftIronCalibration(Phone2, Day6)