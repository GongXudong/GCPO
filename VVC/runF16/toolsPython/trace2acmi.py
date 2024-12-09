# -*- coding: utf-8 -*-

import os, sys
import getopt
import re
import numpy as np
import pandas as pd

from aeroTools import XYZ2Nav
from mpl_toolkits.mplot3d import axes3d
from tools import get_file_names

d2r = 0.01745
LOGPATH = '../logs/'

def get_fname(argv):
    ''' get file name from command line '''
    try:
        opts, args = getopt.getopt(argv, '')
        print(args)
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)  # will print something like "option -a not recognized"
        sys.exit(2)
        
    if len(args) > 1:
        if isinstance(args[1], str):
            return args[1]
        else:
            return str(args[1])
    
    return None

def get_data(fname):
    ''' get data from file name '''
    if fname == 'log':
        files = get_file_names(None, '..', False)
        for fn in files:
            if os.path.split(fn)[1].startswith('log'):
                fname = fn
                break
    if fname.endswith('txt'):
        with open(fname) as f:
            data_names = f.readline()   # get the 1st line of column names 
        del_str = '[#()0-9\n]'  # define the char to delete
        data_names = (re.sub(del_str, '', data_names)).split('\t')  # delete the selected char
        raw_data = np.loadtxt(fname)    # load data txt
        data = pd.DataFrame(raw_data, columns=data_names)   # transfer the data to DataFrame
    elif fname.endswith('csv'):
        data = pd.read_csv(fname)   # load data from csv
    else:
        print('No method to read this file')
        sys.exit()
    return data
    
def get_useful_columns(data):
    data['Lon'] = 56    # longitude reference
    data['Lat'] = 41.36 # latitude reference
    data['north'] = 0   # north direction
    
    # calculate the lon and lat from x y, and save the result to data
    for num,i in enumerate(map(XYZ2Nav, data.x, data.y, data.Lon, data.Lat, data.north)):
        data.at[num, 'Longitude'] = i[0]
        data.at[num, 'Latitude'] = i[1]
    
    useful_columns = ['time','Longitude', 'Latitude', 'h', 'phi', 'theta', 'psi']   # define the useful columns names
    useful_array = np.copy(data[useful_columns])    # copy the useful data
    
    # transfer the useful data to DataFrame and change the columns names
    useful_data = pd.DataFrame(useful_array, columns=['time','Longitude', 'Latitude', 'Altitude', 'Roll', 'Pitch', 'Yaw'])
    useful_data.Altitude = useful_data.Altitude + 100 # add 100 to latitude
    return useful_data

def addlonlat(data):
    data['Lon'] = 56    # longitude reference
    data['Lat'] = 41.36 # latitude reference
    data['north'] = 0   # north direction
    
    # calculate the lon and lat from x y, and save the result to data
    for num,i in enumerate(map(XYZ2Nav, data.x, data.y, data.Lon, data.Lat, data.north)):
        data.at[num, 'Longitude'] = i[0]
        data.at[num, 'Latitude'] = i[1]
    return data

def txt2csv(txtname, csvname=None):
    if csvname is None:
        csvname = txtname.replace('txt', 'csv')
    data = get_data(fname)
    data = addlonlat(data)
    useful_columns = ['time','x','y','h','Longitude','Latitude','Vt','Vh','adjustedTheta','adjustedPhi','adjustedPsi','alpha','beta','p','q','r','nx','ny','nz','posDe','posDa','posDr','posPLA']
    newColumnsNames = ['time(1)','x(2)','y(3)','h(4)','lon(5)','lat(6)','v(7)','vz(8)','theta(9)','phi(10)','psi(11)','alpha(12)','beta(13)','p(14)','q(15)','r(16)','nx(17)','ny(18)','nz(19)','ele(20)','ail(21)','rud(22)','throttle(23)']
    useful_array = np.copy(data[useful_columns])    # copy the useful data
    useful_data = pd.DataFrame(useful_array, columns=newColumnsNames)
    useful_data.to_csv(csvname, index=False)

def txt2acmi(txtname):
    print('file name: ',txtname)
    data = get_data(txtname)
    data['time'] = data['time'].round(2)
    csv_name = 'A-4E (Viper) [Red].csv'
    get_useful_columns(data).to_csv(csv_name, mode = 'w', index=False)   # wirte data to csv file

def csv2acmi(csvname):
    print('file name: ',csvname)
    data = get_data(csvname)
    data['time'] = data['time'].round(2)
    csv_name = LOGPATH + 'A-4E (Viper) [Red].csv'
    get_useful_columns(data).to_csv(csv_name, mode = 'w', index=False)   # wirte data to csv file

if __name__ == '__main__':
    fname = LOGPATH + 'planeStates.csv'         # default file name
    my_fname = get_fname(sys.argv)      # get file name from command line
    if my_fname:
        fname = my_fname
    csv2acmi(fname)