#!/user/bin/python
#-- coding:utf-8 -- 

#this model created to convert the posXYZ to posNAV

from __future__ import division
from math import *
import sys

#常量定义
rad2deg = 180/pi
deg2rad = pi/180
f_eth = 1/298.257223563  #地球WGS-84坐标系中的扁率(Re-Rp)/Re
Re = 6378137   #赤道半径（m）

Const_1 = 1
Const_2 = 2
Const_90 = 90
Const_180 = 180
Const_360 = 360

Epsilon = sys.float_info.epsilon

'''this section defines the utility functions '''

def MyAtan2(x,y):
    '''defined based on atan2'''
    if y ==0:
        if x >0.0:return pi/2.0
        elif x<0.0:return -pi/2.0
        else :return 0.0
    else:return atan2(x,y)

def round(x):
    if (fabs(x)<4.5035996273705E+15):
        return -floor(fabs(x) + 0.5) if x<0.0 else floor(fabs(u)+0.5)
    else:
        return x
    
def remd(x,y):
    '''get the remainder of x in y'''
    z = x/y
    if y<=floor(y):
        return x-fix(z)*y
    elif fabs(z-round(z))<=Epsilon*fabs(z):
        return 0.0
    else:
        return z-fix(z)*y
        
        
def fix(x):
    '''round numbers near 0'''
    return ceil(x) if x<0.0 else floor(x)
  
def XYZ2Nav(x,y,L0,B0,psi):
    '''
    x,y -- 
    L0 -- (0,0) longtitude
    B0 -- (0,0) latitude
    psi --  angle between axis x and north
    '''
    
    psi = psi*deg2rad  
    sinOut = sin(psi)
    cosOut = cos(psi)
    
    e = Const_1 - f_eth  #e in simulink
    eps = sqrt(1-e*e)    #eps in simulink
    
    rB0 = B0*deg2rad     # rB0 in  rad; B0 in deg  
    sin_B0 = sin(rB0)
    
    demon = Const_1 -(eps*eps*sin_B0*sin_B0)
    
    #calc the Lon
    Rn = Re/sqrt(demon)
    dEast = MyAtan2(Const_1,Rn*cos(rB0))
    radLon = (x*sinOut+y*cosOut)*dEast
    muLon = radLon*rad2deg + L0
    
    #calc the Lat
    Rm = (Const_1-eps*eps)*Rn/demon
    dNorth = MyAtan2(Const_1,Rm)
    radLat = (x*cosOut-y*sinOut)*dNorth
    muLat = radLat*rad2deg + B0
    
    #
    muLat = remd(muLat,Const_360)
    muLat1 = muLat/Const_180
    muLat = muLat - fix(muLat1)*Const_360
    
    muLat2 = muLat/Const_180
    muLat3 = Const_2*muLat2
    muLat3 = muLat3*muLat3
    muLat3 = sqrt(muLat3)
    muLat3 = fix(muLat3)
    muLat3 = Const_2*muLat3
    
    mu2Lon = (int(muLat3!=0))*Const_180+muLon
    mu2Lon = remd(mu2Lon,Const_360)
    
    mu2Lon1 = mu2Lon/Const_180

    #Longitude between +/- 180
    Longitude = mu2Lon - fix(mu2Lon1)*Const_360
    
    muLat2 = fix(muLat2)*Const_180
    
    #latitude between +/- 90
    Latitude = muLat -muLat2 -remd(muLat,Const_90)*muLat3
    
    return Longitude,Latitude
    
 #109.538521, 34.830795   
# print XYZ2Nav(600.14318841, 799.736600097,109.538521,34.830795,61.4144555991)
# print XYZ2Nav(599.856625473, 600.263298658,109.538521,34.830795,61.4144555991)