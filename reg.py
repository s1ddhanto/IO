# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: Siddhant
"""

import pandas as pd
import numpy as np
from rust import OLS

if __name__ == '__main__':
    from time import clock as clock
    start_time = clock()

    df = pd.read_csv('ddc_pset.csv')
    print(clock()-start_time)
#    df = df.sort_values(by=['i','t'])
#    print(df)

    beta = 0.95

#==============================================================================
# CCP as fx of x
#==============================================================================
    params = {}
    for t in range(1,101):
        relevantdf = df.loc[df['t']==t,:]
        x = relevantdf['x'].values
        y = relevantdf['a'].values

        X = np.vstack((np.ones(1000),x,x**2,x**3)).T

        params[t] = OLS(y,X,'no')[0]
    print(clock()-start_time)
#
    df2 = df.loc[df['t']!=100,:]

#==============================================================================
# RHS
#==============================================================================
    X = np.vstack([df2['x'].values,-df2['rc'].values]).T

#==============================================================================
# LHS
#==============================================================================
    Y = np.empty((1000,99))
    epsilon = 1e-8
    for ind,(i,t,a,x,rc) in df2.iterrows():

        xt1 = (x+1)%8
        p1 = params[t]@np.array([1,x,x**2,x**3])
        p2 = params[t+1]@np.array([1,1,1,1])
        p3 = params[t+1]@np.array([1,xt1,xt1**2,xt1**3])
        p1,p2,p3 = np.clip([p1,p2,p3],epsilon,1-epsilon)
        Y[int(i)-1,int(t)-1] = np.log(p1)-np.log(1-p1)+beta*(np.log(p2)-np.log(p3))
    Y = Y.reshape(99000)

#==============================================================================
# OLS
#==============================================================================
    res = OLS(Y,X,'no')

    end_time = clock()
    print("It took the program", int((end_time-start_time)//3600), "hour(s)", int(((end_time-start_time)//60)%60), "minute(s)",  '{:f}'.format(((end_time-start_time)%3600)%60),"seconds to run.")















