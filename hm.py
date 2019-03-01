# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: Siddhant
"""

import pandas as pd
import numpy as np
from numpy import log, exp, logaddexp
from scipy.stats import norm
from scipy.optimize import *
from rust import OLS

def logPtilde(theta):
    p = np.empty((7,l))
    for xInd,x in xEnum:
        for rInd, rc in rcEnum:
            num = -theta[1]*rc+v1[xInd,rInd]
            denom = logaddexp(num,-theta[0]*x+v0[xInd,rInd])
#            print(num/denom)
            p[xInd,rInd] = num-denom
    return p

def weightedDistance(theta):
    relevantPtilde = exp(logPtilde(theta)[relevantXRC]).flatten()
    return np.linalg.norm(weights*(relevantPtilde-relevantObsCCP))

def unweightedDistance(theta):
    relevantPtilde = exp(logPtilde(theta)[relevantXRC]).flatten()
    return np.linalg.norm((relevantPtilde-relevantObsCCP))

if __name__ == '__main__':
    from time import clock as clock
    start_time = clock()

    beta = 0.95
    d = 0.5
    gamma = np.euler_gamma
    xEnum = list(enumerate(range(1,8)))
    xGrid = np.array(range(1,8))
    rcGrid = np.arange(10,95.1,d)
    rcEnum = list(enumerate(rcGrid))
    rcInd = {x:i for i,x in rcEnum}
    l = len(rcGrid)

    df = pd.read_csv('ddc_pset.csv')
    df = df.sort_values(by=['i','t'])
    df[['a_lag1','rc_lag1']] = df.groupby(['i']).shift().loc[:,['a','rc']]

#==============================================================================
# Estimating the parameters of the AR(1) process
#==============================================================================
    x = np.vstack((np.ones(99),df['rc_lag1'].values[1:100])).T
    y = df['rc'].values[1:100]
    res = OLS(y,x,'no')
    rho, sigma = res[0], res[-1]**0.5

#==============================================================================
# Unsmoothing the AR(1) process into a matrix of transition probabilities
#==============================================================================
    rcProb = np.zeros((l,l))
    for m,j in rcEnum:
        mean = rho[0]+rho[1]*j
#        print(mean)
        for n,k in rcEnum:
            if n == 0:
                rcProb[m,n] = norm.cdf(k+d/2,mean,sigma)
            elif n == l-1:
                rcProb[m,n] = 1-norm.cdf(k-d/2,mean,sigma)
            else:
                rcProb[m,n] = norm.cdf(k+d/2,mean,sigma)-norm.cdf(k-d/2,mean,sigma)

#==============================================================================
# observed CCP
#==============================================================================
    obsCCP = np.empty((7,l))
    for xInd,x in xEnum:
        for rc in rcGrid:
            rcind = rcInd[rc]
            obsCCP[xInd,rcind] = df.loc[(df['x']==x)&(df['rc']==rc),'a'].mean()

#==============================================================================
# smoothing the CCP using quadratic in rc and x
#==============================================================================
    y = df['a'].values
    rc = df['rc'].values
    x = df['x'].values
    X = np.vstack((np.ones(100000),rc,x,rc*x,rc**2,x**2)).T

    params = OLS(y,X,'no')[0]

    smoothCCP = np.empty((7,l))
    for xInd,x in xEnum:
        for rc, rcind in rcInd.items():
            smoothCCP[xInd,rcind] = params.dot(np.array([1,rc,x,rc*x,rc**2,x**2]))
#            print(xInd,rcind,x,rc,params.dot(np.array([1,rc,x,rc*x,rc**2,x**2])))
#            input()
    epsilon = 1e-10
    smoothCCPclipped = np.clip(smoothCCP,epsilon,1-epsilon)
    longCCP = smoothCCPclipped.flatten() # CCP of dim (777,) (x=1 & rc--> then x=2 rc --> etc.)

#==============================================================================
# Creating the 'Long' (conditional on action)
# state transition matrices of dim (7*l,7*l)
#==============================================================================
    stateTrans = {0:0,1:0}

    blocks0 = [[0]*7 for xInd in range(7)]
    for xInd in range(7):
        blocks0[xInd][(xInd+1)%7] = 1
    stateTrans[0] = np.kron(blocks0,rcProb)

    blocks1 = [[1]+[0]*6 for xInd in range(7)]
    stateTrans[1] = np.kron(blocks1,rcProb)

#==============================================================================
# Inversion
#==============================================================================
    psi = {}
    psi[0] = -np.log(1-longCCP) + gamma

    V = np.linalg.solve(np.eye(7*l)-beta*stateTrans[0],psi[0])

    v0 = (beta*stateTrans[0]@V).reshape((7,l))
    v1 = (beta*stateTrans[1]@V).reshape((7,l))

#==============================================================================
# CCPs on the support
#==============================================================================
    relevantXRC = ~np.isnan(obsCCP)
    relevantObsCCP = obsCCP[relevantXRC].flatten()

    XRC, countsXRC = np.unique(df.loc[:,['x','rc']].values,axis=0,return_counts=True)
    weights = countsXRC/countsXRC.sum()

#==============================================================================
# Setting up the minimization
#==============================================================================
    res = {}
    guesses = [np.array((k,k))*0.2 for k in range(1,31)]

    np.random.seed(123)
    randomTheta1 = [np.random.uniform(0,10) for _ in range(10)]
    randomTheta2 = [np.random.uniform(0,5) for _ in range(10)]
    guesses += list(zip(randomTheta1,randomTheta2))
    print(clock()-start_time,'setup complete')
    start_time=clock()
    weighted = 0
#==============================================================================
# Minimizing the log-likelihood for 40 initial guesses
#==============================================================================
    for initialGuess in guesses:
        if weighted:
            res[str(initialGuess)] = minimize(weightedDistance,initialGuess,bounds=((0,None),(0,None)))
#            res[str(initialGuess)] = basinhopping(weightedDistance,initialGuess,)

        else:
            res[str(initialGuess)] = minimize(unweightedDistance,initialGuess,bounds=((0,None),(0,None)))
#            res[str(initialGuess)] = basinhopping(unweightedDistance,initialGuess,)

    pd.DataFrame.from_dict(data=res, orient='index').to_csv('resHM'+'un'*(1-weighted)+'weighted.csv', header=True)

    end_time = clock()
    print("It took the program", int((end_time-start_time)//3600), "hour(s)", int(((end_time-start_time)//60)%60), "minute(s)",  '{:f}'.format(((end_time-start_time)%3600)%60),"seconds to run.")












