# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: Siddhant
"""

import pandas as pd
#from plotnine import *
#import statsmodels as sm
import numpy as np
from scipy.stats import norm
from scipy.special import logsumexp
from scipy.optimize import *
import timeit

def OLS(y,x,se='hc1'):
    """se could be:
                    no
                    homo (homoskedastic unbiased)
                    hc0  (Eicker-White)
                    hc1  (Hinkley) (dof-adjusted Eicker-White) (default)
                    hc2  (Horn-Horn-Duncan)
                    hc3  (MacKinnon-White)

                    The var-covar matrix is one to which the var-covar of (beta_hat - beta) will asymptotically converge.
                    (Note the absence of the root_n.) So, standard errors are just the roots of the diagonals of this matrix.
    """
    n, k = x.shape
    bread = np.linalg.inv(x.T@x)
    beta_hat = bread@x.T@y
    y_hat = x@beta_hat
    error_hat = y - y_hat
    error_hat_sq = error_hat**2
    mse = sum(error_hat_sq)/n
    if se == 'no':
        return beta_hat, mse
    if se == 'homo':
        sigma_sq_hat = (1/n)*np.sum(error_hat_sq)
        sigma_sq_hat_unbiased = n/(n-k)*sigma_sq_hat
        var = sigma_sq_hat_unbiased*bread
    elif se == 'hc0':
        meat = x.T@np.diag(error_hat_sq)@x
        var = bread@meat@bread
    elif se == 'hc1':
        meat = n/(n-k)*x.T@np.diag(error_hat_sq)@x
        var = bread@meat@bread
    elif se == 'hc2':
        hat_matrix = x@bread@x.T
        leverage = np.diag(hat_matrix)
        meat = x.T@np.diag((1/(1-leverage))*error_hat_sq)@x
        var = bread@meat@bread
    elif se == 'hc3':
        hat_matrix = x@bread@x.T
        leverage = np.diag(hat_matrix)
        meat = x.T@np.diag((1/(1-leverage))*(1/(1-leverage))*error_hat_sq)@x
        var = bread@meat@bread
    se = np.sqrt(np.diag(var))
    return beta_hat, se, mse

def valueFx(theta,tol=1e-12,):
    EV = np.zeros((7,len(rcGrid),2))
    dist = 1e10
    EV_ = np.ones((7,len(rcGrid),2))
    for xInd, x in xEnum:
        for a in [0,1]:
            x_ = a + (1-a)*min(x+1,7)
            x_Ind = np.where(xGrid==x_)[0][0]
            u0 = -theta[0]*x_ + beta*EV[x_Ind,:,0]
            u1 = -theta[1]*rcGrid + beta*EV[x_Ind,:,1]
            EV_[xInd,:,a] = np.logaddexp(u0,u1)@rcProb
        dist = (abs(EV-EV_)).max()
        if dist < tol:
#            print('convergence',dist,)
            break
        EV[:] = EV_
    return EV

def likelihood(theta):
    def logprobiGivenx(x,rc,a):
        xInd = x-1
        temp = -theta*np.array((x,rc)) + beta*np.array(EV[xInd,rcInd[rc],:])
        denom = np.logaddexp(*temp)
        return temp[a] - denom

    EV = valueFx(theta)
    l1 = -sum([c*logprobiGivenx(x,rc,a) for x,rc,a,c in XRCAwithCounts])
    return l1

if __name__ == '__main__':
    from time import clock as clock
    start_time = clock()

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

    beta = 0.95
    d = 0.5
    xEnum = list(enumerate(range(1,8)))
    xGrid = np.array(range(1,8))
    rcGrid = np.arange(10,95.1,d)
    rcInd = {x:i for i,x in enumerate(rcGrid)}
    l = len(rcGrid)

#==============================================================================
# Unsmoothing the AR(1) process into a matrix of transition probabilities
#==============================================================================
    rcProb = np.zeros((l,l))
    for m,j in enumerate(rcGrid):
        mean = rho[0]+rho[1]*j
        for n,k in enumerate(rcGrid):
            if n == 0:
                rcProb[m,n] = norm.cdf(k+d/2,mean,sigma)
            elif n == l-1:
                rcProb[m,n] = 1-norm.cdf(k-d/2,mean,sigma)
            else:
                rcProb[m,n] = norm.cdf(k+d/2,mean,sigma)-norm.cdf(k-d/2,mean,sigma)

#==============================================================================
# Identifying unique (x,rc,a) and their counts in the data
#==============================================================================

    XRCA, counts = np.unique(df.loc[:,['x','rc','a']].values,axis=0,return_counts=True)
    counts = counts.astype(np.int)
    X = XRCA[:,0].astype(np.int)
    RC = XRCA[:,1].astype(np.float16)
    A = XRCA[:,2].astype(np.int)

    XRCAwithCounts = np.rec.fromarrays([X,RC,A,counts])

    res = {}
    guesses = [np.array((k,k))*0.2 for k in range(1,51)]
    guesses += [np.random.normal(loc=5,scale=5,size=(2)) for _ in range(50)]
    print(clock()-start_time,'setup complete')

#==============================================================================
# Minimizing the log-likelihood for 50 initial guesses
#==============================================================================
    for initialGuess in guesses:
        if (initialGuess > 0) == 2:
            res[str(initialGuess)] = minimize(likelihood,initialGuess,bounds=((0,None),(0,None)),tol=1e-10,)

    pd.DataFrame.from_dict(data=res, orient='index').to_csv('resRust.csv', header=False)
    end_time = clock()
    print("It took the program", int((end_time-start_time)//3600), "hour(s)", int(((end_time-start_time)//60)%60), "minute(s)",  '{:f}'.format(((end_time-start_time)%3600)%60),"seconds to run.")















