import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate
import sympy as sm
from sympy.plotting import plot
from sympy import lambdify
from scipy import linalg
from scipy import optimize
from scipy.optimize import brute
from scipy import interpolate
import time
from IPython import display

#-------------- Functions for part 1 ---------------

def maximfigure(hvec,lvec,cvec,vvec,per):
    '''
    This function prints three plots. 1) is the utility in a given period as a function of human capital, 2) is labour supply in a given period as a function of human capital and 3) is a plot of consumption in a given period as a function of human capital

    Parameters
    ----------
    hvec : array
        Vector of human capital in equilibriums
    lvec : array
        Vector of labour supply in equilibriums
    cvec : array
        Vector of consumption in equilibriums
    vvec : array
        Vector of utility in equilibriums
    per : float
        Number indicating the period
        
    Returns
    -------
    Three plots: Utility, labour supply and consumption as a function of human capital

    '''
    if per==1:
        hlab='$h_1$'
        clab='$c_1$'
        vlab='$v_1$'
        llab='$l_1$'
    else:
        hlab='$h_2$'
        clab='$c_2$'
        vlab='$v_2$'
        llab='$l_2$'
    
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(1,3,1)
    ax.plot(hvec,vvec)
    ax.grid()
    ax.set_xlabel(hlab)
    ax.set_ylabel(vlab)
    ax.set_title('Utility')
    #ax.set_xlim([0.1,1.5])
    #ax.set_ylim([-1.5,0]);

    ax = fig.add_subplot(1,3,2)
    ax.plot(hvec,lvec)
    ax.grid()
    ax.set_xlabel(hlab)
    ax.set_ylabel(llab)
    ax.set_title('Labour supply')
    #ax.set_xlim([0.1,1.5])
    #ax.set_ylim([0,2.5]);

    ax = fig.add_subplot(1,3,3)
    ax.plot(hvec,cvec)
    ax.grid()
    ax.set_xlabel(hlab)
    ax.set_ylabel(clab)
    ax.set_title('Consumption')
    #ax.set_xlim([0.1,1.5])
    #ax.set_ylim([0,3.5])
    plt.tight_layout();
    
