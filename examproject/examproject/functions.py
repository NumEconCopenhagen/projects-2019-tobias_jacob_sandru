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
from matplotlib.ticker import PercentFormatter

#-------------- Functions for part 1 ---------------


def consum(w, h, l,b):
    '''
    Computes consumption based on wage, human capital, labour supply and benefit
    
    
    Parameters
    ----------
    w : Float
        The wage in the labour market
    h : Float
        The human capital of the worker
    l : Int/Boolean
        Takes the value 1 (supplying labour) or 0 (unemployed)
    b : Float
        Unemployment benefits
    
    Returns
    -------
    c : Float
        Consumption of the worker
    
    '''
    
    c=w*h*l+(1-l)*b
    
    return c



def utility(w,h,l,rho,b,gamma):
    '''
    Computes total utility from the consumption and diutility from labour supply
    
        
    Parameters
    ----------
    w : Float
        The wage in the labour market
    h : Float
        The human capital of the worker
    l : Int/Boolean
        Takes the value 1 (supplying labour) or 0 (unemployed)
    rho : Float
        The constant risk aversion parameter of the CRRA utility function
    b : Float
        Unemployment benefits
    gamma : Float
        The disutility of labour
    
    Returns
    -------
    u : Float
        The utility of the decisions in current period.
    
    '''
    
    u=consum(w, h, l,b)**(1-rho)/(1-rho)-gamma*l
    return u



def v2(w,h2,l2,rho,b,gamma):
    '''
    Computes utility from period 2
    
        
    Parameters
    ----------
    w : Float
        The wage in the labour market
    h2 : Float
        The human capital of the worker in period 2
    l2 : Int/Boolean
        Takes the value 1 (supplying labour) or 0 (unemployed) in period 2
    rho : Float
        The constant risk aversion parameter of the CRRA utility function
    b : Float
        Unemployment benefits
    gamma : Float
        The disutility of labour
    
    Returns
    -------
    v : Float
        The utility in period 2.
    
    '''
    
    v=utility(w,h2,l2,rho,b,gamma)
    
    return v



def v1(l1,h1,rho,beta,Delta,b,w,gamma,v2_interp,accum):
    '''
    Computes utility from period 1 where the humancapital accumulates to period 2,
    if one have supplied labour in period 1. The accumulation includes a stochastic
    term. The functoin also enables the user to turn
    of accumulation.
        
    Parameters
    ----------
    l1 : int/boolean
        Takes the value 1 (supplying labour) or 0 (unemployed) in period 1
    h1 : float
        The human capital of the worker in period 1
    rho : float
        The constant risk aversion parameter of the CRRA utility function
    Delta : float
        The potential additional increase in human capital from period 1 to period 3
    b : float
        Unemployment benefits
    w : float
        The wage in the labour market
    gamma : float
        The disutility of labour
    v2_interp : array
        Interpolated grid of utility in period 2 based on human capital
    accum : boolean
        Whether or not to accumulate human capital
    
    Returns
    -------
    vs : Float
        The utility in period 1.
    
    '''
    # Expected utility in period 2
    h2_low = h1+l1
    h2_high = h1+l1+Delta
    if accum==False:
        h2_high=h1
        h2_low=h1
    
    v2_low = v2_interp([h2_low])[0]
    v2_high = v2_interp([h2_high])[0]
    
    
    # Expected v2 value
    v2 = 0.5*v2_low + 0.5*v2_high
    # d. total value
    vs = utility(w,h1,l1,rho,b,gamma) + beta*v2
    
    return vs



def solve_period_2(w,rho,b,gamma,h_vec):
    '''
    Maximizes utility in period 2
            
    Parameters
    ----------
    w : float
        The wage in the labour market
    h1 : float
        The human capital of the worker in period 1
    l1 : int/boolean
        Takes the value 1 (supplying labour) or 0 (unemployed) in period 1
    rho : float
        The constant risk aversion parameter of the CRRA utility function
    b : float
        Unemployment benefits
    gamma : float
        The disutility of labour
    h_vec : array
        Vector of human capital
    
    Returns
    -------
    h2_vec : array
        List of human capital in equlibrium
    v2_vec : array
        List of utility in equilibrium
    l2_vec : array
        List of lbaour in equilibrium
    c2_vec : array
        List of lbaour in equilibrium
    
    '''
    # Defining grids
    h2_vec = h_vec
    v2_vec = np.empty(100)
    l2_vec = np.empty(100)
    c2_vec = np.empty(100)

    # Solve for each h2 in grid
    for i,h2 in enumerate(h2_vec):

        v2_1=v2(w,h2,1,rho,b,gamma) # Utility from period 2 if labour is supplied
        v2_0=v2(w,h2,0,rho,b,gamma) # Utility from period 2 if labour is not supplied

        # The below computes the utility from period 2 dependent on whether the worker
        # prefers to supply labour
        if v2_1>v2_0:
            v2_vec[i] = v2_1 
            l2_vec[i] = 1
            c2_vec[i] = consum(w, h2, 1,b)
        
        if v2_1<v2_0:
            v2_vec[i] = v2_0 
            l2_vec[i] = 0
            c2_vec[i] = consum(w, h2, 0,b)
        
    return h2_vec,v2_vec,l2_vec,c2_vec



def solve_period_1(rho,beta,Delta,b,w,gamma,v2_interp,h_vec,accum):
    '''
    Maximizes utility in period 1.
    
                
    Parameters
    ----------
    rho : float
        The constant risk aversion parameter of the CRRA utility function
    beta : float
        The discount factor of future utility
    Delta : float
        The potential additional increase in human capital from period 1 to period 3
    b : float
        Unemployment benefits
    w : float
        The wage in the labour market
    gamma : float
        The disutility of labour
    v2_interp : array
        Interpolated grid of utility in period 2 based on human capital
    h_vec : array
        Vector of human capital
    accum : boolean
        Whether or not to accumulate human capital
    
    Returns
    -------
    h1_vec : array
        List of human capital in equlibrium
    v1_vec : array
        List of utility in equilibrium
    l1_vec : array
        List of lbaour in equilibrium
    c1_vec : array
        List of lbaour in equilibrium
    
    '''
    
    # Defining grids
    h1_vec = h_vec
    v1_vec = np.empty(100)
    l1_vec = np.empty(100)
    c1_vec = np.empty(100)
    
    # Solve for each h1 in grid
    for i,h1 in enumerate(h1_vec):

        v1_1=v1(1,h1,rho,beta,Delta,b,w,gamma,v2_interp,accum) # Utility from period 1 if labour is supplied
        v1_0=v1(0,h1,rho,beta,Delta,b,w,gamma,v2_interp,accum) # Utility from period 2 if labour is not supplied
        
        
        # The below computes the utility from period 2 dependent on whether the worker
        # prefers to supply labour
        if v1_1>v1_0:
            v1_vec[i] = v1_1 
            l1_vec[i] = 1
            c1_vec[i] = consum(w, h1, 1,b)
        
        if v1_1<v1_0:
            v1_vec[i] = v1_0 
            l1_vec[i] = 0
            c1_vec[i] = consum(w, h1, 0,b)
        
     
    return h1_vec,v1_vec,l1_vec,c1_vec



def solve(w,rho,b,gamma,beta,Delta,h_vec,accum=True):
    '''
    A function that first solves problem in period 2 for any level of human capital,
        and then computing the period 1 problem with any level of human capital
        while accounting for the accumulation.
        
                
    Parameters
    ----------
    w : float
        The wage in the labour market
    rho : float
        The constant risk aversion parameter of the CRRA utility function
    b : float
        Unemployment benefits
    gamma : float
        The disutility of labour
    beta : float
        The discount factor of future utility
    Delta : float
        The potential additional increase in human capital from period 1 to period 3
    h_vec : array
        Vector of human capital
    accum : boolean (optional)
        Whether or not to accumulate human capital (default is True)
    
    Returns
    -------
    h1_vec : array
        List of human capital in equlibrium
    v1_vec : array
        List of utility in equilibrium
    l1_vec : array
        List of lbaour in equilibrium
    c1_vec : array
        List of lbaour in equilibrium
    
    '''
    #Solving for period 2 first
    h2_vec,v2_vec,l2_vec,c2_vec=solve_period_2(w,rho,b,gamma,h_vec)
    
    #Creating a interpolated matrix of equilibriums from the problem of period 2
    v2_interp = interpolate.RegularGridInterpolator((h2_vec,), v2_vec,bounds_error=False,fill_value=None)
    
    #Using the results from the period 2 problem to optimize in period 1
    h1_vec,v1_vec,l1_vec,c1_vec=solve_period_1(rho,beta,Delta,b,w,gamma,v2_interp,h_vec,accum)
    
    return h1_vec,v1_vec,l1_vec,c1_vec

    
#-------------- Functions for part 2 ---------------


def single_shock(T,v_shock=0,s_shock=0,alphas=5.76,hs=0.5,bs=0.5,phis=0,gammas=0.075,deltas=0.8,omegas=0.15):
    '''
    This functions enables the user to produce and plot impulse response functions following either a
    demand or supply shock. Both are set to a default value of zero.
    The user is also able to alter the value of the parameter, where the default are the ones given
    in the assignment
            
                
    Parameters
    ----------
    T : Float
        Number of periods in the simulation
    v_shock : Float (optional)
        Size of demand shock in period 1 (default is 0)
    s_shock : Float (optional)
        Size of supply shock in period 1 (default is 0)
    alphas : float (optional)
        Output response to real interest rate (default is 5.76)
    hs : Float (optional)
        Central bank response to inflation gap in period 1 (default is 0.5)
    bs : Float (optional)
        Central bank response to output gap in period 1 (default is 0.5)
    phis : Float (optional)
        Expectation of inflation weighing on last period of expected inflation (default is 0)
    gammas : Float (optional)
        Inflation response to output gap (default is 0.075)
    deltas : float (optional)
        Persistence of demand shocks (default is 0.8)
    omegas : float (optional)
        Persistence of supply shocks (default is 0.15)

    Returns
    -------
    Plot of output and inflation impulse response following shock
    
    '''
    
    
    i = sm.symbols('i_t')            # Nominal interest rate
    pi =sm.symbols('\pi_t')          # Inflation gap
    pi_l =sm.symbols('pi_{t-1}')    # Inflation gap, lagged
    epi =sm.symbols('pi_t')         # Expected inflation gap
    epi_l =sm.symbols('pi_{t-1}')   # Expected inflation gap, lagged 
    epi_pl =sm.symbols('pi_{t-1}')  # Expected inflation gap, future
    r = sm.symbols('r_t')            # Real interest rate
    y =sm.symbols('y_t')             # Output gap
    y_l =sm.symbols('y_{t-1}')       # Output gap, lagged
    v = sm.symbols('v_t')            # Demand disturbance
    v_l = sm.symbols('v_{t-1}')      # Demand disturbance, lagged
    s = sm.symbols('s_t')            # Supply disturbance
    s_l = sm.symbols('s_{t-1}')      # Supply disturbance, lagged
    alpha = sm.symbols('alpha')      
    h = sm.symbols('h')      
    b = sm.symbols('b')      
    phi = sm.symbols('phi')      
    gamma = sm.symbols('gamma')      
    
    AD=sm.Eq(pi,1/(h*alpha)*(v-(1+b*alpha)*y))
    SRAS=sm.Eq(pi,pi_l+gamma*y-phi*gamma*y_l+s-phi*s_l)
    
    par = {}                  # Redefining intital values
    par['alpha'] = alphas
    par['h'] = hs
    par['b'] = bs
    par['phi'] = phis
    par['gamma'] = gammas

    AD_eq=AD.subs(par)         #Solving for the initital equilibriaum
    SRAS_eq=SRAS.subs(par)
    eq_y_pi_eq = sm.solve([AD_eq,SRAS_eq],[y,pi])   #Solving

    par['delta'] = deltas        #Setting persistence values.
    par['omega'] = omegas

    pi_irf=[]                        #Initializing the impulse response functions
    y_irf=[]

    T=100              #Setting time horizon
    for t in range(T):
        if t==1:
            x=v_shock
            c=s_shock
        else:
            x=0
            c=0
        if t==0:                     #Initializing the variables
            par['v_{t-1}']=0
            par['pi_{t-1}']=0
            par['y_{t-1}']=0
            par['s_{t-1}']=0
        else:
            par['y_{t-1}']=y_irf[-1]      #Setting lagged variables
            par['pi_{t-1}']=pi_irf[-1]
            par['s_{t-1}']=par['s_t']
            par['v_{t-1}']=par['v_t']    
            del par['v_t'], par['s_t']

        par['v_t']=par['v_{t-1}']*par['delta']+x
        par['s_t']=par['s_{t-1}']*par['omega']+c

        pi_irf.append(eq_y_pi_eq[pi].subs(par))    #Creating IRFs
        y_irf.append(eq_y_pi_eq[y].subs(par))

    pi_irf_=[float(i) for i in pi_irf]
    y_irf_=[float(i) for i in y_irf]
    fig = plt.figure(figsize=(5,5)) #Plotting    
    ax=fig.add_subplot(1,1,1)
    ax.plot(range(T),pi_irf_,color='b',label='$\pi_t$')
    ax.plot(range(T),y_irf_,color='r',label='$y_t$')
    ax.set_xlabel('T')
    ax.set_ylabel('$\pi_t$/$y_t$')
    ax.axhline(0,color='k')
    ax.set_title('IRF of $\pi_t$ and $y_t$ after a persistent demand shock')
    ax.legend()
    

    
def stochproc(T,s_on=False,v_on=False,sigmax=3.492,sigmac=0.2,alphas=5.76,hs=0.5,bs=0.5,phis=0,gammas=0.075,deltas=0.8,omegas=0.15, plot=True):
    '''
    This functions enables the user to produce and plot impulse response functions following either a
    demand or supply stochastic process. Both are set to a default value of off.
    The user is also able to alter the value of the parameter, where the default are the ones given
    in the assignment. You're are also able to turn off the plot, the default is on.
                
                
    Parameters
    ----------
    T : Float
        Number of periods in the simulation
    s_on : boolean (optional)
        Stochastic supply shocks (default is False)
    v_on : boolean (optional)
        Stochastic demand shocks (default is False)
    sigmax : float (optional)
        Standard deviation of the normal distribution of demand shocks (default is 3.492)
    sigmac : float (optional)
        Standard deviation of the normal distribution of supply shocks (default is 0.2)
    alphas : float (optional)
        Output response to real interest rate (default is 5.76)
    hs : Float (optional)
        Central bank response to inflation gap in period 1 (default is 0.5)
    bs : Float (optional)
        Central bank response to output gap in period 1 (default is 0.5)
    phis : Float (optional)
        Expectation of inflation weighing on last period of expected inflation (default is 0)
    gammas : Float (optional)
        Inflation response to output gap (default is 0.075)
    deltas : float (optional)
        Persistence of demand shocks (default is 0.8)
    omegas : float (optional)
        Persistence of supply shocks (default is 0.15)
    plot : boolean (optional)
        Plot the response of output and inflation (default is True)

    Returns
    -------
    y_bc : list
        List of output gap
    pi_bc : list
        List of inflation gap
    Plot (optional)
    
    '''
        
    
    i = sm.symbols('i_t')            # Nominal interest rate
    pi =sm.symbols('\pi_t')          # Inflation gap
    pi_l =sm.symbols('pi_{t-1}')    # Inflation gap, lagged
    epi =sm.symbols('pi_t')         # Expected inflation gap
    epi_l =sm.symbols('pi_{t-1}')   # Expected inflation gap, lagged 
    epi_pl =sm.symbols('pi_{t-1}')  # Expected inflation gap, future
    r = sm.symbols('r_t')            # Real interest rate
    y =sm.symbols('y_t')             # Output gap
    y_l =sm.symbols('y_{t-1}')       # Output gap, lagged
    v = sm.symbols('v_t')            # Demand disturbance
    v_l = sm.symbols('v_{t-1}')      # Demand disturbance, lagged
    s = sm.symbols('s_t')            # Supply disturbance
    s_l = sm.symbols('s_{t-1}')      # Supply disturbance, lagged
    alpha = sm.symbols('alpha')      
    h = sm.symbols('h')      
    b = sm.symbols('b')      
    phi = sm.symbols('phi')      
    gamma = sm.symbols('gamma')      
    
    AD=sm.Eq(pi,1/(h*alpha)*(v-(1+b*alpha)*y))
    SRAS=sm.Eq(pi,pi_l+gamma*y-phi*gamma*y_l+s-phi*s_l)
    
    par = {}                  # Redefining intital values
    par['alpha'] = alphas
    par['h'] = hs
    par['b'] = bs
    par['phi'] = phis
    par['gamma'] = gammas

    AD_eq=AD.subs(par)         #Solving for the initital equilibriaum
    SRAS_eq=SRAS.subs(par)
    eq_y_pi_eq = sm.solve([AD_eq,SRAS_eq],[y,pi])   #Solving

    par['delta'] = deltas        #Setting persistence values.
    par['omega'] = omegas

    par['sigma_x'] = sigmax     #Setting standard deviations
    par['sigma_c'] = sigmac

    pi_bc=[]                        #Initializing the impulse response functions
    y_bc=[]

    T=1000              #Setting time horizon
    np.random.seed(123)                         #Creating a distribution of stochastic shocks
    if v_on==True:
        xlist=np.random.normal(0,par['sigma_x'],T)
    else:
        xlist=np.zeros(T)
    if s_on==True:
        clist=np.random.normal(0,par['sigma_c'],T)
    else:
        clist=np.zeros(T)
    
    for t in range(T):
        if t>0:
            x=xlist.tolist()[t]
            c=clist.tolist()[t]
        else:
            x=0
            c=0
        if t==0:                     #Initializing the variables
            par['v_{t-1}']=0
            par['pi_{t-1}']=0
            par['y_{t-1}']=0
            par['s_{t-1}']=0
        else:
            par['y_{t-1}']=y_bc[-1]      #Setting lagged variables
            par['pi_{t-1}']=pi_bc[-1]
            par['s_{t-1}']=par['s_t']
            par['v_{t-1}']=par['v_t']    
            del par['v_t']
            del par['s_t']

        par['v_t']=par['v_{t-1}']*par['delta']+x
        par['s_t']=par['s_{t-1}']*par['omega']+c

        pi_bc.append(eq_y_pi_eq[pi].subs(par))    #Creating IRFs
        y_bc.append(eq_y_pi_eq[y].subs(par))

    
    pi_bc_=[float(i) for i in pi_bc]
    y_bc_=[float(i) for i in y_bc]
    if plot==True:
        fig = plt.figure(figsize=(5,5)) #Plotting    
        ax=fig.add_subplot(1,1,1)
        ax.plot(range(T),y_bc,color='r',label='$y_t$')
        ax.plot(range(T),pi_bc,color='b',label='$\pi_t$')
        ax.set_xlabel('T')
        ax.set_ylabel('$\pi_t$/$y_t$')
        ax.axhline(0,color='k')
        ax.set_title('Business cycles of $\pi_t$ and $y_t$')
        ax.legend()
    
    return y_bc_, pi_bc_



def statistics(y_bc,pi_bc):
    '''
    This function prints out the variance of boh output and inflation, their correlation and
    the autocorrelation
    of both following a simulation of the AS-AD model with stochastic supply and demand shocks.
                    
                
    Parameters
    ----------
    y_bc : list
        The process of output
    pi_bc : list
        The process of inflation

    Returns
    -------
    Print of:
        Variance of output and of inflation
        Correlation between output and inflation,
        Autocorrelation of output and autocorrelation of inflation.
    
    '''
    
    print('1. Variance of output')
    print(np.var(y_bc))
    print('2. Variance of inflation')
    print(np.var(pi_bc))
    print('3. Correlation between output and inflation')
    print(np.corrcoef(y_bc, pi_bc)[1,0])
    print('4. Auto-correlation of output')
    print(np.corrcoef(np.array([y_bc[:-1], y_bc[1:]]))[1,0])
    print('5. Auto-correlation of inflation')
    print(np.corrcoef(np.array([pi_bc[:-1], pi_bc[1:]]))[1,0])

    

def corr(phi,corr_out=False):
    '''
    This function calculates computes the corr.coeff. for a given phi between inflation and output.
    Then it computes the absolute diffence betweem the obatined coeff. and 0.31.
    
    The user is able to actively choose to compute both the absolute difference and the obtained
    corr.coeff.
    This is useful when wanting to use the absolute diffence in order to minimize the difference.
    Having to output to the objective function restricts using optimize.
                        
                
    Parameters
    ----------
    phi : float
        Expectation of inflation weighing on last period of expected inflation
    corr_out : boolean (optional)
        Returning the correlation coefficient of output and inflation or not (default is False)

    Returns
    -------
    diff : float
        Absolute difference between correlation between output and inflation
        in the simulation and the US economy
    corr : float (optional)
        The correlation between output and inflation
    
    '''
    y_bc, pi_bc = stochproc(1000,v_on=True,s_on=True,phis=phi,plot=False) # Computing inflation and output lists.
    corr=float(np.corrcoef(y_bc, pi_bc)[0,1])                          # Computing corr.coeff.
    diff = float(abs(corr - 0.31))                              # Computing abs. diff. between corr.coeff. and 0.31
    if corr_out==True:                                             
        return diff,corr
    else:
        return diff
  

    
def us_econ(stat,stat_out=False):
    '''
    This function calculates computes the corr.coeff. for a given phi, sigmax and sigmac
    between inflation and output.
    Then it computes the absolute diffences betweem the obatined coeff. and the corresponding
    value for the US economy.
    
    The user is able to actively choose to compute both the absolute difference and the obtained
    corr.coeff.
    This is useful when wanting to use the absolute diffence in order to minimize the difference.
    Having to output to the objective function restricts using optimize.
                            
                
    Parameters
    ----------
    stat : tuple
        Elemements in the tuple are phi, sigma_x and sigma_c
    stat_out : boolean (optional)
        Whether to return the variance of output and of inflation, correlation between
        output and inflation, autocorrelation of output and autocorrelation
        of inflation. (default is False)

    Returns
    -------
    sums : float
        Sum of absolute difference between the simulation and the US economy in 
        variance of output and of inflation, correlation
        between output and inflation, autocorrelation of output and autocorrelation of inflation
    
    Optionally:    
    vary : float
        Variance of output
    varpi : float
        Variance of inflation
    corr : float
        Correlation between inflation and output
    autoy : float
        Autocorrelation of output
    autopi : float
        Autocorrelation of inflation
    
    
    '''
    phi,sigmax_,sigmac_=stat
    
    # Computing inflation and output lists.
    y_bc, pi_bc = stochproc(1000,v_on=True,s_on=True,phis=phi,sigmax=sigmax_,sigmac=sigmac_,plot=False)
    #Computing different statistics.
    vary=np.var(y_bc)
    varpi=np.var(pi_bc)
    corr=np.corrcoef(y_bc, pi_bc)[1,0]
    autoy=np.corrcoef(np.array([y_bc[:-1], y_bc[1:]]))[1,0]
    autopi=np.corrcoef(np.array([pi_bc[:-1], pi_bc[1:]]))[1,0]
    #Computing the difference from the US economy
    diffvary = abs(vary - 1.64)
    diffvarpi = abs(varpi - 0.21)
    diffcorr = abs(corr - 0.31)
    diffautoy = abs(autoy - 0.84)
    diffautopi = abs(autopi - 0.48)
    sums=sum([diffvary, diffvarpi, diffcorr, diffautoy, diffautopi])
    if stat_out==True:                                             
        return vary, varpi, corr, autoy, autopi
    else:
        return sums

    
#-------------- Functions for part 3 ---------------



def demand_good1(betas,p1,p2,e1,e2,e3):
    '''
    This function calculates the demand of good 1 given the preferences,
    prices and initial endowments                        
                
    Parameters
    ----------
    betas : array
        Array of preferences of each good for each consumer
    p1 : float
        Price of good 1
    p2 : float
        Price of good 2
    e1 : array
        List of endowments of good 1 for each consumer
    e2 : array
        List of endowments of good 2 for each consumer
    e3 : array
        List of endowments of good 3 for each consumer
        
    Returns
    -------
    d : array
        Demand of good 1 for each consumer
    
    '''
        
    I=p1*e1+p2*e2+e3
    d=betas[:,0]*(I/p1)
    return d

def demand_good2(betas,p1,p2,e1,e2,e3):
    '''
    This function calculates the demand of good 2 given the preferences,
    prices and initial endowments                        
                
    Parameters
    ----------
    betas : array
        Array of preferences of each good for each consumer
    p1 : float
        Price of good 1
    p2 : float
        Price of good 2
    e1 : array
        List of endowments of good 1 for each consumer
    e2 : array
        List of endowments of good 2 for each consumer
    e3 : array
        List of endowments of good 3 for each consumer
        
    Returns
    -------
    d : array
        Demand of good 2 for each consumer
    
    '''    
    I=p1*e1+p2*e2+e3
    d=betas[:,1]*(I/p2)
    return d



def demand_good3(betas,p1,p2,e1,e2,e3):
    '''
    This function calculates the demand of good 3 given the preferences,
    prices and initial endowments                        
                
    Parameters
    ----------
    betas : array
        Array of preferences of each good for each consumer
    p1 : float
        Price of good 1
    p2 : float
        Price of good 2
    e1 : array
        List of endowments of good 1 for each consumer
    e2 : array
        List of endowments of good 2 for each consumer
    e3 : array
        List of endowments of good 3 for each consumer
        
    Returns
    -------
    d : array
        Demand of good 3 for each consumer
    
    '''    
    I=p1*e1+p2*e2+e3
    d=betas[:,2]*I
    return d




def ex_demand_good1(betas,p1,p2,e1,e2,e3):
    '''
    This function calculates the excess demand of good 1 given the preferences,
    prices and initial endowments                        
                
    Parameters
    ----------
    betas : array
        Array of preferences of each good for each consumer
    p1 : float
        Price of good 1
    p2 : float
        Price of good 2
    e1 : array
        List of endowments of good 1 for each consumer
    e2 : array
        List of endowments of good 2 for each consumer
    e3 : array
        List of endowments of good 3 for each consumer
        
    Returns
    -------
    excess : float
        Excess demand of good 1
    
    ''' 
    
    total_demand=np.sum(demand_good1(betas,p1,p2,e1,e2,e3))
    total_endow=np.sum(e1)
    excess = total_demand-total_endow
    return excess



def ex_demand_good2(betas,p1,p2,e1,e2,e3):
    '''
    This function calculates the excess demand of good 2 given the preferences,
    prices and initial endowments                        
                
    Parameters
    ----------
    betas : array
        Array of preferences of each good for each consumer
    p1 : float
        Price of good 1
    p2 : float
        Price of good 2
    e1 : array
        List of endowments of good 1 for each consumer
    e2 : array
        List of endowments of good 2 for each consumer
    e3 : array
        List of endowments of good 3 for each consumer
        
    Returns
    -------
    excess : float
        Excess demand of good 2
    
    ''' 
    total_demand=np.sum(demand_good2(betas,p1,p2,e1,e2,e3))
    total_endow=np.sum(e2)
    excess = total_demand-total_endow
    return excess

def ex_demand_good3(betas,p1,p2,e1,e2,e3):
    '''
    This function calculates the excess demand of good 3 given the preferences,
    prices and initial endowments                        
                
    Parameters
    ----------
    betas : array
        Array of preferences of each good for each consumer
    p1 : float
        Price of good 1
    p2 : float
        Price of good 2
    e1 : array
        List of endowments of good 1 for each consumer
    e2 : array
        List of endowments of good 2 for each consumer
    e3 : array
        List of endowments of good 3 for each consumer
        
    Returns
    -------
    excess : float
        Excess demand of good 3
    
    ''' 
    
    total_demand=np.sum(demand_good3(betas,p1,p2,e1,e2,e3))
    total_endow=np.sum(e3)
    excess = total_demand-total_endow
    return excess



def walras_taton(gp1,gp2,betas,e1,e2,e3,N,epsilon=0.0001,kappa=0.2,max_iters=10000):
    '''
    The first two arguments are guesses on the prices, epsilon is the deviation from
    a perfect equilibrium, that we are willing to accept.
    kappa is the adjustment parameter, and max_iters is the amount of loops
    that the function will perform before giving up.
                    
    Parameters
    ----------
    gp1 : float
        Initial guess of Walrasian equilibirum price of good 1
    gp2 : float
        Initial guess of Walrasian equilibirum price of good 2
    betas : array
        Array of preferences of each good for each consumer
    e1 : array
        List of endowments of good 1 for each consumer
    e2 : array
        List of endowments of good 2 for each consumer
    e3 : array
        List of endowments of good 3 for each consumer
    N : Float
        Amount of consumers
    epsilon : float (optional)
        The deviation of zero excess demand that is acceptable (default is 0.0001)
    kappa : float (optional)
        Adjustment aggressivity parameter (default is 0.2)
    max_iters : int (optional)
        Maximum amount of iterations (default is 10000)
        
    Returns
    -------
    gp1 : float
        Walrasian equilibrium price of good 1
    gp2: float
        Walrasian equilibrium price of good 2
    
    ''' 
    
    # We loop though step 2-4 given in the assignment.
    loops=0 # We are counting the iterations
    below_tol=False
    
    while below_tol==False and loops<max_iters+1: # While not having achieved eq. and under max_iters.
        
        loops+=1 # Incrementing number of iterations
        
        # First we calculate the excess demand in market 1 and 2 given the prices, that we guessed on.
        ex_d1=ex_demand_good1(betas,gp1,gp2,e1,e2,e3)
        ex_d2=ex_demand_good2(betas,gp1,gp2,e1,e2,e3)
        
        if abs(ex_d1)>epsilon or abs(ex_d2)>epsilon: # Checking if within tolerance
            gp1=gp1+kappa*ex_d1/N                    # If not, we adjust the prices
            gp2=gp2+kappa*ex_d2/N                   
        else:
            below_tol=True                           # If yes, we end the loop.
    
    if below_tol==True:
        print('Completed. \nThe Walras-equilibrium prices are: \n p1: '+str(gp1) + '\n p2: '+str(gp2))
        print('\nExcess demand in market 1: '+str(ex_demand_good1(betas,gp1,gp2,e1,e2,e3)))
        print('\nExcess demand in market 2: '+str(ex_demand_good2(betas,gp1,gp2,e1,e2,e3)))
        print('\nExcess demand in market 3: '+str(ex_demand_good3(betas,gp1,gp2,e1,e2,e3)))
    else:
        print('Could not find equlibirum within the maximum iterations')
        
    return gp1, gp2



def utilityexch(betas,p1,p2,e1,e2,e3,gamma):
    
    '''
    This function calculate the utility of each N consumer and the output is then a vector.
                    
    Parameters
    ----------
    betas : array
        Array of preferences of each good for each consumer
    p1 : float
        Price of good 1
    p2 : float
        Price of good 2
    e1 : array
        List of endowments of good 1 for each consumer
    e2 : array
        List of endowments of good 2 for each consumer
    e3 : array
        List of endowments of good 3 for each consumer
    gamma : float
        An additional parameter in the utility function
        
    Returns
    -------
    uts : array
        Vector of each consumers utility 
    
    '''
    
    uts=demand_good1(betas,p1,p2,e1,e2,e3)**(betas[:,0]*gamma) * demand_good2(betas,p1,p2,e1,e2,e3)**(betas[:,1]*gamma) * demand_good3(betas,p1,p2,e1,e2,e3)**(betas[:,2]*gamma)
    
    # We use the demand funtions already defined and use the provided utility function
    return uts

def utility_plot_stat(betas,p1,p2,e1,e2,e3,gamma,perc=False):
    
    '''
    This function uses utility() to compute a vector of the consumers utility and then
    plot a histogram of the distribution and compute the mean and variance of utility.
    The default is frequency plot, but you can set it to be percent by setting perc/=0
    
                        
    Parameters
    ----------
    betas : array
        Array of preferences of each good for each consumer
    p1 : float
        Price of good 1
    p2 : float
        Price of good 2
    e1 : array
        List of endowments of good 1 for each consumer
    e2 : array
        List of endowments of good 2 for each consumer
    e3 : array
        List of endowments of good 3 for each consumer
    gamma : float
        An additional parameter in the utility function
    perc : boolean (optional)
        Whether or not to plot a histogram of percentages instead of frequencies (default is False)
        
    Returns
    -------
    Histogram of utility
    
    '''
    
    
    ut_vec=utilityexch(betas,p1,p2,e1,e2,e3,gamma);
    if perc==False:
        plt.hist(ut_vec,bins=100)
        plt.ylabel('Frequency')
        plt.ylim(0,15000)
        plt.xlim(0,6)
        plt.title('Distribution of utility')
    else:
        plt.hist(ut_vec,weights=np.ones(len(ut_vec))/len(ut_vec), bins=100)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.ylabel('Percent')
        plt.ylim(0,1)
        plt.xlim(0,10)
        plt.title('Distribution of utility, $\gamma=$'+str(gamma))
    plt.xlabel('Utility')
    plt.show()
    print('\nMean of utility: '+str(np.mean(ut_vec)))
    print('\nVariance of utility: '+str(np.var(ut_vec)))