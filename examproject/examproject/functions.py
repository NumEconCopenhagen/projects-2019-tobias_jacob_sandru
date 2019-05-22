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
h_vec = np.linspace(0.1,1.5,100) 
rho = 2
beta = 0.96
gamma = 0.1
w = 2
b = 1
Delta = 0.1
def consum(w, h, l,b):
    '''
    Computes consumption based on wage, human capital, labour supply and benefit
    '''
    return w*h*l+(1-l)*b

def utility(w,h,l,rho,b,gamma):
    '''
    Computes total utility from the consumption and diutility from labour supply
    '''
    return consum(w, h, l,b)**(1-rho)/(1-rho)-gamma*l

def v2(w,h2,l2,rho,b,gamma):
    '''
    Computes utility from period 2
    '''
    return utility(w,h2,l2,rho,b,gamma)

def v1(l1,h1,rho,beta,Delta,b,w,gamma,v2_interp,accum):
    '''
    Computes utility from period 1 where the humancapital accumulates to period 2,
        if one have supplied labour in period 1. The accumulation includes a stochastic
        term. The functoin also enables the user to turn
        of accumulation.
    '''
    # Expected utility in period 2
    h2_low = h1+l1
    h2_high = h1+l1+Delta
    if accum==0:
        h2_high=h1
        h2_low=h1
    
    v2_low = v2_interp([h2_low])[0]
    v2_high = v2_interp([h2_high])[0]
    
    
    # Expected v2 value
    v2 = 0.5*v2_low + 0.5*v2_high
    
    # d. total value
    return utility(w,h1,l1,rho,b,gamma) + beta*v2

def solve_period_2(w,rho,b,gamma):
    '''
    Maximizes utility in period 2
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

def solve_period_1(rho,beta,Delta,b,w,gamma,v2_interp,accum):
    '''
    Maximizes utility in period 1.
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

def solve(w,rho,b,gamma,Delta,accum=1):
    '''
    A function that first solves problem in period 2 for any level of human capital,
        and then computing the period 1 problem with any level of human capital
        while accounting for the accumulation.
    '''
    #Solving for period 2 first
    h2_vec,v2_vec,l2_vec,c2_vec=solve_period_2(w,rho,b,gamma)
    
    #Creating a interpolated matrix of equilibriums from the problem of period 2
    v2_interp = interpolate.RegularGridInterpolator((h2_vec,), v2_vec,bounds_error=False,fill_value=None)
    
    #Using the results from the period 2 problem to optimize in period 1
    h1_vec,v1_vec,l1_vec,c1_vec=solve_period_1(rho,beta,Delta,b,w,gamma,v2_interp,accum)
    
    return h1_vec,v1_vec,l1_vec,c1_vec

    
#-------------- Functions for part 2 ---------------

def single_shock(T,v_shock=0,s_shock=0,alphas=5.76,hs=0.5,bs=0.5,phis=0,gammas=0.075,deltas=0.8,omegas=0.15):
    '''
    This functions enables the user to produce and plot impulse response functions following either a
    demand or supply shock. Both are set to a default value of zero.
    The user is also able to alter the value of the parameter, where the default are the ones given
    in the assignment
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
    
    
def stochproc(T,s_on=0,v_on=0,sigmax=3.492,sigmac=0.2,alphas=5.76,hs=0.5,bs=0.5,phis=0,gammas=0.075,deltas=0.8,omegas=0.15, plot=1):
    '''
    This functions enables the user to produce and plot impulse response functions following either a
    demand or supply stochastic process. Both are set to a default value of off.
    The user is also able to alter the value of the parameter, where the default are the ones given
    in the assignment
    
    You're are also able to turn off the plot, the default is on.
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
    if v_on==1:
        xlist=np.random.normal(0,par['sigma_x'],T)
    else:
        xlist=np.zeros(T)
    if s_on==1:
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
    if plot==1:
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


def corr(phi,corr_out=0):
    '''
    This function calculates computes the corr.coeff. for a given phi between inflation and output.
    Then it computes the absolute diffence betweem the obatined coeff. and 0.31.
    
    The user is able to actively choose to compute both the absolute difference and the obtained
    corr.coeff.
    This is useful when wanting to use the absolute diffence in order to minimize the difference.
    Having to output to the objective function restricts using optimize.
    '''
    y_bc, pi_bc = stochproc(1000,v_on=1,s_on=1,phis=phi,plot=0) # Computing inflation and output lists.
    corr=float(np.corrcoef(y_bc, pi_bc)[0,1])                          # Computing corr.coeff.
    diff = float(abs(corr - 0.31))                              # Computing abs. diff. between corr.coeff. and 0.31
    if corr_out==1:                                             
        return diff,corr
    else:
        return diff
    
    
def us_econ(stat,stat_out=0):
    '''
    This function calculates computes the corr.coeff. for a given phi, sigmax and sigmac
    between inflation and output.
    Then it computes the absolute diffences betweem the obatined coeff. and the corresponding
    value for the US economy.
    
    The user is able to actively choose to compute both the absolute difference and the obtained
    corr.coeff.
    This is useful when wanting to use the absolute diffence in order to minimize the difference.
    Having to output to the objective function restricts using optimize.
    '''
    phi,sigmax_,sigmac_=stat
    
    # Computing inflation and output lists.
    y_bc, pi_bc = stochproc(1000,v_on=1,s_on=1,phis=phi,sigmax=sigmax_,sigmac=sigmac_,plot=0)
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
    if stat_out==1:                                             
        return vary, varpi, corr, autoy, autopi
    else:
        return sum([diffvary, diffvarpi, diffcorr, diffautoy, diffautopi])

    
#-------------- Functions for part 3 ---------------

#The price of Good 3 is numeraire, so p3=1.
def demand_good1(betas,p1,p2,e1,e2,e3):
    I=p1*e1+p2*e2+e3
    return betas[:,0]*(I/p1)

def demand_good2(betas,p1,p2,e1,e2,e3):
    I=p1*e1+p2*e2+e3
    return betas[:,1]*(I/p2)

def demand_good3(betas,p1,p2,e1,e2,e3):
    I=p1*e1+p2*e2+e3
    return betas[:,2]*I


# We simple use the equations given in the assigment
def ex_demand_good1(betas,p1,p2,e1,e2,e3):
    
    total_demand=np.sum(demand_good1(betas,p1,p2,e1,e2,e3))
    total_endow=np.sum(e1)   
    return total_demand-total_endow

def ex_demand_good2(betas,p1,p2,e1,e2,e3):
    
    total_demand=np.sum(demand_good2(betas,p1,p2,e1,e2,e3))
    total_endow=np.sum(e2)
    return total_demand-total_endow

def ex_demand_good3(betas,p1,p2,e1,e2,e3):
    
    total_demand=np.sum(demand_good3(betas,p1,p2,e1,e2,e3))
    total_endow=np.sum(e3)
    return total_demand-total_endow


def walras_taton(gp1,gp2,betas,e1,e2,e3,N,epsilon=0.0001,kappa=0.2,max_iters=10000):
    '''
    The first two arguments are guesses on the prices, epsilon is the deviation from a perfect equilibrium,
    that we are willing to accept. kappa is the adjustment parameter, and max_iters is the amount of loops
    that the function will perform before giving up.
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
    '''
    
    # We use the demand funtions already defined and use the provided utility function
    return demand_good1(betas,p1,p2,e1,e2,e3)**(betas[:,0]*gamma) * demand_good2(betas,p1,p2,e1,e2,e3)**(betas[:,1]*gamma) * demand_good3(betas,p1,p2,e1,e2,e3)**(betas[:,2]*gamma)

def utility_plot_stat(betas,p1,p2,e1,e2,e3,gamma,perc=0):
    
    '''
    This function uses utility() to compute a vector of the consumers utility and then
    plot a histogram of the distribution and compute the mean and variance of utility.
    The default is frequency plot, but you can set it to be percent by setting perc/=0
    '''
    
    
    ut_vec=utilityexch(betas,p1,p2,e1,e2,e3,gamma);
    if perc==0:
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