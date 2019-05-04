import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
import time
from scipy import linalg
from scipy import optimize
import sympy as sm

x = sm.symbols('x')             # consumption
h = sm.symbols('h')             # working share
epsilon = sm.symbols('epsilon') # elasticity
tw = sm.symbols('t_w')          # tax on labor income
w = sm.symbols('w')             # wage
t0 = sm.symbols('t_0')          # lump sump tax
a = sm.symbols('a')             # non-labor income

#define function for the budget constraint
def budget_cons(h,tw=0.2,w=10,a=2,t0=1):
    '''The budget constraint'''
    return (1-tw)*w*h-t0+a

#define function for the compensated budget constraint
def budget_cons_comp(h,u_b,u_a,tw=0.5,w=10,a=2,t0=1):
    '''The compensated budget constraint'''
    return (1-tw)*w*h-t0+a+(u_b-u_a)

#define function for the utility function
def utility_function(h,u,epsilon=-0.9):
    '''The utility function'''
    return u-(1/(1+1/epsilon))*(1-h)**(1+1/epsilon)



# _sol_func=sm.lambdify((tw,w,epsilon),sol[0])

# def sol_func(tw=0.2,w=10,epsilon=-0.9,a=2,t0=1):
#     '''
#     This function outputs the optimal fraction of time devoted to labor, consumption
#     and last but not least the utility derived given the optimum.
#     '''
#     opt_h=_sol_func(tw,w,epsilon)
#     opt_x=(1-tw)*w*opt_h-t0+a
#     opt_u=opt_x+1/(1+1/epsilon)*(1-opt_h)**(1+1/epsilon)
#     return opt_h, opt_x, opt_u