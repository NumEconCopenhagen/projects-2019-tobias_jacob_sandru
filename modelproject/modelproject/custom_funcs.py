import sympy as sm



    #Solving function
def sol_func(sol,tw=0.2,w=10,epsilon=-0.5,a=2,t0=1):
    
    def _sol_func(sol,tw=0.2,w=10,epsilon=-0.5):
        res=sm.lambdify((tw,w,epsilon),sol[0])
        return res

    opt_h=_sol_func(sol,tw,w,epsilon)
    opt_x=(1-tw)*w*opt_h-t0+a
    opt_u=opt_x+1/(1+1/epsilon)*(1-opt_h)**(1+1/epsilon)
    return opt_h, opt_x, opt_u

#define function for the budget constraint
def budget_cons(h,tw=0.2,w=10,a=2,t0=1):
    return (1-tw)*w*h-t0+a

#define function for the compensated budget constraint
def budget_cons_comp(h,u_b,u_a,tw=0.5,w=10,a=2,t0=1):
    return (1-tw)*w*h-t0+a+(u_b-u_a)

#define function for the utility function
def utility(h,u,epsilon=-0.5):
    return u-(1/(1+1/epsilon))*(1-h)**(1+1/epsilon)


#make the utility function into a python function
def _utility_func(h):
    return (1-0.2)*10*h-1+2+(1/(1+1/(-0.5)))*(1-h)**(1+1/(-0.5))
def utility_func(x):
    #add a -1 to convert the maximization problem into a minimization problem
    return -1*_utility_func(x[0])