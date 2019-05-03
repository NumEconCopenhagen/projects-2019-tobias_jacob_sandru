import matplotlib.pyplot as plt

from .models import *



def fig_func(epsilon, optimal_h_list, optimal_h_list_1, net_wage, net_wage_1):
    fig = plt.figure(figsize=(8,8));
    ax = fig.add_subplot(1,1,1);
    ax.plot(optimal_h_list_1,net_wage_1,lw=2,linestyle='-',color='blue',label=f'{0.2*100} pct. tax',zorder=-1);
    ax.plot(optimal_h_list,net_wage,lw=2,linestyle='-',color='black',label=f'{0.5*100} pct. tax',zorder=-1);
    ax.set_ylim([0,50]);
    ax.set_xlim([0,1]);
    ax.legend(loc='lower right',fontsize=15);
    ax.set_xlabel('$h$',fontsize=20);
    ax.set_ylabel('Net wage $(w-t \cdot w)$',fontsize=20);
    ax.set_title(f'Labor supply \t $\epsilon$: {round(epsilon, 2)}',fontsize=30);
    return plt.show();