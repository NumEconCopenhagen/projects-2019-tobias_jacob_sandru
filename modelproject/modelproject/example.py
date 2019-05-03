def fig_func(epsilon):
    fig = plt.figure(figsize=(8,8));
    ax = fig.add_subplot(1,1,1);
    wage_list = list(range(1,50))
    optimal_h_list = []
    optimal_h_list_1 = []
    net_wage = []
    net_wage_1 = []

    for wage in wage_list:
        if round(epsilon,2) == -1:
            continue
        else:
            opt_h, opt_x, opt_u = sol_func(tw=0.2, w=wage, epsilon=round(epsilon,2));
            optimal_h_list.append(opt_h);
            net_wage.append(wage-wage*0.2);
    for wage in wage_list:
        if round(epsilon,2) == -1:
            continue
        else:
            opt_h_1, opt_x_1, opt_u_1 = sol_func(tw=0.5, w=wage, epsilon=round(epsilon,2));
            optimal_h_list_1.append(opt_h_1);
            net_wage_1.append(wage-wage*0.2);
    ax.plot(optimal_h_list_1,net_wage_1,lw=2,linestyle='-',color='blue',label=f'{0.2*100} pct. tax',zorder=-1);
    ax.plot(optimal_h_list,net_wage,lw=2,linestyle='-',color='black',label=f'{0.5*100} pct. tax',zorder=-1);
    ax.set_ylim([0,50]);
    ax.set_xlim([0,1]);
    ax.legend(loc='lower right',fontsize=15);
    ax.set_xlabel('$h$',fontsize=20);
    ax.set_ylabel('Net wage $(w-t \cdot w)$',fontsize=20);
    ax.set_title(f'Labor supply \t $\epsilon$: {round(epsilon, 2)}',fontsize=30);
    return plt.show();