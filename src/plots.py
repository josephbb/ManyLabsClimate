import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pymc as pm
from scipy.special import expit, logit
import itertools
def flip(items, ncol):
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])



# Define the list of category names
categories = ['SystemJust', 'SciConsens', 'CollectAction', 'Control', 'NegativeEmotions', 'LetterFuture', 'PluralIgnorance', 'PsychDistance', 'FutureSelfCont', 'NormativeAppeal', 'BindingMoral', 'DynamicNorm', 'aaControl']

# Use the specified color palette
sns.set_palette([
    "#0c344a",  # blue
    "#C3A200",  # orange
    "#009E73",  # green
    "#F0E442",  # yellow
    "#CC79A7",  # pink
    "#6bcaff",  # light blue
    "#B44900",  # red-orange
    "#CC6633",  # brown
    "#999999",  # gray
    "#A8C8E4",  # pale blue
    "#CCB974",  # tan
    "#64B5CD"   # dark blue
])

# Use seaborn's color_palette() method to get the colors in the specified palette
colors = sns.color_palette()

# Bind the category names to the colors in a dictionary
palette = dict(zip(categories, colors))

# Set the color of 'aaControl' to be the same as 'Control'
palette['aaControl'] = palette['Control']
treatment_pal = palette


def plot_by_belief(sim_model, idata_sim,
                   var_name, treatments, 
                   control_idx, 
                   avg=None,
                   ax=None,
                   res=100, pal=None,
                   x_sim_low=0.02,
                   x_sim_high=0.98,
                   convert_func = lambda x: x,
                   country_name='country_sim', 
                   treatment_name='treatment_sim',
                   ylim=(-10,10),
                   mult=100,
                   ylabel='None',relabel=None):
    with sim_model:
        x_sim = np.linspace(x_sim_low, x_sim_high, res)
        x_sim = logit(x_sim)
        plt.sca(ax)
        pm.set_data({country_name:np.repeat(0, res), 
            treatment_name:np.repeat(control_idx, res), 
            'bool_country':np.repeat(0, res),
            'sim_belief':x_sim})
        ppc = pm.sample_posterior_predictive(idata_sim, var_names=[var_name])
        control = mult*convert_func(ppc.posterior_predictive[var_name])
        for idx in range(0,12):
            legend = treatments[idx]
            color = treatment_pal[legend]

            pm.set_data({country_name:np.repeat(1, res), 
                            treatment_name:np.repeat(idx, res), 
                            'bool_country':np.repeat(0, res),
                            'sim_belief':x_sim})
            temp = pm.sample_posterior_predictive(idata_sim, var_names=[var_name])
            treatment_belief = mult*convert_func(temp.posterior_predictive[var_name])
            above = np.where(np.percentile(treatment_belief-control, 3, axis=(0,1)) > 0)[0]
            below = np.where(np.percentile(treatment_belief-control, 97, axis=(0,1)) < 0)[0]
            neither = np.where(np.logical_and(np.percentile(treatment_belief-control, 3, axis=(0,1)) < 0, 
                                            np.percentile(treatment_belief-control, 97, axis=(0,1)) > 0))[0]
            if len(above) > 1:
                plt.plot(100*convert_func(x_sim[above]), np.median(treatment_belief-control, axis=(0,1))[above], color=color, alpha=1, label=legend, zorder=3)
                plt.fill_between(100*convert_func(x_sim[above]), np.percentile(treatment_belief-control, 3, axis=(0,1))[above],
                                np.percentile(treatment_belief-control, 97, axis=(0,1))[above], linewidth=0, alpha=0.04, color=color, zorder=2)
                legend=None
            if len(below) > 1:
                plt.plot(100*convert_func(x_sim[below]), np.median(treatment_belief-control, axis=(0,1))[below], color=color,label=legend, alpha=1,zorder=3)
                plt.fill_between(100*convert_func(x_sim[below]), np.percentile(treatment_belief-control, 3, axis=(0,1))[below],
                                np.percentile(treatment_belief-control, 97, axis=(0,1))[below], linewidth=0, color=color, alpha=0.04, zorder=2)  
            if (len(below) > 1) & (len(above) > 1):
                if len(neither) > 0:
                    plt.plot(100*convert_func(x_sim[neither.min()-1:neither.max()+1]), 
                            np.median(treatment_belief-control, axis=(0,1))[neither.min()-1:neither.max()+1], color=color, alpha=0.5, ls='--',zorder=3)
                    plt.fill_between(100*convert_func(x_sim[neither.min()-1:neither.max()+1]),
                                    np.percentile(treatment_belief-control, 3, axis=(0,1))[neither.min()-1:neither.max()+1],
                                    np.percentile(treatment_belief-control, 97, axis=(0,1))[neither.min()-1:neither.max()+1], 
                                    linewidth=0, color=color, alpha=0.02, zorder=2)
                
        handles, labels = plt.gca().get_legend_handles_labels()
        if relabel is not None:
            labels = relabel(labels)
        plt.legend(flip(handles, 2), flip(labels, 2),loc=4, ncol=2,title='Intervention',prop={'size': 8})

        avg = avg
        plt.ylim(ylim)
        plt.xlim(0,100)
        plt.plot([avg, avg],plt.ylim(), ls='--', color='k', zorder=0, alpha=0.5)
        plt.plot([0, 100], [0, 0], color='k')
        plt.xlabel('Initial Belief (%)')
        plt.ylabel(ylabel)




def plot_country_forest(idata, ls, mean, ax, xlabel='Sharing(%)', wept=False): 
    plt.sca(ax)
    
    if wept:
        az.plot_forest(idata, filter_vars='like', 
                        linewidth=1.5, markersize=6, combined=True,textsize=9, colors='crimson', ax=ax)
    else:
        az.plot_forest(idata.sel({'countries':ls}), filter_vars='like', 
                    linewidth=1.5, markersize=6, combined=True,textsize=9, colors='crimson', ax=ax)

    locs, labels =plt.yticks()
    plt.yticks(locs, ls[::-1])
    plt.plot([np.mean(mean), np.mean(mean)],plt.ylim(), ls='--', color='k', zorder=0)
    plt.fill_betweenx(plt.ylim(), np.percentile(mean, 2.5), np.percentile(mean, 97.5), color='k', linewidth=0, alpha=0.2, zorder=0)
    plt.title('')
    plt.xlim(0,100)
    plt.xlabel(xlabel, fontsize=16)
    plt.tight_layout()
    
def plot_ATE_data(idata, ls, ax, xlabel='Sharing(%)', xlim=(-15,15), xticks=[-15,-10,-5,0,5,10,15], color='green'):
    plt.sca(ax)
    treatments = idata.coords["treatments"].values
    az.plot_forest( [idata.sel({"treatments": t}) for t in treatments], linewidth=4, 
                   textsize=10, markersize=10, combined=True, colors=color, legend=False, ax=ax)
    locs, labels =plt.yticks()
    ax.set_facecolor('white')
    ylim = plt.ylim()
    plt.yticks(locs,ls[::-1])
    plt.title('')
    plt.xlim(xlim)
    plt.fill_between(plt.xlim(),[35, 35],color='white', zorder=1)
    plt.plot([0, 0],plt.ylim(), '--', color='k', zorder=1)
    plt.xticks(xticks)
    plt.xlabel(xlabel)
        
