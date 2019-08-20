import multipanel_layout
import numpy as np
from label_format import *
from scipy.optimize import curve_fit

BOLTZMANN = 1.38065e-16
toinch = 0.393700787

suffix = 'pdf'

colors = ['blue', 'red']


def fit_func(x, a, b):
    return a * x + b


def plot_mass_diameter(files, suffix):
    figure = multipanel_layout.multipanel_layout(nrows=1, ncols=1, npanels=1, hspace=0.07, wspace=0.07, left=0.15, right=0.95, bottom=0.11, top=0.95,
                                                 scale=2)
    figure.set_figure_layout()
    figure.set_axis_limits_and_aspect(xlim=[0.5, 20.0], ylim=[0.8, 20], logaxis=True)
    figure.set_fontsize(8.)
    figure.set_axis_labels(xlabel=r"${\rm SFR\,[M_\odot\,yr^{-1}]}$", ylabel=r"${\rm h_{HI} \,[kpc]}$")
    
    ax = figure.axes[0]
    
    sfr = np.loadtxt(files[0], usecols=[6])
    hz = np.loadtxt(files[0], usecols=[3])
    sfrGK = np.loadtxt(files[1], usecols=[6])
    hzGK = np.loadtxt(files[1], usecols=[3])
    
    popt, pcov = curve_fit(fit_func, np.log10(sfr), np.log10(hz))
    poptGK, pcovGK = curve_fit(fit_func, np.log10(sfrGK), np.log10(hzGK))
    
    print
    popt
    print
    poptGK
    
    d = np.linspace(0.1, 20, num=20, endpoint=True)
    
    ax.loglog(d, 10. ** fit_func(np.log10(d), popt[0], popt[1]), color='blue', linestyle='dashed', lw=1.5)
    ax.loglog(d, 10. ** fit_func(np.log10(d), poptGK[0], poptGK[1]), color='red', linestyle='dashed', lw=1.5)
    fitlab = "%.2f%s + %.3f" % (popt[0], r'${\rm \,\,log\,\,SFR}$', popt[1])
    fitlabGK = "%.2f%s + %.3f" % (poptGK[0], r'${\rm \,\,log\,\,SFR}$', poptGK[1])
    l1 = ax.legend(labels=[fitlab, fitlabGK], loc='upper left', frameon=False, fontsize=figure.fontsize)
    
    ax.loglog(sfr, hz, marker='^', mfc='blue', mec='None', ms=8, linestyle='None', label='L08')
    ax.loglog(sfrGK, hzGK, marker='^', mfc='red', mec='None', ms=8, linestyle='None', label='GK11')
    
    ax.legend(loc='lower right', frameon=False, fontsize=figure.fontsize, numpoints=1)
    ax.add_artist(l1)
    
    ax.xaxis.set_major_formatter(UserLogFormatterMathtext())
    ax.yaxis.set_major_formatter(UserLogFormatterMathtext())
    
    # tok = file.split('-')[1].split('.')[0]
    figure.fig.savefig('../plots/height_sfr.%s' % (suffix), dpi=400, quality=95)


# file = 'disc-thick.txt'
files = ['disc-thick.txt', 'disc-thickGK.txt']
plot_mass_diameter(files, suffix)