import multipanel_layout
import numpy as np
from label_format import *
from scipy.optimize import curve_fit

BOLTZMANN = 1.38065e-16
toinch = 0.393700787

suffix = 'pdf'


def fit_func(x, a, b):
    return a * x + b


def plot_mass_diameter(suffix):
    file = 'mass_diameter_scal.txt'
    mass = 1.0e9 * np.loadtxt(file, usecols=[2])
    diameter = 2. * np.loadtxt(file, usecols=[1])
    massGK = 1.0e9 * np.loadtxt(file, usecols=[5])
    diameterGK = 2. * np.loadtxt(file, usecols=[4])
    
    fileres = 'mass_diameter_res.txt'
    massres = 1.0e9 * np.loadtxt(fileres, usecols=[2])
    diameterres = 2. * np.loadtxt(fileres, usecols=[1])
    massGKres = 1.0e9 * np.loadtxt(fileres, usecols=[5])
    diameterGKres = 2. * np.loadtxt(fileres, usecols=[4])
    
    popt, pcov = curve_fit(fit_func, np.log10(diameter), np.log10(mass))
    poptGK, pcovGK = curve_fit(fit_func, np.log10(diameterGK), np.log10(massGK))
    
    print
    popt
    
    figure = multipanel_layout.multipanel_layout(nrows=1, ncols=1, npanels=1, hspace=0.07, wspace=0.07, left=0.16, right=0.95, bottom=0.11, top=0.95,
                                                 scale=2)
    figure.set_figure_layout()
    figure.set_axis_limits_and_aspect(xlim=[10.0, 200.0], ylim=[1.0e8, 1.0e11], logaxis=True)
    figure.set_fontsize(8.)
    figure.set_axis_labels(xlabel=r"${\rm D_{HI}\,[kpc]}$", ylabel=r"${\rm M_{HI} \,[M_{\odot}]}$")
    
    ax = figure.axes[0]
    
    d = np.linspace(10, 200, num=20, endpoint=True)
    ax.loglog(d, 10. ** fit_func(np.log10(d), popt[0], popt[1]), color='blue', linestyle='dashed', lw=1.5)
    ax.loglog(d, 10. ** fit_func(np.log10(d), poptGK[0], poptGK[1]), linestyle='dashed', color='red', lw=1.5)
    
    fitlab = "%.2f%s + %.2f" % (popt[0], r'${\rm \,\,log\,\,D_{HI}}$', popt[1])
    fitlabGK = "%.2f%s + %.2f" % (poptGK[0], r'${\rm \,\,log\,\,D_{HI}}$', poptGK[1])
    l1 = ax.legend(labels=[fitlab, fitlabGK], loc='lower right', frameon=False, fontsize=figure.fontsize)
    
    ax.loglog(diameter, mass, marker='^', mfc='blue', mec='None', ms=8, linestyle='None', label='L08')
    ax.loglog(diameterGK, massGK, marker='^', mfc='red', mec='None', ms=6, linestyle='None', label='GK11')
    
    ax.loglog(diameterres[[0, 3]], massres[[0, 3]], marker='o', mfc='blue', mec='None', ms=8, linestyle='None')
    ax.loglog(diameterres[[2, 5]], massres[[2, 5]], marker='s', mfc='blue', mec='None', ms=8, linestyle='None')
    
    ax.loglog(diameterGKres[[0, 3]], massGKres[[0, 3]], marker='o', mfc='red', mec='None', ms=6, linestyle='None')
    ax.loglog(diameterGKres[[2, 5]], massGKres[[2, 5]], marker='s', mfc='red', mec='None', ms=6, linestyle='None')
    
    ax.legend(loc='upper left', frameon=False, numpoints=1, fontsize=figure.fontsize)
    ax.add_artist(l1)
    
    ax.xaxis.set_major_formatter(UserLogFormatterMathtext())
    
    figure.fig.savefig('../plots/mass_diameter.%s' % (suffix), dpi=400, quality=95)


plot_mass_diameter(suffix)