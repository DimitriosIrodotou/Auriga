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
    massGK = 1.0e9 * np.loadtxt(file, usecols=[5])
    sigma = np.loadtxt(file, usecols=[3])
    diameter = 2. * np.loadtxt(file, usecols=[1])
    sigmaGK = np.loadtxt(file, usecols=[6])
    diameterGK = 2. * np.loadtxt(file, usecols=[4])
    
    fileres = 'mass_diameter_res.txt'
    sigmares = np.loadtxt(fileres, usecols=[3])
    diameterres = 2. * np.loadtxt(fileres, usecols=[1])
    sigmaGKres = np.loadtxt(fileres, usecols=[6])
    diameterGKres = 2. * np.loadtxt(fileres, usecols=[4])
    
    popt, pcov = curve_fit(fit_func, np.log10(diameter), np.log10(mass))
    poptGK, pcovGK = curve_fit(fit_func, np.log10(diameterGK), np.log10(massGK))
    
    print
    popt
    
    figure = multipanel_layout.multipanel_layout(nrows=1, ncols=1, npanels=1, hspace=0.07, wspace=0.07, left=0.15, right=0.95, bottom=0.11, top=0.95,
                                                 scale=2)
    figure.set_figure_layout()
    figure.set_axis_limits_and_aspect(xlim=[10.0, 200.0], ylim=[1.0, 10.0], logaxis=True)
    figure.set_fontsize(8.)
    figure.set_axis_labels(xlabel=r"${\rm D_{HI}\,[kpc]}$", ylabel=r"${\rm \Sigma_{HI} \,[M_{\odot}\, pc^{-2}]}$")
    
    ax = figure.axes[0]
    
    ax.loglog(diameter, sigma, marker='^', mfc='blue', mec='None', ms=8, linestyle='None')
    ax.loglog(diameterGK, sigmaGK, marker='^', mfc='red', mec='None', ms=6, linestyle='None')
    
    ax.loglog(diameterres[[0, 3]], sigmares[[0, 3]], marker='o', mfc='blue', mec='None', ms=8, linestyle='None')
    ax.loglog(diameterres[[2, 5]], sigmares[[2, 5]], marker='s', mfc='blue', mec='None', ms=8, linestyle='None')
    
    ax.loglog(diameterGKres[[0, 3]], sigmaGKres[[0, 3]], marker='o', mfc='red', mec='None', ms=6, linestyle='None')
    ax.loglog(diameterGKres[[2, 5]], sigmaGKres[[2, 5]], marker='s', mfc='red', mec='None', ms=6, linestyle='None')
    
    l1 = ax.legend(labels=['L08', 'GK11'], loc='upper right', frameon=False, numpoints=1, fontsize=figure.fontsize)
    
    A = popt[0] - 2
    AGK = poptGK[0] - 2
    B = popt[1] + np.log10(4. / np.pi) - 6.
    BGK = poptGK[1] + np.log10(4. / np.pi) - 6.
    
    fitlab = "%.2f%s + %.2f" % (A, r'${\rm \,\,log\,\,D_{HI}}$', B)
    fitlabGK = "%.2f%s + %.2f" % (AGK, r'${\rm \,\,log\,\,D_{HI}}$', BGK)
    
    d = np.linspace(10, 200, num=20, endpoint=True)
    ax.loglog(d, 10. ** fit_func(np.log10(d), A, B), color='blue', linestyle='dashed', lw=1.5, label=fitlab)
    ax.loglog(d, 10. ** fit_func(np.log10(d), AGK, BGK), linestyle='dashed', color='red', lw=1.5, label=fitlabGK)
    
    ax.legend(loc='lower left', frameon=False, fontsize=figure.fontsize)
    ax.add_artist(l1)
    
    ax.xaxis.set_major_formatter(UserLogFormatterMathtext())
    ax.yaxis.set_major_formatter(UserLogFormatterMathtext())
    
    figure.fig.savefig('../plots/sigma_diameter.%s' % (suffix), dpi=400, quality=95)


plot_mass_diameter(suffix)