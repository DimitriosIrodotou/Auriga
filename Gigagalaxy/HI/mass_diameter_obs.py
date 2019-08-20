import multipanel_layout
import numpy as np
from label_format import *

BOLTZMANN = 1.38065e-16
toinch = 0.393700787

suffix = 'pdf'


def ponomareva(x, a=1.62, b=7.06):
    return a * x + b


def verheijen(x, a=1.86, b=6.7):
    return a * x + b


def martinsson(x, a=1.72, b=6.92):
    return a * x + b


def broeils(x, a=1.96, b=6.52):
    return a * x + b


def plot_mass_diameter_obs(suffix):
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
    
    filelelli = 'Lelli.txt'
    masslelli = 1.0e9 * np.loadtxt(filelelli, usecols=[13])
    diameterlelli = 2. * np.loadtxt(filelelli, usecols=[14])
    
    figure = multipanel_layout.multipanel_layout(nrows=1, ncols=1, npanels=1, hspace=0.07, wspace=0.07, left=0.16, right=0.95, bottom=0.11, top=0.95,
                                                 scale=2)
    figure.set_figure_layout()
    figure.set_axis_limits_and_aspect(xlim=[10.0, 200.0], ylim=[1.0e8, 1.0e11], logaxis=True)
    figure.set_fontsize(8.)
    figure.set_axis_labels(xlabel=r"${\rm D_{HI}\,[kpc]}$", ylabel=r"${\rm M_{HI} \,[M_{\odot}]}$")
    
    ax = figure.axes[0]
    
    d = np.linspace(10, 200, num=20, endpoint=True)
    ax.loglog(diameterlelli, masslelli, marker='h', mfc='None', mec='darkgrey', ms=5, linestyle='None')
    ax.loglog(d, 10. ** verheijen(np.log10(d)), linestyle='dashdot', color='darkgrey', lw=1.5)
    ax.loglog(d, 10. ** martinsson(np.log10(d)), linestyle='dashed', color='darkgrey', lw=1.5)
    ax.loglog(d, 10. ** ponomareva(np.log10(d)), linestyle='solid', color='darkgrey', lw=1.5)
    # ax.loglog(d, 10.**broeils(np.log10(d)), linestyle='dotted', color='darkgrey', lw=1.5)
    l1 = ax.legend(labels=['Lelli+ 16', 'Verheijen & Sancisi 01', 'Martinsson+ 16', 'Ponomareva+ 16'], loc='lower right', frameon=False,
                   fontsize=figure.fontsize, numpoints=1)
    
    ax.loglog(diameter, mass, marker='^', mfc='blue', mec='None', ms=8, linestyle='None', label='L08')
    ax.loglog(diameterGK, massGK, marker='^', mfc='red', mec='None', ms=6, linestyle='None', label='GK11')
    
    ax.loglog(diameterres[[0, 3]], massres[[0, 3]], marker='o', mfc='blue', mec='None', ms=8, linestyle='None')
    ax.loglog(diameterres[[2, 5]], massres[[2, 5]], marker='s', mfc='blue', mec='None', ms=8, linestyle='None')
    
    ax.loglog(diameterGKres[[0, 3]], massGKres[[0, 3]], marker='o', mfc='red', mec='None', ms=6, linestyle='None')
    ax.loglog(diameterGKres[[2, 5]], massGKres[[2, 5]], marker='s', mfc='red', mec='None', ms=6, linestyle='None')
    
    ax.legend(loc='upper left', frameon=False, numpoints=1, fontsize=figure.fontsize)
    
    ax.set_xlabel(r"${\rm D_{HI}\,[kpc]}$")
    ax.set_ylabel(r"${\rm M_{HI} \,[M_{\odot}]}$")
    
    ax.add_artist(l1)
    
    ax.xaxis.set_major_formatter(UserLogFormatterMathtext())
    
    figure.fig.savefig('../plots/mass_diameter_obs.%s' % (suffix), dpi=400, quality=95)


plot_mass_diameter_obs(suffix)