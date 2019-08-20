import multipanel_layout
import numpy as np
from label_format import *

BOLTZMANN = 1.38065e-16
toinch = 0.393700787

suffix = 'pdf'


def ponomareva(x, a=-0.38, b=(1.06 + np.log10(4 / np.pi))):
    return a * x + b


def verheijen(x, a=-0.14, b=(0.7 + np.log10(4 / np.pi))):
    return a * x + b


def martinsson(x, a=-0.28, b=(0.92 + np.log10(4 / np.pi))):
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
    
    filelelli = 'Lelli.txt'
    masslelli = 1.0e9 * np.loadtxt(filelelli, usecols=[13])
    diameterlelli = 2. * np.loadtxt(filelelli, usecols=[14])
    masslelli /= 0.25e6 * np.pi * diameterlelli * diameterlelli
    
    figure = multipanel_layout.multipanel_layout(nrows=1, ncols=1, npanels=1, hspace=0.07, wspace=0.07, left=0.15, right=0.95, bottom=0.11, top=0.95,
                                                 scale=2)
    figure.set_figure_layout()
    figure.set_axis_limits_and_aspect(xlim=[10.0, 200.0], ylim=[1.0, 10.0], logaxis=True)
    figure.set_fontsize(8.)
    figure.set_axis_labels(xlabel=r"${\rm D_{HI}\,[kpc]}$", ylabel=r"${\rm \Sigma_{HI} \,[M_{\odot}\, pc^{-2}]}$")
    
    ax = figure.axes[0]
    
    d = np.linspace(10, 200, num=20, endpoint=True)
    ax.loglog(diameterlelli, masslelli, marker='h', mfc='None', mec='darkgrey', ms=5, linestyle='None')
    ax.loglog(d, 10. ** verheijen(np.log10(d)), linestyle='dashdot', color='darkgrey', lw=1.5)
    ax.loglog(d, 10. ** martinsson(np.log10(d)), linestyle='dashed', color='darkgrey', lw=1.5)
    ax.loglog(d, 10. ** ponomareva(np.log10(d)), linestyle='solid', color='darkgrey', lw=1.5)
    l1 = ax.legend(labels=['Lelli+ 2016', 'Verheijen & Sancisi 01', 'Martinsson+ 16', 'Ponomareva+ 16'], loc='lower left', frameon=False,
                   fontsize=figure.fontsize, numpoints=1)
    
    ax.loglog(diameter, sigma, marker='^', mfc='blue', mec='None', ms=8, linestyle='None', label='L08')
    ax.loglog(diameterGK, sigmaGK, marker='^', mfc='red', mec='None', ms=6, linestyle='None', label='GK11')
    
    ax.loglog(diameterres[[0, 3]], sigmares[[0, 3]], marker='o', mfc='blue', mec='None', ms=8, linestyle='None')
    ax.loglog(diameterres[[2, 5]], sigmares[[2, 5]], marker='s', mfc='blue', mec='None', ms=8, linestyle='None')
    
    ax.loglog(diameterGKres[[0, 3]], sigmaGKres[[0, 3]], marker='o', mfc='red', mec='None', ms=6, linestyle='None')
    ax.loglog(diameterGKres[[2, 5]], sigmaGKres[[2, 5]], marker='s', mfc='red', mec='None', ms=6, linestyle='None')
    
    ax.legend(loc='upper right', frameon=False, numpoints=1, fontsize=figure.fontsize)
    
    ax.add_artist(l1)
    
    ax.xaxis.set_major_formatter(UserLogFormatterMathtext())
    ax.yaxis.set_major_formatter(UserLogFormatterMathtext())
    
    figure.fig.savefig('../plots/sigma_diameter_obs.%s' % (suffix), dpi=400, quality=95)


plot_mass_diameter(suffix)