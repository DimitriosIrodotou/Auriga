import numpy as np
from label_format import *
from matplotlib.ticker import MultipleLocator

BOLTZMANN = 1.38065e-16
toinch = 0.393700787

suffix = 'pdf'


def fit_func(x, a, b):
    return a * x + b


def plot_mass_comparison(suffix):
    file = 'mass_diameter_scal.txt'
    
    mass = 1.0e9 * np.loadtxt(file, usecols=[2])
    massGK = 1.0e9 * np.loadtxt(file, usecols=[5])
    
    fileres = 'mass_diameter_res.txt'
    massres = 1.0e9 * np.loadtxt(fileres, usecols=[2])
    massGKres = 1.0e9 * np.loadtxt(fileres, usecols=[5])
    
    pxsize = 16.1
    pysize = 24.1
    
    psizex = 0.76
    psizey = 0.55
    
    fig = pylab.figure(figsize=(np.array([pxsize, pysize]) * toinch * 0.45), dpi=400)
    
    ax = axes([0.2, 0.38, psizex, psizey])
    ax2 = axes([0.2, 0.1, psizex, 0.46 * psizey])
    
    ax.set_xlim((1.0e8, 1.0e11))
    ax2.set_xlim((1.0e8, 1.0e11))
    ax.set_ylim((1.0e8, 1.0e11))
    ax2.set_ylim((-0.45, 0.1))
    
    xlim = ax2.get_xlim()
    
    d = np.linspace(1e8, 1e11, num=20, endpoint=True)
    ax.loglog(d, d, linestyle='dashed', color='lightgrey', lw=1.5)
    ax.loglog(massres[[0, 3]], massGKres[[0, 3]], marker='s', mfc='green', mec='None', ms=7, linestyle='None', label='lv5')
    ax.loglog(mass[:15], massGK[:15], marker='^', mfc='blue', mec='None', ms=7, linestyle='None')
    ax.loglog(mass[15], massGK[15], marker='s', mfc='blue', mec='None', ms=7, linestyle='None', label='lv4')
    ax.loglog(mass[16:23], massGK[16:23], marker='^', mfc='blue', mec='None', ms=7, linestyle='None')
    ax.loglog(mass[23], massGK[23], marker='s', mfc='blue', mec='None', ms=7, linestyle='None')
    ax.loglog(mass[24:], massGK[24:], marker='^', mfc='blue', mec='None', ms=7, linestyle='None')
    ax.loglog(massres[[2, 5]], massGKres[[2, 5]], marker='s', mfc='red', mec='None', ms=7, linestyle='None', label='lv3')
    
    ax2.hlines(0, xlim[0], xlim[1], linestyle='dashed', color='lightgrey', lw=1.5)
    ax2.semilogx(massres[[0, 3]], (massres[[0, 3]] - massGKres[[0, 3]]) / massres[[0, 3]], marker='s', mfc='green', mec='None', ms=7,
                 linestyle='None')
    ax2.semilogx(mass[:15], (mass[:15] - massGK[:15]) / mass[:15], marker='^', mfc='blue', mec='None', ms=7, linestyle='None')
    ax2.semilogx(mass[15], (mass[15] - massGK[15]) / mass[15], marker='s', mfc='blue', mec='None', ms=7, linestyle='None')
    ax2.semilogx(mass[16:23], (mass[16:23] - massGK[16:23]) / mass[16:23], marker='^', mfc='blue', mec='None', ms=7, linestyle='None')
    ax2.semilogx(mass[23], (mass[23] - massGK[23]) / mass[23], marker='s', mfc='blue', mec='None', ms=7, linestyle='None')
    ax2.semilogx(mass[24:], (mass[24:] - massGK[24:]) / mass[24:], marker='^', mfc='blue', mec='None', ms=7, linestyle='None')
    ax2.semilogx(massres[[2, 5]], (massres[[2, 5]] - massGKres[[2, 5]]) / massres[[2, 5]], marker='s', mfc='red', mec='None', ms=7, linestyle='None')
    
    fontsize = 8
    ax.legend(loc='lower right', fontsize=fontsize, frameon=False, numpoints=1)
    
    ax2.set_xlabel(r"${\rm M_{HI}^{L} \,[M_{\odot}]}$", fontsize=fontsize)
    ax.set_ylabel(r"${\rm M_{HI}^{GK} \,[M_{\odot}]}$", fontsize=fontsize)
    ax2.set_ylabel(r"${\rm \Delta M_{HI} / M_{HI}^{L}}$", fontsize=fontsize)
    
    for label in ax.xaxis.get_ticklabels():
        label.set_fontsize(fontsize)
    for label in ax.yaxis.get_ticklabels():
        label.set_fontsize(fontsize)
    for label in ax2.xaxis.get_ticklabels():
        label.set_fontsize(fontsize)
    for label in ax2.yaxis.get_ticklabels():
        label.set_fontsize(fontsize)
    
    ax.xaxis.set_major_formatter(NullFormatter())
    
    locator = MultipleLocator(0.05)
    ax2.yaxis.set_minor_locator(locator)
    mlocator = MultipleLocator(0.2)
    ax2.yaxis.set_major_locator(mlocator)
    
    fig.savefig('../plots/mass_comparison.%s' % (suffix), dpi=400, quality=95)


plot_mass_comparison(suffix)