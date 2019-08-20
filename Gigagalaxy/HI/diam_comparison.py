import numpy as np
from label_format import *
from matplotlib.ticker import MultipleLocator

BOLTZMANN = 1.38065e-16
toinch = 0.393700787

suffix = 'pdf'


def fit_func(x, a, b):
    return a * x + b


def plot_diam_comparison(suffix):
    file = 'mass_diameter_scal.txt'
    
    diam = 2. * np.loadtxt(file, usecols=[1])
    diamGK = 2. * np.loadtxt(file, usecols=[4])
    
    fileres = 'mass_diameter_res.txt'
    diameterres = 2. * np.loadtxt(fileres, usecols=[1])
    diameterGKres = 2. * np.loadtxt(fileres, usecols=[4])
    
    pxsize = 16.1
    pysize = 24.1
    
    psizex = 0.76
    psizey = 0.55
    
    fig = pylab.figure(figsize=(np.array([pxsize, pysize]) * toinch * 0.45), dpi=400)
    
    ax = axes([0.21, 0.38, psizex, psizey])
    ax2 = axes([0.21, 0.1, psizex, 0.46 * psizey])
    
    ax.set_xlim((1.0e1, 2.0e2))
    ax2.set_xlim((1.0e1, 2.0e2))
    ax2.set_ylim((-0.06, 0.018))
    ax.set_ylim((1.0e1, 2.0e2))
    
    xlim = ax2.get_xlim()
    
    d = np.linspace(1e1, 2e2, num=20, endpoint=True)
    ax.loglog(d, d, linestyle='dashed', color='lightgrey', lw=1.5)
    ax.loglog(diameterres[[0, 3]], diameterGKres[[0, 3]], marker='s', mfc='green', mec='None', ms=7, linestyle='None', label='lv5')
    ax.loglog(diam[:15], diamGK[:15], marker='^', mfc='blue', mec='None', ms=7, linestyle='None')
    ax.loglog(diam[15], diamGK[15], marker='s', mfc='blue', mec='None', ms=7, linestyle='None', label='lv4')
    ax.loglog(diam[16:23], diamGK[16:23], marker='^', mfc='blue', mec='None', ms=7, linestyle='None')
    ax.loglog(diam[23], diamGK[23], marker='s', mfc='blue', mec='None', ms=7, linestyle='None')
    ax.loglog(diam[24:], diamGK[24:], marker='^', mfc='blue', mec='None', ms=7, linestyle='None')
    ax.loglog(diameterres[[2, 5]], diameterGKres[[2, 5]], marker='s', mfc='red', mec='None', ms=7, linestyle='None', label='lv3')
    
    ax2.hlines(0, xlim[0], xlim[1], linestyle='dashed', color='lightgrey', lw=1.5)
    ax2.semilogx(diameterres[[0, 3]], (diameterres[[0, 3]] - diameterGKres[[0, 3]]) / diameterres[[0, 3]], marker='^', mfc='green', mec='None', ms=7,
                 linestyle='None')
    ax2.semilogx(diam[:15], (diam[:15] - diamGK[:15]) / diam[:15], marker='^', mfc='blue', mec='None', ms=7, linestyle='None')
    ax2.semilogx(diam[15], (diam[15] - diamGK[15]) / diam[15], marker='s', mfc='blue', mec='None', ms=7, linestyle='None')
    ax2.semilogx(diam[16:23], (diam[16:23] - diamGK[16:23]) / diam[16:23], marker='^', mfc='blue', mec='None', ms=7, linestyle='None')
    ax2.semilogx(diam[23], (diam[23] - diamGK[23]) / diam[23], marker='s', mfc='blue', mec='None', ms=7, linestyle='None')
    ax2.semilogx(diam[24:], (diam[24:] - diamGK[24:]) / diam[24:], marker='^', mfc='blue', mec='None', ms=7, linestyle='None')
    ax2.semilogx(diameterres[[2, 5]], (diameterres[[2, 5]] - diameterGKres[[2, 5]]) / diameterres[[2, 5]], marker='^', mfc='red', mec='None', ms=7,
                 linestyle='None')
    
    fontsize = 8
    ax.legend(loc='lower right', fontsize=fontsize, frameon=False, numpoints=1)
    
    ax2.set_xlabel(r"${\rm D_{HI}^{L} \,[kpc]}$", fontsize=fontsize)
    ax.set_ylabel(r"${\rm D_{HI}^{GK} \,[kpc]}$", fontsize=fontsize)
    ax2.set_ylabel(r"${\rm \Delta D_{HI} / D_{HI}^{L}}$", fontsize=fontsize)
    
    for label in ax.xaxis.get_ticklabels():
        label.set_fontsize(fontsize)
    for label in ax.yaxis.get_ticklabels():
        label.set_fontsize(fontsize)
    for label in ax2.xaxis.get_ticklabels():
        label.set_fontsize(fontsize)
    for label in ax2.yaxis.get_ticklabels():
        label.set_fontsize(fontsize)
    
    locator = MultipleLocator(0.005)
    ax2.yaxis.set_minor_locator(locator)
    mlocator = MultipleLocator(0.02)
    ax2.yaxis.set_major_locator(mlocator)
    
    ax.yaxis.set_major_formatter(UserLogFormatterMathtext())
    ax2.xaxis.set_major_formatter(UserLogFormatterMathtext())
    ax.xaxis.set_major_formatter(NullFormatter())
    
    fig.savefig('../plots/diam_comparison.%s' % (suffix), dpi=400, quality=95)


plot_diam_comparison(suffix)