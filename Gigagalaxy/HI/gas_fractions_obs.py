import multipanel_layout
import numpy as np
from label_format import *

BOLTZMANN = 1.38065e-16
toinch = 0.393700787

suffix = 'pdf'


def fit_func(x, a, b):
    return a * x + b


def colin_2016(x):
    Mstar = x / (10. ** 9.22)
    f = 1.83 / (Mstar ** 0.13 + Mstar ** 0.65)
    f /= 1.4
    fgas = f / (1. + f)
    return fgas


def colin_2016_up(x):
    return colin_2016(x) * 10. ** 0.33


def colin_2016_down(x):
    return colin_2016(x) / 10. ** 0.33


def plot_gas_fractions(dirs, runs, snap, suffix):
    file = 'mass_diameter_scal.txt'
    mass = 1.0e9 * np.loadtxt(file, usecols=[2])
    massGK = 1.0e9 * np.loadtxt(file, usecols=[5])
    
    fileres = 'gas_fractions_res.txt'
    stellarmass = np.loadtxt(fileres, usecols=[1])
    gfres = np.loadtxt(fileres, usecols=[2])
    gfGKres = np.loadtxt(fileres, usecols=[3])
    
    filegass = 'HIfrac.txt'
    starmass = np.loadtxt(filegass, usecols=[3])
    gfrac = np.loadtxt(filegass, usecols=[4])
    
    himass = 10. ** starmass
    starmass -= gfrac
    starmass = 10. ** starmass
    gfrac = himass / (himass + starmass)
    
    filebd = 'Bluediskdataset.txt'
    starmassbd = np.loadtxt(filebd, usecols=[5])
    gfracbd = np.loadtxt(filebd, usecols=[6])
    flag = np.loadtxt(filebd, usecols=[3])
    
    himassbd = 10. ** starmassbd
    starmassbd -= gfracbd
    starmassbd = 10. ** starmassbd
    gfracbd = himassbd / (himassbd + starmassbd)
    
    figure = multipanel_layout.multipanel_layout(nrows=1, ncols=1, npanels=1, hspace=0.07, wspace=0.07, left=0.16, right=0.95, bottom=0.11, top=0.95,
                                                 scale=2)
    figure.set_figure_layout()
    figure.set_axis_limits_and_aspect(xlim=[1.0e10, 2.0e11], ylim=[1.0e-3, 1.0], logaxis=True)
    figure.set_fontsize(8.)
    figure.set_axis_labels(xlabel=r"${\rm M_{*} \,[M_{\odot}]}$", ylabel=r"${\rm f_{gas}}$")
    
    ax = figure.axes[0]
    
    mstar = np.zeros(mass.shape)
    mtot = np.zeros(mass.shape)
    
    f = open("gas_fractions.txt", "w")
    
    for d in range(len(runs)):
        dd = dirs[d] + runs[d]
        
        print
        "Doing dir %s, snap %d." % (dd, snap)
        
        s = gadget_readsnap(snap, snappath=dd + '/output/', hdf5=True, loadonlytype=[0, 4], loadonly=['pos', 'mass', 'age'])
        sf = load_subfind(snap, dir=dd + '/output/', hdf5=True, loadonly=['fpos', 'frc2'])
        
        s.center = sf.data['fpos'][0, :]
        rad = 0.1 * sf.data['frc2'][0]
        
        istars, = np.where((s.r() < rad) & (s.type == 4))
        
        # remove wind particles from stars ...
        first_star = s.nparticlesall[:4].sum()
        istars -= first_star
        j, = np.where(s.data['age'][istars] > 0.)
        
        istars = istars[j]
        istars += first_star
        
        mstar[d] = s.data['mass'][istars].astype('float64').sum()
        
        print
        "%10.1f" % mstar[d]
        
        mstar[d] *= 1.0e10
        
        header = "%12s%17.7e%17.7e%17.7e\n" % (runs[d], mstar[d], mass[d] / (mass[d] + mstar[d]), massGK[d] / (massGK[d] + mstar[d]))
        f.write(header)
    
    f.close()
    
    ax.loglog(starmass, gfrac, marker='o', mfc='lightgray', mec='None', ms=5, linestyle='None')
    idx, = np.where(flag == 1)
    ax.loglog(starmassbd[idx], gfracbd[idx], marker='s', mfc='gray', mec='None', ms=5, linestyle='None')
    
    mtot = mstar + mass
    ax.loglog(mstar, mass / mtot, marker='^', mfc='blue', mec='None', ms=8, linestyle='None')
    mtot = mstar + massGK
    ax.loglog(mstar, massGK / mtot, marker='^', mfc='red', mec='None', ms=6, linestyle='None')
    
    ax.loglog(stellarmass[[0, 3]], gfres[[0, 3]], marker='o', mfc='blue', mec='None', ms=8, linestyle='None')
    ax.loglog(stellarmass[[2, 5]], gfres[[2, 5]], marker='s', mfc='blue', mec='None', ms=8, linestyle='None')
    
    ax.loglog(stellarmass[[0, 3]], gfGKres[[0, 3]], marker='o', mfc='red', mec='None', ms=6, linestyle='None')
    ax.loglog(stellarmass[[2, 5]], gfGKres[[2, 5]], marker='s', mfc='red', mec='None', ms=6, linestyle='None')
    
    idx, = np.where(flag != 1)
    ax.loglog(starmassbd[idx], gfracbd[idx], marker='s', mfc='None', mec='gray', ms=5, linestyle='None')
    
    m = np.logspace(10, 11.5, num=20)
    ax.loglog(m, colin_2016(m), 'k-', lw=1.5)
    ax.loglog(m, colin_2016_up(m), 'k--', lw=0.8)
    ax.loglog(m, colin_2016_down(m), 'k--', lw=0.8)
    
    ax.legend(labels=['Catinella+ 13', 'Wang+ 14', 'L08', 'GK11'], loc='lower left', frameon=False, numpoints=1, fontsize=figure.fontsize)
    
    ax.yaxis.set_major_formatter(UserLogFormatterMathtext())
    
    figure.fig.savefig('../plots/gas_fractions_obs.%s' % (suffix), dpi=400, quality=95)


runs = ['halo_%s' % i for i in range(1, 31)]
prefix = ['/hits/universe/GigaGalaxy/level4_MHD/'] * len(runs)
snap = 127

plot_gas_fractions(prefix, runs, snap, suffix)