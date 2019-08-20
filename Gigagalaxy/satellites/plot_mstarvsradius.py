from pylab import *
from util.subhalos_utilities import subhalos_properties

colors = ['g', 'b', 'r', 'k', 'c', 'y', 'm', 'purple']
toinch = 0.393700787
base = '/output/fof_subhalo_tab_'


def plot_mstarvsradius(runs, dirs, outpath, snap, suffix):
    fac = 1.0
    fig = figure(figsize=np.array([16.1 * fac, 16.1]) * toinch * 0.5, dpi=300)
    
    for d in range(len(runs)):
        print
        'doing halo', runs[d]
        dd = dirs[d] + runs[d]
        subhalos = subhalos_properties(snap, base=base, directory=dd, hdf5=True)
        subhalos.set_subhalos_properties(parent_group=0, main_halo=False, sfr=True)
        starmass, radius, i, j, k = subhalos.get_mstar_vs_position()
        
        ax = axes([0.15 / fac, 0.13, 0.8 / fac, 0.8])
        
        ax.loglog(1.0e3 * radius[i], 1.0e10 * starmass[i], 'o', ms=3.0, color='r', label="without gas")
        ax.loglog(1.0e3 * radius[j], 1.0e10 * starmass[j], 'o', ms=3.0, color='b', label="with gas")
        ax.loglog(1.0e3 * radius[k], 1.0e10 * starmass[k], '^', ms=3.0, color='b', label="with SF gas")
        
        limits = ax.get_ylim()
        ax.vlines(1.0e3 * subhalos.rvir, limits[0], limits[1], linestyle=':', color='darkgray', lw=0.7)
        
        legend(loc='upper left', fontsize=6, frameon=False, numpoints=1)
        
        xlabel("$\\rm{r\\,[kpc]}$")
        ylabel("$\\rm{M_{star}\\,[M_{\\odot}]}$")
        
        savefig("%s/mstarvsradius_%s.%s" % (outpath, runs[d], suffix), dpi=300)
        fig.clf()