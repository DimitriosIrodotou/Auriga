from pylab import *
from util.subhalos_utilities import subhalos_properties

colors = ['g', 'b', 'r', 'k', 'c', 'y', 'm', 'purple']
toinch = 0.393700787
base = '/output/fof_subhalo_tab_'


def plot_gasfracvsradius(runs, dirs, outpath, snap, suffix):
    fac = 1.0
    fig = figure(figsize=np.array([16.1 * fac, 16.1]) * toinch * 0.5, dpi=300)
    
    for d in range(len(runs)):
        print
        'doing halo', runs[d]
        dd = dirs[d] + runs[d]
        subhalos = subhalos_properties(snap, base=base, directory=dd, hdf5=True)
        subhalos.set_subhalos_properties(parent_group=0, main_halo=False, sfr=True)
        gasfrac, radius = subhalos.get_gasfrac_vs_position()
        
        ax = axes([0.15 / fac, 0.13, 0.8 / fac, 0.8])
        
        ax.semilogx(1.0e3 * radius, gasfrac, 'o', ms=3.0, color=colors[d], label="$\\rm{%s}$" % runs[d])
        
        minorLocator = MultipleLocator(0.05)
        ax.yaxis.set_minor_locator(minorLocator)
        
        limits = ax.get_ylim()
        ax.vlines(1.0e3 * subhalos.rvir, limits[0], limits[1], linestyle=':', color='darkgray', lw=0.7)
        
        legend(loc='upper left', fontsize=6, frameon=False, numpoints=1)
        
        xlabel("$\\rm{r\\,[kpc]}$")
        ylabel("$\\rm{M_{gas}\\,[M_{\\odot}]}$")
        
        savefig("%s/gasfracvsradius_%s.%s" % (outpath, runs[d], suffix), dpi=300)
        fig.clf()