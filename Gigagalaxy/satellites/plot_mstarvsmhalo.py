from pylab import *
from util.subhalos_utilities import subhalos_properties

colors = ['g', 'b', 'r', 'k', 'c', 'y', 'm', 'purple']
toinch = 0.393700787
base = '/output/fof_subhalo_tab_'


def plot_mstarvsmhalo(runs, dirs, outpath, snap, suffix):
    fac = 1.0
    fig = figure(figsize=np.array([16.1 * fac, 16.1]) * toinch * 0.5, dpi=300)
    
    for d in range(len(runs)):
        print
        'doing halo', runs[d]
        dd = dirs[d] + runs[d]
        subhalos = subhalos_properties(snap, base=base, directory=dd, hdf5=True)
        subhalos.set_subhalos_properties(parent_group=0)
        totalmass, stellarmass = subhalos.get_masses_vs_stellar_masses(rvcirc_masses=False)
        
        axes([0.17 / fac, 0.13, 0.8 / fac, 0.8])
        
        loglog(1.0e10 * totalmass, 1.0e10 * stellarmass, 'o', ms=3.0, color=colors[d], label="$\\rm{%s}$" % runs[d])
        
        legend(loc='upper left', fontsize=6, frameon=False, numpoints=1)
        
        xlabel("$\\rm{M_{tot}\\,[M_\\odot]}$")
        ylabel("$\\rm{M_{stars}\\,[M_\\odot]}$")
        
        savefig("%s/mstarvsmhalo_%s.%s" % (outpath, runs[d], suffix), dpi=300)
        fig.clf()