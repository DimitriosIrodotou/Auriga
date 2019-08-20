from pylab import *
from util.subhalos_utilities import subhalos_properties

colors = ['g', 'b', 'r', 'k', 'c', 'y', 'm', 'purple']
bands = ['U', 'B', 'V', 'K', 'g', 'r', 'i', 'z']
band_array = {'U': 0, 'B': 1, 'V': 2, 'K': 3, 'g': 4, 'r': 5, 'i': 6, 'z': 7}
toinch = 0.393700787
base = '/output/fof_subhalo_tab_'


def plot_mstarvsvcirc(runs, dirs, outpath, snap, suffix):
    fac = 1.0
    fig = figure(figsize=np.array([16.1 * fac, 16.1]) * toinch * 0.5, dpi=300)
    
    for d in range(len(runs)):
        print
        'doing halo', runs[d]
        dd = dirs[d] + runs[d]
        subhalos = subhalos_properties(snap, base=base, directory=dd, hdf5=True)
        subhalos.set_subhalos_properties(parent_group=0, main_halo=False, photometry=True)
        starmass, vcirc = subhalos.get_mstar_vs_vcirc()
        
        ax = axes([0.15 / fac, 0.13, 0.8 / fac, 0.8])
        
        ax.loglog(vcirc, 1.0e10 * starmass, 'o', ms=3.0, color=colors[d], label="$\\rm{%s}$" % runs[d])
        
        legend(loc='upper left', fontsize=6, frameon=False, numpoints=1)
        
        xlabel("$\\rm{v_{imax}\\,[km\\,s^{-1}]}$")
        ylabel("$\\rm{M_{star}\\,[M_{\\odot}]}$")
        
        savefig("%s/mstarvsvcirc_%s.%s" % (outpath, runs[d], suffix), dpi=300)
        fig.clf()