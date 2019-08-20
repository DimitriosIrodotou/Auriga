from gadget import *
from gadget_subfind import *
from pylab import *
from util.subhalos_utilities import subhalos_properties

colors = ['g', 'b', 'r', 'k', 'c', 'y', 'm', 'purple']
toinch = 0.393700787
base = 'fof_subhalo_tab_'


def plot_number_vs_vcirc_differential(runs, dirs, outpath, snap, suffix, rcut=False, logscale=True):
    fac = 1.0
    fig = figure(figsize=np.array([16.1 * fac, 16.1]) * toinch * 0.5, dpi=300)
    
    axes([0.17 / fac, 0.13, 0.8 / fac, 0.8])
    
    for d in range(len(runs)):
        print
        'doing halo', runs[d]
        dd = dirs[d] + runs[d]
        subhalos = subhalos_properties(snap, base=base, directory=dd + '/output/', hdf5=True)
        subhalos.set_subhalos_properties(parent_group=0, main_halo=False)
        number, vcirc, edges = subhalos.get_differential_subhalo_vcirc_function(bins=15, range=None, rcut=rcut, logscale=logscale)
        
        if logscale:
            # normalize bins (such that the number per mass interval is plotted)
            edges = 10. ** edges
            binwidth = edges[1:] - edges[:-1]
            number = (number * 1.0) / binwidth
            
            loglog(10. ** vcirc, number, color=colors[d], label="$\\rm{%s}$" % runs[d])
        else:
            binwidth = edges[1:] - edges[:-1]
            number = (number * 1.0) / binwidth
            
            loglog(vcirc, number, color=colors[d], label="$\\rm{%s}$" % runs[d])
    
    # xlim( 1.0, 1.0e3 )
    # ylim( 0.9, 1.0e3 )
    
    legend(loc='upper right', fontsize=6, frameon=False, numpoints=1)
    
    xlabel("$\\rm{v_{circ}\\,[km\\,s^{-1}]}$")
    ylabel("$\\rm{dN_{sub}/dv\\,\\,[km^{-1}\\,s]}$")
    
    savefig("%s/differential_vcirc_distrib.%s" % (outpath, suffix), dpi=300)


def plot_number_vs_vcirc_cumulative(runs, dirs, outpath, snap, suffix, rcut=False, logscale=True):
    fac = 1.0
    fig = figure(figsize=np.array([16.1 * fac, 16.1]) * toinch * 0.5, dpi=300)
    
    axes([0.17 / fac, 0.13, 0.8 / fac, 0.8])
    
    for d in range(len(runs)):
        print
        'doing halo', runs[d]
        dd = dirs[d] + runs[d]
        subhalos = subhalos_properties(snap, base=base, directory=dd + '/output/', hdf5=True)
        subhalos.set_subhalos_properties(parent_group=0)
        number, vcirc = subhalos.get_cumulative_subhalo_vcirc_function(bins=100, range=None, rcut=rcut, logscale=logscale, inverse=True)
        
        if logscale:
            loglog(10. ** vcirc, number, color=colors[d], label="$\\rm{%s}$" % runs[d])
        else:
            loglog(vcirc, number, color=colors[d], label="$\\rm{%s}$" % runs[d])
    
    xlim(1.0, 1.0e3)
    ylim(0.9, 1.0e3)
    
    legend(loc='upper right', fontsize=6, frameon=False, numpoints=1)
    
    xlabel("$\\rm{v_{circ}\\,[km\\,s^{-1}]}$")
    ylabel("$\\rm{N_{sub}\\,[> v_{circ}]}$")
    
    savefig("%s/cumulative_vcirc_distrib.%s" % (outpath, suffix), dpi=300)