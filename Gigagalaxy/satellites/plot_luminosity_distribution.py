from gadget import *
from gadget_subfind import *
from pylab import *
from util.subhalos_utilities import subhalos_properties

colors = ['g', 'b', 'r', 'k', 'c', 'y', 'm', 'purple', 'orange']
bands = ['U', 'B', 'V', 'K', 'g', 'r', 'i', 'z']
band_array = {'U': 0, 'B': 1, 'V': 2, 'K': 3, 'g': 4, 'r': 5, 'i': 6, 'z': 7}
toinch = 0.393700787
base = 'fof_subhalo_tab_'


def plot_number_vs_luminosity_differential(runs, dirs, outpath, snap, suffix, rcut, mhd, newfig, band):
    fac = 1.0
    fig = figure(figsize=np.array([16.1 * fac, 16.1]) * toinch * 0.5, dpi=300)
    
    for d in range(len(runs)):
        print
        'doing halo', runs[d]
        dd = dirs[d] + runs[d]
        print
        dd
        subhalos = subhalos_properties(snap, base=base, directory=dd + '/output/', hdf5=True)
        subhalos.set_subhalos_properties(parent_group=0, photometry=True, main_halo=False)
        number, magnitude, edges = subhalos.get_differential_subhalo_luminosity_function(bins=15, range=None, band=band_array[band])
        
        ax = axes([0.15 / fac, 0.13, 0.8 / fac, 0.8])
        
        # normalize bins (such that the number per mass interval is plotted)
        # edges = 10.**edges
        binwidth = edges[1:] - edges[:-1]
        number = (number * 1.0) / binwidth
        
        semilogy(magnitude, number, color=colors[d], label="$\\rm{%s}$" % runs[d])
        
        # xlim( 1.0, 1.0e3 )
        ylim(0.9, 1.0e2)
        
        legend(loc='upper right', fontsize=6, frameon=False, numpoints=1)
        
        xlabel("$\\rm{M_{%s}}$" % band)
        ylabel("$\\rm{dN_{sub}/dM_{%s}\\,\\,[mag^{-1}]}$" % band)
        
        if newfig:
            ax.invert_xaxis()
            savefig("%s/differential_luminosity_distrib_%s_%s.%s" % (outpath, runs[d], band, suffix), dpi=300)
            fig.clf()
    if not newfig:
        ax.invert_xaxis()
        if mhd:
            savefig("%s/differential_luminosity_distrib_MHD_%s.%s" % (outpath, band, suffix), dpi=300)
        else:
            savefig("%s/differential_luminosity_distrib_%s.%s" % (outpath, band, suffix), dpi=300)


def plot_number_vs_luminosity_cumulative(runs, dirs, outpath, snap, suffix, rcut, mhd, newfig, band):
    fac = 1.0
    fig = figure(figsize=np.array([16.1 * fac, 16.1]) * toinch * 0.5, dpi=300)
    
    for d in range(len(runs)):
        print
        'doing halo', runs[d]
        dd = dirs[d] + runs[d]
        subhalos = subhalos_properties(snap, base=base, directory=dd + '/output/', hdf5=True)
        subhalos.set_subhalos_properties(parent_group=0, photometry=True, main_halo=False)
        number, magnitude = subhalos.get_cumulative_subhalo_luminosity_function(bins=20, range=None, band=band_array[band], inverse=False, rcut=rcut)
        
        ax = axes([0.15 / fac, 0.13, 0.8 / fac, 0.8])
        # ax.plot( magnitude, number, color=colors[d], drawstyle='steps-post', label="$\\rm{%s}$" % runs[d] )
        ax.semilogy(magnitude, number, color=colors[d], label="$\\rm{%s}$" % runs[d])
        
        minorLocator = MultipleLocator(0.5)
        ax.xaxis.set_minor_locator(minorLocator)
        # minorLocator = MultipleLocator(1.0)
        # ax.yaxis.set_minor_locator(minorLocator)
        
        # xlim( 1.0, 1.0e3 )
        # ylim( 0.0, 35. )
        
        legend(loc='upper right', fontsize=6, frameon=False, numpoints=1)
        
        xlabel("$\\rm{M_{%s}}$" % band)
        ylabel("$\\rm{N_{sub}\\,[< M_{%s}]}$" % band)
        
        if newfig:
            ax.invert_xaxis()
            savefig("%s/cumulative_luminosity_distrib_%s_%s.%s" % (outpath, runs[d], band, suffix), dpi=300)
            fig.clf()
    if not newfig:
        ax.invert_xaxis()
        if mhd:
            savefig("%s/cumulative_luminosity_distrib_MHD_%s.%s" % (outpath, band, suffix), dpi=300)
        else:
            savefig("%s/cumulative_luminosity_distrib_%s.%s" % (outpath, band, suffix), dpi=300)


# wrappers to do the luminosity function in all bands
def differential_luminosity_function(runs, dirs, outpath, snap, suffix, rcut=False, mhd=False, newfig=True):
    for i in bands:
        plot_number_vs_luminosity_differential(runs, dirs, outpath, snap, suffix, rcut, mhd, newfig, i)
    
    return


def cumulative_luminosity_function(runs, dirs, outpath, snap, suffix, rcut=False, mhd=False, newfig=True):
    for i in bands:
        plot_number_vs_luminosity_cumulative(runs, dirs, outpath, snap, suffix, rcut, mhd, newfig, i)
    
    return