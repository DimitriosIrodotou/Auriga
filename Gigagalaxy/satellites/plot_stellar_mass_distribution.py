from gadget import *
from gadget_subfind import *
from pylab import *
from util.subhalos_utilities import subhalos_properties

colors = ['g', 'b', 'r', 'k', 'c', 'y', 'm', 'purple']
toinch = 0.393700787
base = 'fof_subhalo_tab_'


def plot_number_vs_stellar_mass_differential(runs, dirs, outpath, snap, suffix, rcut=False, logscale=True):
    fac = 1.0
    fig = figure(figsize=np.array([16.1 * fac, 16.1]) * toinch * 0.5, dpi=300)
    
    axes([0.16 / fac, 0.13, 0.8 / fac, 0.8])
    
    for d in range(len(runs)):
        print
        'doing halo', runs[d]
        dd = dirs[d] + runs[d]
        subhalos = subhalos_properties(snap, base=base, directory=dd + '/output/', hdf5=True)
        if rcut:
            subhalos.set_subhalos_properties(parent_group=0, main_halo=False)
        else:
            subhalos.set_subhalos_properties()
        number, mass, edges = subhalos.get_differential_subhalo_stellar_mass_function(bins=15, range=None, rcut=rcut, logscale=logscale)
        
        if logscale:
            # normalize bins (such that the number per mass interval is plotted)
            # edges = 10.**edges
            binwidth = edges[1:] - edges[:-1]
            # number = (number * 1.0) / (binwidth * 1.0e10)
            number = (number * 1.0) / (binwidth)
            
            print
            number
            print
            mass
            
            loglog(10. ** (mass + 10.), number, color=colors[d], label="$\\rm{%s}$" % runs[d])
        else:
            # normalize bins (such that the number per mass interval is plotted)
            binwidth = edges[1:] - edges[:-1]
            number /= (binwidth * 1.0e10)
            
            loglog(1.0e10 * mass, number, color=colors[d], label="$\\rm{%s}$" % runs[d])
    
    # xlim( 1.0, 1.0e3 )
    # ylim( 0.9, 1.0e3 )
    
    legend(loc='upper right', fontsize=6, frameon=False, numpoints=1)
    
    xlabel("$\\rm{M_{\\star}\\,[M_{\\odot}]}$")
    ylabel("$\\rm{dN_{sub}/dM_{\\star}\\,\\,[dex^{-1}]}$")
    
    savefig("%s/differential_stellar_mass_distrib.%s" % (outpath, suffix), dpi=300)


def plot_number_vs_stellar_mass_cumulative(runs, dirs, outpath, snap, suffix, rcut=False, logscale=True):
    fac = 1.
    fig = figure(figsize=np.array([16.1 * fac, 16.1]) * toinch * 0.5, dpi=300)
    
    axes([0.2 / fac, 0.18, 0.75 / fac, 0.75])
    
    # MW satellite masses (McConnachie+ 2012) [10**6 Msub units] #
    mwsat_dict = {'CanisMajor': 49., 'Sagittarius': 21., 'SegueI': 0.00034, 'UrsaMajorII': 0.0041, 'BootesII': 0.001, 'SegueII': 0.00086,
                  'WillmanI':   0.001, 'ComaB': 0.0037, 'BootesIII': 0.017, 'LMC': 1500., 'SMC': 460., 'BootesI': 0.029, 'Draco': 0.29,
                  'UrsaMinor':  0.29, 'Sculptor': 2.3, 'SextansI': 0.44, 'UrsaMajorI': 0.014, 'Carina': 0.38, 'Hercules': 0.037, 'Fornax': 20.,
                  'LeoIV':      0.019, 'CanesVenII': 0.0079, 'LeoV': 0.019, 'PiscesII': 0.0086, 'CanesVenI': 0.23, 'LeoII': 0.74, 'LeoI': 5.5}
    mwsatmass = np.array(mwsat_dict.values()) * 1e6
    mwsatmass = log10(mwsatmass)
    nsat, edges = np.histogram(mwsatmass, bins=100)
    mwsatbin = edges[:-1]
    mwsatcum = np.cumsum(nsat[::-1])
    
    for d in range(len(runs)):
        print
        'doing halo', runs[d]
        dd = dirs[d] + runs[d]
        subhalos = subhalos_properties(snap, base=base, directory=dd + '/output/', hdf5=True)
        # if rcut:
        # subhalos.set_subhalos_properties( parent_group=0, main_halo=False )
        # else:
        subhalos.set_subhalos_properties()
        number, mass = subhalos.get_cumulative_subhalo_stellar_mass_function(bins=100, range=None, rcut=rcut, logscale=logscale, inverse=True)
        
        if logscale:
            semilogx(10. ** (mass + 10.), number, color=colors[d], label="$\\rm{%s}$" % runs[d])
        else:
            semilogx(1.0e10 * mass, number, color=colors[d], label="$\\rm{%s}$" % runs[d])
    
    if logscale:
        semilogx(10. ** (mwsatbin), mwsatcum[::-1], color='k', linestyle='--', label=r"$\rm MW$")
    else:
        semilogx(mwsatbin, mwsatcum[::-1], color='k', linestyle='--', label=r"$\rm MW$")
    
    xlim(1.0e5, 1.0e10)
    ylim(1., 30.)
    
    legend(loc='upper right', fontsize=6, frameon=False, numpoints=1)
    
    xlabel("$\\rm{M_{\\star}\\,[M_{\\odot}]}$")
    ylabel("$\\rm{N_{sub}\\,[> M_{\\star}]}$")
    
    savefig("%s/cumulative_stellar_mass_distrib.%s" % (outpath, suffix), dpi=300)