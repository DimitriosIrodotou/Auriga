from gadget import *
from pylab import *
from util.subhalos_utilities import subhalos_properties

colors = ['g', 'b', 'r', 'k', 'c', 'y', 'm', 'purple']
toinch = 0.393700787
base = '/output/fof_subhalo_tab_'


def plot_fracsfrgasvsradius(runs, dirs, outpath, snap, suffix):
    fac = 1.0
    fig = figure(figsize=np.array([16.1 * fac, 16.1]) * toinch * 0.5, dpi=300)
    
    for d in range(len(runs)):
        print
        'doing halo', runs[d]
        dd = dirs[d] + runs[d]
        s = gadget_readsnap(snap, snappath=dd + '/output/', hdf5=True)
        subhalos = subhalos_properties(snap, base=base, directory=dd, hdf5=True)
        # subhalos.set_subhalos_properties( parent_group=0, main_halo=True, sfr=True )
        subhalos.set_subhalos_properties(parent_group=0, main_halo=False, sfr=True)
        
        sfrmass = np.zeros(subhalos.numbersubhalos[subhalos.groupid])
        radius = sqrt((subhalos.subhalospos[:, :] ** 2.0).sum(axis=1))
        gasmass = subhalos.subhalosmassestype[:, 0]
        starmass = subhalos.subhalosmassestype[:, 4]
        
        ax = axes([0.15 / fac, 0.13, 0.8 / fac, 0.8])
        
        # selecting the star-forming gas for each of the subhalos
        for i in range(subhalos.numbersubhalos[subhalos.groupid]):
            if subhalos.subhalossfr[i] <= 0.0:
                continue
            
            mass = s.data['mass'][subhalos.particlesoffsets[i, 0]:subhalos.particlesoffsets[i, 0] + subhalos.subhaloslentype[i, 0]].astype('float64')
            sfr = s.data['sfr'][subhalos.particlesoffsets[i, 0]:subhalos.particlesoffsets[i, 0] + subhalos.subhaloslentype[i, 0]].astype('float64')
            j, = np.where(sfr > 0)
            sfrmass[i] = mass[j].sum()
        
        # only plot subhalos with ongoing star formation
        j, = np.where(sfrmass > 0.0)
        ax.semilogx(1.0e3 * radius[j], sfrmass[j] / (gasmass[j] + starmass[j]), 'o', ms=3.0, color=colors[d], label="$\\rm{%s}$" % runs[d])
        
        limits = ax.get_ylim()
        ax.vlines(1.0e3 * subhalos.rvir, limits[0], limits[1], linestyle=':', color='darkgray', lw=0.7)
        
        # xlim( 1.0e8, 1.0e13 )
        ylim(0.0, 0.14)
        
        minorLocator = MultipleLocator(0.005)
        ax.yaxis.set_minor_locator(minorLocator)
        
        legend(loc='upper left', fontsize=6, frameon=False, numpoints=1)
        
        xlabel("$\\rm{r\\,[kpc]}$")
        ylabel("$\\rm{f_{gas}}$")
        
        savefig("%s/fracsfrgasvsradius_%s.%s" % (outpath, runs[d], suffix), dpi=300)
        fig.clf()