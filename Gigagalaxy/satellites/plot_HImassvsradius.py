from gadget import *
from pylab import *
from util.subhalos_utilities import subhalos_properties

colors = ['g', 'b', 'r', 'k', 'c', 'y', 'm', 'purple']
toinch = 0.393700787
base = '/output/fof_subhalo_tab_'

element = {'H': 0, 'He': 1, 'C': 2, 'N': 3, 'O': 4, 'Ne': 5, 'Mg': 6, 'Si': 7, 'Fe': 8}


def plot_HImassvsradius(runs, dirs, outpath, snap, suffix):
    fac = 1.0
    fig = figure(figsize=np.array([16.1 * fac, 16.1]) * toinch * 0.5, dpi=300)
    
    for d in range(len(runs)):
        print
        'doing halo', runs[d]
        dd = dirs[d] + runs[d]
        s = gadget_readsnap(snap, snappath=dd + '/output/', hdf5=True)
        subhalos = subhalos_properties(snap, base=base, directory=dd, hdf5=True)
        subhalos.set_subhalos_properties(parent_group=0, main_halo=False)
        
        radius = sqrt((subhalos.subhalospos[:, :] ** 2).sum(axis=1))
        HImass = np.zeros(subhalos.numbersubhalos[subhalos.groupid])
        
        ax = axes([0.15 / fac, 0.13, 0.8 / fac, 0.8])
        
        # selecting the star-forming gas for each of the subhalos
        for i in range(subhalos.numbersubhalos[subhalos.groupid]):
            if subhalos.subhaloslentype[i, 0] <= 0.0:
                continue
            
            mass = s.data['mass'][subhalos.particlesoffsets[i, 0]:subhalos.particlesoffsets[i, 0] + subhalos.subhaloslentype[i, 0]].astype('float64')
            XH = s.data['gmet'][subhalos.particlesoffsets[i, 0]:subhalos.particlesoffsets[i, 0] + subhalos.subhaloslentype[i, 0],
                 element['H']].astype('float64')
            nh = s.data['nh'][subhalos.particlesoffsets[i, 0]:subhalos.particlesoffsets[i, 0] + subhalos.subhaloslentype[i, 0]].astype('float64')
            mass = mass * nh * XH
            HImass[i] = mass.sum()
        
        # only plot subhalos with neutral hydrogen
        j, = np.where(HImass > 0.0)
        ax.loglog(1.0e3 * radius[j], 1.0e10 * HImass[j], 'o', ms=3.0, color=colors[d], label="$\\rm{%s}$" % runs[d])
        
        limits = ax.get_ylim()
        ax.vlines(1.0e3 * subhalos.rvir, limits[0], limits[1], linestyle=':', color='darkgray', lw=0.7)
        
        # xlim( 1.0e8, 1.0e13 )
        # ylim( 0.0, 0.14 )
        
        # minorLocator = MultipleLocator(0.005)
        # ax.yaxis.set_minor_locator(minorLocator)
        
        legend(loc='upper left', fontsize=6, frameon=False, numpoints=1)
        
        xlabel("$\\rm{r\\,[kpc]}$")
        ylabel("$\\rm{M_{HI}\\,[M_{\\odot}]}$")
        
        savefig("%s/HImassvsradius_%s.%s" % (outpath, runs[d], suffix), dpi=300)
        fig.clf()