from gadget import *
from pylab import *
from util.subhalos_utilities import subhalos_properties

colors = ['g', 'b', 'r', 'k', 'c', 'y', 'm', 'purple']
toinch = 0.393700787
base = '/output/fof_subhalo_tab_'

Gcosmo = 43.0071
ZSUN = 0.0127

element = {'H': 0, 'He': 1, 'C': 2, 'N': 3, 'O': 4, 'Ne': 5, 'Mg': 6, 'Si': 7, 'Fe': 8}

# from Asplund et al. (2009) Table 5
SUNABUNDANCES = {'H': 12.0, 'He': 10.98, 'C': 8.47, 'N': 7.87, 'O': 8.73, 'Ne': 7.97, 'Mg': 7.64, 'Si': 7.55, 'Fe': 7.54}


def plot_oxygenvsiron(runs, dirs, outpath, snap, suffix):
    fac = 1.0
    fig = figure(figsize=np.array([16.1 * fac, 16.1]) * toinch * 0.5, dpi=300)
    
    for d in range(len(runs)):
        print
        'doing halo', runs[d]
        dd = dirs[d] + runs[d]
        s = gadget_readsnap(snap, snappath=dd + '/output/', hdf5=True)
        subhalos = subhalos_properties(snap, base=base, directory=dd, hdf5=True)
        subhalos.set_subhalos_properties(parent_group=0, main_halo=False)
        
        fileoffsets = np.zeros(6)
        fileoffsets[1:] = np.cumsum(s.nparticlesall[:-1])
        subhalos.particlesoffsets += fileoffsets
        
        iron = np.zeros(subhalos.numbersubhalos[subhalos.groupid])
        oxygen = np.zeros(subhalos.numbersubhalos[subhalos.groupid])
        iron[:] = 1.0e31
        oxygen[:] = 1.0e31
        
        ax = axes([0.15 / fac, 0.13, 0.8 / fac, 0.8])
        
        for i in range(subhalos.numbersubhalos[subhalos.groupid]):
            if subhalos.subhaloslentype[i, 4] > 0:
                mass = s.data['mass'][subhalos.particlesoffsets[i, 4]:subhalos.particlesoffsets[i, 4] + subhalos.subhaloslentype[i, 4]].astype(
                    'float64')
                
                istarsbeg = subhalos.particlesoffsets[i, 4] - fileoffsets[4] + fileoffsets[1]
                istarsend = istarsbeg + subhalos.subhaloslentype[i, 4]
                abundances = s.data['gmet'][istarsbeg:istarsend]
                
                istarsbeg = subhalos.particlesoffsets[i, 4] - fileoffsets[4]
                istarsend = istarsbeg + subhalos.subhaloslentype[i, 4]
                j, = np.where(s.data['age'][istarsbeg:istarsend] > 0.)
                mass = mass[j]
                abundances = abundances[j]
                
                totabundances = (abundances[:, :] * mass[:, None]).sum(axis=0) / mass.sum()
                
                iron[i] = log10(totabundances[element['Fe']] / totabundances[element['H']] / 56.0)
                oxygen[i] = log10(totabundances[element['O']] / totabundances[element['H']] / 16.)
                
                iron[i] -= (SUNABUNDANCES['Fe'] - SUNABUNDANCES['H'])
                oxygen[i] -= (SUNABUNDANCES['O'] - SUNABUNDANCES['H'])
                oxygen[i] -= iron[i]
        
        j, = np.where(iron < 1.0e30)
        ax.plot(iron[j], oxygen[j], 'o', ms=3.0, color=colors[d], label="$\\rm{%s}$" % runs[d])
        
        # xlim( 1.0e8, 1.0e13 )
        ylim(0.25, 0.35)
        
        minorLocator = MultipleLocator(0.05)
        ax.xaxis.set_minor_locator(minorLocator)
        minorLocator = MultipleLocator(0.005)
        ax.yaxis.set_minor_locator(minorLocator)
        
        legend(loc='best', fontsize=6, frameon=False, numpoints=1)
        
        xlabel("$\\rm{[Fe/H]}$")
        ylabel("$\\rm{[O/Fe]}$")
        
        savefig("%s/oxygenvsiron_%s.%s" % (outpath, runs[d], suffix), dpi=300)
        fig.clf()