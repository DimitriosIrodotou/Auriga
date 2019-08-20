from gadget import *
from subhalos_utilities import subhalos_properties

colors = ['g', 'b', 'r', 'k', 'c', 'y', 'm', 'purple']
toinch = 0.393700787
base = '/output/fof_subhalo_tab_'


def test_offsets(runs, dirs, snap):
    for d in range(len(runs)):
        print('doing halo', runs[d])
        dd = dirs[d] + runs[d]
        s = gadget_readsnap(snap, snappath=dd + '/output/', hdf5=True)
        subhalos = subhalos_properties(snap, base=base, directory=dd, hdf5=True)
        subhalos.set_subhalos_properties(parent_group=0, main_halo=False, sfr=True)
        
        sfrmass = np.zeros(subhalos.numbersubhalos[subhalos.groupid])
        
        fileoffsets = np.zeros(6)
        fileoffsets[1:] = np.cumsum(s.nparticlesall[:-1])
        subhalos.particlesoffsets += fileoffsets
        
        totmass = np.zeros(6)
        submass = np.zeros(6)
        
        for i in range(subhalos.numbersubhalos[subhalos.groupid]):
            print('Subhalo number', i)
            
            for ptype in range(6):
                if subhalos.subhaloslentype[i, ptype] > 0:
                    mass = s.data['mass'][
                           subhalos.particlesoffsets[i, ptype]:subhalos.particlesoffsets[i, ptype] + subhalos.subhaloslentype[i, ptype]].astype(
                        'float64')
                    
                    # to accont for wind particles that it the catalogue are counted as gas
                    if ptype == 4:
                        istarsbeg = subhalos.particlesoffsets[i, ptype] - fileoffsets[ptype]
                        istarsend = istarsbeg + subhalos.subhaloslentype[i, ptype]
                        j, = np.where(s.data['age'][istarsbeg:istarsend] > 0.)
                        k, = np.where(s.data['age'][istarsbeg:istarsend] <= 0.)
                        totmass[ptype] = mass[j].sum()
                        totmass[0] += mass[k].sum()
                    else:
                        totmass[ptype] = mass.sum()
                    
                    submass[ptype] = subhalos.subhalosmassestype[i, ptype]
            
            for ptype in range(6):
                print('particle type', ptype, 'totalmass', totmass[ptype], 'from catalogue', submass[ptype], 'difference',
                      totmass[ptype] - submass[ptype])
                totmass[ptype] = 0.0
                submass[ptype] = 0.0
            
            print("")


dir1 = '/hits/tap/marinafo/Aquarius/'

runs = ['Aq-A_5']
dirs = [dir1]

snap = 63

test_offsets(runs, dirs, snap)