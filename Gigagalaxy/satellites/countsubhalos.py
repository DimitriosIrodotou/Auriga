def plot_gasdensity(s, sf):
    s.pos = s.pos[:s.nparticlesall[0], :]
    s.rho = s.rho[:s.nparticlesall[0]]
    
    # read in satellite information
    Nsubhalos = sf.data['fnsh'][0]
    subhalosLenGas = sf.data['slty'][0:Nsubhalos, 0]
    subhalosLenStars = sf.data['slty'][0:Nsubhalos, 4]
    
    # select gas containing satellites
    isub, = np.where((subhalosLenGas > 0) & (subhalosLenStars > 0))
    isubg, = np.where(subhalosLenGas > 0)
    isubs, = np.where(subhalosLenStars > 0)
    
    print
    'Subhalo number', Nsubhalos, 'with gas', len(isubg), 'with stars', len(isubs), 'with both', len(isub)
    
    return


def plot_disk(prefix, run, snap, suffix, func, name):
    path = prefix + run + '/output/'
    s = gadget_readsnap(snap, snappath=path, hdf5=True)
    sf = load_subfind(snap, dir=path, hdf5=True)
    s.calc_sf_indizes(sf)
    rotmatrix = s.select_halo(sf, use_principal_axis=True, do_rotation=True)
    
    # center the satellites
    sf.data['spos'][:, 0] -= sf.data['fpos'][0, 0]
    sf.data['spos'][:, 1] -= sf.data['fpos'][0, 1]
    sf.data['spos'][:, 2] -= sf.data['fpos'][0, 2]
    
    func[0](s, sf)


snap = 255

dir1 = '/hits/tap/marinafo/Aquarius/'
dir2 = '/hits/tap_duo/pakmorrr/Aquarius/'
dir3 = '/hits/universe/pakmorrr/Aquarius/Aq-4.MHD/'

# runs = ['Aq-A_5', 'Aq-B_5', 'Aq-C_5', 'Aq-D_5','Aq-E_5', 'Aq-F_5', 'Aq-G_5', 'Aq-H_5']
# dirs = [ dir1, dir1, dir1, dir1, dir2, dir2, dir2, dir2]
runs = ['Aq-A_4.MHD']
dirs = [dir3]
nruns = len(runs)

suffix = 'pdf'

functions = [plot_gasdensity]

for i in range(nruns):
    run = runs[i]
    prefix = dirs[i]
    
    print
    print
    "Doing run ", run, "first plot"
    plot_disk(prefix, run, snap, suffix, functions, "diskA")