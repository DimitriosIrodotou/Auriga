import calcGrid
import pylab

toinch = 0.393700787
BOLTZMANN = 1.38065e-16
PROTONMASS = 1.67262178e-24
GAMMA = 5.0 / 3.0
GAMMA_MINUS1 = GAMMA - 1.0
MSUN = 1.989e33
MPC = 3.085678e24
KPC = 3.085678e21
ZSUN = 0.0127
Gcosmo = 43.0071


def rotate_value(value, matrix):
    new_value = pylab.zeros(pylab.shape(value))
    for i in range(3):
        new_value[:, i] = (value * matrix[i, :][None, :]).sum(axis=1)
    return new_value


# NOTE: all gas particles within the assigen projection box are selected. Then since the projection routine
# assigens to each cell in the cube the density of the closest neighbour, the particles failing the other
# selection criteria have their densities put to zero

def plot_stardensity(ax1, s, sf, boxsize, fact, res):
    dist = np.max(np.abs(s.pos), axis=1)
    istars, = np.where((s.type == 4) & (dist < fact * boxsize * 1.0e-3))
    
    # remove wind particles from stars ...
    first_star = s.nparticlesall[:4].sum()
    istars -= first_star
    j, = np.where(s.data['age'][istars] > 0.)
    jstars = j + first_star
    
    mass = s.data['mass'][jstars].astype('float64')
    print
    "Stellar projection plot selected particles in box", len(j)
    
    rad = sf.data['frc2'][0] * 1.0e3
    maxsatrad = 0.15 * rad
    minsatrad = 0.05 * rad
    
    # read in satellite information
    Nsubhalos = sf.data['fnsh'][0]
    subhalosPos = sf.data['spos'][0:Nsubhalos, :]
    subhalosVel = sf.data['svel'][0:Nsubhalos, :]
    subhalosLen = sf.data['slty'][0:Nsubhalos, 4]
    subhalosLenGas = sf.data['slty'][0:Nsubhalos, 0]
    subhalosMass = sf.data['smty'][0:Nsubhalos, 4]
    
    # select gas containing satellites
    isub, = np.where(subhalosLen > 0)
    subhalosPos = subhalosPos[isub, :]
    subhalosVel = subhalosVel[isub, :]
    subhalosMass = subhalosMass[isub]
    subhalosLen = subhalosLen[isub]
    subhalosLenGas = subhalosLenGas[isub]
    subhalosMassmax = subhalosMass.max()
    
    print
    'Subhalo number', Nsubhalos, 'with stars', len(isub)
    
    # convert to kpc
    subhalosPos *= 1.0e3
    
    temp_pos = s.pos[jstars, :].astype('float64') * 1.0e3
    pos = np.zeros((size(jstars), 3))
    pos[:, 0] = temp_pos[:, 0]
    pos[:, 1] = temp_pos[:, 1]
    pos[:, 2] = temp_pos[:, 2]
    
    print
    pos.max(), pos.min()
    
    tree = makeTree(pos)
    hsml = tree.calcHsmlMulti(pos, pos, mass, 48)
    hsml = np.minimum(hsml, 4. * boxsize / res)
    hsml = np.maximum(hsml, 1.001 * boxsize / res)
    rho = np.ones(size(jstars))
    
    # datarange = np.array( [ [4003.36,800672.], [199.370,132913.], [133.698,200548.] ] )
    datarange = np.array([[40.0336, 800672.], [1.99370, 132913.], [1.33698, 200548.]])
    fac = (512. / res) ** 2 * (0.5 * boxsize * 1e-3 / (0.025 * s.hubbleparam)) ** 2
    datarange *= fac
    
    data = np.zeros((res, res, 3))
    
    for k in range(3):
        iband = [3, 1, 0][k]
        band = 10 ** (-2.0 * s.data['gsph'][j, iband].astype('float64') / 5.0)
        grid = calcGrid.calcGrid(pos * 1e-3, hsml * 1e-3, band, rho, rho, res, res, res, boxsize * 1e-3, boxsize * 1e-3, boxsize * 1e-3, s.center[0],
                                 s.center[1], s.center[2], 1, 1)
        
        print
        grid.sum(), band.sum()
        print
        grid.min(), grid.max(), datarange[k].min(), datarange[k].max()
        
        drange = datarange[k]
        grid = np.minimum(np.maximum(grid, drange[0]), drange[1])
        loggrid = np.log10(grid)
        logdrange = np.log10(drange)
        data[:, :, k] = (loggrid - logdrange[0]) / (logdrange[1] - logdrange[0])
        
        print
        loggrid.min(), loggrid.max(), logdrange.min(), logdrange.max()
    
    ax = axes(ax1)
    ax.imshow(data, interpolation='nearest', origin='lower', extent=[-0.5 * boxsize, 0.5 * boxsize, -0.5 * boxsize, 0.5 * boxsize])
    
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.tick_params(axis='both', which='both', color='white', labelcolor='black')
    ax.yaxis.set_major_formatter(NullFormatter())
    
    minorLocator = MultipleLocator(50.0)
    ax.xaxis.set_minor_locator(minorLocator)
    ax.yaxis.set_minor_locator(minorLocator)
    ax.set_xlabel("$\\rm{x\\,[kpc]}$", size=6)
    
    for label in ax1.xaxis.get_ticklabels():
        label.set_fontsize(6)
    
    rvircircle = matplotlib.patches.Circle((0.0, 0.0), rad, color='white', fill=False, lw=0.5)
    gca().add_artist(rvircircle)
    
    for i in range(len(isub)):
        fac = subhalosMass[i] / subhalosMassmax
        radius = minsatrad + (maxsatrad - minsatrad) * fac ** 0.3
        # rsatcircle = matplotlib.patches.Circle( (subhalosPos[i,1], subhalosPos[i,0]), radius, color='white', fill=False, lw=0.5 )
        if subhalosLenGas[i] > 0:
            rsatcircle = matplotlib.patches.RegularPolygon((subhalosPos[i, 1], subhalosPos[i, 0]), numVertices=4, radius=radius,
                                                           orientation=0.25 * pi, color='white', fill=False, lw=0.3)
        else:
            rsatcircle = matplotlib.patches.RegularPolygon((subhalosPos[i, 1], subhalosPos[i, 0]), numVertices=3, radius=radius, orientation=0.,
                                                           color='white', fill=False, lw=0.3)
        
        gca().add_artist(rsatcircle)
        """
        norm = np.sqrt((subhalosVel[:,:2]**2.0).sum(axis=1)) * 0.025
        if norm[i] > 0.0:
            subhalosarrow = matplotlib.patches.Arrow(subhalosPos[i,1], subhalosPos[i,0], subhalosVel[i,1] / norm[i], subhalosVel[i,0] / norm[i],
            color='white', lw=0.3)
        gca().add_artist(subhalosarrow)
        """
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
    # rotate subhalo positions
    sf.data['spos'] = rotate_value(sf.data['spos'], rotmatrix)
    sf.data['svel'] = rotate_value(sf.data['svel'], rotmatrix)
    
    # rotate around spin axis of 45 degrees
    rotmatrix[0, 0] = 1.0
    rotmatrix[0, 1] = 0.0
    rotmatrix[0, 2] = 0.0
    rotmatrix[1, 0] = 0.0
    rotmatrix[1, 1] = 0.5 * sqrt(2.0)
    rotmatrix[1, 2] = 0.5 * sqrt(2.0)
    rotmatrix[2, 0] = 0.0
    rotmatrix[2, 1] = -rotmatrix[1, 2]
    rotmatrix[2, 2] = 0.5 * sqrt(2.0)
    
    pxsize = 11.45
    pysize = 4.6
    
    psize = 1.8
    offsetx = 0.1
    offsety = 0.4
    offset = 0.3
    
    fig = pylab.figure(figsize=(np.array([pxsize, pysize]) * toinch), dpi=900)
    
    res = 512
    # res = 256
    # res = 128
    boxsize = 0.6e3  # 1.0Mpc
    fact = 0.5  # projection lenght will be 2.0 * fact * boxsize
    
    for iplot in range(3):
        ix = iplot % 4
        x = ix * (2. * psize + offsetx) / pxsize + offsetx / pysize
        
        y = offsety / pysize
        y = (2. * offsety) / pysize
        ax1 = axes([x, y, 2. * psize / pxsize, 2. * psize / pysize], frameon=True)
        
        if iplot > 0:
            s.pos = rotate_value(s.pos, rotmatrix)
            sf.data['spos'] = rotate_value(sf.data['spos'], rotmatrix)
            sf.data['svel'] = rotate_value(sf.data['svel'], rotmatrix)
        
        func[0](ax1, s, sf, boxsize, fact, res)
    
    fig.savefig('../../plots/satellites/%s_%s.%s' % (name, run, suffix), transparent=True, dpi=900)


toinch = 0.393700787
snap = 63
snap = 255

dir1 = '/hits/tap/marinafo/Aquarius/'
dir2 = '/hits/tap_duo/pakmorrr/Aquarius/'
dir2 = '/hits/universe/pakmorrr/Aquarius/Aq-4.MHD/'

# runs = ['Aq-A_5', 'Aq-B_5', 'Aq-C_5', 'Aq-D_5','Aq-E_5', 'Aq-F_5', 'Aq-G_5', 'Aq-H_5']
# dirs = [ dir1, dir1, dir1, dir1, dir2, dir2, dir2, dir2]
runs = ['Aq-A_4.MHD']
dirs = [dir2]
nruns = len(runs)

suffix = 'pdf'

functions = [plot_stardensity]

for i in range(nruns):
    run = runs[i]
    prefix = dirs[i]
    
    print
    print
    "Doing run ", run, "first plot"
    plot_disk(prefix, run, snap, suffix, functions, "stars")