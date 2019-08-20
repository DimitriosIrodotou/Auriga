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

element = {'H': 0, 'He': 1, 'C': 2, 'N': 3, 'O': 4, 'Ne': 5, 'Mg': 6, 'Si': 7, 'Fe': 8}


def rotate_value(value, matrix):
    new_value = pylab.zeros(pylab.shape(value))
    for i in range(3):
        new_value[:, i] = (value * matrix[i, :][None, :]).sum(axis=1)
    return new_value


# NOTE: all gas particles within the assigen projection box are selected. Then since the projection routine
# assigens to each cell in the cube the density of the closest neighbour, the particles failing the other
# selection criteria have their densities put to zero

def plot_gasdensity(ax1, cax, s, sf, boxsize, fact, res):
    s.pos = s.pos[:s.nparticlesall[0], :]
    s.rho = s.rho[:s.nparticlesall[0]]
    
    rad = sf.data['frc2'][0] * 1.0e3
    maxsatrad = 0.15 * rad
    minsatrad = 0.05 * rad
    
    # read in satellite information
    Nsubhalos = sf.data['fnsh'][0]
    subhalosPos = sf.data['spos'][0:Nsubhalos, :]
    subhalosVel = sf.data['svel'][0:Nsubhalos, :]
    subhalosLen = sf.data['slty'][0:Nsubhalos, 0]
    subhalosMass = sf.data['smty'][0:Nsubhalos, 0]
    
    # select gas containing satellites
    isub, = np.where(subhalosLen > 0)
    subhalosPos = subhalosPos[isub, :]
    subhalosVel = subhalosVel[isub, :]
    subhalosMass = subhalosMass[isub]
    subhalosMassmax = subhalosMass.max()
    
    print
    'Subhalo number', Nsubhalos, 'with gas', len(isub)
    
    # convert to kpc
    subhalosPos *= 1.0e3
    
    box = 0.5 * boxsize * 1.0e-3  # reconvert to Mpc for the first selection
    igas, = np.where((np.abs(s.pos[:, 0]) < box) & (np.abs(s.pos[:, 1]) < box) & (np.abs(s.pos[:, 2]) < box))
    npart = len(igas)
    print
    "Gas density plot selected particles in box", len(igas)
    
    sfr = s.data['sfr'][igas].astype('float64')
    s.data['pos'] = s.pos[igas] * 1.0e3
    ne = s.data['ne'][igas].astype('float64')
    metallicity = s.data['gz'][igas].astype('float64')
    XH = s.data['gmet'][igas, element['H']].astype('float64')
    yhelium = (1 - XH - metallicity) / (4. * XH);
    mu = (1 + 4 * yhelium) / (1 + yhelium + ne)
    # this conversion seems strange but basically rho is first converted to 10 Msun / kpc^3 by
    # multiplying by then and then in cm^-2 with all the other factors (this holds also for the
    # other projection functions). The factor boxsize / res is the dl of the projection
    s.data['rho'] = s.rho[igas].astype('float64') * 1.0e1 * MSUN / KPC ** 2.0 / mu / PROTONMASS * boxsize / res
    
    # choose non star-forming particles
    igas, = np.where(sfr > 0.0)
    print
    "star forming particles in box", len(igas)
    print
    "non star forming particles in box", npart - len(igas)
    s.data['rho'][igas] = 0.0
    
    axes(ax1)
    s.plot_Aslice("rho", colorbar=False, res=res, proj=True, axes=[1, 0], box=[boxsize, boxsize], proj_fact=fact, logplot=True, rasterized=True,
                  newfig=False)
    # clim( 1.0e21, 1.0e25 )
    
    rvircircle = matplotlib.patches.Circle((0.0, 0.0), rad, color='k', fill=False, lw=0.5)
    gca().add_artist(rvircircle)
    
    for i in range(len(isub)):
        fac = subhalosMass[i] / subhalosMassmax
        radius = minsatrad + (maxsatrad - minsatrad) * fac ** 0.3
        # rsatcircle = matplotlib.patches.Circle( (subhalosPos[i,1], subhalosPos[i,0]), radius, color='k', fill=False, lw=0.5 )
        rsatcircle = matplotlib.patches.RegularPolygon((subhalosPos[i, 1], subhalosPos[i, 0]), numVertices=4, radius=radius, orientation=0.25 * pi,
                                                       color='k', fill=False, lw=0.3)
        gca().add_artist(rsatcircle)
        
        norm = np.sqrt((subhalosVel[:, :2] ** 2.0).sum(axis=1)) * 0.025
        if norm[i] > 0.0:
            subhalosarrow = matplotlib.patches.Arrow(subhalosPos[i, 1], subhalosPos[i, 0], subhalosVel[i, 1] / norm[i], subhalosVel[i, 0] / norm[i],
                                                     color='k', lw=0.3)
        gca().add_artist(subhalosarrow)
    
    colorbar(cax=cax, orientation='horizontal', format=matplotlib.ticker.LogFormatterMathtext())
    cax.set_title('$\\rm{N\\,[cm^{-2}]}$', size=8)
    for label in cax.xaxis.get_ticklabels():
        label.set_fontsize(8)
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
    pysize = 5.7
    
    psize = 1.8
    offsetx = 0.1
    offsety = 0.38
    offset = 0.3
    
    fig = pylab.figure(figsize=(np.array([pxsize, pysize]) * toinch), dpi=300)
    
    res = 256
    # res = 128
    boxsize = 1.0e3  # 1.0Mpc
    fact = 0.5  # projection lenght will be 2.0 * fact * boxsize
    
    for iplot in range(3):
        ix = iplot % 4
        x = ix * (2. * psize + offsetx) / pxsize + offsetx / pysize
        
        y = offsety / pysize
        y = (2. * offsety) / pysize
        ax1 = axes([x, y, 2. * psize / pxsize, 2. * psize / pysize], frameon=True)
        
        y = (2. * psize + 3. * offset) / pysize + 0.15 * psize / pysize
        cax = axes([x, y, 2. * psize / pxsize, psize / pysize / 15.], frameon=False)
        
        if iplot > 0:
            s.pos = rotate_value(s.pos, rotmatrix)
            sf.data['spos'] = rotate_value(sf.data['spos'], rotmatrix)
            sf.data['svel'] = rotate_value(sf.data['svel'], rotmatrix)
        
        func[0](ax1, cax, s, sf, boxsize, fact, res)
        
        for label in cax.xaxis.get_ticklabels():
            label.set_fontsize(6)
        
        for label in ax1.xaxis.get_ticklabels():
            label.set_fontsize(6)
        
        # ax1.xaxis.set_major_formatter( NullFormatter() )
        ax1.yaxis.set_major_formatter(NullFormatter())
        
        minorLocator = MultipleLocator(50.0)
        ax1.xaxis.set_minor_locator(minorLocator)
        ax1.yaxis.set_minor_locator(minorLocator)
        ax1.set_xlabel("$\\rm{x\\,[kpc]}$", size=6)
    
    fig.savefig('../../plots/satellites/%s_%s.%s' % (name, run, suffix), transparent=True, dpi=300)


toinch = 0.393700787
snap = 63

dir1 = '/hits/tap/marinafo/Aquarius/'
dir2 = '/hits/tap_duo/pakmorrr/Aquarius/'

# runs = ['Aq-A_5', 'Aq-B_5', 'Aq-C_5', 'Aq-D_5','Aq-E_5', 'Aq-F_5', 'Aq-G_5', 'Aq-H_5']
# dirs = [ dir1, dir1, dir1, dir1, dir2, dir2, dir2, dir2]
runs = ['Aq-A_5']
dirs = [dir1]
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