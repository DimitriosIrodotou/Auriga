import matplotlib.ticker
from loadmodules import *
from projections.ionization_fractions import *
from pylab import axes, colorbar, gca

BOLTZMANN = 1.38065e-16
PROTONMASS = 1.67262178e-24
GAMMA = 5.0 / 3.0
GAMMA_MINUS1 = GAMMA - 1.0
MSUN = 1.989e33
MPC = 3.085678e24
KPC = 3.085678e21
ZSUN = 0.0127

element = {'H': 0, 'He': 1, 'C': 2, 'N': 3, 'O': 4, 'Ne': 5, 'Mg': 6, 'Si': 7, 'Fe': 8}


# NOTE: all gas particles within the assigen projection box are selected. Then since the projection routine
# assigens to each cell in the cube the density of the closest neighbour, the particles failing the other
# selection criteria have their densities put to zero

def plot_gasdensity(ax1, cax, s, sf, boxsize, fact, res, dextoshow, numthreads):
    s.pos = s.pos[:s.nparticlesall[0], :]
    s.rho = s.rho[:s.nparticlesall[0]]
    
    rad = sf.data['frc2'][0] * 1.0e3
    maxsatrad = 0.15 * rad
    minsatrad = 0.05 * rad
    
    # read in satellite information
    Nsubhalos = sf.data['fnsh'][0]
    subhalosPos = sf.data['spos'][0:Nsubhalos, :]
    subhalosLen = sf.data['slty'][0:Nsubhalos, 0]
    subhalosMass = sf.data['smty'][0:Nsubhalos, 0]
    
    # select gas containing satellites
    isub, = np.where(subhalosLen > 0)
    subhalosPos = subhalosPos[isub, :]
    subhalosMass = subhalosMass[isub]
    subhalosMassmax = subhalosMass.max()
    
    print('Subhalo number', Nsubhalos, 'with gas', len(isub))
    
    # convert to kpc
    subhalosPos *= 1.0e3
    
    box = 0.5 * boxsize * 1.0e-3  # reconvert to Mpc for the first selection
    igas, = np.where((np.abs(s.pos[:, 0]) < box) & (np.abs(s.pos[:, 1]) < box) & (np.abs(s.pos[:, 2]) < box))
    npart = len(igas)
    print("Gas density plot selected particles in box", len(igas))
    
    sfr = s.data['sfr'][igas].astype('float64')
    s.data['pos'] = s.pos[igas] * 1.0e3
    s.data['type'] = s.type[igas]
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
    print("star forming particles in box", len(igas))
    print("non star forming particles in box", npart - len(igas))
    s.data['rho'][igas] = 0.0
    
    axes(ax1)
    s.plot_Aslice("rho", colorbar=False, res=res, proj=True, axes=[1, 0], box=[boxsize, boxsize], proj_fact=fact, logplot=True, rasterized=True,
                  newfig=False, dextoshow=dextoshow, numthreads=numthreads)
    # clim( 1.0e21, 1.0e25 )
    
    rvircircle = matplotlib.patches.Circle((0.0, 0.0), rad, color='k', fill=False, lw=0.5)
    gca().add_artist(rvircircle)
    
    for i in range(len(isub)):
        fac = subhalosMass[i] / subhalosMassmax
        radius = minsatrad + (maxsatrad - minsatrad) * fac ** 0.3
        # rsatcircle = matplotlib.patches.Circle( (subhalosPos[i,1], subhalosPos[i,0]), radius, color='k', fill=False, lw=0.5 )
        rsatcircle = matplotlib.patches.RegularPolygon((subhalosPos[i, 1], subhalosPos[i, 0]), numVertices=4, radius=radius, orientation=0.25 * np.pi,
                                                       color='k', fill=False, lw=0.3)
        gca().add_artist(rsatcircle)
    
    colorbar(cax=cax, orientation='horizontal', format=matplotlib.ticker.LogFormatterMathtext())
    cax.set_title('$\\rm{N\\,[cm^{-2}]}$', size=8)
    for label in cax.xaxis.get_ticklabels():
        label.set_fontsize(8)
    return


def plot_hydrogen(ax1, cax, s, sf, boxsize, fact, res, dextoshow, numthreads):
    s.pos = s.pos[:s.nparticlesall[0], :]
    s.rho = s.rho[:s.nparticlesall[0]]
    
    box = 0.5 * boxsize * 1.0e-3  # reconvert to Mpc for the first selection
    igas, = np.where((np.abs(s.pos[:, 0]) < box) & (np.abs(s.pos[:, 1]) < box) & (np.abs(s.pos[:, 2]) < box))
    npart = len(igas)
    print("hydrogen density plot selected particles in box", len(igas))
    
    sfr = s.data['sfr'][igas].astype('float64')
    s.data['pos'] = s.pos[igas] * 1.0e3
    s.data['type'] = s.type[igas]
    ne = s.data['ne'][igas].astype('float64')
    metallicity = s.data['gz'][igas].astype('float64')
    XH = s.data['gmet'][igas, element['H']].astype('float64')
    nh = s.data['nh'][igas].astype('float64')
    yhelium = (1 - XH - metallicity) / (4. * XH);
    mu = 1.0
    # this conversion seems strange but basically rho is first converted to 10 Msun / kpc^3 by
    # multiplying by then and then in cm^-2 with all the other factors (this holds also for the
    # other projection functions). The factor boxsize / res is the dl of the projection
    s.data['rho'] = XH * nh * s.rho[igas].astype('float64') * 1.0e1 * MSUN / KPC ** 2.0 / mu / PROTONMASS * boxsize / res
    
    # choose non star-forming particles
    igas, = np.where(sfr > 0.0)
    print("star forming particles in box", len(igas))
    print("non star forming particles in box", npart - len(igas))
    s.data['rho'][igas] = 0.0
    
    axes(ax1)
    s.plot_Aslice("rho", colorbar=False, res=res, proj=True, axes=[1, 0], box=[boxsize, boxsize], proj_fact=fact, logplot=True, rasterized=True,
                  newfig=False, dextoshow=dextoshow, numthreads=numthreads)
    # s.plot_Aweightedslice( "rho", "mass", colorbar=False, res=res, proj=True, axes=[1,0], box=[boxsize,boxsize], proj_fact=fact,
    # logplot=True, rasterized=True, newfig=False )
    # clim( 1.0e21, 1.0e25 )
    
    colorbar(cax=cax, orientation='horizontal', format=matplotlib.ticker.LogFormatterMathtext())
    cax.set_title('$\\rm{N_{HI}\\,[cm^{-2}]}$', size=8)
    for label in cax.xaxis.get_ticklabels():
        label.set_fontsize(8)
    return


def plot_carbon(ax1, cax, s, sf, boxsize, fact, res, dextoshow, numthreads):
    s.pos = s.pos[:s.nparticlesall[0], :]
    s.rho = s.rho[:s.nparticlesall[0]]
    
    rad = sf.data['frc2'][0] * 1.0e3
    
    box = 0.5 * boxsize * 1.0e-3  # reconvert to Mpc for the first selection
    igas, = np.where((np.abs(s.pos[:, 0]) < box) & (np.abs(s.pos[:, 1]) < box) & (np.abs(s.pos[:, 2]) < box))
    npart = len(igas)
    print("carbon density plot selected particles in box", len(igas))
    
    sfr = s.data['sfr'][igas].astype('float64')
    s.data['pos'] = s.pos[igas] * 1.0e3
    s.data['type'] = s.type[igas]
    ne = s.data['ne'][igas].astype('float64')
    metallicity = s.data['gz'][igas].astype('float64')
    XH = s.data['gmet'][igas, element['H']].astype('float64')
    XC = s.data['gmet'][igas, element['C']].astype('float64')
    yhelium = (1 - XH - metallicity) / (4. * XH);
    mu = 12.0
    # this conversion seems strange but basically rho is first converted to 10 Msun / kpc^3 by
    # multiplying by then and then in cm^-2 with all the other factors (this holds also for the
    # other projection functions). The factor boxsize / res is the dl of the projection
    s.data['rho'] = XC * s.rho[igas].astype('float64') * 1.0e1 * MSUN / KPC ** 2.0 / mu / PROTONMASS * boxsize / res
    
    # choose non star-forming particles
    igas, = np.where(sfr > 0.0)
    print("star forming particles in box", len(igas))
    print("non star forming particles in box", npart - len(igas))
    s.data['rho'][igas] = 0.0
    
    axes(ax1)
    s.plot_Aslice("rho", colorbar=False, res=res, proj=True, axes=[1, 0], box=[boxsize, boxsize], proj_fact=fact, logplot=True, rasterized=True,
                  newfig=False, dextoshow=dextoshow, numthreads=numthreads)
    # clim( 1.0e21, 1.0e25 )
    
    rvircircle = matplotlib.patches.Circle((0.0, 0.0), rad, color='k', fill=False, lw=0.5)
    gca().add_artist(rvircircle)
    
    colorbar(cax=cax, orientation='horizontal', format=matplotlib.ticker.LogFormatterMathtext())
    cax.set_title('$\\rm{N_{C}\\,[cm^{-2}]}$', size=8)
    for label in cax.xaxis.get_ticklabels():
        label.set_fontsize(8)
    return


def plot_oxygen(ax1, cax, s, sf, boxsize, fact, res, dextoshow, numthreads):
    s.pos = s.pos[:s.nparticlesall[0], :]
    s.rho = s.rho[:s.nparticlesall[0]]
    
    box = 0.5 * boxsize * 1.0e-3  # reconvert to Mpc for the first selection
    igas, = np.where((np.abs(s.pos[:, 0]) < box) & (np.abs(s.pos[:, 1]) < box) & (np.abs(s.pos[:, 2]) < box))
    npart = len(igas)
    print("oxygen density plot selected particles in box", len(igas))
    
    sfr = s.data['sfr'][igas].astype('float64')
    s.data['pos'] = s.pos[igas] * 1.0e3
    s.data['type'] = s.type[igas]
    ne = s.data['ne'][igas].astype('float64')
    metallicity = s.data['gz'][igas].astype('float64')
    XH = s.data['gmet'][igas, element['H']].astype('float64')
    XO = s.data['gmet'][igas, element['O']].astype('float64')
    yhelium = (1 - XH - metallicity) / (4. * XH);
    mu = 16.0
    # this conversion seems strange but basically rho is first converted to 10 Msun / kpc^3 by
    # multiplying by then and then in cm^-2 with all the other factors (this holds also for the
    # other projection functions). The factor boxsize / res is the dl of the projection
    s.data['rho'] = XO * s.rho[igas].astype('float64') * 1.0e1 * MSUN / KPC ** 2.0 / mu / PROTONMASS * boxsize / res
    
    # choose non star-forming particles
    igas, = np.where(sfr > 0.0)
    print("star forming particles in box", len(igas))
    print("non star forming particles in box", npart - len(igas))
    s.data['rho'][igas] = 0.0
    
    axes(ax1)
    s.plot_Aslice("rho", colorbar=False, res=res, proj=True, axes=[1, 0], box=[boxsize, boxsize], proj_fact=fact, logplot=True, rasterized=True,
                  newfig=False, dextoshow=dextoshow, numthreads=numthreads)
    # clim( 1.0e21, 1.0e25 )
    
    colorbar(cax=cax, orientation='horizontal', format=matplotlib.ticker.LogFormatterMathtext())
    cax.set_title('$\\rm{N_{O}\\,[cm^{-2}]}$', size=8)
    for label in cax.xaxis.get_ticklabels():
        label.set_fontsize(8)
    return


def plot_silicon(ax1, cax, s, sf, boxsize, fact, res, dextoshow, numthreads):
    s.pos = s.pos[:s.nparticlesall[0], :]
    s.rho = s.rho[:s.nparticlesall[0]]
    
    box = 0.5 * boxsize * 1.0e-3  # reconvert to Mpc for the first selection
    igas, = np.where((np.abs(s.pos[:, 0]) < box) & (np.abs(s.pos[:, 1]) < box) & (np.abs(s.pos[:, 2]) < box))
    npart = len(igas)
    print("silicon density plot selected particles in box", len(igas))
    
    sfr = s.data['sfr'][igas].astype('float64')
    s.data['pos'] = s.pos[igas] * 1.0e3
    s.data['type'] = s.type[igas]
    ne = s.data['ne'][igas].astype('float64')
    metallicity = s.data['gz'][igas].astype('float64')
    XH = s.data['gmet'][igas, element['H']].astype('float64')
    XSi = s.data['gmet'][igas, element['Si']].astype('float64')
    yhelium = (1 - XH - metallicity) / (4. * XH);
    mu = 28.0
    # this conversion seems strange but basically rho is first converted to 10 Msun / kpc^3 by
    # multiplying by then and then in cm^-2 with all the other factors (this holds also for the
    # other projection functions). The factor boxsize / res is the dl of the projection
    s.data['rho'] = XSi * s.rho[igas].astype('float64') * 1.0e1 * MSUN / KPC ** 2.0 / mu / PROTONMASS * boxsize / res
    
    # choose non star-forming particles
    igas, = np.where(sfr > 0.0)
    print("star forming particles in box", len(igas))
    print("non star forming particles in box", npart - len(igas))
    s.data['rho'][igas] = 0.0
    
    axes(ax1)
    s.plot_Aslice("rho", colorbar=False, res=res, proj=True, axes=[1, 0], box=[boxsize, boxsize], proj_fact=fact, logplot=True, rasterized=True,
                  newfig=False, dextoshow=dextoshow, numthreads=numthreads)
    # clim( 1.0e21, 1.0e25 )
    
    colorbar(cax=cax, orientation='horizontal', format=matplotlib.ticker.LogFormatterMathtext())
    cax.set_title('$\\rm{N_{Si}\\,[cm^{-2}]}$', size=8)
    for label in cax.xaxis.get_ticklabels():
        label.set_fontsize(8)
    return


def plot_iron(ax1, cax, s, sf, boxsize, fact, res, dextoshow, numthreads):
    s.pos = s.pos[:s.nparticlesall[0], :]
    s.rho = s.rho[:s.nparticlesall[0]]
    
    box = 0.5 * boxsize * 1.0e-3  # reconvert to Mpc for the first selection
    igas, = np.where((np.abs(s.pos[:, 0]) < box) & (np.abs(s.pos[:, 1]) < box) & (np.abs(s.pos[:, 2]) < box))
    npart = len(igas)
    print("iron density plot selected particles in box", len(igas))
    
    sfr = s.data['sfr'][igas].astype('float64')
    s.data['pos'] = s.pos[igas] * 1.0e3
    s.data['type'] = s.type[igas]
    ne = s.data['ne'][igas].astype('float64')
    metallicity = s.data['gz'][igas].astype('float64')
    XH = s.data['gmet'][igas, element['H']].astype('float64')
    XFe = s.data['gmet'][igas, element['Fe']].astype('float64')
    yhelium = (1 - XH - metallicity) / (4. * XH);
    mu = 56.0
    # this conversion seems strange but basically rho is first converted to 10 Msun / kpc^3 by
    # multiplying by then and then in cm^-2 with all the other factors (this holds also for the
    # other projection functions). The factor boxsize / res is the dl of the projection
    s.data['rho'] = XFe * s.rho[igas].astype('float64') * 1.0e1 * MSUN / KPC ** 2.0 / mu / PROTONMASS * boxsize / res
    
    # choose non star-forming particles
    igas, = np.where(sfr > 0.0)
    print("star forming particles in box", len(igas))
    print("non star forming particles in box", npart - len(igas))
    s.data['rho'][igas] = 0.0
    
    axes(ax1)
    s.plot_Aslice("rho", colorbar=False, res=res, proj=True, axes=[1, 0], box=[boxsize, boxsize], proj_fact=fact, logplot=True, rasterized=True,
                  newfig=False, dextoshow=dextoshow, numthreads=numthreads)
    # clim( 1.0e21, 1.0e25 )
    
    colorbar(cax=cax, orientation='horizontal', format=matplotlib.ticker.LogFormatterMathtext())
    cax.set_title('$\\rm{N_{Fe}\\,[cm^{-2}]}$', size=8)
    for label in cax.xaxis.get_ticklabels():
        label.set_fontsize(8)
    return


def plot_NV(ax1, cax, s, sf, boxsize, fact, res, dextoshow, numthreads):
    s.pos = s.pos[:s.nparticlesall[0], :]
    s.rho = s.rho[:s.nparticlesall[0]]
    
    rad = sf.data['frc2'][0] * 1.0e3
    
    box = 0.5 * boxsize * 1.0e-3  # reconvert to Mpc for the first selection
    igas, = np.where((np.abs(s.pos[:, 0]) < box) & (np.abs(s.pos[:, 1]) < box) & (np.abs(s.pos[:, 2]) < box))
    npart = len(igas)
    print("NV density plot selected particles in box", len(igas))
    
    sfr = s.data['sfr'][igas].astype('float64')
    s.data['pos'] = s.pos[igas] * 1.0e3
    s.data['type'] = s.type[igas]
    ne = s.data['ne'][igas].astype('float64')
    metallicity = s.data['gz'][igas].astype('float64')
    XH = s.data['gmet'][igas, element['H']].astype('float64')
    XN = s.data['gmet'][igas, element['N']].astype('float64')
    AN = 14.0
    nh = s.data['nh'][igas].astype('float64')
    yhelium = (1 - XH - metallicity) / (4. * XH);
    mu = (1 + 4 * yhelium) / (1 + yhelium + ne)
    # this conversion seems strange but basically rho is first converted to 10 Msun / kpc^3 by
    # multiplying by then and then in cm^-2 with all the other factors (this holds also for the
    # other projection functions). The factor boxsize / res is the dl of the projection
    temp = np.log10(s.data['u'][igas].astype('float64') * 1.0e10 * mu * PROTONMASS / BOLTZMANN)
    XN *= 10.0 ** get_cie_ionization_fractions(element='nitrogen', ion='V', temperatures=temp)
    s.data['rho'] = XN * s.rho[igas].astype('float64') * 1.0e1 * MSUN / KPC ** 2.0 / AN / PROTONMASS * boxsize / res
    
    # choose non star-forming particles
    igas, = np.where(sfr > 0.0)
    print("star forming particles in box", len(igas))
    print("non star forming particles in box", npart - len(igas))
    s.data['rho'][igas] = 0.0
    
    axes(ax1)
    
    s.plot_Aslice("rho", logplot=True, colorbar=False, res=res, axes=[1, 0], proj_fact=fact, newfig=False, rasterized=True, box=[boxsize, boxsize],
                  proj=True, dextoshow=dextoshow, numthreads=numthreads)
    
    colorbar(cax=cax, orientation='horizontal', format=matplotlib.ticker.LogFormatterMathtext())
    cax.set_title('$\\rm{N_{NV}\\,[cm^{-2}]}$', size=8)
    for label in cax.xaxis.get_ticklabels():
        label.set_fontsize(8)
    return


def plot_OVI(ax1, cax, s, sf, boxsize, fact, res, dextoshow, numthreads):
    s.pos = s.pos[:s.nparticlesall[0], :]
    s.rho = s.rho[:s.nparticlesall[0]]
    
    box = 0.5 * boxsize * 1.0e-3  # reconvert to Mpc for the first selection
    igas, = np.where((np.abs(s.pos[:, 0]) < box) & (np.abs(s.pos[:, 1]) < box) & (np.abs(s.pos[:, 2]) < box))
    npart = len(igas)
    print("OVI density plot selected particles in box", len(igas))
    
    sfr = s.data['sfr'][igas].astype('float64')
    s.data['pos'] = s.pos[igas] * 1.0e3
    s.data['type'] = s.type[igas]
    ne = s.data['ne'][igas].astype('float64')
    metallicity = s.data['gz'][igas].astype('float64')
    XH = s.data['gmet'][igas, element['H']].astype('float64')
    XO = s.data['gmet'][igas, element['O']].astype('float64')
    AO = 16.0
    nh = s.data['nh'][igas].astype('float64')
    yhelium = (1 - XH - metallicity) / (4. * XH);
    mu = (1 + 4 * yhelium) / (1 + yhelium + ne)
    # this conversion seems strange but basically rho is first converted to 10 Msun / kpc^3 by
    # multiplying by then and then in cm^-2 with all the other factors (this holds also for the
    # other projection functions). The factor boxsize / res is the dl of the projection
    temp = np.log10(s.data['u'][igas].astype('float64') * 1.0e10 * mu * PROTONMASS / BOLTZMANN)
    XO *= 10.0 ** get_cie_ionization_fractions(element='oxygen', ion='VI', temperatures=temp)
    s.data['rho'] = XO * s.rho[igas].astype('float64') * 1.0e1 * MSUN / KPC ** 2.0 / AO / PROTONMASS * boxsize / res
    
    # choose non star-forming particles
    igas, = np.where(sfr > 0.0)
    print("star forming particles in box", len(igas))
    print("non star forming particles in box", npart - len(igas))
    s.data['rho'][igas] = 0.0
    
    axes(ax1)
    
    s.plot_Aslice("rho", logplot=True, colorbar=False, res=res, axes=[1, 0], proj_fact=fact, newfig=False, rasterized=True, box=[boxsize, boxsize],
                  proj=True, dextoshow=dextoshow, numthreads=numthreads)
    
    colorbar(cax=cax, orientation='horizontal', format=matplotlib.ticker.LogFormatterMathtext())
    cax.set_title('$\\rm{N_{OVI}\\,[cm^{-2}]}$', size=8)
    for label in cax.xaxis.get_ticklabels():
        label.set_fontsize(8)
    return


def plot_OVII(ax1, cax, s, sf, boxsize, fact, res, dextoshow, numthreads):
    s.pos = s.pos[:s.nparticlesall[0], :]
    s.rho = s.rho[:s.nparticlesall[0]]
    
    box = 0.5 * boxsize * 1.0e-3  # reconvert to Mpc for the first selection
    igas, = np.where((np.abs(s.pos[:, 0]) < box) & (np.abs(s.pos[:, 1]) < box) & (np.abs(s.pos[:, 2]) < box))
    npart = len(igas)
    print("OVII density plot selected particles in box", len(igas))
    
    sfr = s.data['sfr'][igas].astype('float64')
    s.data['pos'] = s.pos[igas] * 1.0e3
    s.data['type'] = s.type[igas]
    ne = s.data['ne'][igas].astype('float64')
    metallicity = s.data['gz'][igas].astype('float64')
    XH = s.data['gmet'][igas, element['H']].astype('float64')
    XO = s.data['gmet'][igas, element['O']].astype('float64')
    AO = 16.0
    nh = s.data['nh'][igas].astype('float64')
    yhelium = (1 - XH - metallicity) / (4. * XH);
    mu = (1 + 4 * yhelium) / (1 + yhelium + ne)
    # this conversion seems strange but basically rho is first converted to 10 Msun / kpc^3 by
    # multiplying by then and then in cm^-2 with all the other factors (this holds also for the
    # other projection functions). The factor boxsize / res is the dl of the projection
    temp = np.log10(s.data['u'][igas].astype('float64') * 1.0e10 * mu * PROTONMASS / BOLTZMANN)
    XO *= 10.0 ** get_cie_ionization_fractions(element='oxygen', ion='VII', temperatures=temp)
    s.data['rho'] = XO * s.rho[igas].astype('float64') * 1.0e1 * MSUN / KPC ** 2.0 / AO / PROTONMASS * boxsize / res
    
    # choose non star-forming particles
    igas, = np.where(sfr > 0.0)
    print("star forming particles in box", len(igas))
    print("non star forming particles in box", npart - len(igas))
    s.data['rho'][igas] = 0.0
    
    axes(ax1)
    
    s.plot_Aslice("rho", logplot=True, colorbar=False, res=res, axes=[1, 0], proj_fact=fact, newfig=False, rasterized=True, box=[boxsize, boxsize],
                  proj=True, dextoshow=dextoshow, numthreads=numthreads)
    
    colorbar(cax=cax, orientation='horizontal', format=matplotlib.ticker.LogFormatterMathtext())
    cax.set_title('$\\rm{N_{OVII}\\,[cm^{-2}]}$', size=8)
    for label in cax.xaxis.get_ticklabels():
        label.set_fontsize(8)
    return


def plot_SiIII(ax1, cax, s, sf, boxsize, fact, res, dextoshow, numthreads):
    s.pos = s.pos[:s.nparticlesall[0], :]
    s.rho = s.rho[:s.nparticlesall[0]]
    
    rad = sf.data['frc2'][0] * 1.0e3
    
    box = 0.5 * boxsize * 1.0e-3  # reconvert to Mpc for the first selection
    igas, = np.where((np.abs(s.pos[:, 0]) < box) & (np.abs(s.pos[:, 1]) < box) & (np.abs(s.pos[:, 2]) < box))
    npart = len(igas)
    print("SiIII density plot selected particles in box", len(igas))
    
    sfr = s.data['sfr'][igas].astype('float64')
    s.data['pos'] = s.pos[igas] * 1.0e3
    s.data['type'] = s.type[igas]
    ne = s.data['ne'][igas].astype('float64')
    metallicity = s.data['gz'][igas].astype('float64')
    XH = s.data['gmet'][igas, element['H']].astype('float64')
    XSi = s.data['gmet'][igas, element['Si']].astype('float64')
    ASi = 28.0
    nh = s.data['nh'][igas].astype('float64')
    yhelium = (1 - XH - metallicity) / (4. * XH);
    mu = (1 + 4 * yhelium) / (1 + yhelium + ne)
    # this conversion seems strange but basically rho is first converted to 10 Msun / kpc^3 by
    # multiplying by then and then in cm^-2 with all the other factors (this holds also for the
    # other projection functions). The factor boxsize / res is the dl of the projection
    temp = np.log10(s.data['u'][igas].astype('float64') * 1.0e10 * mu * PROTONMASS / BOLTZMANN)
    XSi *= 10.0 ** get_cie_ionization_fractions(element='silicon', ion='III', temperatures=temp)
    s.data['rho'] = XSi * s.rho[igas].astype('float64') * 1.0e1 * MSUN / KPC ** 2.0 / ASi / PROTONMASS * boxsize / res
    
    # choose non star-forming particles
    igas, = np.where(sfr > 0.0)
    print("star forming particles in box", len(igas))
    print("non star forming particles in box", npart - len(igas))
    s.data['rho'][igas] = 0.0
    
    axes(ax1)
    
    s.plot_Aslice("rho", logplot=True, colorbar=False, res=res, axes=[1, 0], proj_fact=fact, newfig=False, rasterized=True, box=[boxsize, boxsize],
                  proj=True, dextoshow=dextoshow, numthreads=numthreads)
    
    colorbar(cax=cax, orientation='horizontal', format=matplotlib.ticker.LogFormatterMathtext())
    cax.set_title('$\\rm{N_{SiIII}\\,[cm^{-2}]}$', size=8)
    for label in cax.xaxis.get_ticklabels():
        label.set_fontsize(8)
    return


def plot_SiIV(ax1, cax, s, sf, boxsize, fact, res, dextoshow, numthreads):
    s.pos = s.pos[:s.nparticlesall[0], :]
    s.rho = s.rho[:s.nparticlesall[0]]
    
    box = 0.5 * boxsize * 1.0e-3  # reconvert to Mpc for the first selection
    igas, = np.where((np.abs(s.pos[:, 0]) < box) & (np.abs(s.pos[:, 1]) < box) & (np.abs(s.pos[:, 2]) < box))
    npart = len(igas)
    print("SiIV density plot selected particles in box", len(igas))
    
    sfr = s.data['sfr'][igas].astype('float64')
    s.data['pos'] = s.pos[igas] * 1.0e3
    s.data['type'] = s.type[igas]
    ne = s.data['ne'][igas].astype('float64')
    metallicity = s.data['gz'][igas].astype('float64')
    XH = s.data['gmet'][igas, element['H']].astype('float64')
    XSi = s.data['gmet'][igas, element['Si']].astype('float64')
    ASi = 28.0
    nh = s.data['nh'][igas].astype('float64')
    yhelium = (1 - XH - metallicity) / (4. * XH);
    mu = (1 + 4 * yhelium) / (1 + yhelium + ne)
    # this conversion seems strange but basically rho is first converted to 10 Msun / kpc^3 by
    # multiplying by then and then in cm^-2 with all the other factors (this holds also for the
    # other projection functions). The factor boxsize / res is the dl of the projection
    temp = np.log10(s.data['u'][igas].astype('float64') * 1.0e10 * mu * PROTONMASS / BOLTZMANN)
    XSi *= 10.0 ** get_cie_ionization_fractions(element='silicon', ion='IV', temperatures=temp)
    s.data['rho'] = XSi * s.rho[igas].astype('float64') * 1.0e1 * MSUN / KPC ** 2.0 / ASi / PROTONMASS * boxsize / res
    
    # choose non star-forming particles
    igas, = np.where(sfr > 0.0)
    print("star forming particles in box", len(igas))
    print("non star forming particles in box", npart - len(igas))
    s.data['rho'][igas] = 0.0
    
    axes(ax1)
    
    s.plot_Aslice("rho", logplot=True, colorbar=False, res=res, axes=[1, 0], proj_fact=fact, newfig=False, rasterized=True, box=[boxsize, boxsize],
                  proj=True, dextoshow=dextoshow, numthreads=numthreads)
    
    colorbar(cax=cax, orientation='horizontal', format=matplotlib.ticker.LogFormatterMathtext())
    cax.set_title('$\\rm{N_{SiIV}\\,[cm^{-2}]}$', size=8)
    for label in cax.xaxis.get_ticklabels():
        label.set_fontsize(8)
    return


def plot_CIV(ax1, cax, s, sf, boxsize, fact, res, dextoshow, numthreads):
    s.pos = s.pos[:s.nparticlesall[0], :]
    s.rho = s.rho[:s.nparticlesall[0]]
    
    box = 0.5 * boxsize * 1.0e-3  # reconvert to Mpc for the first selection
    igas, = np.where((np.abs(s.pos[:, 0]) < box) & (np.abs(s.pos[:, 1]) < box) & (np.abs(s.pos[:, 2]) < box))
    npart = len(igas)
    print("CIV density plot selected particles in box", len(igas))
    
    sfr = s.data['sfr'][igas].astype('float64')
    s.data['pos'] = s.pos[igas] * 1.0e3
    s.data['type'] = s.type[igas]
    ne = s.data['ne'][igas].astype('float64')
    metallicity = s.data['gz'][igas].astype('float64')
    XH = s.data['gmet'][igas, element['H']].astype('float64')
    XC = s.data['gmet'][igas, element['C']].astype('float64')
    AC = 12.0
    nh = s.data['nh'][igas].astype('float64')
    yhelium = (1 - XH - metallicity) / (4. * XH);
    mu = (1 + 4 * yhelium) / (1 + yhelium + ne)
    # this conversion seems strange but basically rho is first converted to 10 Msun / kpc^3 by
    # multiplying by then and then in cm^-2 with all the other factors (this holds also for the
    # other projection functions). The factor boxsize / res is the dl of the projection
    temp = np.log10(s.data['u'][igas].astype('float64') * 1.0e10 * mu * PROTONMASS / BOLTZMANN)
    XC *= 10.0 ** get_cie_ionization_fractions(element='carbon', ion='IV', temperatures=temp)
    s.data['rho'] = XC * s.rho[igas].astype('float64') * 1.0e1 * MSUN / KPC ** 2.0 / AC / PROTONMASS * boxsize / res
    
    # choose non star-forming particles
    igas, = np.where(sfr > 0.0)
    print("star forming particles in box", len(igas))
    print("non star forming particles in box", npart - len(igas))
    s.data['rho'][igas] = 0.0
    
    axes(ax1)
    
    s.plot_Aslice("rho", logplot=True, colorbar=False, res=res, axes=[1, 0], proj_fact=fact, newfig=False, rasterized=True, box=[boxsize, boxsize],
                  proj=True, dextoshow=dextoshow, numthreads=numthreads)
    
    colorbar(cax=cax, orientation='horizontal', format=matplotlib.ticker.LogFormatterMathtext())
    cax.set_title('$\\rm{N_{CIV}\\,[cm^{-2}]}$', size=8)
    for label in cax.xaxis.get_ticklabels():
        label.set_fontsize(8)
    return