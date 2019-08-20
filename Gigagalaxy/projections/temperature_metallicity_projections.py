import matplotlib.ticker
import numpy as np
from loadmodules import *
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

def plot_gasdensity(ax1, cax, s, sf, boxsize, fact, res, dextoshow, daxes, numthreads):
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
    s.plot_Aslice("rho", colorbar=False, res=res, proj=True, axes=daxes, box=[boxsize, boxsize], proj_fact=fact, logplot=True, rasterized=True,
                  newfig=False, dextoshow=dextoshow, numthreads=numthreads)
    # clim( 1.0e21, 1.0e25 )
    
    rvircircle = matplotlib.patches.Circle((0.0, 0.0), rad, color='k', fill=False, lw=0.5)
    gca().add_artist(rvircircle)
    
    for i in range(len(isub)):
        fac = subhalosMass[i] / subhalosMassmax
        radius = minsatrad + (maxsatrad - minsatrad) * fac ** 0.3
        # rsatcircle = matplotlib.patches.Circle( (subhalosPos[i,1], subhalosPos[i,0]), radius, color='k', fill=False, lw=0.5 )
        rsatcircle = matplotlib.patches.RegularPolygon((subhalosPos[i, daxes[0]], subhalosPos[i, daxes[1]]), numVertices=4, radius=radius,
                                                       orientation=0.25 * np.pi, color='k', fill=False, lw=0.3)
        gca().add_artist(rsatcircle)
    
    colorbar(cax=cax, orientation='horizontal', format=matplotlib.ticker.LogFormatterMathtext())
    cax.set_title('$\\rm{N\\,[cm^{-2}]}$', size=8)
    for label in cax.xaxis.get_ticklabels():
        label.set_fontsize(8)
    return


def plot_temperature(ax1, cax, s, sf, boxsize, fact, res, dextoshow, daxes, numthreads):
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
    mu = (1 + 4 * yhelium) / (1 + yhelium + ne)
    # this conversion seems strange but basically rho is first converted to 10 Msun / kpc^3 by
    # multiplying by then and then in cm^-2 with all the other factors (this holds also for the
    # other projection functions). The factor boxsize / res is the dl of the projection
    rhotmp = s.data['rho']
    utmp = s.data['u']
    s.data['rho'] = s.rho[igas].astype('float64')
    s.data['u'] = s.data['u'][igas].astype('float64') * 1.0e10 * mu * PROTONMASS / BOLTZMANN
    
    # choose non star-forming particles
    igas, = np.where(sfr > 0.0)
    print("star forming particles in box", len(igas))
    print("non star forming particles in box", npart - len(igas))
    s.data['rho'][igas] = 0.0
    
    axes(ax1)
    
    s.plot_Aweightedslice("u", "rho", logplot=True, colorbar=False, res=res, axes=daxes, proj_fact=fact, newfig=False, rasterized=True,
                          box=[boxsize, boxsize], vrange=[1.0e4, 10. ** 6.5], proj=True, dextoshow=dextoshow, numthreads=numthreads)
    s.data['rho'] = rhotmp
    s.data['u'] = utmp
    
    colorbar(cax=cax, orientation='horizontal', format=matplotlib.ticker.LogFormatterMathtext())
    cax.set_title('$\\rm{T\\,[K]}$', size=8)
    for label in cax.xaxis.get_ticklabels():
        label.set_fontsize(8)
    return


def plot_metallicity(ax1, cax, s, sf, boxsize, fact, res, dextoshow, daxes, numthreads=1):
    s.pos = s.pos[:s.nparticlesall[0], :]
    s.rho = s.rho[:s.nparticlesall[0]]
    
    box = 0.5 * boxsize * 1.0e-3  # reconvert to Mpc for the first selection
    igas, = np.where((np.abs(s.pos[:, 0]) < box) & (np.abs(s.pos[:, 1]) < box) & (np.abs(s.pos[:, 2]) < box))
    npart = len(igas)
    print("oxygen density plot selected particles in box", len(igas))
    
    sfr = s.data['sfr'][igas].astype('float64')
    s.data['pos'] = s.pos[igas] * 1.0e3
    s.data['type'] = s.type[igas]
    # this conversion seems strange but basically rho is first converted to 10 Msun / kpc^3 by
    # multiplying by then and then in cm^-2 with all the other factors (this holds also for the
    # other projection functions). The factor boxsize / res is the dl of the projection
    rhotmp = s.data['rho']
    gztmp = s.data['gz']
    s.data['rho'] = s.rho[igas].astype('float64')
    s.data['gz'] = s.data['gz'][igas].astype('float64') / ZSUN
    
    # choose non star-forming particles
    igas, = np.where(sfr > 0.0)
    print("star forming particles in box", len(igas))
    print("non star forming particles in box", npart - len(igas))
    s.data['rho'][igas] = 0.0
    
    axes(ax1)
    
    s.plot_Aweightedslice("gz", "rho", logplot=True, colorbar=False, res=res, axes=daxes, proj_fact=fact, newfig=False, rasterized=True,
                          box=[boxsize, boxsize], proj=True, vrange=[1.0e-3, 10. ** 0.5], dextoshow=dextoshow, numthreads=numthreads)
    s.data['rho'] = rhotmp
    s.data['gz'] = gztmp
    
    colorbar(cax=cax, orientation='horizontal', format=matplotlib.ticker.LogFormatterMathtext())
    cax.set_title('$\\rm{Z\\,[Z_{\\odot}]}$', size=8)
    for label in cax.xaxis.get_ticklabels():
        label.set_fontsize(8)
    return


def plot_radial_velocity(ax1, cax, s, sf, boxsize, fact, res, dextoshow, daxes, numthreads=1):
    s.pos = s.pos[:s.nparticlesall[0], :]
    s.vel = s.vel[:s.nparticlesall[0], :]
    s.rho = s.rho[:s.nparticlesall[0]]
    
    box = 0.5 * boxsize * 1.0e-3  # reconvert to Mpc for the first selection
    
    igas, = np.where((np.abs(s.pos[:, 0]) < box) & (np.abs(s.pos[:, 1]) < box) & (np.abs(s.pos[:, 2]) < box))
    npart = len(igas)
    print("oxygen density plot selected particles in box", len(igas))
    
    sfr = s.data['sfr'][igas].astype('float64')
    s.data['pos'] = s.pos[igas] * 1.0e3
    s.data['vel'] = s.vel[igas]
    s.data['type'] = s.type[igas]
    rr = np.sqrt((s.pos[igas, :] ** 2).sum(axis=1))
    vrad = np.zeros(npart)
    print("len pos, vel, rr, vrad=", len(s.data['pos']), len(s.data['vel']), len(rr), len(vrad))
    vrad[:] = ((s.pos[igas, 0] * s.vel[igas, 0]) + (s.pos[igas, 1] * s.vel[igas, 1]) + (s.pos[igas, 2] * s.vel[igas, 2])) / rr[:]
    # this conversion seems strange but basically rho is first converted to 10 Msun / kpc^3 by
    # multiplying by then and then in cm^-2 with all the other factors (this holds also for the
    # other projection functions). The factor boxsize / res is the dl of the projection
    rhotmp = s.data['rho']
    gztmp = s.data['gz']
    s.data['rho'] = s.rho[igas].astype('float64')
    s.data['gz'] = s.data['gz'][igas].astype('float64') / ZSUN
    
    # choose non star-forming particles
    igas, = np.where(sfr > 0.0)
    print("star forming particles in box", len(igas))
    print("non star forming particles in box", npart - len(igas))
    s.data['rho'][igas] = 0.0
    
    axes(ax1)
    
    s.plot_Aweightedslice(vrad, "rho", logplot=False, colorbar=False, res=res, axes=daxes, proj_fact=fact, newfig=False, rasterized=True,
                          box=[boxsize, boxsize], proj=True, vrange=[-120., 120.], dextoshow=dextoshow, numthreads=numthreads)
    s.data['rho'] = rhotmp
    s.data['gz'] = gztmp
    
    colorbar(cax=cax, orientation='horizontal', format=matplotlib.ticker.LogFormatterMathtext())
    cax.set_title('$\\rm{V_{R}\\,[km \, s^{-1}]}$', size=8)
    for label in cax.xaxis.get_ticklabels():
        label.set_fontsize(8)
    return