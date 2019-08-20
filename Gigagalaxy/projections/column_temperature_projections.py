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

def plot_coldgas(ax1, cax, s, sf, boxsize, fact, res, dextoshow, numthreads):
    s.pos = s.pos[:s.nparticlesall[0], :]
    s.rho = s.rho[:s.nparticlesall[0]]
    
    rad = sf.data['frc2'][0] * 1.0e3
    
    box = 0.5 * boxsize * 1.0e-3  # reconvert to Mpc for the first selection
    igas, = np.where((np.abs(s.pos[:, 0]) < box) & (np.abs(s.pos[:, 1]) < box) & (np.abs(s.pos[:, 2]) < box))
    npart = len(igas)
    print("cold gas density plot selected particles in box", len(igas))
    
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
    temp = s.data['u'][igas].astype('float64') * 1.0e10 * mu * PROTONMASS / BOLTZMANN
    
    # choose non star-forming particles
    igas, = np.where(sfr > 0.0)
    print("star forming particles in box", len(igas))
    print("non star forming particles in box", npart - len(igas))
    s.data['rho'][igas] = 0.0
    
    igas, = np.where(temp > 1.0e5)
    s.data['rho'][igas] = 0.0
    
    axes(ax1)
    s.plot_Aslice("rho", colorbar=False, res=res, proj=True, axes=[1, 0], box=[boxsize, boxsize], proj_fact=fact, logplot=True, rasterized=True,
                  newfig=False, dextoshow=dextoshow, numthreads=numthreads)
    # clim( 1.0e21, 1.0e25 )
    
    rvircircle = matplotlib.patches.Circle((0.0, 0.0), rad, color='k', fill=False, lw=0.5)
    gca().add_artist(rvircircle)
    
    colorbar(cax=cax, orientation='horizontal', format=matplotlib.ticker.LogFormatterMathtext())
    cax.set_title('$\\rm{N_{cold}\\,[cm^{-2}]}$', size=8)
    for label in cax.xaxis.get_ticklabels():
        label.set_fontsize(8)
    return


def plot_warmgas(ax1, cax, s, sf, boxsize, fact, res, dextoshow, numthreads):
    s.pos = s.pos[:s.nparticlesall[0], :]
    s.rho = s.rho[:s.nparticlesall[0]]
    
    box = 0.5 * boxsize * 1.0e-3  # reconvert to Mpc for the first selection
    igas, = np.where((np.abs(s.pos[:, 0]) < box) & (np.abs(s.pos[:, 1]) < box) & (np.abs(s.pos[:, 2]) < box))
    npart = len(igas)
    print("warm gas density plot selected particles in box", len(igas))
    
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
    s.data['rho'] = s.rho[igas].astype('float64') * 1.0e1 * MSUN / KPC ** 2.0 / mu / PROTONMASS * boxsize / res
    temp = s.data['u'][igas].astype('float64') * 1.0e10 * mu * PROTONMASS / BOLTZMANN
    
    # choose non star-forming particles
    igas, = np.where(sfr > 0.0)
    print("star forming particles in box", len(igas))
    print("non star forming particles in box", npart - len(igas))
    s.data['rho'][igas] = 0.0
    
    igas, = np.where((temp <= 1.0e5) | (temp > 1.0e6))
    s.data['rho'][igas] = 0.0
    
    axes(ax1)
    
    s.plot_Aslice("rho", logplot=True, colorbar=False, res=res, axes=[1, 0], proj_fact=fact, newfig=False, rasterized=True, box=[boxsize, boxsize],
                  proj=True, dextoshow=dextoshow, numthreads=numthreads)
    
    colorbar(cax=cax, orientation='horizontal', format=matplotlib.ticker.LogFormatterMathtext())
    cax.set_title('$\\rm{N_{warm}\\,[cm^{-2}]}$', size=8)
    for label in cax.xaxis.get_ticklabels():
        label.set_fontsize(8)
    return


def plot_hotgas(ax1, cax, s, sf, boxsize, fact, res, dextoshow, numthreads):
    s.pos = s.pos[:s.nparticlesall[0], :]
    s.rho = s.rho[:s.nparticlesall[0]]
    
    box = 0.5 * boxsize * 1.0e-3  # reconvert to Mpc for the first selection
    igas, = np.where((np.abs(s.pos[:, 0]) < box) & (np.abs(s.pos[:, 1]) < box) & (np.abs(s.pos[:, 2]) < box))
    npart = len(igas)
    print("hot gas density plot selected particles in box", len(igas))
    
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
    s.data['rho'] = s.rho[igas].astype('float64') * 1.0e1 * MSUN / KPC ** 2.0 / mu / PROTONMASS * boxsize / res
    temp = s.data['u'][igas].astype('float64') * 1.0e10 * mu * PROTONMASS / BOLTZMANN
    
    # choose non star-forming particles
    igas, = np.where(sfr > 0.0)
    print("star forming particles in box", len(igas))
    print("non star forming particles in box", npart - len(igas))
    s.data['rho'][igas] = 0.0
    
    igas, = np.where(temp <= 1.0e6)
    s.data['rho'][igas] = 0.0
    
    axes(ax1)
    
    s.plot_Aslice("rho", logplot=True, colorbar=False, res=res, axes=[1, 0], proj_fact=fact, newfig=False, rasterized=True, box=[boxsize, boxsize],
                  proj=True, dextoshow=dextoshow, numthreads=numthreads)
    
    colorbar(cax=cax, orientation='horizontal', format=matplotlib.ticker.LogFormatterMathtext())
    cax.set_title('$\\rm{N_{hot}\\,[cm^{-2}]}$', size=8)
    for label in cax.xaxis.get_ticklabels():
        label.set_fontsize(8)
    return