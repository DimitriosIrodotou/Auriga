import numpy as np
import read_McConnachie_table as read_data
from gadget import *
from pylab import *
from util.subhalos_utilities import subhalos_properties

colors = ['g', 'b', 'r', 'k', 'c', 'y', 'm', 'purple']
toinch = 0.393700787

bands = ['U', 'B', 'V', 'K', 'g', 'r', 'i', 'z']
band_array = {'U': 0, 'B': 1, 'V': 2, 'K': 3, 'g': 4, 'r': 5, 'i': 6, 'z': 7}
Msunabs = [5.61, 5.48, 4.83, 3.30, 5.12, 4.68, 4.57, 4.60]


def halflightradii(runs, dirs, outpath, snap, suffix, band):
    fac = 1.0
    fig = figure(figsize=np.array([16.1 * fac, 16.1]) * toinch * 0.5, dpi=300)
    
    for d in range(len(runs)):
        print
        'doing halo', runs[d]
        dd = dirs[d] + runs[d]
        s = gadget_readsnap(snap, snappath=dd + '/output/', hdf5=True, loadonlytype=[4], loadonly=['pos', 'gmet', 'gsph', 'age'])
        subhalos = subhalos_properties(snap, directory=dd + '/output/', hdf5=True, loadonly=['fnsh', 'flty', 'spos', 'slty', 'ssph'])
        subhalos.set_subhalos_properties(parent_group=0, main_halo=False, center_to_main_halo=False, photometry=True, verbose=True)
        
        fileoffsets = np.zeros(6)
        fileoffsets[1:] = np.cumsum(s.nparticlesall[:-1])
        subhalos.particlesoffsets += fileoffsets
        
        magnitudecat = np.zeros(subhalos.numbersubhalos[subhalos.groupid])
        halflightradius = np.zeros(subhalos.numbersubhalos[subhalos.groupid])
        magnitudecat[:] = 1.0e31
        
        ax = axes([0.15 / fac, 0.13, 0.8 / fac, 0.8])
        
        for i in range(subhalos.numbersubhalos[subhalos.groupid]):
            if subhalos.subhaloslentype[i, 4] > 0:
                pos = s.data['pos'][subhalos.particlesoffsets[i, 4]:subhalos.particlesoffsets[i, 4] + subhalos.subhaloslentype[i, 4]].astype(
                    'float64')
                
                # center to the subhalo position
                pos[:, :] -= subhalos.subhalospos[i, :]
                
                istarsbeg = subhalos.particlesoffsets[i, 4] - fileoffsets[4]
                istarsend = istarsbeg + subhalos.subhaloslentype[i, 4]
                magnitudes = s.data['gsph'][istarsbeg:istarsend]
                
                j, = np.where(s.data['age'][istarsbeg:istarsend] > 0.)
                magnitudes = magnitudes[j]
                radius = sqrt(((pos[j, :]) ** 2.0).sum(axis=1))
                
                istar = np.argsort(radius)
                magnitudes = magnitudes[istar]
                radius = radius[istar]
                
                luminosity = (10 ** (-0.4 * (magnitudes[:, band_array[band]] - Msunabs[band_array[band]])))
                totluminosity = luminosity.sum()
                
                if len(istar) == 1:
                    continue
                
                j = 0
                mm = 0.0
                while mm < 0.5 * totluminosity:
                    mm += luminosity[j]
                    # print mm, 0.5 * totluminosity
                    j += 1
                
                halflightradius[i] = radius[j - 1]
                
                magnitudecheck = -2.5 * log10(totluminosity) + Msunabs[band_array[band]]
                magnitudecat[i] = subhalos.subhalosphotometry[i, band_array[band]]
        
        # print magnitudecheck, magnitudecat[i], magnitudecheck - magnitudecat[i]  # print halflightradius[i], radius.max()
        
        j, = np.where((magnitudecat < 1.0e30) & (halflightradius > 0.0))
        ax.semilogx(1.0e6 * halflightradius[j], magnitudecat[j], 's', ms=5.0, mec='None', mfc='black', color=colors[d], label="$\\rm{%s}$" % runs[d])
        
        if band == 'V':
            plot_Mvvshalflight(ax)
        
        legend(loc='upper left', fontsize=6, frameon=False, numpoints=1)
        
        minorLocator = MultipleLocator(1.0)
        ax.yaxis.set_minor_locator(minorLocator)
        
        xlabel("$\\rm{r_{h}\\,[pc]}$")
        ylabel("$\\rm{M_{V}}$")
        
        ax.invert_yaxis()
        
        savefig("%s/halflightradii_%s_%s.%s" % (outpath, runs[d], band, suffix), dpi=300)
        fig.clf()


# wrappers to do the luminosity function in all bands
def plot_halflightradii(runs, dirs, outpath, snap, suffix, bandlist=[]):
    if not bandlist:
        bandlist = bands
    
    for i in bandlist:
        halflightradii(runs, dirs, outpath, snap, suffix, i)
    
    return


def plot_Mvvshalflight(ax):
    data = read_data.read_data_catalogue(filename='./data/McConnachie2012_extended.dat')
    data.read_table()
    
    i, = np.where((data.halflightrad < 9999.) & (data.Vabsmagnitude < 99))
    data.halflightrad = data.halflightrad[i]
    data.Vabsmagnitude = data.Vabsmagnitude[i]
    data.associated = data.associated[i]
    
    i, = np.where((data.associated == 'G') | (data.associated == 'G,L'))
    ax.semilogx(data.halflightrad[i], data.Vabsmagnitude[i], 'o', ms=5.0, mec='None', mfc='r', label='Milky Way')
    i, = np.where((data.associated == 'A') | (data.associated == 'A,L'))
    ax.semilogx(data.halflightrad[i], data.Vabsmagnitude[i], 'o', ms=5.0, mec='None', mfc='b', label='Andromeda')
    i, = np.where((data.associated == 'L') | (data.associated == 'L,G') | (data.associated == 'L,A'))
    ax.semilogx(data.halflightrad[i], data.Vabsmagnitude[i], 'o', ms=5.0, mec='None', mfc='g', label='Local Group')
    i, = np.where((data.associated == 'N'))
    ax.semilogx(data.halflightrad[i], data.Vabsmagnitude[i], 'o', ms=5.0, mec='None', mfc='magenta', label='Nearby Galaxies')