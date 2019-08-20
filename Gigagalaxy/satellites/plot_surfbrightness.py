import numpy as np
import read_McConnachie_table as read_data
from gadget import *
from pylab import *
from util.subhalos_utilities import subhalos_properties

toinch = 0.393700787

band_array = {'U': 0, 'B': 1, 'V': 2, 'K': 3, 'g': 4, 'r': 5, 'i': 6, 'z': 7}
Msunabs = [5.61, 5.48, 4.83, 3.30, 5.12, 4.68, 4.57, 4.60]


def plot_surfacebrightness(runs, dirs, outpath, snap, suffix, band='V'):
    fac = 1.0
    fig = figure(figsize=np.array([16.1 * fac, 16.1]) * toinch * 0.5, dpi=300)
    
    for d in range(len(runs)):
        print('doing halo', runs[d])
        dd = dirs[d] + runs[d]
        s = gadget_readsnap(snap, snappath=dd + '/output/', hdf5=True, loadonlytype=[4], loadonly=['pos', 'gsph', 'age'])
        subhalos = subhalos_properties(snap, directory=dd + '/output/', hdf5=True, loadonly=['fnsh', 'flty', 'spos', 'slty', 'ssph'])
        subhalos.set_subhalos_properties(parent_group=0, main_halo=False, center_to_main_halo=False, photometry=True)
        
        fileoffsets = np.zeros(6)
        fileoffsets[1:] = np.cumsum(s.nparticlesall[:-1])
        subhalos.particlesoffsets += fileoffsets
        
        magnitudecat = np.zeros(subhalos.numbersubhalos[subhalos.groupid])
        halflightradius = np.zeros(subhalos.numbersubhalos[subhalos.groupid])
        totluminosity = np.zeros(subhalos.numbersubhalos[subhalos.groupid])
        massstar = np.zeros(subhalos.numbersubhalos[subhalos.groupid])
        massgas = np.zeros(subhalos.numbersubhalos[subhalos.groupid])
        massdm = np.zeros(subhalos.numbersubhalos[subhalos.groupid])
        magnitudecat[:] = 1.0e31
        
        ax = axes([0.15 / fac, 0.13, 0.8 / fac, 0.8])
        
        for i in range(subhalos.numbersubhalos[subhalos.groupid]):
            if subhalos.subhaloslentype[i, 4] > 0:
                starpos = s.data['pos'][subhalos.particlesoffsets[i, 4]:subhalos.particlesoffsets[i, 4] + subhalos.subhaloslentype[i, 4]].astype(
                    'float64')
                
                # center to the subhalo position
                starpos[:, :] -= subhalos.subhalospos[i, :]
                
                istarsbeg = subhalos.particlesoffsets[i, 4] - fileoffsets[4]
                istarsend = istarsbeg + subhalos.subhaloslentype[i, 4]
                magnitudes = s.data['gsph'][istarsbeg:istarsend]
                
                j, = np.where(s.data['age'][istarsbeg:istarsend] > 0.)
                magnitudes = magnitudes[j]
                starradius = sqrt(((starpos[j, :]) ** 2.0).sum(axis=1))
                
                istar = np.argsort(starradius)
                magnitudes = magnitudes[istar]
                starradius = starradius[istar]
                
                luminosity = (10 ** (-0.4 * (magnitudes[:, band_array[band]] - Msunabs[band_array[band]])))
                totluminosity[i] = luminosity.sum()
                magnitudecat[i] = subhalos.subhalosphotometry[i, band_array[band]]
                
                if len(istar) == 1:
                    continue
                
                j = 0
                mm = 0.0
                while mm < 0.5 * totluminosity[i]:
                    mm += luminosity[j]
                    # print(mm, 0.5 * totluminosity)
                    j += 1
                
                halflightradius[i] = starradius[j - 1]
                k, = np.where(starradius <= halflightradius[i])
                
                # this is the luminosity at the half-light radius
                totluminosity[i] = luminosity[k].sum()
                area = (np.pi * (1.0e6 * halflightradius[i]) ** 2.0)
                
                # 1 pc seen at 10 pc is 0.1 radians
                totluminosity[i] = lum_square_pc_to_mag_squarearcsec(totluminosity[i] / area, 1.0, band)
        
        j, = np.where((magnitudecat < 1.0e30) & (halflightradius > 0.0))
        ax.plot(magnitudecat[j], totluminosity[j], 's', ms=5.0, mec='None', mfc='black', label="$\\rm{%s}$" % runs[d])
        
        plot_surfbrightvsMv(ax, band)
        
        legend(loc='upper left', fontsize=6, frameon=False, numpoints=1)
        
        # xlim(0.0, 600.)
        # ylim(16.0, 31.)
        
        minorLocator = MultipleLocator(1.0)
        ax.xaxis.set_minor_locator(minorLocator)
        minorLocator = MultipleLocator(1.0)
        ax.yaxis.set_minor_locator(minorLocator)
        majorLocator = MultipleLocator(5.0)
        ax.yaxis.set_major_locator(majorLocator)
        
        ax.invert_xaxis()
        ax.invert_yaxis()
        
        xlabel("$\\rm{M_{V}}$")
        ylabel("$\\rm{\\mu_{eff}\\,[mag\\,\\,arcsec^{-2}]}$")
        
        savefig("%s/surfacebrightness_%s_%s.%s" % (outpath, runs[d], band, suffix), dpi=300)
        fig.clf()


def rad_to_arcsec(rad):
    return rad * (3600. * 180. / np.pi)


def lum_square_pc_to_mag_squarearcsec(surf_bright, rad, filter):
    mag_sqarcsec = -2.5 * log10(surf_bright) + 5.0 * log10(rad_to_arcsec(rad)) + Msunabs[band_array[filter]] - 5.0
    return mag_sqarcsec


def plot_surfbrightvsMv(ax, band):
    data = read_data.read_data_catalogue(filename='./data/McConnachie2012_extended.dat')
    data.read_table()
    
    i, = np.where((data.halflightrad < 9999.) & (data.Vabsmagnitude < 99))
    data.halflightrad = data.halflightrad[i]
    data.Vabsmagnitude = data.Vabsmagnitude[i]
    data.associated = data.associated[i]
    
    # this is the luminosity at the half-light radius
    luminosity = 0.5 * 10 ** (-0.4 * (data.Vabsmagnitude - Msunabs[band_array[band]]))
    area = (np.pi * data.halflightrad ** 2.0)
    
    # 1 pc seen at 10 pc is 0.1 radians
    luminosity = lum_square_pc_to_mag_squarearcsec(luminosity / area, 1.0, band)
    
    i, = np.where((data.associated == 'G') | (data.associated == 'G,L'))
    ax.plot(data.Vabsmagnitude[i], luminosity[i], 'o', ms=5.0, mec='None', mfc='r', label='Milky Way')
    i, = np.where((data.associated == 'A') | (data.associated == 'A,L'))
    ax.plot(data.Vabsmagnitude[i], luminosity[i], 'o', ms=5.0, mec='None', mfc='b', label='Andromeda')
    i, = np.where((data.associated == 'L') | (data.associated == 'L,G') | (data.associated == 'L,A'))
    ax.plot(data.Vabsmagnitude[i], luminosity[i], 'o', ms=5.0, mec='None', mfc='g', label='Local Group')
    i, = np.where((data.associated == 'N'))
    ax.plot(data.Vabsmagnitude[i], luminosity[i], 'o', ms=5.0, mec='None', mfc='magenta', label='Nearby Galaxies')