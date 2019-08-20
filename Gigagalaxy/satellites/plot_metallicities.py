import numpy as np
import read_McConnachie_table as read_data
from gadget import *
from pylab import *
from util.subhalos_utilities import subhalos_properties

toinch = 0.393700787

# from Asplund et al. (2009) Table 5
SUNABUNDANCES = {'H': 12.0, 'He': 10.98, 'C': 8.47, 'N': 7.87, 'O': 8.73, 'Ne': 7.97, 'Mg': 7.64, 'Si': 7.55, 'Fe': 7.54}
element = {'H': 0, 'He': 1, 'C': 2, 'N': 3, 'O': 4, 'Ne': 5, 'Mg': 6, 'Si': 7, 'Fe': 8}

band_array = {'U': 0, 'B': 1, 'V': 2, 'K': 3, 'g': 4, 'r': 5, 'i': 6, 'z': 7}
Msunabs = [5.61, 5.48, 4.83, 3.30, 5.12, 4.68, 4.57, 4.60]


def plot_metallicities(runs, dirs, outpath, snap, suffix, band='V'):
    fac = 1.0
    fig = figure(figsize=np.array([16.1 * fac, 16.1]) * toinch * 0.5, dpi=300)
    
    for d in range(len(runs)):
        print('doing halo', runs[d])
        dd = dirs[d] + runs[d]
        s = gadget_readsnap(snap, snappath=dd + '/output/', hdf5=True, loadonlytype=[4], loadonly=['pos', 'mass', 'gmet', 'gsph', 'age'])
        subhalos = subhalos_properties(snap, directory=dd + '/output/', hdf5=True, loadonly=['fnsh', 'flty', 'spos', 'slty', 'ssph'])
        subhalos.set_subhalos_properties(parent_group=0, main_halo=False, center_to_main_halo=False, photometry=True)
        
        fileoffsets = np.zeros(6)
        fileoffsets[1:] = np.cumsum(s.nparticlesall[:-1])
        subhalos.particlesoffsets += fileoffsets
        
        magnitudecat = np.zeros(subhalos.numbersubhalos[subhalos.groupid])
        halflightradius = np.zeros(subhalos.numbersubhalos[subhalos.groupid])
        totluminosity = np.zeros(subhalos.numbersubhalos[subhalos.groupid])
        metallicity = np.zeros(subhalos.numbersubhalos[subhalos.groupid])
        magnitudecat[:] = 1.0e31
        
        ax = axes([0.15 / fac, 0.13, 0.8 / fac, 0.8])
        
        for i in range(subhalos.numbersubhalos[subhalos.groupid]):
            if subhalos.subhaloslentype[i, 4] > 0:
                starmass = s.data['mass'][subhalos.particlesoffsets[i, 4]:subhalos.particlesoffsets[i, 4] + subhalos.subhaloslentype[i, 4]].astype(
                    'float64')
                
                starpos = s.data['pos'][subhalos.particlesoffsets[i, 4]:subhalos.particlesoffsets[i, 4] + subhalos.subhaloslentype[i, 4]].astype(
                    'float64')
                
                # center to the subhalo position
                starpos[:, :] -= subhalos.subhalospos[i, :]
                
                istarsbeg = subhalos.particlesoffsets[i, 4] - fileoffsets[4] + fileoffsets[1]
                istarsend = istarsbeg + subhalos.subhaloslentype[i, 4]
                abundances = s.data['gmet'][istarsbeg:istarsend]
                
                istarsbeg = subhalos.particlesoffsets[i, 4] - fileoffsets[4]
                istarsend = istarsbeg + subhalos.subhaloslentype[i, 4]
                magnitudes = s.data['gsph'][istarsbeg:istarsend]
                
                j, = np.where(s.data['age'][istarsbeg:istarsend] > 0.)
                starmass = starmass[j]
                abundances = abundances[j]
                magnitudes = magnitudes[j]
                starradius = sqrt(((starpos[j, :]) ** 2.0).sum(axis=1))
                
                istar = np.argsort(starradius)
                starmass = starmass[istar]
                abundances = abundances[istar]
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
                    # print mm, 0.5 * totluminosity
                    j += 1
                
                halflightradius[i] = starradius[j - 1]
                
                k, = np.where(starradius <= halflightradius[i])
                starmass = starmass[k]
                abundances = abundances[k]
                massstar = starmass.sum()
                
                totabundances = (abundances[:, :] * starmass[:, None]).sum(axis=0) / massstar
                k, = np.where(totabundances <= 0.0)
                totabundances[k] = 1.0e-30
                metallicity[i] = log10(totabundances[element['Fe']] / totabundances[element['H']] / 56.0)
                metallicity[i] -= (SUNABUNDANCES['Fe'] - SUNABUNDANCES['H'])
        
        j, = np.where((magnitudecat < 1.0e30) & (halflightradius > 0.0))
        ax.plot(magnitudecat[j], metallicity[j], 's', ms=5.0, mec='None', mfc='black', label="$\\rm{%s}$" % runs[d])
        
        plot_metallicityvsMv(ax, band)
        
        xlim(-20.0, 0.)
        ylim(-4.0, 0.5)
        
        minorLocator = MultipleLocator(0.1)
        ax.yaxis.set_minor_locator(minorLocator)
        majorLocator = MultipleLocator(5.0)
        ax.xaxis.set_major_locator(majorLocator)
        minorLocator = MultipleLocator(1.0)
        ax.xaxis.set_minor_locator(minorLocator)
        
        ax.invert_xaxis()
        
        legend(loc='upper left', fontsize=6, frameon=False, numpoints=1)
        
        xlabel("$\\rm{M_{V}}$")
        ylabel("$\\rm{<[Fe/H]>}$")
        
        savefig("%s/metallicities_%s_%s.%s" % (outpath, runs[d], band, suffix), dpi=300)
        fig.clf()


def plot_metallicityvsMv(ax, band):
    data = read_data.read_data_catalogue(filename='./data/McConnachie2012_extended.dat')
    data.read_table()
    
    i, = np.where((data.metallicity < 99.) & (data.errormetallicity < 90.) & (data.Vabsmagnitude < 99.) & (data.errorVabsmagnitudeup < 99.) & (
                data.errorVabsmagnitudedown < 99.))
    yerr = np.zeros(len(i))
    xerr = np.zeros((2, len(i)))
    yerr[:] = data.errormetallicity[i]
    xerr[0, :] = data.errorVabsmagnitudedown[i]
    xerr[1, :] = data.errorVabsmagnitudeup[i]
    
    data.Vabsmagnitude = data.Vabsmagnitude[i]
    data.metallicity = data.metallicity[i]
    data.associated = data.associated[i]
    
    i, = np.where((data.associated == 'G') | (data.associated == 'G,L'))
    ax.errorbar(data.Vabsmagnitude[i], data.metallicity[i], yerr=yerr[i], xerr=xerr[:, i], marker='s', ls='None', ms=3.0, mec='None', mfc='r',
                ecolor='r', label='Milky Way')
    i, = np.where((data.associated == 'A') | (data.associated == 'A,L'))
    ax.errorbar(data.Vabsmagnitude[i], data.metallicity[i], yerr=yerr[i], xerr=xerr[:, i], marker='s', ls='None', ms=3.0, mec='None', mfc='b',
                ecolor='b', label='Andromeda')
    i, = np.where((data.associated == 'L') | (data.associated == 'L,G') | (data.associated == 'L,A'))
    ax.errorbar(data.Vabsmagnitude[i], data.metallicity[i], yerr=yerr[i], xerr=xerr[:, i], marker='s', ls='None', ms=3.0, mec='None', mfc='g',
                ecolor='g', label='Local Group')
    i, = np.where((data.associated == 'N'))
    ax.errorbar(data.Vabsmagnitude[i], data.metallicity[i], yerr=yerr[i], xerr=xerr[:, i], marker='s', ls='None', ms=3.0, mec='None', mfc='magenta',
                ecolor='magenta', label='Nearby Galaxies')