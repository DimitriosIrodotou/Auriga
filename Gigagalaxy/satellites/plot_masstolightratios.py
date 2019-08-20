import numpy as np
import read_McConnachie_table as read_data
from gadget import *
from pylab import *
from util.subhalos_utilities import subhalos_properties

toinch = 0.393700787

band_array = {'U': 0, 'B': 1, 'V': 2, 'K': 3, 'g': 4, 'r': 5, 'i': 6, 'z': 7}
Msunabs = [5.61, 5.48, 4.83, 3.30, 5.12, 4.68, 4.57, 4.60]


def plot_masstolightratios(runs, dirs, outpath, snap, suffix, band='V'):
    fac = 1.0
    fig = figure(figsize=np.array([16.1 * fac, 16.1]) * toinch * 0.5, dpi=300)
    
    for d in range(len(runs)):
        print
        'doing halo', runs[d]
        dd = dirs[d] + runs[d]
        s = gadget_readsnap(snap, snappath=dd + '/output/', hdf5=True, loadonlytype=[0, 1, 4], loadonly=['pos', 'mass', 'gsph', 'age'])
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
                starmass = s.data['mass'][subhalos.particlesoffsets[i, 4]:subhalos.particlesoffsets[i, 4] + subhalos.subhaloslentype[i, 4]].astype(
                    'float64')
                gasmass = s.data['mass'][subhalos.particlesoffsets[i, 0]:subhalos.particlesoffsets[i, 0] + subhalos.subhaloslentype[i, 0]].astype(
                    'float64')
                dmmass = s.data['mass'][subhalos.particlesoffsets[i, 1]:subhalos.particlesoffsets[i, 1] + subhalos.subhaloslentype[i, 1]].astype(
                    'float64')
                
                starpos = s.data['pos'][subhalos.particlesoffsets[i, 4]:subhalos.particlesoffsets[i, 4] + subhalos.subhaloslentype[i, 4]].astype(
                    'float64')
                gaspos = s.data['pos'][subhalos.particlesoffsets[i, 0]:subhalos.particlesoffsets[i, 0] + subhalos.subhaloslentype[i, 0]].astype(
                    'float64')
                dmpos = s.data['pos'][subhalos.particlesoffsets[i, 1]:subhalos.particlesoffsets[i, 1] + subhalos.subhaloslentype[i, 1]].astype(
                    'float64')
                
                # center to the subhalo position
                starpos[:, :] -= subhalos.subhalospos[i, :]
                gaspos[:, :] -= subhalos.subhalospos[i, :]
                dmpos[:, :] -= subhalos.subhalospos[i, :]
                
                gasradius = sqrt(((gaspos[:, :]) ** 2.0).sum(axis=1))
                dmradius = sqrt(((dmpos[:, :]) ** 2.0).sum(axis=1))
                
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
                    # print mm, 0.5 * totluminosity
                    j += 1
                
                halflightradius[i] = starradius[j - 1]
                
                k, = np.where(starradius <= halflightradius[i])
                massstar[i] = starmass[k].sum()
                k, = np.where(gasradius <= halflightradius[i])
                massgas[i] = gasmass[k].sum()
                k, = np.where(dmradius <= halflightradius[i])
                massdm[i] = dmmass[k].sum()
        
        j, = np.where((magnitudecat < 1.0e30) & (halflightradius > 0.0))
        ax.semilogy(magnitudecat[j], 1.0e10 * (massstar[j] + massgas[j] + massdm[j]) / totluminosity[j], 's', ms=5.0, mec='None', mfc='black',
                    label="$\\rm{%s}$" % runs[d])
        
        plot_masstolightvsMv(ax, band)
        
        xlim(-20.0, 0.)
        # ylim(-500.0, 500.)
        
        majorLocator = MultipleLocator(5.0)
        ax.xaxis.set_major_locator(majorLocator)
        minorLocator = MultipleLocator(1.0)
        ax.xaxis.set_minor_locator(minorLocator)
        
        ax.invert_xaxis()
        
        legend(loc='upper right', fontsize=6, frameon=False, numpoints=1)
        
        xlabel("$\\rm{M_{%s}}$" % band)
        ylabel("$\\rm{\\Gamma_{dyn}(\\leq\,r_{h})\\,[M_{\\odot}\\,L_{\\odot}^{-1}]}$")
        
        savefig("%s/masstolightratio_%s_%s.%s" % (outpath, runs[d], band, suffix), dpi=300)
        fig.clf()


def plot_masstolightvsMv(ax, band):
    data = read_data.read_data_catalogue(filename='./data/McConnachie2012_extended.dat')
    data.read_table()
    
    # computed at the half-light radius
    luminosity = 0.5 * 10 ** (-0.4 * (data.Vabsmagnitude - Msunabs[band_array[band]]))
    
    i, = np.where((data.dynamicalmass < 999.99e6) & (data.Vabsmagnitude < 99))
    data.Vabsmagnitude = data.Vabsmagnitude[i]
    data.dynamicalmass = data.dynamicalmass[i]
    data.associated = data.associated[i]
    luminosity = luminosity[i]
    
    i, = np.where((data.associated == 'G') | (data.associated == 'G,L'))
    ax.semilogy(data.Vabsmagnitude[i], data.dynamicalmass[i] / luminosity[i], 'o', ms=5.0, mec='None', mfc='r', label='Milky Way')
    i, = np.where((data.associated == 'A') | (data.associated == 'A,L'))
    ax.semilogy(data.Vabsmagnitude[i], data.dynamicalmass[i] / luminosity[i], 'o', ms=5.0, mec='None', mfc='b', label='Andromeda')
    i, = np.where((data.associated == 'L') | (data.associated == 'L,G') | (data.associated == 'L,A'))
    ax.semilogy(data.Vabsmagnitude[i], data.dynamicalmass[i] / luminosity[i], 'o', ms=5.0, mec='None', mfc='g', label='Local Group')