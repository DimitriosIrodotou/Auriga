import numpy as np
import read_McConnachie_table as read_data
from gadget import *
from pylab import *
from util.subhalos_utilities import subhalos_properties

toinch = 0.393700787

band_array = {'U': 0, 'B': 1, 'V': 2, 'K': 3, 'g': 4, 'r': 5, 'i': 6, 'z': 7}
Msunabs = [5.61, 5.48, 4.83, 3.30, 5.12, 4.68, 4.57, 4.60]


def plot_sigmastars(runs, dirs, outpath, snap, suffix, band='V'):
    fac = 1.0
    fig = figure(figsize=np.array([16.1 * fac, 16.1]) * toinch * 0.5, dpi=300)
    
    for d in range(len(runs)):
        print
        'doing halo', runs[d]
        dd = dirs[d] + runs[d]
        s = gadget_readsnap(snap, snappath=dd + '/output/', hdf5=True, loadonlytype=[4], loadonly=['mass', 'pos', 'vel', 'gsph', 'age'])
        subhalos = subhalos_properties(snap, directory=dd + '/output/', hdf5=True, loadonly=['fnsh', 'flty', 'spos', 'svel', 'slty', 'ssph'])
        subhalos.set_subhalos_properties(parent_group=0, main_halo=True, center_to_main_halo=False, photometry=True)
        
        fileoffsets = np.zeros(6)
        fileoffsets[1:] = np.cumsum(s.nparticlesall[:-1])
        subhalos.particlesoffsets += fileoffsets
        
        magnitudecat = np.zeros(subhalos.numbersubhalos[subhalos.groupid])
        halflightradius = np.zeros(subhalos.numbersubhalos[subhalos.groupid])
        totluminosity = np.zeros(subhalos.numbersubhalos[subhalos.groupid])
        sigma = np.zeros(subhalos.numbersubhalos[subhalos.groupid])
        magnitudecat[:] = 1.0e31
        
        ax = axes([0.15 / fac, 0.13, 0.8 / fac, 0.8])
        
        # star from one to exclude main halo (we read this info because we needed a line of sight)
        for i in range(1, subhalos.numbersubhalos[subhalos.groupid]):
            if subhalos.subhaloslentype[i, 4] > 0:
                starmass = s.data['mass'][subhalos.particlesoffsets[i, 4]:subhalos.particlesoffsets[i, 4] + subhalos.subhaloslentype[i, 4]].astype(
                    'float64')
                starpos = s.data['pos'][subhalos.particlesoffsets[i, 4]:subhalos.particlesoffsets[i, 4] + subhalos.subhaloslentype[i, 4]].astype(
                    'float64')
                starvel = s.data['vel'][subhalos.particlesoffsets[i, 4]:subhalos.particlesoffsets[i, 4] + subhalos.subhaloslentype[i, 4]].astype(
                        'float64')
                
                # center to the subhalo position
                starpos[:, :] -= subhalos.subhalospos[i, :]
                starvel[:, :] -= subhalos.subhalosvel[i, :]
                
                los = subhalos.subhalospos[i, :] - subhalos.subhalospos[0, :]
                los /= sqrt((los * los).sum())
                
                istarsbeg = subhalos.particlesoffsets[i, 4] - fileoffsets[4]
                istarsend = istarsbeg + subhalos.subhaloslentype[i, 4]
                magnitudes = s.data['gsph'][istarsbeg:istarsend]
                
                j, = np.where(s.data['age'][istarsbeg:istarsend] > 0.)
                starmass = starmass[j]
                magnitudes = magnitudes[j]
                starradius = sqrt(((starpos[j, :]) ** 2.0).sum(axis=1))
                starvel = starvel[j]
                vel = (starvel[:, :] * los[None, :]).sum(axis=1)
                
                istar = np.argsort(starradius)
                starmass = starmass[istar]
                magnitudes = magnitudes[istar]
                vel = vel[istar]
                starradius = starradius[istar]
                
                print
                vel.shape
                
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
                vel = vel[k]
                massstar = starmass.sum()
                
                avgvel = ((starmass[:] * vel[:]).sum()) / massstar
                
                sigma[i] = sqrt((starmass[:] * (vel[:] - avgvel) ** 2.0).sum() / massstar)
        
        j, = np.where((magnitudecat < 1.0e30) & (halflightradius > 0.0))
        ax.loglog(1.0e6 * halflightradius[j], sigma[j], 's', ms=5.0, mec='None', mfc='black', label="$\\rm{%s}$" % runs[d])
        
        plot_sigmastarsvshalflight(ax, band)
        
        # xlim(10.0, 3000.)
        ylim(1.0, 100.)
        
        # minorLocator = MultipleLocator(0.1)
        # ax.yaxis.set_minor_locator(minorLocator)
        # majorLocator = MultipleLocator(5.0)
        # ax.xaxis.set_major_locator(majorLocator)
        # minorLocator = MultipleLocator(1.0)
        # ax.xaxis.set_minor_locator(minorLocator)
        
        legend(loc='upper left', fontsize=6, frameon=False, numpoints=1)
        
        # ax.set_xscale('log')
        # ax.set_yscale('log')
        
        xlabel("$\\rm{r_{h}\\,[pc]}$")
        ylabel("$\\rm{\\sigma_{*}\\,[km\\,s^{-1}]}$")
        
        savefig("%s/sigmastars_%s_%s.%s" % (outpath, runs[d], band, suffix), dpi=300)
        fig.clf()


def plot_sigmastarsvshalflight(ax, band):
    data = read_data.read_data_catalogue(filename='./data/McConnachie2012_extended.dat')
    data.read_table()
    
    i, = np.where((data.halflightrad < 9999.) & (data.sigmastars < 90.) & (data.errorsigmastarsdown < 99.) & (data.errorsigmastarsup < 99.) & (
                data.errorhalflightdown < 999.) & (data.errorhalflightup < 999.))
    yerr = np.zeros((2, len(i)))
    xerr = np.zeros((2, len(i)))
    yerr[0, :] = data.errorsigmastarsdown[i]
    yerr[1, :] = data.errorsigmastarsup[i]
    xerr[0, :] = data.errorhalflightdown[i]
    xerr[1, :] = data.errorhalflightup[i]
    
    data.halflightrad = data.halflightrad[i]
    data.sigmastars = data.sigmastars[i]
    data.associated = data.associated[i]
    
    k, = np.where(data.sigmastars - yerr[0, :] <= 0.0)
    yerr[0, k] -= 0.1
    
    print
    data.sigmastars - yerr[0, :]
    
    i, = np.where((data.associated == 'G') | (data.associated == 'G,L'))
    ax.errorbar(data.halflightrad[i], data.sigmastars[i], yerr=yerr[:, i], xerr=xerr[:, i], marker='s', ls='None', ms=3.0, mec='None', mfc='r',
                ecolor='r', label='Milky Way')
    i, = np.where((data.associated == 'A') | (data.associated == 'A,L'))
    ax.errorbar(data.halflightrad[i], data.sigmastars[i], yerr=yerr[:, i], xerr=xerr[:, i], marker='s', ls='None', ms=3.0, mec='None', mfc='b',
                ecolor='b', label='Andromeda')
    i, = np.where((data.associated == 'L') | (data.associated == 'L,G') | (data.associated == 'L,A'))
    ax.errorbar(data.halflightrad[i], data.sigmastars[i], yerr=yerr[:, i], xerr=xerr[:, i], marker='s', ls='None', ms=3.0, mec='None', mfc='g',
                ecolor='g', label='Local Group')