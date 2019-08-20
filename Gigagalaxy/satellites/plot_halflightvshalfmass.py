from gadget import *
from pylab import *
from util.subhalos_utilities import subhalos_properties

colors = ['g', 'b', 'r', 'k', 'c', 'y', 'm', 'purple']
toinch = 0.393700787

bands = ['U', 'B', 'V', 'K', 'g', 'r', 'i', 'z']
band_array = {'U': 0, 'B': 1, 'V': 2, 'K': 3, 'g': 4, 'r': 5, 'i': 6, 'z': 7}
Msunabs = [5.61, 5.48, 4.83, 3.30, 5.12, 4.68, 4.57, 4.60]


def halflightvshalfmass(runs, dirs, outpath, snap, suffix, band):
    fac = 1.0
    fig = figure(figsize=np.array([16.1 * fac, 16.1]) * toinch * 0.5, dpi=300)
    
    for d in range(len(runs)):
        print
        'doing halo', runs[d]
        dd = dirs[d] + runs[d]
        s = gadget_readsnap(snap, snappath=dd + '/output/', hdf5=True, loadonlytype=[4], loadonly=['pos', 'gsph', 'age'])
        subhalos = subhalos_properties(snap, directory=dd + '/output/', hdf5=True, loadonly=['fnsh', 'flty', 'spos', 'slty', 'ssph', 'shmt'])
        subhalos.set_subhalos_properties(parent_group=0, main_halo=False, center_to_main_halo=False, photometry=True)
        
        fileoffsets = np.zeros(6)
        fileoffsets[1:] = np.cumsum(s.nparticlesall[:-1])
        subhalos.particlesoffsets += fileoffsets
        
        magnitudecat = np.zeros(subhalos.numbersubhalos[subhalos.groupid])
        halflightradius = np.zeros(subhalos.numbersubhalos[subhalos.groupid])
        halfmassradius = np.zeros(subhalos.numbersubhalos[subhalos.groupid])
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
                halfmassradius[i] = subhalos.subhaloshalfmassradiustype[i, 4]
        
        # print magnitudecheck, magnitudecat[i], magnitudecheck - magnitudecat[i]  # print halflightradius[i], radius.max()
        
        j, = np.where((magnitudecat < 1.0e30) & (halflightradius > 0.0))
        ax.loglog(1.0e6 * halfmassradius[j], 1.0e6 * halflightradius[j], 'o', ms=3.0, color=colors[d], label="$\\rm{%s}$" % runs[d])
        
        # 1-to-1 realtion
        radii = arange(ax.get_xlim()[0], ax.get_xlim()[1])
        ax.loglog(radii, radii, marker='None', color='gray', ls='--')
        
        # xlim( 1.0e8, 1.0e13 )
        # ylim( 0.25, 0.35 )
        
        # minorLocator = MultipleLocator(0.5)
        # ax.xaxis.set_minor_locator(minorLocator)
        # minorLocator = MultipleLocator(0.05)
        # ax.yaxis.set_minor_locator(minorLocator)
        
        # legend( loc='best', fontsize=6, frameon=False, numpoints=1 )
        
        xlabel("$\\rm{r_{h}\\,[pc]}$")
        ylabel("$\\rm{r_{h}\\,(%s\\,band)\\,[pc]}$" % band)
        
        savefig("%s/halfightvshalfmass_%s_%s.%s" % (outpath, runs[d], band, suffix), dpi=300)
        fig.clf()


# wrappers to do the luminosity function in all bands
def plot_halflightvshalfmass(runs, dirs, outpath, snap, suffix, bandlist=[]):
    if not bandlist:
        bandlist = bands
    
    for i in bandlist:
        halflightvshalfmass(runs, dirs, outpath, snap, suffix, i)
    
    return