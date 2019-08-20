import matplotlib as mpl
from const import *
from gadget import *
from gadget_subfind import *
from matplotlib.colors import LogNorm
from util import multipanel_layout

mpl.rcParams['image.cmap'] = 'gray_r'

toinch = 0.393700787
BOLTZMANN = 1.38065e-16
PROTONMASS = 1.67262178e-24
GAMMA = 5.0 / 3.0
GAMMA_MINUS1 = GAMMA - 1.0
MSUN = 1.989e33
MPC = 3.085678e24
ZSUN = 0.0127
Gcosmo = 43.0071
# ZSUN = 0.02

element = {'H': 0, 'He': 1, 'C': 2, 'N': 3, 'O': 4, 'Ne': 5, 'Mg': 6, 'Si': 7, 'Fe': 8}


def compute_percentile_radii(min_radius, max_radius, nshells):
    hradius = np.zeros(nshells)
    dr = log10(max_radius / min_radius) / nshells
    
    i = np.arange(nshells)
    hradius[:] = min_radius * 10 ** ((i[:] + 0.5) * dr)
    
    return log10(hradius)


def avg_metalprofiles(radius, mass, metallicity, min_radius, max_radius, min_metal, max_metal, nshells, dispersions=True):
    i, = np.where((metallicity > min_metal) & (metallicity < max_metal))
    gradius = radius[i]
    gmass = mass[i]
    gmetallicity = metallicity[i] * ZSUN
    log_rmax = log10(max_radius)
    log_rmin = log10(min_radius)
    
    dr = (log_rmax - log_rmin) / nshells
    
    gasmetallicity = np.zeros(nshells)
    massbin = np.zeros(nshells)
    hradius = np.zeros(nshells)
    
    gasmetallicity[:], edge = np.histogram(np.log10(gradius[:]), bins=nshells, range=(log_rmin, log_rmax), weights=(gmass[:] * gmetallicity[:]))
    massbin[:], edge = np.histogram(np.log10(gradius[:]), bins=nshells, range=(log_rmin, log_rmax), weights=gmass[:])
    
    index, = np.where(massbin > 0.)
    gasmetallicity[index] /= massbin[index]
    
    if dispersions:
        gasmetallicitydisp = np.zeros(nshells)
        binind = np.digitize(np.log10(gradius), edge) - 1
        gasmetallicitydisp[:], edge = np.histogram(np.log10(gradius), bins=nshells, range=(log_rmin, log_rmax),
                                                   weights=(gmass[:] * (gmetallicity[:] - gasmetallicity[binind]) ** 2))
        
        gasmetallicitydisp[index] /= massbin[index]
        gasmetallicitydisp[:] = sqrt(gasmetallicitydisp[:]) / ZSUN
    
    gasmetallicity[:] /= ZSUN
    
    hradius[:] = 10. ** (0.5 * (edge[1:] + edge[:-1]))
    
    if dispersions:
        return hradius, gasmetallicity, gasmetallicitydisp
    else:
        return hradius, gasmetallicity


def plot_gasmetalposition_histogram(runs, dirs, outpath, snap, suffix, nrows, ncols):
    panels = len(runs)
    
    figure = multipanel_layout.multipanel_layout(nrows=nrows, ncols=ncols, npanels=panels)
    figure.set_figure_layout()
    figure.set_axis_limits_and_aspect(ylim=[-2.0, 1.0], xlim=[0.0, 3.0], logaxis=False)
    figure.set_axis_locators(xminloc=0.1, yminloc=0.1)
    figure.set_axis_labels(xlabel="$\\rm{log\\,d\\,[kpc]}$", ylabel="$\\rm{log(Z / Z_{\\odot})}$")
    figure.set_fontsize()
    
    save_snap = snap
    
    for d in range(len(runs)):
        dd = dirs[d] + runs[d]
        ax = figure.fig.axes[d]
        
        # if runs[d] == 'Aq-G_5':
        #	snap = 60
        # else:
        #	snap = save_snap
        
        print
        "Doing dir %s. snapshot %d" % (dd, snap)
        
        s = gadget_readsnap(snap, snappath=dd + '/output/', hdf5=True, loadonly=['pos', 'vel', 'mass', 'sfr', 'gz'])
        sf = load_subfind(snap, dir=dd + '/output/', hdf5=True, loadonly=['fpos', 'frc2', 'svel', 'flty', 'fnsh', 'slty'])
        s.calc_sf_indizes(sf)
        na = s.nparticlesall
        
        s.select_halo(sf, use_principal_axis=True, use_cold_gas_spin=False, do_rotation=True)
        
        # s.center = sf.data['fpos'][0,:]
        rad = sf.data['frc2'][0]
        
        print
        'Select gas cells'
        # this selects the diffuse gas particles and center them
        # i, = np.where( (s.r() < 2.0 * rad) & (s.type == 0) )
        i, = np.where((s.r() < 1.0) & (s.type == 0))
        mass = s.data['mass'][i].astype('float64')
        radius = s.r()[i]
        pos = s.data['pos'][i]
        # pos -= sf.data['fpos'][0,:]
        sfr = s.data['sfr'][i].astype('float64')
        metallicity = s.data['gz'][i].astype('float64')
        
        igas, = np.where(sfr <= 0.0)
        mass = mass[igas]
        radius = log10(1.0e3 * radius[igas])
        pos = pos[igas]
        metallicity = (metallicity[igas] / ZSUN)
        ngas = len(igas)
        
        print
        'original values'
        print
        'metal abundances: max', max(metallicity), 'min', min(metallicity)
        kk, = np.where(metallicity < 0.0)
        metallicity[kk] = 1.0e-40
        print
        'metallicity fix applied'
        print
        'metal abundances: max', max(metallicity), 'min', min(metallicity)
        print
        
        metallicity = np.log10(metallicity)
        
        print
        'Compute profiles'
        
        min_metal = -3.0
        max_metal = 1.0
        
        # cone aperture from xy plane
        m = tan(pi / 3.0)
        m *= m
        print
        m
        
        iperp, = np.where(((m * pos[:, 0] * pos[:, 0]) > (pos[:, 1] * pos[:, 1] + pos[:, 2] * pos[:, 2])))
        ipar, = np.where(((m * pos[:, 0] * pos[:, 0]) <= (pos[:, 1] * pos[:, 1] + pos[:, 2] * pos[:, 2])))
        
        (hradperp, avgmetperp, disp) = avg_metalprofiles(10. ** radius[iperp], mass[iperp], 10. ** metallicity[iperp], 1.0, 1000.0, 10. ** min_metal,
                                                         10. ** max_metal, 60)
        (hradpar, avgmetpar, disp) = avg_metalprofiles(10. ** radius[ipar], mass[ipar], 10. ** metallicity[ipar], 1.0, 1000.0, 10. ** min_metal,
                                                       10. ** max_metal, 60)
        ax.hist2d(radius, metallicity, bins=(60, 80), range=([0.0, 3.0], [min_metal, max_metal]), weights=mass, normed=False, rasterized=True,
                  norm=LogNorm())
        
        kk, = np.where(avgmetperp > 0.0)
        hradperp = hradperp[kk]
        avgmetperp = avgmetperp[kk]
        kk, = np.where(avgmetpar > 0.0)
        hradpar = hradpar[kk]
        avgmetpar = avgmetpar[kk]
        
        # up = avgmet + disp
        # down = avgmet - disp
        
        # print hrad
        # print avgmet
        # print avgmet + disp
        # print avgmet - disp
        
        # plotting in a log scale so no negative values allowed
        # kk, = np.where(down <= 0.0)
        # down[kk] = 1.0e-40
        
        # plot( hrad, quartiles[0,:], '-', color='r')
        ax.plot(log10(hradperp), log10(avgmetperp), '-', lw=1.0, color='r', label='perpendicular')
        ax.plot(log10(hradpar), log10(avgmetpar), '-', lw=1.0, color='b', label='parallel')
        # ax.plot( log10(hrad), log10(down), '-', lw=1.0, color='b')
        ax.legend(loc='upper right', frameon=False, prop={'size': 5}, numpoints=1)
        
        limits = figure.ylim
        ax.vlines(log10(1.0e3 * rad), limits[0], limits[1], linestyle=':', color='darkgray', lw=0.7)
        ax.vlines(log10(1.0e3 * 0.1 * rad), limits[0], limits[1], linestyle=':', color='darkgray', lw=0.7)
        
        figure.set_panel_title(panel=d, title="$\\rm{%s-{%s}}$" % (runs[d].split('_')[0], runs[d].split('_')[1]), position='bottom left')
    
    figure.reset_axis_limits()
    
    figure.fig.savefig("%s/metalpositionhist_multi%03d.%s" % (outpath, snap, suffix), dpi=300)