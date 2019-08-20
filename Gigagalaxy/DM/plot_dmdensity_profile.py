from const import *
from gadget import *
from gadget_subfind import *
from pylab import *
from scipy.optimize import curve_fit
from util import multipanel_layout

toinch = 0.393700787
Gcosmo = 43.0071
ZSUN = 0.0127


def nfw_profile(x, rhocdelta, rs):
    y = rhocdelta / ((x / rs) * (1. + x / rs) ** 2)
    return y


def plot_dmdensity_profile(runs, dirs, outpath, snaps, suffix, nrows, ncols):
    panels = len(runs)
    figure = multipanel_layout.multipanel_layout(nrows=nrows, ncols=ncols, npanels=panels)
    figure.set_figure_layout()
    figure.set_axis_locators(xminloc=1.0, xmajloc=5.0, yminloc=1., ymajloc=5.)
    figure.set_fontsize(5)
    figure.set_axis_labels(xlabel="$\\rm{R [kpc]}$", ylabel="$\\rm{ [M_{\odot} kpc^{-3}]}$")
    
    colors = ['m', 'r', 'y', 'g', 'c', 'b', 'k']
    print
    "snaps =", snaps
    
    apath = outpath + '/plotall/'
    
    for d in range(len(runs)):
        dd = dirs[d] + runs[d]
        
        ax = figure.fig.axes[d]
        
        nshells = 35
        
        for isnap in range(len(snaps)):
            snap = snaps[isnap]
            print
            "Doing dir %s snap %d." % (dd, snap)
            
            s = gadget_readsnap(snap, snappath=dd + '/output/', hdf5=True, loadonlytype=[1, 4], loadonly=['pos', 'vel', 'mass', 'sfr', 'pot', 'age'])
            time[isnap] = s.cosmology_get_lookback_time_from_a(s.time.astype('float64'))
            if time[isnap] < 0.0001:
                time[isnap] = 0.0
            t0 = str(time[isnap])
            t0l = 'ti=' + t0[:4] + ' Gyr'
            sf = load_subfind(snap, dir=dd + '/output/', hdf5=True)
            s.calc_sf_indizes(sf)
            na = s.nparticlesall
            
            galrad = 0.1 * sf.data['frc2'][0]
            print
            'galrad:', galrad
            
            Rcut = galrad
            rd = np.linspace(0.0, Rcut, nshells)
            mnow = np.zeros(len(rd))
            
            center = None
            age_select = 3.
            s.select_halo(sf, center, age_select, use_principal_axis=True, use_cold_gas_spin=False, do_rotation=True)
            mass = s.data['mass'].astype('float64')
            print
            "Time = ", time[isnap]
            st = na[:4].sum();
            en = st + na[4]
            age = np.zeros(s.npartall)
            age[st:en] = s.data['age']
            rad = pylab.sqrt((s.pos[:, :] ** 2).sum(axis=1))
            z = s.pos[:, 0]
            
            iall, = np.where((rad < sf.data['frc2'][0]) & (rad > 0.))
            istars, = np.where((rad < Rcut) & (rad > 0.) & (s.type == 4) & (age > 0.))
            idm, = np.where((rad < sf.data['frc2'][0]) & (rad > 0.) & (s.type == 1))
            
            nstars = size(istars)
            nall = size(iall)
            rr = pylab.sqrt((s.pos[iall, :] ** 2).sum(axis=1))
            print
            "nstars=", nstars
            
            pos = s.pos[iall, :].astype('float64')
            vel = s.vel[iall, :].astype('float64')
            mass = s.data['mass'][iall].astype('float64')
            ptype = s.data['type'][iall].astype('int32')
            pot = s.data['pot'][iall].astype('float64')
            radius = pylab.sqrt((s.pos[iall, :] ** 2).sum(axis=1))
            age = age[iall]
            
            nn, = np.where((ptype[:] == 1) & (radius < sf.data['frc2'][0]))
            ndm = size(nn)
            
            dmass = np.zeros(ndm)
            dradius = np.zeros(ndm)
            
            dmass = mass[nn]
            dradius = radius[nn]
            
            mshell, edges = np.histogram(dradius, weights=dmass, bins=nshells, range=[0., sf.data['frc2'][0]])
            
            x = np.zeros(nshells)
            x[:] = 0.5 * (edges[:-1] + edges[1:])
            vol = np.zeros(len(x))
            den = np.zeros(len(x))
            vol[:] = (4. / 3.) * np.pi * (edges[i + 1] ** 3 - edges[i] ** 3)
            den[:] = mshell[:] / vol[:]
            
            den *= 1e-9
            r = x * 1e3
            
            rfit = sf.data['frc2'][0] * 1e3
            guess = (1e-3, 15.)
            
            indy = find_nearest(x, [sf.data['frc2'][0]]).astype('int64')
            if snap == snaps[-1]:
                (popt, pcov) = curve_fit(nfw_profile, r[:indy], (den[:indy]), guess)
                
                print
                "popt=", popt
                print
                "pcov=", pcov
            
            figure.set_axis_limits_and_aspect(xlim=[1e1, 3e2], ylim=[1e1, 1e8], logaxis=True)
            ax.loglog(r, 1e10 * den, linewidth=0.7, color=colors[isnap], linestyle='-')
            if snap == snaps[-1]:
                ax.loglog(r, 1e10 * nfw_profile(r, popt[0], popt[1]), color='r', linestyle='--',
                          label='$\\rm{ c = %.1f }$' % (1e3 * sf.data['frc2'][0] / popt[1]))
            ax.xaxis.set_ticks([1., 10., 100.])
            ax.legend(loc='upper right', frameon=False, prop={'size': 5})
            
            figure.set_panel_title(panel=d, title="$\\rm %s\, %s$" % ("Au", runs[d].split('_')[1]), position='bottom right')
    
    figure.reset_axis_limits()
    
    print
    "Saving figures"
    figure.fig.savefig('%s/dmden_prof%03d.%s' % (apath, snap, suffix))
    figure.fig.savefig('%s/dmden_prof%03d.ps' % (apath, snap))


def find_nearest(array, value):
    if len(value) == 1:
        idx = (np.abs(array - value)).argmin()
    else:
        idx = np.zeros(len(value))
        for i in range(len(value)):
            idx[i] = (np.abs(array - value[i])).argmin()
    
    return idx