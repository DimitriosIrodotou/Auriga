from const import *
from loadmodules import *
from pylab import *
from scipy.optimize import curve_fit
from util import *


def plot_radial_DMdensity(runs, dirs, outpath, outputlistfile, tlist, suffix, nrows, ncols, logfit=False):
    readflag = np.zeros(len(runs))
    lstyle = [':', '--', '-']
    
    p = plot_helper.plot_helper()
    
    snaps = np.int_(select_snapshot_number.select_snapshot_number_gyr(outputlistfile[0], tlist))
    
    if isinstance(snaps, int):
        snaps = [snaps]
    
    for isnap, snap in enumerate(snaps):
        
        panels = ncols * nrows
        figure1 = multipanel_layout.multipanel_layout(nrows=nrows, ncols=ncols, npanels=panels, left=0.14, bottom=0.2)
        figure1.set_figure_layout()
        figure1.set_axis_locators(xminloc=25.0, xmajloc=50.0, yminloc=1., ymajloc=1.)
        figure1.set_axis_labels(xlabel=r"$\rm{r [kpc]}$", ylabel=r"$\rm{\rho [M_{\odot} pc^{-3}]}$")
        figure1.set_fontsize()
        figure1.set_axis_limits_and_aspect(xlim=[0.0, 199.], ylim=[1e-6, 2e1], logyaxis=True)
        
        clist = ['r', 'b', 'c', 'g', 'k']
        
        for d in range(len(runs)):
            # snap = select_snapshot_number.select_snapshot_number( outputlistfile[d], redshift )
            
            dd = dirs[d] + runs[d]
            ax = figure1.fig.axes[d]
            
            figure1.set_panel_title(panel=d, title="$\\rm %s\,%s$" % ("Au", runs[d].split('_')[1]), position='bottom left', color='k', fontsize=10)
            
            wpath = outpath + runs[d] + '/densityprof/'
            if not os.path.exists(wpath):
                os.makedirs(wpath)
            
            filename = wpath + 'fitDM_density.txt'
            if isnap == 0:
                f3 = open(filename, 'w')
                header = "%12s%12s%12s%12s\n" % ("Time", "rho0", "a", "MencRsun")
                f3.write(header)
            else:
                f3 = open(filename, 'a')
            
            print("Doing dir %s snap %d." % (dd, snap))
            attrs = ['pos', 'vel', 'mass', 'age']
            
            s = gadget_readsnap(snap, snappath=dd + '/output/', hdf5=True, loadonlytype=[1, 4], loadonly=attrs)
            sf = load_subfind(snap, dir=dd + '/output/', hdf5=True, loadonly=['fpos', 'frc2', 'svel', 'flty', 'fnsh', 'slty', 'fmc2'])
            s.calc_sf_indizes(sf)
            s.select_halo(sf, [], 3., use_principal_axis=True, use_cold_gas_spin=False, do_rotation=True)
            time = s.cosmology_get_lookback_time_from_a(s.time.astype('float64'), is_flat=True)
            g = parse_particledata(s, sf, attrs)  # , radialcut = 0.1*sf.data['frc2'][0] )
            g.prep_data()
            
            rad = sf.data['frc2'][0]
            nr = 40
            
            ddata = g.sgdata['ddata']
            mass = ddata['mass']
            pos = ddata['pos']
            vel = ddata['vel']
            
            dm_radius = pylab.sqrt((pos[:, :] ** 2).sum(axis=1))
            
            rbin = np.zeros(nr)
            
            coltab = ['b', 'k']
            symtab = ['o', '^']
            
            pn, edges = np.histogram(dm_radius, bins=nr, weights=mass, range=[0., rad])
            rbin = 0.5 * (edges[:-1] + edges[1:])
            vol = (4. * np.pi / 3.) * (edges[1:] ** 3 - edges[:-1] ** 3)
            rho = 1e-8 * pn / vol
            
            ax.semilogy(rbin * 1e3, rho, color='k', marker='o', markersize=2., linewidth=0.)
            
            # both together
            guess = (.001, 10.)
            
            sigma = rho
            # bounds = ([0., 0.1], [20., 30.])
            bounds = ([0., 0.1], [700., 30.])
            (popt, pcov) = curve_fit(p.NFW, rbin * 1e3, rho, guess, bounds=bounds, sigma=sigma)
            print("popt=", popt)
            ax.semilogy(rbin * 1e3, p.NFW(rbin * 1e3, popt[0], popt[1]), linestyle=':', color='b', lw=1., label="$\\rm{%.1f kpc}$" % (popt[1]))
            
            masscum = np.cumsum(mass[np.argsort(dm_radius)])
            massencRsun = masscum[dm_radius[np.argsort(dm_radius)] > 0.008][0]
            
            with open(filename, "a") as f3:
                param = "%12.3f%18.3f%12.3f%16.3f\n" % (time, popt[0] * 1e9, popt[1], massencRsun)
                f3.write(param)
            
            # ax.yaxis.set_ticks([1e-4, 1e-3, 1e-2, 1e-1, 1.])
            figure1.reset_axis_limits()
            
            ax.legend(loc='upper right', fontsize=8, frameon=False, ncol=1)
            
            figure1.fig.savefig('%s/DMrho_prof%03d.%s' % (outpath, snap, suffix))
    
    f3.close()