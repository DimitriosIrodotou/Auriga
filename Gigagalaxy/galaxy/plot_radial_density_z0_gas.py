from const import *
from loadmodules import *
from parse_particledata import parse_particledata
from pylab import *
from scipy.optimize import curve_fit
from util import *

p = plot_helper.plot_helper()
fitdict = {"exp_only":  p.exp_prof, "exp_exp": p.total_profilede, "nfw_exp": p.total_NFW_exp, "hern_exp": p.total_hernquist_exp,
           "hern_only": p.hernquist_bulge}
colors = ['b', 'r']


def plot_radial_density_z0_gas(runs, dirs, outpath, outputlistfile, tlist, suffix, nrows, ncols, rran=[0.0075, 0.0085], fit="exp_only", logfit=False):
    readflag = np.zeros(len(runs))
    lstyle = [':', '--', '-']
    
    snaps = np.int_(select_snapshot_number.select_snapshot_number_gyr(outputlistfile[0], tlist))
    
    if isinstance(snaps, int):
        snaps = [snaps]
    
    for isnap, snap in enumerate(snaps):
        
        panels = ncols * nrows
        figure1 = multipanel_layout.multipanel_layout(nrows=nrows, ncols=ncols, npanels=panels, left=0.14, bottom=0.2)
        figure1.set_figure_layout()
        figure1.set_axis_locators(xminloc=1.0, xmajloc=5.0, yminloc=1., ymajloc=1.)
        figure1.set_axis_labels(xlabel=r"$\rm{R [kpc]}$", ylabel=r"$\rm{\rho [M_{\odot} pc^{-3}]}$")
        figure1.set_fontsize()
        figure1.set_axis_limits_and_aspect(xlim=[0.0, 14.5], ylim=[1e-4, 5e1], logyaxis=True)
        
        clist = ['r', 'b', 'c', 'g', 'k']
        
        profs = fit.split('_')
        
        for d in range(len(runs)):
            # snap = select_snapshot_number.select_snapshot_number( outputlistfile[d], redshift )
            
            dd = dirs[d] + runs[d]
            ax = figure1.fig.axes[d]
            
            figure1.set_panel_title(panel=d, title="$\\rm %s\,%s$" % ("Au", runs[d].split('_')[1]), position='bottom left', color='k', fontsize=10)
            
            wpath = outpath + runs[d] + '/densityprof/'
            if not os.path.exists(wpath):
                os.makedirs(wpath)
            
            filename = wpath + 'fitgas_hR_z0.txt'
            
            if isnap == 0:
                f3 = open(filename, 'w')
                header = "%12s%12s%12s%12s%12s\n" % ("Time", "rho01", "h1", "rho02", "hz")
                f3.write(header)
            else:
                f3 = open(filename, 'a')
            
            nr = 30
            galrad = 0.02
            zmax = 0.0005
            
            print("Doing dir %s snap %d." % (dd, snap))
            attrs = ['pos', 'vel', 'mass', 'age']
            
            loadtype = [0, 4]
            attrs.extend(['sfr'])
            
            s = gadget_readsnap(snap, snappath=dd + '/output/', hdf5=True, loadonlytype=loadtype, loadonly=attrs)
            sf = load_subfind(snap, dir=dd + '/output/', hdf5=True, loadonly=['fpos', 'frc2', 'svel', 'flty', 'fnsh', 'slty', 'fmc2'])
            s.calc_sf_indizes(sf)
            s.select_halo(sf, 3., use_principal_axis=True, use_cold_gas_spin=False, do_rotation=True)
            time = s.cosmology_get_lookback_time_from_a(s.time.astype('float64'), is_flat=True)
            g = parse_particledata(s, sf, attrs, radialcut=0.1 * sf.data['frc2'][0])
            g.prep_data()
            
            sdata = g.sgdata['sdata']
            mass = sdata['mass']
            pos = sdata['pos']
            vel = sdata['vel']
            star_age = sdata['age']
            
            zmax = 0.001
            
            gdata = g.sgdata['gdata']
            sfr = gdata['sfr']
            sfind, = np.where((abs(gdata['pos'][:, 0]) < zmax) & (sfr > 0.))
            gmass = gdata['mass'][sfind]
            gpos = gdata['pos'][sfind]
            gradius = pylab.sqrt((gpos[:, 1:] ** 2).sum(axis=1))
            
            rbin = np.zeros(nr)
            
            coltab = ['b', 'k']
            symtab = ['o', '^']
            
            pn, edges = np.histogram(gradius, bins=nr, weights=gmass, range=[0., galrad])
            rbin = 0.5 * (edges[:-1] + edges[1:])
            vol = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
            rho = 1e-5 * pn / vol
            
            ax.semilogy(rbin * 1e3, rho, color='k', marker='o', markersize=2., linewidth=0.)
            
            guess = (0.1, 3.)
            sigma = rho
            (popt, pcov) = curve_fit(fitdict[fit], rbin * 1e3, rho, guess, sigma=sigma)
            print("popt=", popt)
            
            ax.semilogy(rbin * 1e3, fitdict[fit](rbin * 1e3, popt[0], popt[1]), linestyle='-', color='k', lw=1., label="$\\rm{%.1f kpc}$" % (popt[1]))
            
            # calc vertical scale height at Rsun
            nz = 10
            pn, edges = np.histogram(abs(gpos[:, 0]), bins=nz, weights=gmass, range=[0., zmax])
            zbin = 0.5 * (edges[:-1] + edges[1:])
            vol = np.pi * (rran[1] ** 2 - rran[0] ** 2) * (edges[1:] - edges[:-1])
            rho = 1e-8 * pn / vol
            guess = (0.01, 0.3)
            sigma = rho
            (poptz, pcovz) = curve_fit(fitdict[fit], zbin * 1e3, rho, guess, sigma=sigma)
            print("poptz=", poptz)
            
            with open(filename, "a") as f3:
                param = "%12.3f%18.3f%12.3f%18.3f%12.3f\n" % (time, popt[0] * 1e9, popt[1], poptz[0] * 1e9, poptz[1])
                f3.write(param)
            
            # ax.yaxis.set_ticks([1e-4, 1e-3, 1e-2, 1e-1, 1.])
            figure1.reset_axis_limits()
            
            ax.legend(loc='upper right', fontsize=8, frameon=False, ncol=1)
        
        f3.close()
        
        figure1.fig.savefig('%s/gasrho_R_prof_z0%03d.%s' % (outpath, snap, suffix))