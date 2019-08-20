from const import *
from loadmodules import *
from pylab import *
from scipy.optimize import curve_fit
from util import *

p = plot_helper.plot_helper()

fitdict = {"exp_only":   p.exp_prof, "sechsq_only": p.sechsq, "exp_exp": p.total_profilede, "sechsq_sechsq": p.total_profile_doublesechsq,
           "sechsq_exp": p.sech2_exp}  # , "exp":p.exp_prof, "sechsq"}

colors = ['b', 'r']


def plot_vertical_density_R0(runs, dirs, outpath, outputlistfile, tlist, suffix, nrows, ncols, rran=[0.0075, 0.0085], fit="sechsq_sechsq",
                             disc_stars=True, do_stars=True, do_gas=False, writetofile=False):
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
        figure1.set_axis_labels(xlabel=r"$\rm{z [kpc]}$", ylabel=r"$\rm{\rho [M_{\odot} pc^{-3}]}$")
        figure1.set_fontsize()
        if do_stars and not do_gas:
            figure1.set_axis_limits_and_aspect(xlim=[0.0, 3.9], ylim=[1e-4, 0.09], logyaxis=True)
        else:
            figure1.set_axis_limits_and_aspect(xlim=[0.0, 3.9], ylim=[1e-5, 0.5], logyaxis=True)
        
        clist = ['r', 'b', 'c', 'g', 'k']
        
        h_all = np.array([])
        h_young = np.array([])
        
        profs = fit.split('_')
        
        for d in range(len(runs)):
            dd = dirs[d] + runs[d]
            ax = figure1.fig.axes[d]
            
            figure1.set_panel_title(panel=d, title="$\\rm %s\,%s$" % ("Au", runs[d].split('_')[1]), position='bottom left', color='k', fontsize=10)
            
            wpath = outpath + runs[d] + '/densityprof/'
            if not os.path.exists(wpath):
                os.makedirs(wpath)
            
            if writetofile:
                if do_stars and not do_gas:
                    filename = wpath + 'fit%sstars_hz_R0.txt' % (fit)
                
                if do_gas and not do_stars:
                    filename = wpath + 'fit%sgas_hz_R0.txt' % (fit)
                
                if do_gas and do_stars:
                    filename = wpath + 'fit%sbaryon_hz_R0.txt' % (fit)
                
                if profs[1] == 'only':
                    if isnap == 0:
                        f3 = open(filename, 'w')
                        header = "%12s%12s%12s\n" % ("Time", "rho01", "h1")
                        f3.write(header)
                    else:
                        f3 = open(filename, 'a')
                else:
                    if isnap == 0:
                        f3 = open(filename, 'w')
                        header = "%12s%12s%12s%12s%12s%12s\n" % ("starsel", "Time", "rho01", "h1", "rho02", "h2")
                        f3.write(header)
                    else:
                        f3 = open(filename, 'a')
            
            nr = 10
            nx = 2 * nr
            nz = 10
            rc = np.linspace(0., 16., nr)
            hz = np.zeros(len(rc))
            hz2 = np.zeros(len(rc))
            
            print("Doing dir %s snap %d." % (dd, snap))
            attrs = ['pos', 'vel', 'mass', 'age', 'gmet', 'gz']
            
            if disc_stars:
                attrs.extend(['pot'])
            
            if do_gas:
                loadtype = [0, 4]
                attrs.extend(['sfr'])
            else:
                loadtype = [4]
            s = gadget_readsnap(snap, snappath=dd + '/output/', hdf5=True, loadonlytype=loadtype, loadonly=attrs)
            sf = load_subfind(snap, dir=dd + '/output/', hdf5=True, loadonly=['fpos', 'frc2', 'svel', 'flty', 'fnsh', 'slty', 'fmc2'])
            s.calc_sf_indizes(sf)
            s.select_halo(sf, 3., use_principal_axis=True, use_cold_gas_spin=False, do_rotation=True)
            time = s.cosmology_get_lookback_time_from_a(s.time.astype('float64'), is_flat=True)
            g = parse_particledata(s, sf, attrs, radialcut=0.1 * sf.data['frc2'][0])
            g.prep_data()
            
            if do_gas:
                gdata = g.sgdata['gdata']
                sfr = gdata['sfr']
                sfind, = np.where(sfr > 0.)
                gmass = gdata['mass'][sfind]
                gpos = gdata['pos'][sfind]
            
            galrad = 0.016
            zmax = 0.004
            dz = zmax / float(nz)
            rbin = np.zeros(nr)
            zbin = np.zeros(nz)
            
            if do_stars:
                sdata = g.sgdata['sdata']
                mass = sdata['mass']
                pos = sdata['pos']
                star_age = sdata['age']
                alphafe = sdata['alpha'] - sdata['Fe']
                
                if disc_stars:
                    eps = sdata['eps2']
                    ii, = np.where((eps > 0.7))
                else:
                    ii = np.arange(len(mass))
                
                star_radius = pylab.sqrt((pos[ii, 1:] ** 2).sum(axis=1))
                star_height = pos[ii, 0]
                star_x = pos[ii, 1]
                star_y = pos[ii, 2]
                smass = mass[ii]
                star_age = star_age[ii]
                
                kk, = np.where((star_age < 14.) & (star_radius > rran[0]) & (star_radius < rran[1]))  # all
                
                pp = ['young', 'all']
                pp = ['all']
                pdict = {'young': ii, 'all': kk}
                rdict = {}
                
                for ind, val in enumerate(pp):
                    mpf = pp[ind]
                    pn, edges = np.histogram(abs(star_height[pdict[pp[ind]]]), bins=nz, weights=smass[pdict[mpf]], range=[0., zmax])
                    pn /= 2.
                    zbin = 0.5 * (edges[:-1] + edges[1:])
                    vol = np.pi * (rran[1] ** 2 - rran[0] ** 2) * (edges[1:] - edges[:-1])
                    rho = 1e-8 * pn / vol
                    rdict[val] = rho
                
                # plot old "thick" disc
                # ll, = np.where( (star_age > 7.5) & (star_radius > rran[0]) & (star_radius < rran[1]) )
                ll, = np.where((alphafe > 0.06) & (star_age > 7.5) & (star_radius > rran[0]) & (star_radius < rran[1]))
                pn, edges = np.histogram(abs(star_height[ll]), bins=nz, weights=smass[ll], range=[0., zmax])
                pn /= 2.
                zbin = 0.5 * (edges[:-1] + edges[1:])
                vol = np.pi * (rran[1] ** 2 - rran[0] ** 2) * (edges[1:] - edges[:-1])
                rho_old = 1e-8 * pn / vol
                ax.semilogy(zbin * 1e3, rho_old, marker='^', color='g', markersize=3., lw=0.)  #
            
            if do_gas:
                pn, edges = np.histogram(abs(gpos[:, 0]), bins=nz, weights=gmass, range=[0., zmax])
                pn /= 2.
                zbin = 0.5 * (edges[:-1] + edges[1:])
                vol = np.pi * (rran[1] ** 2 - rran[0] ** 2) * (edges[1:] - edges[:-1])
                rhogas = 1e-8 * pn / vol
            
            # do fitting
            if do_stars and do_gas:
                rho += rhogas
            elif do_gas:
                rho = rhogas
            
            if profs[1] != 'only':
                guess = (0.01, 0.3, 0.001, 3.)
                sigma = 0.1 * np.log(rho * 1e9 + 0.1) - 1.
                (popt, pcov) = curve_fit(fitdict[fit], zbin * 1e3, rho, guess, sigma=sigma)
                print("** fit params=", popt)
            else:
                guess = (.001, 0.5)
                sigma = 0.1 * np.log(rho * 1e9 + 0.1) - 1.
                (popt, pcov) = curve_fit(fitdict[fit], zbin * 1e3, rho, guess, sigma=sigma)
                print("** fit params=", popt)
            
            # increase number of plot points
            pn, edges = np.histogram(abs(star_height[pdict[pp[ind]]]), bins=15, weights=smass[pdict[mpf]], range=[0., zmax])
            pn /= 2.
            zbin = 0.5 * (edges[:-1] + edges[1:])
            vol = np.pi * (rran[1] ** 2 - rran[0] ** 2) * (edges[1:] - edges[:-1])
            rho = 1e-8 * pn / vol
            if do_gas:
                pn, edges = np.histogram(abs(gpos[:, 0]), bins=15, weights=gmass, range=[0., zmax])
                pn /= 2.
                zbin = 0.5 * (edges[:-1] + edges[1:])
                vol = np.pi * (rran[1] ** 2 - rran[0] ** 2) * (edges[1:] - edges[:-1])
                rhogas = 1e-8 * pn / vol
                if do_gas and do_stars:
                    rho += rhogas
                else:
                    rho = rhogas
            
            ax.semilogy(zbin * 1e3, rho, marker='o', color='k', markersize=3., lw=0.)
            
            zbin = np.insert(zbin, 0, 0)
            if profs[1] != 'only':
                for i, prof in enumerate(profs):
                    popttmp = (popt[2 * i], popt[2 * i + 1])
                    proftmp = prof + '_only'
                    ax.semilogy(zbin * 1e3, fitdict[proftmp](zbin * 1e3, popttmp[0], popttmp[1]), dashes=(i + 1, i + 1), color=colors[i], lw=1.,
                                label="$\\rm{%.1f pc}$" % (abs(popt[2 * i + 1]) * 1e3))
                ax.semilogy(zbin * 1e3, fitdict[fit](zbin * 1e3, popt[0], popt[1], popt[2], popt[3]), linestyle='-', color='k', lw=1.)
            else:
                ax.semilogy(zbin * 1e3, fitdict[fit](zbin * 1e3, popt[0], popt[1]), linestyle='-', color='k', lw=1.,
                            label="$\\rm{%.1f pc}$" % (abs(popt[1]) * 1e3))
            
            if writetofile:
                with open(filename, "a") as f3:
                    if len(popt) == 2:
                        param = "%12.3f%12.6f%12.3f\n" % (time, popt[0], abs(popt[1]))
                    elif len(popt) == 4:
                        param = "%12s%12.3f%12.6f%12.3f%12.6f%12.3f\n" % (pp[ind], time, abs(popt[0]), abs(popt[1]), abs(popt[2]), abs(popt[3]))
                    f3.write(param)
            
            ax.xaxis.set_ticks([0, 1, 2, 3, 4])
            figure1.reset_axis_limits()
            
            ax.legend(loc='upper right', fontsize=8, frameon=False, ncol=1)
        
        if do_gas and do_stars:
            figure1.fig.savefig('%s/baryonrho_z_%sfit_R0%03d.%s' % (outpath, fit, snap, suffix))
        elif do_stars:
            figure1.fig.savefig('%s/starsrho_z_%sfit_R0%03d_halo_23.%s' % (outpath, fit, snap, suffix))
        elif do_gas:
            figure1.fig.savefig('%s/gasrho_z_%sfit_R0%03d.%s' % (outpath, fit, snap, suffix))
    
    if writetofile:
        f3.close()