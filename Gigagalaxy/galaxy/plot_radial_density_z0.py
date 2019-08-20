from const import *
from loadmodules import *
from pylab import *
from scipy.optimize import curve_fit
from util import *

p = plot_helper.plot_helper()
fitdict = {"exp_only":  p.exp_prof, "exp_exp": p.total_profilede, "nfw_exp": p.total_NFW_exp, "hern_exp": p.total_hernquist_exp,
           "hern_only": p.hernquist_bulge}
colors = ['b', 'r']


def plot_radial_density_z0(runs, dirs, outpath, outputlistfile, tlist, suffix, nrows, ncols, rran=[0.0075, 0.0085], fit="hern_exp", disc_stars=True,
                           do_stars=True, do_gas=False):
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
            
            if do_gas and do_stars:
                filename = wpath + 'fitbaryon_hR_z0.txt'
            elif do_gas:
                filename = wpath + 'fitgas_hR_z0.txt'
            else:
                filename = wpath + 'fitstars_hR_z0.txt'
            
            if isnap == 0:
                if do_stars:
                    f3 = open(filename, 'w')
                    header = "%12s%12s%12s%12s%12s%12s\n" % ("Time", "rho01", "h1", "rho02", "h2", "zrmsRsun")
                    f3.write(header)
                else:
                    f3 = open(filename, 'w')
                    header = "%12s%12s%12s%12s\n" % ("Time", "rho01", "h1", "zrmsRsun")
                    f3.write(header)
            else:
                f3 = open(filename, 'a')
            
            nr = 30
            # galrad = 0.02
            zmax = 0.0005
            
            print("Doing dir %s snap %d." % (dd, snap))
            attrs = ['pos', 'vel', 'mass', 'age']
            
            if do_stars:
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
            
            galrad = 0.1 * sf.data['frc2'][0]
            
            sdata = g.sgdata['sdata']
            mass = sdata['mass']
            pos = sdata['pos']
            vel = sdata['vel']
            star_age = sdata['age']
            star_radius = pylab.sqrt((pos[:, 1:] ** 2).sum(axis=1))
            star_height = pos[:, 0]
            
            if do_gas:
                gdata = g.sgdata['gdata']
                sfr = gdata['sfr']
                gmass = gdata['mass']
                gpos = gdata['pos']
                gradius = pylab.sqrt((gpos[:, 1:] ** 2).sum(axis=1))
                sfind, = np.where((abs(gdata['pos'][:, 0]) < zmax) & (sfr > 0.))
                if galrad > rran[1]:
                    sfindz, = np.where((gradius < rran[1]) & (gradius > rran[0]) & (abs(gdata['pos'][:, 0]) < 0.004) & (sfr > 0.))
                else:
                    sfindz, = np.where((gradius < galrad) & (gradius > (galrad - 0.001)) & (abs(gdata['pos'][:, 0]) < 0.004) & (sfr > 0.))
            
            if do_stars and disc_stars:
                eps = sdata['eps2']
                ii, = np.where((eps > 0.7) & (abs(pos[:, 0]) < zmax))
                if galrad > rran[1]:
                    iiz, = np.where((eps > 0.7) & (star_radius < rran[1]) & (star_radius > rran[0]) & (abs(pos[:, 0]) < 0.004))
                else:
                    iiz, = np.where((eps > 0.7) & (star_radius < galrad) & (star_radius > (galrad - 0.001)) & (abs(pos[:, 0]) < 0.004))
            elif disc_stars and not do_stars:
                raise ValueError('Cannot define disc stars without do_stars.')
            elif do_stars:
                ii, = np.where((abs(pos[:, 0]) < zmax))
                if galrad > rran[1]:
                    iiz, = np.where((sdata['eps2'] > 0.7) & (star_radius < rran[1]) & (star_radius > rran[0]) & (abs(pos[:, 0]) < 0.004))
                else:
                    iiz, = np.where((star_radius < galrad) & (star_radius > (galrad - 0.001)) & (abs(pos[:, 0]) < 0.004))
            
            rbin = np.zeros(nr)
            
            coltab = ['b', 'k']
            symtab = ['o', '^']
            
            if do_stars:
                pn, edges = np.histogram(star_radius[ii], bins=nr, weights=mass[ii], range=[0., galrad])
                rbin = 0.5 * (edges[:-1] + edges[1:])
                vol = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
                rho = 1e-5 * pn / vol
            
            if do_gas:
                pn, edges = np.histogram(gradius[sfind], bins=nr, weights=gmass[sfind], range=[0., galrad])
                rbin = 0.5 * (edges[:-1] + edges[1:])
                vol = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
                rhogas = 1e-5 * pn / vol
                if do_stars:
                    rho += rhogas
                else:
                    rho = rhogas
            
            ax.semilogy(rbin * 1e3, rho, color='k', marker='o', markersize=2., linewidth=0.)
            
            if do_stars:
                # fit both profiles together
                # guess = (.2, 0.5, 0.001, 3.)
                guess = (1., 0.5, 0.1, 3.)
                sigma = rho
                bounds = ([0., 0.1, 0., 0.5], [20., 5., 10., 15.])
                (popt, pcov) = curve_fit(fitdict[fit], rbin * 1e3, rho, guess, sigma=sigma, bounds=bounds)
                (poptd, pcovd) = curve_fit(fitdict["exp_only"], rbin * 1e3, rho, (0.1, 3.), sigma=sigma, bounds=([0., 0.5], [10., 10.]))
                (poptb, pcovb) = curve_fit(fitdict["hern_only"], rbin * 1e3, rho, (1., 0.5), sigma=sigma, bounds=([0., 0.1], [20., 5.]))
                perr = np.sqrt(np.diag(pcov)).sum() / 2.
                perrd = np.sqrt(np.diag(pcovd)).sum()
                perrb = np.sqrt(np.diag(pcovb)).sum()
                print("perr,perrd,perrb=", perr, perrd, perrb)
                
                # if ( (perrd < perr) & (perrd < perrb) ):
                #    ax.semilogy(rbin*1e3,fitdict["exp_only"](rbin*1e3,poptd[0],poptd[1]),dashes=(2,2),color='b',lw=1., label="$\\rm{%.1f kpc}$"%(
                #    poptd[1]))
                # if (perrb < perr):
                #    ax.semilogy(rbin*1e3,fitdict["hern_only"](rbin*1e3,poptb[0],poptb[1]),dashes=(1,1),color='r',lw=1., label="$\\rm{%.1f kpc}$"%(
                #    poptb[1]))
                # else:
                
                for i, prof in enumerate(profs):
                    popttmp = (popt[2 * i], popt[2 * i + 1])
                    proftmp = prof + '_only'
                    ax.semilogy(rbin * 1e3, fitdict[proftmp](rbin * 1e3, popttmp[0], popttmp[1]), dashes=(i + 1, i + 1), color=colors[i], lw=1.,
                                label="$\\rm{%.1f kpc}$" % (popttmp[1]))
                ax.semilogy(rbin * 1e3, fitdict[fit](rbin * 1e3, popt[0], popt[1], popt[2], popt[3]), linestyle='-', color='k', lw=1.)
            else:
                guess = (0.01, 3.)
                sigma = rho
                bounds = ([0.0001, 0.], [1., 10.])
                rmin = 0.25 * galrad
                indy = find_nearest(rbin, [rmin]).astype('int64')
                indy2 = find_nearest(rbin, [galrad * .75]).astype('int64')
                
                (popt, pcov) = curve_fit(fitdict[fit], rbin[indy:indy2] * 1e3, rho[indy:indy2], guess, sigma=sigma[indy:indy2], bounds=bounds)
                ax.semilogy(rbin * 1e3, fitdict[fit](rbin * 1e3, popt[0], popt[1]), dashes=(2, 2), color='k', lw=1.,
                            label="$\\rm{%.1f kpc}$" % (popt[1]))
            
            print("popt=", popt)
            
            if do_stars and do_gas:
                z = np.hstack((star_height[iiz], gpos[sfindz, 0]))
                weight = np.hstack((mass[iiz], gmass[sfindz]))
            elif do_stars:
                z = star_height[iiz]
                weight = mass[iiz]
            elif do_gas:
                z = gpos[sfindz, 0]
                weight = gmass[sfindz]
            
            zrms = np.std(z) * 1e3
            
            # calc vertical scale height at Rsun                                     
            nz = 10
            pn, edges = np.histogram(abs(z), bins=nz, weights=weight, range=[0., 0.003])
            zbin = 0.5 * (edges[:-1] + edges[1:])
            vol = np.pi * (rran[1] ** 2 - rran[0] ** 2) * (edges[1:] - edges[:-1])
            rho = 1e-8 * pn / vol
            guess = (0.001, 0.3)
            sigma = rho
            
            (poptz, pcovz) = curve_fit(fitdict['exp_only'], zbin * 1e3, rho, guess, sigma=sigma)
            print("poptz=", poptz)
            
            with open(filename, "a") as f3:
                if do_stars:
                    param = "%12.3f%18.3f%12.3f%18.3f%12.3f%12.3f\n" % (time, popt[0] * 1e9, popt[1], popt[2] * 1e9, popt[3], zrms)
                else:
                    param = "%12.3f%18.3f%12.3f%12.3f\n" % (time, popt[0] * 1e9, popt[1], zrms)
                f3.write(param)
            
            # ax.yaxis.set_ticks([1e-4, 1e-3, 1e-2, 1e-1, 1.])
            figure1.reset_axis_limits()
            
            ax.legend(loc='upper right', fontsize=8, frameon=False, ncol=1)
        
        f3.close()
        
        if do_stars and not do_gas:
            figure1.fig.savefig('%s/starsrho_R_prof_z0%03d.%s' % (outpath, snap, suffix))
        elif do_stars and do_gas:
            figure1.fig.savefig('%s/baryonrho_R_prof_z0%03d.%s' % (outpath, snap, suffix))
        elif do_gas and not do_stars:
            figure1.fig.savefig('%s/gasrho_R_prof_z0%03d.%s' % (outpath, snap, suffix))


def find_nearest(array, value):
    if len(value) == 1:
        idx = (np.abs(array - value)).argmin()
    else:
        idx = np.zeros(len(value))
        for i in range(len(value)):
            idx[i] = (np.abs(array - value[i])).argmin()
    
    return idx