import matplotlib

matplotlib.use('Agg')
import numpy as np

import matplotlib as plt
from pylab import *
from const import *
from loadmodules import *
from scipy.special import gamma
from scipy.optimize import curve_fit
from parse_particledata import parse_particledata
from scripts.gigagalaxy.util import plot_helper
from scripts.gigagalaxy.util import multipanel_layout
from scripts.gigagalaxy.util import select_snapshot_number


lws = [0.7, 1., 1.3]
colors = ['g', 'b', 'r', 'k', 'c', 'y', 'm', 'purple']
Msunabs = [5.61, 5.48, 4.83, 3.30, 5.12, 4.68, 4.57, 4.60]
band = {'Umag': 0, 'Bmag': 1, 'Vmag': 2, 'Kmag': 3, 'gmag': 4, 'rmag': 5, 'imag': 6, 'zmag': 7}


def plot_stellar_surfden(runs, dirs, zlist, nrows, ncols):
    
    plt.close()
    figure = plt.figure(figsize=(8, 8), dpi=100)
    
    figure = multipanel_layout.multipanel_layout(nrows=nrows, ncols=ncols)
    figure.set_fontsize(3)
    figure.set_figure_layout()
    figure.set_axis_locators(xminloc=5.0, xmajloc=5.0, yminloc=0.1, ymajloc=0.5)
    figure.set_fontsize(10)
    figure.set_axis_labels(xlabel="$\\rm{R [kpc]}$", ylabel="$\\rm{\Sigma [M_{\odot} pc^{-2}]}$")
    figure.set_axis_limits_and_aspect(xlim=[0.0, 19.0], ylim=[1e0, 2e4], logyaxis=True)

    p = plot_helper.plot_helper()
    
    for d in range(len(runs)):
        
        figure = multipanel_layout.multipanel_layout(nrows=nrows, ncols=ncols)
        figure.set_figure_layout()
        figure.set_axis_locators(xminloc=5.0, xmajloc=5.0)
        figure.set_axis_labels(xlabel="$\\rm{R [kpc]}$", ylabel="$\\rm{\Sigma [M_{\odot} pc^{-2}]}$")
        figure.set_axis_limits_and_aspect(xlim=[0.0, 19.0], ylim=[1e0, 2e4], logyaxis=True)
        figure.set_fontsize()
        
        dd = dirs[d] + runs[d]
        ax = figure.axes[d]
        
        nshells = 60  # 35 up to galrad is OK
        
        snaps = [127]
        
        time = np.zeros(len(snaps))
        
        loadonlytype = [4]
        
        for isnap in range(len(snaps)):
            snap = snaps[isnap]
            print("Doing dir %s snap %d." % (dd, snap))
            
            attrs = ['pos', 'vel', 'mass', 'age']
            if 1 in loadonlytype:
                attrs.append('id')
            
            s = gadget_readsnap(snap, snappath=dd + '/output/', hdf5=True, loadonlytype=loadonlytype, loadonly=attrs)
            sf = load_subfind(snap, dir=dd + '/output/', hdf5=True, loadonly=['fpos', 'frc2', 'svel', 'flty', 'fnsh', 'slty', 'fmc2', 'spos', 'ffsh'])
            s.calc_sf_indizes(sf)
            
            if 1 in loadonlytype:
                if isnap == 0:
                    s.select_halo(sf, 3., use_principal_axis=True, use_cold_gas_spin=False, do_rotation=True)
                    idmb = s.get_most_bound_dm_particles()
                else:
                    centre = s.pos[np.where(np.in1d(s.id, [idmb[0]]) == True), :].ravel()
                    centre = list(centre)
                    s.select_halo(sf, 3., centre=centre, use_principal_axis=True, use_cold_gas_spin=False, do_rotation=True)
            else:
                s.select_halo(sf, 3., use_principal_axis=True, use_cold_gas_spin=False, do_rotation=True)
            
            g = parse_particledata(s, sf, attrs, radialcut=0.1 * sf.data['frc2'][0])
            g.prep_data()
            
            sdata = g.sgdata['sdata']
            mass = sdata['mass']
            pos = sdata['pos']
            
            time[isnap] = np.around(s.cosmology_get_lookback_time_from_a(s.time.astype('float64'), is_flat=True), decimals=3)
            
            zcut = 0.001  # vertical cut in Mpc
            Rcut = 0.040
            
            rd = np.linspace(0.0, Rcut, nshells)
            mnow = np.zeros(len(rd))
            
            rad = pylab.sqrt((pos[:, 1:] ** 2).sum(axis=1))
            z = pos[:, 0]
            
            ii, = np.where((abs(z) < zcut))
            
            weight = mass
            
            bins = nshells
            sden, edges = np.histogram(rad[ii], bins=bins, range=(0., Rcut), weights=weight[ii])
            sa = np.zeros(len(edges) - 1)
            sa[:] = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
            sden /= sa
            
            x = np.zeros(len(edges) - 1)
            x[:] = 0.5 * (edges[1:] + edges[:-1])
            
            sden *= 1e-6
            r = x * 1e3
            
            print("Surface density at R=8 kpc:", sden[find_nearest(r, [8.])])
            
            sdlim = 1.
            indy = find_nearest(sden * 1e4, [sdlim]).astype('int64')
            
            rfit = x[indy] * 1e3
            sdfit = sden[:indy]
            r = r[:indy][sdfit > 0.]
            sdfit = sdfit[sdfit > 0.]
            
            print("fitting carried out in radial range %f - %f" % (x[0] * 1e3, x[indy] * 1e3))
            
            flag = 0
            try:
                guess = (0.1, 2., 0.4, 0.6, 1.)
                
                bounds = ([0.01, 0., 0.01, 0.5, 0.25], [1., 6., 10., 2., 10.])
                sigma = 0.1 * sdfit
                (popt, pcov) = curve_fit(p.total_profile, r, sdfit, guess, sigma=sigma, bounds=bounds)
                
                # compute total masses from the fit
                disk_mass = 2.0 * np.pi * popt[0] * popt[1] * popt[1]
                bulge_mass = np.pi * popt[2] * popt[3] * popt[3] * gamma(2.0 / popt[4] + 1)
                disk_to_total = disk_mass / (bulge_mass + disk_mass)
                # Set to nan any crazy values (can happen at earlier times)
                plist = np.array([disk_mass, bulge_mass, popt[0], popt[1], popt[2], popt[3], popt[4], disk_to_total])
                aa, = np.where((plist > 30.) | (plist < 0.))
                plist[aa] = np.nan
                disk_mass, bulge_mass, dnorm, r_d, bnorm, r_b, sers, disk_to_total = plist[0], plist[1], plist[2], plist[3], plist[4], plist[5], \
                                                                                     plist[6], plist[7]
                
                param = "%12s%12.3f%12.3f%12.3f%12.3f%12.3f%12.3f%12.3f%12.3f%12.3f%12.2f%12.2f\n" % (
                    ("Au%s" % runs[d].split('o')[1]), time[isnap], sf.data['fmc2'][0], sf.data['frc2'][0] * 1e3, mass.sum(), disk_mass, r_d,
                    bulge_mass, r_b * p.sersic_b_param(1.0 / sers) ** (1.0 / sers), 1.0 / sers, disk_to_total, rfit)
            
            except:
                popt = np.zeros(5)
                print("fitting failed")
            
            indy = find_nearest(x, rd).astype('int64')
            mnow[:] = sden[indy]
            
            ax.semilogy(r, 1e10 * sdfit * 1e-6, 'o', markersize=1.5, color='k', linewidth=0.)
            if flag == 1:
                ax.semilogy(r, 1e10 * p.sersic_prof1(r, popt[0], popt[1], popt[2]) * 1e-6, 'r--', label=r'n=%.2f' % (1. / popt[2]))
            if flag == 0 or flag == 2:
                ax.semilogy(r, 1e10 * p.exp_prof(r, popt[0], popt[1]) * 1e-6, 'b-', label=r'$R_d = %.2f$' % (popt[1]))
            if flag == 0:
                ax.semilogy(r, 1e10 * p.sersic_prof1(r, popt[2], popt[3], popt[4]) * 1e-6, 'r--', label=r'n=%.2f' % (1. / popt[4]))
                ax.semilogy(r, 1e10 * p.total_profile(r, popt[0], popt[1], popt[2], popt[3], popt[4]) * 1e-6, 'k-')
            
            ax.axvline(rfit, color='gray', linestyle='--')
            ax.xaxis.set_ticks([0, 5, 10, 15, 20])
            ax.legend(loc='upper right', frameon=False, prop={'size': 6})
            if len(zlist) > 1:
                figure.set_panel_title(panel=panels - isnap - 1, title="$\\rm %s\, %.1f$" % ("Time=", time[isnap]), position='bottom right')
            else:
                figure.set_panel_title(panel=d, title="$\\rm{%s\,{%s}}$" % ("Au", runs[d].split('_')[1]), position='bottom left', color='k',
                                       fontsize=8)
        
        ax.legend(loc='upper left', frameon=False, prop={'size': 7})
    
        print("Saving figures")
        pdf.savefig(f)  # Save figure.

def find_nearest(array, value):
    if len(value) == 1:
        idx = (np.abs(array - value)).argmin()
    else:
        idx = np.zeros(len(value))
        for i in range(len(value)):
            idx[i] = (np.abs(array - value[i])).argmin()
    
    return idx