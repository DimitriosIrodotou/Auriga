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

Msunabs = [5.61, 5.48, 4.83, 3.30, 5.12, 4.68, 4.57, 4.60]
band = {'Umag': 0, 'Bmag': 1, 'Vmag': 2, 'Kmag': 3, 'gmag': 4, 'rmag': 5, 'imag': 6, 'zmag': 7}


def plot_stellar_surfden(pdf, runs, dirs):
    p = plot_helper.plot_helper()
    
    for d in range(len(runs)):
        
        plt.close()
        figure = plt.figure(figsize=(10, 7.5), dpi=100)
        plt.xlabel("$\\rm{R [kpc]}$")
        plt.ylabel("$\\rm{\Sigma [M_{\odot} pc^{-2}]}$")
        plt.xlim(0.0, 25.0)
        plt.ylim(1e0, 1e6)
        
        dd = dirs[d] + runs[d]
        
        nshells = 60  # 35 up to galrad is OK
        
        snaps = [251]
        
        time = np.zeros(len(snaps))
        
        loadonlytype = [4]
        
        for isnap in range(len(snaps)):
            snap = snaps[isnap]
            print("Doing dir %s snap %d." % (dd, snap))
            
            attrs = ['pos', 'vel', 'mass', 'age']
            attrs.append('gsph')
            s = gadget_readsnap(snap, snappath=dd + '/output/', hdf5=True, loadonlytype=loadonlytype, loadonly=attrs)
            sf = load_subfind(snap, dir=dd + '/output/', hdf5=True, loadonly=['fpos', 'frc2', 'svel', 'flty', 'fnsh', 'slty', 'fmc2', 'spos', 'ffsh'])
            s.calc_sf_indizes(sf)
            s.select_halo(sf, 3., use_principal_axis=True, rotate_disk=True, do_rotation=True)
            
            g = parse_particledata(s, sf, attrs, radialcut=0.1 * sf.data['frc2'][0])
            g.prep_data()
            sdata = g.sgdata['sdata']
            mass = sdata['mass']
            pos = sdata['pos']
            
            time[isnap] = np.around(s.cosmology_get_lookback_time_from_a(s.time.astype('float64'), is_flat=True), decimals=3)
            
            Rcut = 0.040
            rad = pylab.sqrt((pos[:, 1:] ** 2).sum(axis=1))
            ii, = np.where((abs(pos[:, 0]) < 0.005))  # Vertical cut in Mpc.
            iband=1
            luminosity = 10 ** (-0.4 * (sdata[iband] - Msunabs[band[iband]]))
            weight = luminosity
            
            bins = nshells
            sden, edges = np.histogram(rad[ii], bins=bins, range=(0., Rcut), weights=mass[ii])
            sa = np.zeros(len(edges) - 1)
            sa[:] = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
            sden /= sa
            
            x = np.zeros(len(edges) - 1)
            x[:] = 0.5 * (edges[1:] + edges[:-1])
            
            sden *= 1e-6
            r = x * 1e3
            
            sdlim = 1.0
            indy = find_nearest(sden * 1e4, [sdlim]).astype('int64')
            rfit = x[indy] * 1e3
            sdfit = sden[:indy]
            r = r[:indy][sdfit > 0.0]
            sdfit = sdfit[sdfit > 0.0]
            
            try:
                sigma = 0.1 * sdfit
                bounds = ([0.01, 0.0, 0.01, 0.5, 0.25], [1.0, 6.0, 10.0, 2.0, 10.0])
                
                (popt, pcov) = curve_fit(p.total_profile, r, sdfit, sigma=sigma, bounds=bounds)
                
                # compute total masses from the fit
                disk_mass = 2.0 * np.pi * popt[0] * popt[1] * popt[1]
                bulge_mass = np.pi * popt[2] * popt[3] * popt[3] * gamma(2.0 / popt[4] + 1)
                disk_to_total = disk_mass / (bulge_mass + disk_mass)
            
            except:
                popt = np.zeros(5)
                print("fitting failed")
            
            plt.semilogy(r, 1e10 * sdfit * 1e-6, 'o', markersize=5, color='k', linewidth=0.0)
            plt.semilogy(r, 1e10 * p.exp_prof(r, popt[0], popt[1]) * 1e-6, 'b-', label=r'$R_d = %.2f$' % (popt[1]))
            plt.semilogy(r, 1e10 * p.sersic_prof1(r, popt[2], popt[3], popt[4]) * 1e-6, 'r--', label=r'$R_{eff}=%.2f$' % (1. / popt[4]))
            plt.semilogy(r, 1e10 * p.total_profile(r, popt[0], popt[1], popt[2], popt[3], popt[4]) * 1e-6, 'k-')
            
            plt.axvline(rfit, color='gray', linestyle='--')
            plt.legend(loc='upper right', frameon=False, prop={'size': 12})
        
        plt.legend(loc='upper left', frameon=False, prop={'size': 12})
        plt.savefig('/u/di43/Auriga/plots/' + 'Test' + str(d) + '.png')


def find_nearest(array, value):
    if len(value) == 1:
        idx = (np.abs(array - value)).argmin()
    else:
        idx = np.zeros(len(value))
        for i in range(len(value)):
            idx[i] = (np.abs(array - value[i])).argmin()
    
    return idx