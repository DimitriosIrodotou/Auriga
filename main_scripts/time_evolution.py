from __future__ import division

import os
import re
import glob
import pickle
import projections

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from const import *
from sfigure import *
from loadmodules import *
from parallel_decorators import vectorize_parallel

element = {'H': 0, 'He': 1, 'C': 2, 'N': 3, 'O': 4, 'Ne': 5, 'Mg': 6, 'Si': 7, 'Fe': 8}


def get_names_sorted(names):
    if list(names)[0].find("_"):
        names_sorted = np.array(list(names))
        names_sorted.sort()
        
        return names_sorted
    else:
        values = np.zeros(len(names))
        for i in range(len(names)):
            name = names[i]
            value = 0
            while not name[0].isdigit():
                value = value * 256 + ord(name[0])
                name = name[1:]
            values[i] = value * 256 + np.int32(name)
        isort = values.argsort()
        return np.array(names)[isort]


def create_axis(f, idx, ncol=5):
    ix = idx % ncol
    iy = idx // ncol
    
    s = 1.
    
    ax = f.iaxes(0.5 + ix * (s + 0.5), 0.3 + s + iy * (s + 0.6), s, s, top=False)
    ax2 = ax.twiny()
    return ax, ax2


def set_axis(s, ax, ax2, ylabel, ylim=None, ncol=5):
    z = np.array([5., 3., 2., 1., 0.5, 0.2, 0.0])
    a = 1. / (1 + z)
    
    times = np.zeros(len(a))
    for i in range(len(a)):
        times[i] = s.cosmology_get_lookback_time_from_a(a[i])
    
    lb = []
    for v in z:
        if v >= 1.0:
            lb += ["%.0f" % v]
        else:
            if v != 0:
                lb += ["%.1f" % v]
            else:
                lb += ["%.0f" % v]
    
    ax.set_xlim(0., 13.)
    ax.invert_xaxis()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(times)
    ax2.set_xticklabels(lb)
    
    ax.set_ylabel(ylabel, size=6)
    ax.set_xlabel("$t_\mathrm{look}\,\mathrm{[Gyr]}$", size=6)
    ax2.set_xlabel("$z$", size=6)
    
    for a in [ax, ax2]:
        for label in a.xaxis.get_ticklabels():
            label.set_size(6)
        for label in a.yaxis.get_ticklabels():
            label.set_size(6)
    
    if ylim is not None:
        ax.set_ylim(ylim)
    
    return None


def sfr(pdf, data, levels):
    nlevels = len(levels)
    
    nhalos = 0
    for il in range(nlevels):
        data.select_haloes(levels[il], 0.)
        nhalos += data.selected_current_nsnaps
    
    f = plt.figure(FigureClass=sfig, figsize=(8.2, 1.6 * (((nhalos - 1) // 5) + 1) + 0.3))
    
    nbins = 100
    tmin = 0
    tmax = 13.
    timebin = (tmax - tmin) / nbins
    
    for il in range(nlevels):
        level = levels[il]
        data.select_haloes(level, 0., loadonlytype=[4], loadonlyhalo=0)
        
        isnap = 0
        for s in data:
            s.centerat(s.subfind.data['fpos'][0, :])
            
            mask, = np.where((s.data['age'] > 0.) & (s.r() < 0.005) & (s.pos[:, 2] < 0.003))
            age = s.cosmology_get_lookback_time_from_a(s.data['age'][mask], is_flat=True)
            
            ax, ax2 = create_axis(f, isnap)
            ax.hist(age, weights=s.data['gima'][mask] * 1e10 / 1e9 / timebin, histtype='step', bins=nbins, range=[tmin, tmax])
            set_axis(s, ax, ax2, "$\\mathrm{Sfr}\,\mathrm{[M_\odot\,yr^{-1}]}$")
            
            ax.text(0.05, 0.92, "Au%s-%d r < 5kpc" % (s.haloname, level), color='k', fontsize=6, transform=ax.transAxes)
            
            isnap += 1
    
    pdf.savefig(f, bbox_inches='tight')  # Save the figure.
    plt.close()
    return None


@vectorize_parallel(method='processes', num_procs=8)
def compute_bfld_halo(snapid, halo, offset):
    s = halo.snaps[snapid].loadsnap(loadonlytype=[0], loadonlyhalo=0)
    s.centerat(s.subfind.data['fpos'][0, :])
    
    time = s.cosmology_get_lookback_time_from_a(s.time, is_flat=True)
    i, = np.where(s.r() < 0.001)
    bfld = np.sqrt(((s.data['bfld'][i, :] ** 2.).sum(axis=1) * s.data['vol'][i]).sum() / s.data['vol'][i].sum()) * bfac
    return time, bfld


def bfld(pdf, data, levels):
    nlevels = len(levels)
    
    nhalos = 0
    for il in range(nlevels):
        data.select_haloes(levels[il], 0.)
        nhalos += data.selected_current_nsnaps
    
    f = plt.figure(FigureClass=sfig, figsize=(8.2, 1.6 * (((nhalos - 1) // 5) + 1) + 0.3))
    
    for il in range(nlevels):
        level = levels[il]
        
        res = {}
        rpath = "../plots/data/bfld_%d.npy" % level
        if os.path.exists(rpath):
            with open(rpath, 'rb') as ff:
                res = pickle.load(ff)
        
        halos = data.get_haloes(level)
        for name, halo in halos.items():
            if name not in res:
                redshifts = halo.get_redshifts()
                
                i, = np.where(redshifts < 10.)
                snapids = np.array(list(halo.snaps.keys()))[i]
                
                dd = np.array(compute_bfld_halo(snapids, halo, snapids.min()))
                res[name] = {}
                res[name]["time"] = dd[:, 0]
                res[name]["bfld"] = dd[:, 1]
        
        with open(rpath, 'wb') as ff:
            pickle.dump(res, ff, pickle.HIGHEST_PROTOCOL)
        
        names = get_names_sorted(res.keys())
        for idx, name in enumerate(names):
            ax, ax2 = create_axis(f, idx)
            
            ax.semilogy(res[name]["time"], res[name]["bfld"] * 1e6)
            set_axis(list(halos[name].snaps.values())[0].loadsnap(), ax, ax2, "$B_\mathrm{r<1\,kpc}\,\mathrm{[\mu G]}$", [1e-2, 1e2])
            ax.set_xlim([13., 11.])
    
    pdf.savefig(f, bbox_inches='tight')  # Save the figure.
    plt.close()
    
    f = plt.figure(FigureClass=sfig, figsize=(8.2, 1.6 * (((nhalos - 1) // 5) + 1) + 0.3))
    
    for il in range(nlevels):
        level = levels[il]
        
        res = {}
        rpath = "../plots/data/bfld_%d.npy" % level
        if os.path.exists(rpath):
            with open(rpath, 'rb') as ff:
                res = pickle.load(ff)
        
        names = get_names_sorted(res.keys())
        for idx, name in enumerate(names):
            ax, ax2 = create_axis(f, idx)
            
            ax.plot(res[name]["time"], res[name]["bfld"] * 1e6)
            set_axis(list(halos[name].snaps.values())[0].loadsnap(), ax, ax2, "$B_\mathrm{r<1\,kpc}\,\mathrm{[\mu G]}$", [0., 100.])
    
    pdf.savefig(f, bbox_inches='tight')  # Save the figure.
    plt.close()
    return None


@vectorize_parallel(method='processes', num_procs=8)
def get_bh_mass(snapid, halo, bhid):
    s = halo.snaps[snapid].loadsnap(loadonlytype=[5], loadonly=['mass', 'id'])
    if 'id' not in s.data:
        return s.cosmology_get_lookback_time_from_a(s.time, is_flat=True), 0.
    
    i, = np.where(s.data['id'] == bhid)
    if len(i) > 0:
        mass = s.mass[i[0]]
    else:
        mass = 0.
    return s.cosmology_get_lookback_time_from_a(s.time, is_flat=True), mass


def bh_mass(pdf, data, levels):
    nlevels = len(levels)
    
    nhalos = 0
    for il in range(nlevels):
        data.select_haloes(levels[il], 0.)
        nhalos += data.selected_current_nsnaps
    
    f = plt.figure(FigureClass=sfig, figsize=(8.2, 1.6 * (((nhalos - 1) // 5) + 1) + 0.3))
    
    for il in range(nlevels):
        level = levels[il]
        
        res = {}
        rpath = "../plots/data/bhmass_%d.npy" % level
        if os.path.exists(rpath):
            with open(rpath, 'rb') as ff:
                res = pickle.load(ff)
        
        halos = data.get_haloes(level)
        for name, halo in halos.items():
            if name not in res:
                redshifts = halo.get_redshifts()
                
                i, = np.where(redshifts < 10.)
                snapids = np.array(list(halo.snaps.keys()))[i]
                
                s = halo.snaps[snapids.argmax()].loadsnap(loadonlytype=[5], loadonlyhalo=0)
                bhid = s.data['id'][s.mass.argmax()]
                
                dd = np.array(get_bh_mass(snapids, halo, bhid))
                res[name] = {}
                res[name]["time"] = dd[:, 0]
                res[name]["mass"] = dd[:, 1]
        
        with open(rpath, 'wb') as ff:
            pickle.dump(res, ff, pickle.HIGHEST_PROTOCOL)
        
        names = get_names_sorted(res.keys())
        for idx, name in enumerate(names):
            ax, ax2 = create_axis(f, idx)
            
            ax.semilogy(res[name]["time"], res[name]["mass"] * 1e10)
            set_axis(list(halos[name].snaps.values())[0].loadsnap(), ax, ax2, "$M_\mathrm{BH}\,\mathrm{[M_\odot]}$")
            ax.text(0.05, 0.92, "Au%s" % name, color='k', fontsize=6, transform=ax.transAxes)
    
    pdf.savefig(f, bbox_inches='tight')  # Save the figure.
    plt.close()
    return None


def bar_strength_evolution(pdf, data, read):
    """
        Calculate bar strength from Fourier modes of surface density.
        :param pdf:
        :param data:
        :param read: boolean.
        :return:
        """
    # Check if a folder to save the data exists, if not create one #
    path = '/u/di43/Auriga/plots/data/' + 'bse/'
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Read the data #
    if read is True:
        redshift_cut = 5.0
        A2, names = [], []  # Declare lists to store the data.
        # Get all available redshifts #
        haloes = data.get_haloes(4)
        for name, halo in haloes.items():
            redshifts = halo.get_redshifts()
        
        for redshift in redshifts[np.where(redshifts <= redshift_cut)]:
            # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
            particle_type = [4]
            attributes = ['age', 'mass', 'pos']
            data.select_haloes(4, redshift, loadonlytype=particle_type, loadonlyhalo=0, loadonly=attributes)
            
            # Loop over all haloes #
            for s in data:
                # Select the halo and rotate it based on its principal axes so galaxy's spin is aligned to the z-axis #
                s.calc_sf_indizes(s.subfind)
                s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)
                
                mask, = np.where(s.data['age'] > 0.0)  # Mask the data: select stellar particles.
                
                # Rotate the particle positions so the bar is along the x-axis #
                z_rotated, y_rotated, x_rotated = projections.rotate_bar(s.pos[mask, 0] * 1e3, s.pos[mask, 1] * 1e3,
                                                                         s.pos[mask, 2] * 1e3)  # Distances are in Mpc.
                s.pos = np.vstack((z_rotated, y_rotated, x_rotated)).T  # Rebuild the s.pos attribute in kpc.
                x, y = s.pos[:, 2] * 1e3, s.pos[:, 1] * 1e3  # Load positions and convert from Mpc to Kpc.
                
                # Split up galaxy in radius bins and calculate the Fourier components #
                nbins = 40  # Number of radial bins.
                r = np.sqrt(x[:] ** 2 + y[:] ** 2)  # Radius of each particle.
                
                # Initialise Fourier components #
                r_m = np.zeros(nbins)
                beta_2 = np.zeros(nbins)
                alpha_0 = np.zeros(nbins)
                alpha_2 = np.zeros(nbins)
                
                # Calculate the Fourier components for each bin as in sec 2.3.2 from Athanassoula et al. 2013 #
                for i in range(0, nbins):
                    r_s = float(i) * 0.25
                    r_b = float(i) * 0.25 + 0.25
                    r_m[i] = float(i) * 0.25 + 0.125
                    xfit = x[(r < r_b) & (r > r_s)]
                    yfit = y[(r < r_b) & (r > r_s)]
                    for k in range(0, len(xfit)):
                        th_i = np.arctan2(yfit[k], xfit[k])
                        alpha_0[i] = alpha_0[i] + 1
                        alpha_2[i] = alpha_2[i] + np.cos(2 * th_i)
                        beta_2[i] = beta_2[i] + np.sin(2 * th_i)
                
                # Calculate bar strength A_2 #
                A2.append(max(np.divide(np.sqrt(alpha_2[:] ** 2 + beta_2[:] ** 2), alpha_0[:])))
                
                # Save data for each halo in numpy arrays #
                np.save(path + 'A2_' + str(s.haloname), A2)
                np.save(path + 'name_' + str(s.haloname), s.haloname)
                np.save(path + 'redshifts_' + str(s.haloname), redshifts[np.where(redshifts <= redshift_cut)])
    
    # Generate the figure #
    f, ax = plt.subplots(1, figsize=(10, 7.5))
    plt.xlim(0, 2)
    plt.ylim(0, 1)
    plt.ylabel(r'$\mathrm{A_{2}}$', size=16)
    plt.xlabel(r'Redshift', size=16)
    plt.grid(True)
    
    # Load and plot the data #
    names = glob.glob(path + '/name_*')
    names.sort()
    colors = iter(cm.rainbow(np.linspace(0, 1, len(names))))
    for i in range(len(names)):
        A2 = np.load(path + 'A2_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        redshifts = np.load(path + 'redshifts_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        # Plot bar strength as a function of radius #
        plt.plot(redshifts, A2, color=next(colors), label='Au-' + str(re.split('_|.npy', names[i])[1]))
    ax.legend(loc='upper left', fontsize=12, frameon=False, numpoints=1)
    
    pdf.savefig(f, bbox_inches='tight')  # Save the figure.
    plt.close()
    return None


def gas_temperature_fraction_evolution(pdf, data, read):
    """
        Calculate bar strength from Fourier modes of surface density.
        :param pdf:
        :param data:
        :param read: boolean.
        :return:
        """
    # Check if a folder to save the data exists, if not create one #
    path = '/u/di43/Auriga/plots/data/' + 'gtfe/'
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Read the data #
    if read is True:
        redshift_cut = 5.0
        sfg_ratios, wg_ratios, hg_ratios, names, masses = [], [], [], [], []  # Declare lists to store the data.
        
        # Get all available redshifts #
        haloes = data.get_haloes(4)
        for name, halo in haloes.items():
            redshifts = halo.get_redshifts()
        
        for redshift in redshifts[np.where(redshifts <= redshift_cut)]:
            # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
            particle_type = [0, 4]
            attributes = ['age', 'mass', 'ne', 'pos', 'rho', 'u']
            data.select_haloes(4, redshift, loadonlytype=particle_type, loadonlyhalo=0, loadonly=attributes)
            
            # Loop over all haloes #
            for s in data:
                # Select the halo and rotate it based on its principal axes so galaxy's spin is aligned to the z-axis #
                s.calc_sf_indizes(s.subfind, verbose=False)
                s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)
                
                mask, = np.where(
                    (s.r() < s.subfind.data['frc2'][0]) & (s.type == 0))  # Mask the data: select gas cells within the virial radius R200 #
                
                # Calculate the temperature of the gas cells #
                ne = s.data['ne'][mask]
                metallicity = s.data['gz'][mask]
                XH = s.data['gmet'][mask, element['H']]
                yhelium = (1 - XH - metallicity) / (4. * XH)
                mu = (1 + 4 * yhelium) / (1 + yhelium + ne)
                u = GAMMA_MINUS1 * s.data['u'][mask] * 1.0e10 * mu * PROTONMASS / BOLTZMANN
                
                # Calculate the mass of the gas cells within three temperatures regimes #
                mass = s.data['mass'][mask]
                masses.append(np.sum(mass))
                sfg_ratios.append(np.sum(mass[np.where((u < 2e4))]) / np.sum(mass))
                wg_ratios.append(np.sum(mass[np.where((u >= 2e4) & (u < 5e5))]) / np.sum(mass))
                hg_ratios.append(np.sum(mass[np.where((u >= 5e5))]) / np.sum(mass))
        
        # Save data for each halo in numpy arrays #
        np.save(path + 'masses_' + str(s.haloname), masses)
        np.save(path + 'name_' + str(s.haloname), s.haloname)
        np.save(path + 'sfg_ratios_' + str(s.haloname), sfg_ratios)
        np.save(path + 'wg_ratios_' + str(s.haloname), wg_ratios)
        np.save(path + 'hg_ratios_' + str(s.haloname), hg_ratios)
        np.save(path + 'redshifts_' + str(s.haloname), redshifts[np.where(redshifts <= redshift_cut)])
    
    # # Generate the figure #
    f, ax = plt.subplots(1, figsize=(10, 7.5))
    plt.grid(True)
    plt.ylim(-0.2, 1.2)
    plt.xlim(0, 5)
    plt.grid(True)
    plt.ylabel(r'Gas fraction', size=16)
    plt.xlabel(r'Redshift', size=16)
    
    # Load and plot the data #
    names = glob.glob(path + '/name_18N*')
    names.sort()
    
    for i in range(len(names)):
        sfg_ratios = np.load(path + 'sfg_ratios_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        wg_ratios = np.load(path + 'wg_ratios_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        hg_ratios = np.load(path + 'hg_ratios_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        redshifts = np.load(path + 'redshifts_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        redshifts = np.insert(redshifts, 0, 5.1)
        sfg_ratios = np.flip(sfg_ratios)
        wg_ratios = np.flip(wg_ratios)
        hg_ratios = np.flip(hg_ratios)
        redshifts = np.flip(redshifts)
        
        for j in range(len(redshifts) - 1):
            b1, = plt.bar(redshifts[j], sfg_ratios[j], width=redshifts[j + 1] - redshifts[j], alpha=0.6, color='blue', align='edge', edgecolor='none')
            b2, = plt.bar(redshifts[j], wg_ratios[j], bottom=sfg_ratios[j], width=redshifts[j + 1] - redshifts[j], alpha=0.6, color='green',
                          align='edge', edgecolor='none')
            b3, = plt.bar(redshifts[j], hg_ratios[j], bottom=np.sum(np.vstack([sfg_ratios[j], wg_ratios[j]]).T),
                          width=redshifts[j + 1] - redshifts[j], alpha=0.6, align='edge', color='red', edgecolor='none')
    
    plt.legend([b3, b2, b1], [r'Hot gas', r'Warm gas', r'Cold star-forming gas'], loc='upper left', fontsize=12, frameon=False, numpoints=1)
    ax.text(0.0, 1.01, 'Au-' + str(re.split('_|.npy', names[0])[1]), color='k', fontsize=16, transform=ax.transAxes)
    pdf.savefig(f, bbox_inches='tight')  # Save the figure.
    plt.close()
    return None