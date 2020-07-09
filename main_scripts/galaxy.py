import os
import re
import glob
import calcGrid
import plot_tools
import matplotlib

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from const import *
from sfigure import *
from loadmodules import *
from scipy.special import gamma
from scipy.optimize import curve_fit
from scripts.gigagalaxy.util import plot_helper
from parse_particledata import parse_particledata

default_level = 4
color_array = iter(cm.rainbow(np.linspace(0, 1, 9)))
photo_band = ['U', 'B', 'V', 'K', 'g', 'r', 'i', 'z']
colors = ['black', 'tab:red', 'tab:green', 'tab:blue']
Msunabs = [5.61, 5.48, 4.83, 3.30, 5.12, 4.68, 4.57, 4.60]
element = {'H':0, 'He':1, 'C':2, 'N':3, 'O':4, 'Ne':5, 'Mg':6, 'Si':7, 'Fe':8}


def circularity(pdf, data, redshift, read):
    """
    Plot orbital circularity distribution for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param data: data from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :param read: boolean to read new data.
    :return: None
    """
    path = '/u/di43/Auriga/plots/data/' + 'c/'
    
    # Read the data #
    if read is True:
        # Check if a folder to save the data exists, if not create one #
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
        particle_type = [4]
        attributes = ['age', 'mass', 'pot', 'pos', 'type']
        data.select_haloes(default_level, redshift, loadonlyhalo=0, loadonlytype=particle_type, loadonly=attributes)
        
        # Loop over all available haloes #
        for s in data:
            # Check if any of the haloes' data already exists, if not then read and save it #
            names = glob.glob(path + '/name_*')
            names = [re.split('_|.npy', name)[1] for name in names]
            if str(s.haloname) in names:
                continue
            else:
                print("Analysing halo:", str(s.haloname))
            
            # Select the halo and rotate it based on its principal axes so galaxy's spin is aligned with the z-axis #
            s.calc_sf_indizes(s.subfind)
            s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)
            
            # Mask the data: select all and stellar particles inside 0.1*R200c #
            age = np.zeros(s.npartall)
            age[s.data['type'] == 4] = s.data['age']
            all_mask, = np.where((s.r() < 0.1 * s.subfind.data['frc2'][0]))
            stellar_mask, = np.where((s.data['type'] == 4) & (age > 0.) & (s.r() < 0.1 * s.subfind.data['frc2'][0]))
            n_stars = np.size(stellar_mask)
            
            # Sort particles's radii and calculate the cumulative mass #
            n_all = np.size(all_mask)
            cumulative_mass = np.zeros(n_all)
            sorted_r = s.r()[all_mask].argsort()
            cumulative_mass[sorted_r] = np.cumsum(s.data['mass'][all_mask][sorted_r])
            
            # Calculate the z component of the specific angular momentum and the total (kinetic plus potential) energy of stellar particles #
            stellar_mask, = np.where((s.data['type'][all_mask] == 4) & (age[all_mask] > 0.))  # Mask the data: select stellar particles.
            stellar_masses = s.data['mass'][all_mask][stellar_mask]
            specific_angular_momentum_z = np.cross(s.data['pos'][all_mask, :][stellar_mask, :], s.data['vel'][all_mask, :][stellar_mask, :])[:, 0]
            total_energy = 0.5 * (s.data['vel'][all_mask, :][stellar_mask, :] ** 2.).sum(axis=1) + s.data['pot'][all_mask][stellar_mask]
            sorted_total_energy = total_energy.argsort()
            specific_angular_momentum_z = specific_angular_momentum_z[sorted_total_energy]
            stellar_masses = stellar_masses[sorted_total_energy]
            
            # Calculate the orbital circularity #
            max_circular_angular_momentum = np.zeros(n_stars)
            for i in range(n_stars):
                if i < 50:
                    left = 0
                    right = 100
                elif i > n_stars - 50:
                    left = n_stars - 100
                    right = n_stars
                else:
                    left = i - 50
                    right = i + 50
                
                max_circular_angular_momentum[i] = np.max(specific_angular_momentum_z[left:right])
            epsilon = specific_angular_momentum_z / max_circular_angular_momentum
            
            # Calculate the disc component and the disc to total mass ratio #
            disc_mask, = np.where((epsilon > 0.7) & (epsilon < 1.7))
            all_mask, = np.where((epsilon > -1.7) & (epsilon < 1.7))
            disc_fraction = np.sum(stellar_masses[disc_mask]) / np.sum(stellar_masses[all_mask])
            
            # Save data for each halo in numpy arrays #
            np.save(path + 'name_' + str(s.haloname), s.haloname)
            np.save(path + 'epsilon_' + str(s.haloname), epsilon)
            np.save(path + 'disc_fraction_' + str(s.haloname), disc_fraction)
            np.save(path + 'stellar_masses_' + str(s.haloname), stellar_masses)
    
    # Get the names and sort them #
    names = glob.glob(path + '/name_*')
    names.sort()
    
    # Loop over all available haloes #
    for i in range(len(names)):
        # Generate the figure and set its parameters #
        figure, axis = plt.subplots(1, figsize=(10, 7.5))
        plot_tools.set_axis(axis, xlim=[-2, 2], ylim=[0, 5], xlabel=r'$\mathrm{\epsilon}$', ylabel=r'$\mathrm{f(\epsilon)}$', aspect=None)
        figure.text(0.0, 0.9, r'$\mathrm{Au-%s}$''\n' r'$\mathrm{z=%s}$' % (str(re.split('_|.npy', names[i])[1]), str(redshift)), fontsize=16,
                    transform=axis.transAxes)
        
        # Load the data #
        epsilon = np.load(path + 'epsilon_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        disc_fraction = np.load(path + 'disc_fraction_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        stellar_masses = np.load(path + 'stellar_masses_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        
        # Plot orbital circularity distribution #
        y_data, edges = np.histogram(epsilon, weights=stellar_masses / np.sum(stellar_masses), bins=100, range=[-1.7, 1.7])
        y_data /= edges[1:] - edges[:-1]
        axis.plot(0.5 * (edges[1:] + edges[:-1]), y_data, color=next(color_array), label=r'$\mathrm{D/T = %.2f}$' % disc_fraction)
        
        # Create the legend, save and close the figure #
        axis.legend(loc='upper right', fontsize=16, frameon=False, numpoints=1)
        pdf.savefig(figure, bbox_inches='tight')  # Save the figure.
        plt.close()
    return None


def obs_tullyfisher_fit(masses):
    """
    Observational fit from equation 24 of Dutton et al. 2011
    :param masses: stellar mass
    :return: velocity
    """
    velocity = 10 ** (2.179 + 0.259 * np.log10(masses / 10 ** 0.3))
    return velocity


def pizagno_convert_color_to_mass(color, magnitude, band=5):
    """
    Convert color to mass based on the table in the appendix of Bell et al. 2003.
    :param color: color
    :param magnitude: magnitude
    :param band: r-band
    :return: 10^stellar_mass
    """
    mass_to_light = 10 ** (-0.306 + 1.097 * color)
    luminosity = 10 ** (2.0 * (Msunabs[band] - magnitude) / 5.0)
    stellar_mass = np.log10(mass_to_light * luminosity * 1.0e-10)  # In 1e10 Msun.
    stellar_mass = (stellar_mass - 0.230) / 0.922  # Convert in the MPA stellar masses eq. 20 of Dutton et al. 2011.
    return 10 ** stellar_mass


def verheijen_convert_color_to_mass(color, magnitude, band=1):
    """
    Convert color to mass based on the table in the appendix of Bell et al. 2003.
    :param color: color
    :param magnitude: magnitude
    :param band: U-band
    :return: :return: 10^stellar_mass
    """
    mass_to_light = 10 ** (-0.976 + 1.111 * color)
    luminosity = 10 ** (2.0 * (Msunabs[band] - magnitude) / 5.0)
    stellar_mass = np.log10(mass_to_light * luminosity * 1.0e-10)  # In 1e10 Msun.
    stellar_mass = (stellar_mass - 0.230) / 0.922
    return 10 ** stellar_mass


def courteau_convert_luminosity_to_mass(loglum):
    """
    Convert luminosity to mass from equation 28 of Dutton et al. 2007
    :param loglum: luminosity
    :return: 10^stellar_mass
    """
    mass_to_light = 10 ** (0.172 + 0.144 * (loglum - 10.3))
    luminosity = 10 ** (loglum)
    stellar_mass = np.log10(mass_to_light * luminosity * 1.0e-10)  # In 1e10 Msun.
    stellar_mass = (stellar_mass - 0.230) / 0.922
    return 10 ** stellar_mass


def tully_fisher(pdf, data, redshift, read):
    """
    Plot the Tully-Fisher relation
    :param pdf: path to save the pdf from main.make_pdf
    :param data: data from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :param read: boolean to read new data.
    :return: None
    """
    
    path = '/u/di43/Auriga/plots/data/' + 'tf/'
    
    # Read the data #
    if read is True:
        # Check if a folder to save the data exists, if not create one #
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
        particle_type = [4]
        attributes = ['age', 'mass', 'pot', 'pos', 'type']
        data.select_haloes(default_level, redshift, loadonlyhalo=0, loadonlytype=particle_type, loadonly=attributes)
        
        # Loop over all available haloes #
        for s in data:
            # Check if any of the haloes' data already exists, if not then read and save it #
            names = glob.glob(path + '/name_*')
            names = [re.split('_|.npy', name)[1] for name in names]
            # if str(s.haloname) in names:
            #     continue
            # else:
            #     print("Analysing halo:", str(s.haloname))
            
            # Select the halo and rotate it based on its principal axes so galaxy's spin is aligned with the z-axis #
            s.calc_sf_indizes(s.subfind)
            s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)
            
            # Mask the data: select all and stellar particles inside 0.1*R200c #
            age = np.zeros(s.npartall)
            age[s.data['type'] == 4] = s.data['age']
            stellar_mask, = np.where((s.data['type'] == 4) & (age > 0.) & (s.r() < 0.1 * s.subfind.data['frc2'][0]))
            stellar_mass = np.sum(s.data['mass'][stellar_mask])
            
            # Calculate the cumulative mass and circular velocity #
            mass, edges = np.histogram(s.r(), weights=s.data['mass'], bins=100, range=[0., 0.025])
            cumulative_mass = np.cumsum(mass)
            circular_velocity = np.sqrt(G * cumulative_mass * 1e10 * msol / (edges[1:] * 1e6 * parsec)) / 1e5
            
            # Calculate the cumulative stellar mass and total circular velocity #
            smass, edges = np.histogram(s.r()[stellar_mask], weights=s.data['mass'][stellar_mask], bins=100, range=[0., 0.025])
            cumulative_stellar_mass = np.cumsum(smass)
            total_circular_velocity = circular_velocity[np.abs(cumulative_stellar_mass - 0.8 * stellar_mass).argmin()]
            
            # Save data for each halo in numpy arrays #
            np.save(path + 'name_' + str(s.haloname), s.haloname)
            np.save(path + 'stellar_mass_' + str(s.haloname), stellar_mass)
            np.save(path + 'total_circular_velocity_' + str(s.haloname), total_circular_velocity)
        
    # Get the names and sort them #
    names = glob.glob(path + '/name_*')
    names.sort()
    
    # Loop over all available haloes #
    for i in range(len(names)):
        # Generate the figure and set its parameters #
        figure, axis = plt.subplots(1, figsize=(10, 7.5))
        plot_tools.set_axis(axis, xlabel=r'$\mathrm{M_{\bigstar}/M_{\odot}}$',
                            ylabel=r'$\mathrm{log_{10}(v_{circ}/(km\;s^{-1}))}$', aspect=None)
        figure.text(0.0, 0.9, r'$\mathrm{Au-%s}$''\n' r'$\mathrm{z=%s}$' % (str(re.split('_|.npy', names[i])[1]), str(redshift)), fontsize=16,
                    transform=axis.transAxes)
        
        # Load the data #
        stellar_mass = np.load(path + 'stellar_mass_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        total_circular_velocity = np.load(path + 'total_circular_velocity_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        
        axis.semilogx(stellar_mass * 1e10, np.log10(total_circular_velocity), color=next(color_array), linestyle="None", marker='*', ms=15.0)
        axis.legend(loc='lower right', fontsize=16, frameon=False, numpoints=1)
        
        # plot Pizagno et al. 2007 sample
        tablename = "./data/pizagno.txt"
        rmag_p = np.genfromtxt(tablename, comments='#', usecols=3)
        vcirc_p = np.genfromtxt(tablename, comments='#', usecols=9)
        color_p = np.genfromtxt(tablename, comments='#', usecols=5)
        mass_p = pizagno_convert_color_to_mass(color_p, rmag_p)
        
        axis.semilogx(1.0e10 * mass_p, np.log10(vcirc_p), '^', mfc='lightgray', ms=2.5, mec='None')
        
        # plot Verheijen 2001 sample
        tablename = "./data/verheijen.txt"
        Bmag_v = np.genfromtxt(tablename, comments='#', usecols=1)
        Rmag_v = np.genfromtxt(tablename, comments='#', usecols=2)
        vcirc_v = np.genfromtxt(tablename, comments='#', usecols=7)
        color_v = Bmag_v - Rmag_v
        mass_v = verheijen_convert_color_to_mass(color_v, Bmag_v)
        
        axis.semilogx(1.0e10 * mass_v, np.log10(vcirc_v), 's', mfc='lightgray', ms=2.5, mec='None')
        
        # plot Courteau et al. 2007 sample
        tablename = "./data/courteau.txt"
        loglum_c = np.genfromtxt(tablename, comments='#', usecols=6)
        vcirc_c = np.genfromtxt(tablename, comments='#', usecols=8)
        mass_c = courteau_convert_luminosity_to_mass(loglum_c)
        
        axis.semilogx(1.0e10 * mass_c, vcirc_c, 'o', mfc='lightgray', ms=2.5, mec='None')
        
        # Plot best fit from Dutton et al. (2011)
        masses = np.arange(0.1, 50.)
        axis.semilogx(1.0e10 * masses, np.log10(obs_tullyfisher_fit(masses)), ls='--', color='darkgray', lw=0.8)
        label = ['Pizagno+ 07', 'Verheijen 01', 'Courteau+ 07']
        l1 = axis.legend(label, loc='upper right', fontsize=16, frameon=False, numpoints=1)
        
        pdf.savefig(figure, bbox_inches='tight')  # Save the figure.
        plt.close()
    return None


def guo_abundance_matching(mass):
    # equation 3 of Guo et al. 2010. Mass MUST be given in 10^10 Msun
    c = 0.129
    M_zero = 10 ** 1.4
    alpha = -0.926
    beta = 0.261
    gamma = -2.44
    
    val = c * mass * ((mass / M_zero) ** alpha + (mass / M_zero) ** beta) ** gamma
    return val


def stellar_vs_total(pdf, data, levels):
    nlevels = len(levels)
    
    # Generate the figure and set its parameters #
    figure, axis = plt.subplots(1, figsize=(10, 7.5))
    plt.grid(True, color='gray', linestyle='-')
    plt.xlabel("$\\rm{M_{halo}\\,[M_\\odot]}$")
    plt.ylabel("$\\rm{M_{stars}\\,[M_\\odot]}$")
    
    masses = np.arange(15., 300.)
    cosmic_baryon_frac = 0.048 / 0.307
    
    axis.loglog(1.0e10 * masses, 1.0e10 * masses * cosmic_baryon_frac, 'k--', )
    
    guo_high = guo_abundance_matching(masses) * 10 ** (+0.2)
    guo_low = guo_abundance_matching(masses) * 10 ** (-0.2)
    
    axis.fill_between(1.0e10 * masses, 1.0e10 * guo_low, 1.0e10 * guo_high, color='lightgray', edgecolor='None')
    
    axis.loglog(1.0e10 * masses, 1.0e10 * guo_abundance_matching(masses), 'k:')
    
    labels = ["$\\rm{M_{200}\\,\\,\\,\\Omega_b / \\Omega_m}$", "Guo+ 10"]
    l1 = axis.legend(labels, loc='upper right', fontsize=16, frameon=False)
    
    for il in range(nlevels):
        default_level = levels[il]
        
        data.select_haloes(default_level, 0.)
        nhalos = data.selected_current_nsnaps
        colors = iter(cm.rainbow(np.linspace(0, 1, nhalos)))
        
        mstar = np.zeros(nhalos)
        mhalo = np.zeros(nhalos)
        
        data.select_haloes(default_level, 0., loadonlyhalo=0)
        
        ihalo = 0
        for s in data:
            s.centerat(s.subfind.data['fpos'][0, :])
            
            age = np.zeros(s.npartall)
            age[s.data['type'] == 4] = s.data['age']
            istars, = np.where((s.r() < 0.1 * s.subfind.data['frc2'][0]) & (s.data['type'] == 4) & (age > 0.))
            mstar[ihalo] = s.data['mass'][istars].sum()
            
            all_mask, = np.where(s.r() < s.subfind.data['frc2'][0])
            mhalo[ihalo] = s.data['mass'][all_mask].sum()
            
            axis.loglog(mhalo[ihalo] * 1e10, mstar[ihalo] * 1e10, color=next(colors), linestyle="None", marker='*', ms=15.0,
                        label="Au%s" % s.haloname)
            axis.legend(loc='lower right', fontsize=16, frameon=False, numpoints=1)
            axis.add_artist(l1)
            
            ihalo += 1
    
    pdf.savefig(figure, bbox_inches='tight')  # Save the figure.
    plt.close()
    return None


def convert_rband_to_Rband_mag(r, g):
    """
    Convert Gunn r-band to Mould R-band magnitude using formula A8 from Windhorst+ 1991.
    :param r: Gunn r band magnitude.
    :param g: Gunn g band magnitude.
    :return: Mould R-band magnitude.
    """
    R = r - 0.51 - 0.15 * (g - r)
    return R


def gas_fraction(pdf, data, levels):
    nlevels = len(levels)
    
    # Generate the figure and set its parameters #
    figure, axis = plt.subplots(1, figsize=(10, 7.5))
    plt.grid(True, color='gray', linestyle='-')
    plt.xlabel("$\\rm{M_{R}\\,[mag]}$", size=16)
    plt.ylabel("$\\rm{f_{gas}}$", size=16)
    
    for il in range(nlevels):
        default_level = levels[il]
        
        data.select_haloes(default_level, 0.)
        nhalos = data.selected_current_nsnaps
        colors = iter(cm.rainbow(np.linspace(0, 1, nhalos)))
        
        MR = np.zeros(nhalos)
        fgas = np.zeros(nhalos)
        
        data.select_haloes(default_level, 0., loadonlyhalo=0)
        
        ihalo = 0
        for s in data:
            s.centerat(s.subfind.data['fpos'][0, :])
            
            age = np.zeros(s.npartall)
            age[s.data['type'] == 4] = s.data['age']
            istars, = np.where((s.r() < 0.1 * s.subfind.data['frc2'][0]) & (s.data['type'] == 4) & (age > 0.)) - s.nparticlesall[:4].sum()
            # Calculate the bolometric R-band magnitude in units of the bolometric magnitude of the Sun #
            Rband = convert_rband_to_Rband_mag(s.data['gsph'][istars, 5], s.data['gsph'][istars, 4])
            MR[ihalo] = -2.5 * np.log10((10. ** (- 2.0 * Rband / 5.0)).sum())
            
            istars, = np.where((s.r() < 0.1 * s.subfind.data['frc2'][0]) & (s.data['type'] == 4) & (age > 0.))
            mask, = np.where((s.r() < 0.1 * s.subfind.data['frc2'][0]) & (s.data['type'] == 0))
            fgas[ihalo] = s.data['mass'][mask].sum() / (s.data['mass'][mask].sum() + s.data['mass'][istars].sum())
            
            axis.plot(MR[ihalo], fgas[ihalo], color=next(colors), linestyle="None", marker='*', ms=15.0, label="Au%s" % s.haloname)
            axis.legend(loc='lower right', fontsize=16, frameon=False, numpoints=1)
            
            ihalo += 1
    
    pdf.savefig(figure, bbox_inches='tight')  # Save the figure.
    plt.close()
    return None


def central_bfld(pdf, data, levels):
    nlevels = len(levels)
    
    figure = plt.figure(figsize=(8.2, 8.2))
    plt.grid(True, color='gray', linestyle='-')
    axis = figure.iaxes(1.0, 1.0, 6.8, 6.8, top=True)
    axis.set_xlabel("$\\rm{M_{stars}\\,[M_\\odot]}$")
    axis.set_ylabel("$B_\mathrm{r<1\,kpc}\,\mathrm{[\mu G]}$")
    
    for il in range(nlevels):
        default_level = levels[il]
        
        data.select_haloes(default_level, 0.)
        nhalos = data.selected_current_nsnaps
        colors = iter(cm.rainbow(np.linspace(0, 1, nhalos)))
        
        mstar = np.zeros(nhalos)
        bfld = np.zeros(nhalos)
        
        data.select_haloes(default_level, 0., loadonlyhalo=0, loadonlytype=[0, 4])
        
        ihalo = 0
        for s in data:
            s.centerat(s.subfind.data['fpos'][0, :])
            
            i, = np.where((s.r() < 0.001) & (s.data['type'] == 0))
            bfld[ihalo] = np.sqrt(((s.data['bfld'][i, :] ** 2.).sum(axis=1) * s.data['vol'][i]).sum() / s.data['vol'][i].sum()) * bfac
            
            age = np.zeros(s.npartall)
            age[s.data['type'] == 4] = s.data['age']
            istars, = np.where((s.r() < 0.1 * s.subfind.data['frc2'][0]) & (s.data['type'] == 4) & (age > 0.))
            mstar[ihalo] = s.data['mass'][istars].sum()
            
            axis.loglog(mstar[ihalo] * 1e10, bfld[ihalo] * 1e6, color=next(colors), linestyle="None", marker='*', ms=15.0,
                        label="Au%s-%d" % (s.haloname, levels[0]))
            axis.legend(loc='lower right', fontsize=16, frameon=False, numpoints=1)
            
            ihalo += 1
    
    pdf.savefig(figure, bbox_inches='tight')  # Save the figure.
    plt.close()
    return None


def bar_strength(pdf, data, read):
    """
    Calculate bar strength from Fourier modes of surface density
    :param pdf: path to save the pdf from main.make_pdf
    :param data: data from main.make_pdf
    :param read: boolean to read new data.
    :return: None
    """
    # Read the data #
    if read is True:
        # Check if a folder to save the data exists, if not create one #
        path = '/u/di43/Auriga/plots/data/' + 'bs/'
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
        particle_type = [4]
        attributes = ['age', 'mass', 'pos']
        data.select_haloes(default_level, 0, loadonlyhalo=0, loadonlytype=particle_type, loadonly=attributes)
        
        # Loop over all available haloes #
        for s in data:
            # Select the halo and rotate it based on its principal axes so galaxy's spin is aligned with the z-axis #
            s.calc_sf_indizes(s.subfind)
            s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)
            
            mask, = np.where(s.data['age'] > 0.)  # Mask the data: select stellar particles.
            z_rotated, y_rotated, x_rotated = plot_tools.rotate_bar(s.data['pos'][mask, 0] * 1e3, s.data['pos'][mask, 1] * 1e3,
                                                                    s.data['pos'][mask, 2] * 1e3)  # Distances are in Mpc.
            s.data['pos'] = np.vstack((z_rotated, y_rotated, x_rotated)).T  # Rebuild the s.data['pos'] attribute in kpc.
            x, y = s.data['pos'][:, 2] * 1e3, s.data['pos'][:, 1] * 1e3  # Load positions and convert from Mpc to Kpc.
            
            # Split up galaxy in radius bins and calculate the Fourier components #
            n_bins = 40  # Number of radial bins.
            r = np.sqrt(x[:] ** 2 + y[:] ** 2)  # Radius of each particle.
            
            # Initialise Fourier components #
            r_m = np.zeros(n_bins)
            beta_2 = np.zeros(n_bins)
            alpha_0 = np.zeros(n_bins)
            alpha_2 = np.zeros(n_bins)
            
            # Split up galaxy in radius bins and calculate Fourier components #
            for i in range(0, n_bins):
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
            A2 = np.divide(np.sqrt(alpha_2[:] ** 2 + beta_2[:] ** 2), alpha_0[:])
            
            # Save data for each halo in numpy arrays #
            np.save(path + 'A2_' + str(s.haloname), A2)
            np.save(path + 'r_m_' + str(s.haloname), r_m)
            np.save(path + 'name_' + str(s.haloname), s.haloname)
    
    # Generate the figure and set its parameters #
    figure, axis = plt.subplots(1, figsize=(10, 7.5))
    plt.grid(True, color='gray', linestyle='-')
    plt.ylim(-0.2, 1.2)
    plt.xlim(0, 10)
    plt.ylabel(r'$\mathrm{A_{2}}$', size=16)
    plt.xlabel(r'$\mathrm{R\,[kpc]}$', size=16)
    
    # Plot bar strength as a function of radius plot r_m versus A2 #
    names = glob.glob(path + '/name_18*')
    names.sort()
    colors = iter(cm.rainbow(np.linspace(0, 1, len(names))))
    
    # Loop over all available haloes #
    for i in range(len(names)):
        A2 = np.load(path + 'A2_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        r_m = np.load(path + 'r_m_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        plt.plot(r_m, A2, color=next(colors), label='Au-%s bar strength: %.2f' % (str(re.split('_|.npy', names[i])[1]), max(A2)))
        plt.plot([r_m[np.where(A2 == max(A2))], r_m[np.where(A2 == max(A2))]], [-0.0, max(A2)], color=next(colors), linestyle='dashed',
                 label='Au-%s bar length= %.2f kpc' % (str(re.split('_|.npy', names[i])[1]), r_m[np.where(A2 == max(A2))]))
    
    axis.legend(loc='upper left', fontsize=16, frameon=False, numpoints=1)
    
    pdf.savefig(figure, bbox_inches='tight')  # Save the figure.
    plt.close()
    return None


def stellar_surface_density_decomposition(pdf, data, redshift):
    particle_type = [4]
    attributes = ['pos', 'vel', 'mass', 'age', 'gsph']
    data.select_haloes(default_level, redshift, loadonlyhalo=0, loadonlytype=particle_type, loadonly=attributes)
    
    # Loop over all available haloes #
    for s in data:
        # Generate the figure and set its parameters #
        figure = plt.figure(0, figsize=(10, 7.5))
        plt.ylim(1e0, 1e6)
        plt.xlim(0.0, 30.0)
        plt.grid(True, color='gray', linestyle='-')
        plt.xlabel("$\mathrm{R [kpc]}$", size=16)
        plt.ylabel("$\mathrm{\Sigma [M_{\odot} pc^{-2}]}$", size=16)
        plt.tick_params(direction='out', which='both', top='on', right='on')
        
        s.calc_sf_indizes(s.subfind)
        s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)
        
        g = parse_particledata(s, s.subfind, attributes, radialcut=0.1 * s.subfind.data['frc2'][0])
        g.prep_data()
        sdata = g.sgdata['sdata']
        
        # Define the radial and vertical cuts and calculate the mass surface density #
        radial_cut = 0.1 * s.subfind.data['frc2'][0]  # Radial cut in Mpc.
        ii, = np.where((abs(sdata['pos'][:, 0]) < 0.005))  # Vertical cut in Mpc.
        rad = np.sqrt((sdata['pos'][:, 1:] ** 2).sum(axis=1))
        mass, edges = np.histogram(rad[ii], bins=50, range=(0., radial_cut), weights=sdata['mass'][ii])
        surface = np.zeros(len(edges) - 1)
        surface[:] = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
        sden = np.divide(mass, surface)
        
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
        p = plot_helper.plot_helper()  # Load the helper.
        
        try:
            sigma = 0.1 * sdfit
            bounds = ([0.01, 0.0, 0.01, 0.5, 0.25], [1.0, 6.0, 10.0, 2.0, 10.0])
            (popt, pcov) = curve_fit(p.total_profile, r, sdfit, sigma=sigma, bounds=bounds)
            
            # Compute component masses from the fit #
            disc_mass = 2.0 * np.pi * popt[0] * popt[1] * popt[1]
            bulge_mass = np.pi * popt[2] * popt[3] * popt[3] * gamma(2.0 / popt[4] + 1)
            disc_to_total = disc_mass / (bulge_mass + disc_mass)
        
        except:
            popt = np.zeros(5)
            print('Fitting failed')
        
        plt.axvline(rfit, color='gray', linestyle='--')
        plt.semilogy(r, 1e10 * sdfit * 1e-6, 'o', markersize=5, color='k', linewidth=0.0)
        plt.semilogy(r, 1e10 * p.exp_prof(r, popt[0], popt[1]) * 1e-6, 'b-')
        plt.semilogy(r, 1e10 * p.sersic_prof1(r, popt[2], popt[3], popt[4]) * 1e-6, 'r-')
        plt.semilogy(r, 1e10 * p.total_profile(r, popt[0], popt[1], popt[2], popt[3], popt[4]) * 1e-6, 'k-')
        
        figure.text(0.15, 0.8, 'Au-%s' '\n' r'$\mathrm{n} = %.2f$' '\n' r'$\mathrm{R_{d}} = %.2f$' '\n' r'$\mathrm{R_{eff}} = %.2f$' '\n' % (
            s.haloname, 1. / popt[4], popt[1], popt[3] * p.sersic_b_param(1.0 / popt[4]) ** (1.0 / popt[4])))
        
        pdf.savefig(figure, bbox_inches='tight')  # Save the figure.
        plt.close()
    return None


def circular_velocity_curves(pdf, data, redshift):
    particle_type = [0, 1, 4, 5]
    attributes = ['pos', 'vel', 'mass', 'age']
    data.select_haloes(default_level, redshift, loadonlyhalo=0, loadonlytype=particle_type, loadonly=attributes)
    
    # Loop over all available haloes #
    for s in data:
        # Generate the figure and set its parameters #
        figure, axis = plt.subplots(1, figsize=(10, 7.5))
        plt.xlim(0.0, 24.0)
        plt.ylim(0.0, 700.0)
        plt.grid(True, color='gray', linestyle='-')
        plt.xlabel("$\mathrm{R [kpc]}$", size=16)
        plt.ylabel("$\mathrm{V_{c} [km\, s^{-1}]}$", size=16)
        plt.tick_params(direction='out', which='both', top='on', right='on')
        axis.set_yticks([150, 160, 170, 180], minor=True)
        figure.text(0.0, 1.01, 'Au-' + str(s.haloname), color='k', fontsize=16, transform=axis.transAxes)
        
        s.calc_sf_indizes(s.subfind)
        s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)
        
        nshells = 400
        radius = 0.04
        dr = radius / nshells
        
        na = s.nparticlesall
        end = na.copy()
        
        for i in range(1, len(end)):
            end[i] += end[i - 1]
        
        start = np.zeros(len(na), dtype='int32')
        for i in range(1, len(start)):
            start[i] = end[i - 1]
        
        shmass = np.zeros((nshells, 6))
        shvel = np.zeros((nshells, 6))
        for i in range(6):
            rp = calcGrid.calcRadialProfile(s.data['pos'][start[i]:end[i], :].astype('float64'), s.data['mass'][start[i]:end[i]].astype('float64'), 0,
                                            nshells, dr, s.center[0], s.center[1], s.center[2])
            
            radius = rp[1, :]
            shmass[:, i] = rp[0, :]
            for j in range(1, nshells):
                shmass[j, i] += shmass[j - 1, i]
            shvel[:, i] = np.sqrt(G * shmass[:, i] * 1e10 * msol / (radius * 1e6 * parsec)) / 1e5
        
        rp = calcGrid.calcRadialProfile(s.data['pos'].astype('float64'), s.data['mass'].astype('float64'), 0, nshells, dr, s.center[0], s.center[1],
                                        s.center[2])
        radius = rp[1, :]
        mtot = rp[0, :]
        
        for j in range(1, nshells):
            mtot[j] += mtot[j - 1]
        
        vtot = np.sqrt(G * mtot * 1e10 * msol / (radius * 1e6 * parsec)) / 1e5
        plt.plot(radius * 1e3, vtot, 'k-', linewidth=4, label='Total')
        plt.plot(radius * 1e3, shvel[:, 0], 'b--', linewidth=4, label='Gas')
        plt.plot(radius * 1e3, shvel[:, 4], 'g:', linewidth=4, label='Stars')
        plt.plot(radius * 1e3, shvel[:, 1], 'r-.', linewidth=4, label='Dark matter')
        axis.legend(loc='upper right', fontsize=16, frameon=False, numpoints=1)
        
        pdf.savefig(figure, bbox_inches='tight')  # Save the figure.
        plt.close()
    return None


def gas_temperature_histogram(pdf, data, read):
    """
    Mass- and volume-weighted histograms of gas temperature
    :param pdf: path to save the pdf from main.make_pdf
    :param data: data from main.make_pdf
    :param read: boolean to read new data.
    :return: None
    """
    # Read the data #
    if read is True:
        volumes, masses, temperatures = [], [], []  # Declare empty arrays to store the data.
        
        # Check if a folder to save the data exists, if not create one #
        path = '/u/di43/Auriga/plots/data/' + 'gth/'
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Get all available redshifts #
        haloes = data.get_haloes(default_level)
        for name, halo in haloes.items():
            redshifts = halo.get_redshifts()
        
        for i in range(0, 1):
            redshift = np.flip(redshifts)[i]
            
            # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
            particle_type = [0, 4]
            attributes = ['age', 'mass', 'ne', 'pos', 'rho', 'u', 'vol']
            data.select_haloes(default_level, redshift, loadonlyhalo=0, loadonlytype=particle_type, loadonly=attributes)
            
            # Loop over all available haloes #
            for s in data:
                # Check if any of the haloes' data already exists, if not then read and save it #
                names = glob.glob(path + '/name_*')
                names = [re.split('_|.npy', name)[1] for name in names]
                # if str(s.haloname) in names:
                #     continue
                
                # Select the halo and rotate it based on its principal axes so galaxy's spin is aligned with the z-axis #
                s.calc_sf_indizes(s.subfind)
                s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)
                
                # Mask the data: select gas cells within the virial radius R200 #
                mask, = np.where((s.data['type'] == 0) & (s.r() < s.subfind.data['frc2'][0]))
                
                # Calculate the temperature of the gas cells #
                ne = s.data['ne'][mask]
                metallicity = s.data['gz'][mask]
                XH = s.data['gmet'][mask, element['H']]
                yhelium = (1 - XH - metallicity) / (4 * XH)
                mu = (1 + 4 * yhelium) / (1 + yhelium + ne)
                temperature = GAMMA_MINUS1 * s.data['u'][mask] * 1.0e10 * mu * PROTONMASS / BOLTZMANN
                
                # Append the properties for all redshifts #
                temperatures.append(temperature)
                volumes.append(s.data['vol'][mask])
                masses.append(s.data['mass'][mask])
        
        # Save data for each halo in numpy arrays #
        np.save(path + 'masses_' + str(s.haloname), masses)
        np.save(path + 'volumes_' + str(s.haloname), volumes)
        np.save(path + 'name_' + str(s.haloname), s.haloname)
        np.save(path + 'temperatures_' + str(s.haloname), temperatures)
    
    # Generate the figure and set its parameters #
    figure = plt.figure(figsize=(10, 7.5))
    plt.grid(True, color='gray', linestyle='-')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'Temperature [K]')
    plt.tick_params(direction='out', which='both', top='on', right='on')
    
    # Load and plot the data #
    names = glob.glob(path + '/name_06*')
    names.sort()
    
    # Loop over all available haloes #
    for i in range(len(names)):
        masses = np.load(path + 'masses_' + str(re.split('_|.npy', names[i])[1]) + '.npy', allow_pickle=True)
        volumes = np.load(path + 'volumes_' + str(re.split('_|.npy', names[i])[1]) + '.npy', allow_pickle=True)
        temperatures = np.load(path + 'temperatures_' + str(re.split('_|.npy', names[i])[1]) + '.npy', allow_pickle=True)
        
        # for i in range(len(masses)):
        #     print(len(masses[i]))
        # average_masses = np.sum(masses, axis=0) / 10
        # average_volumes = np.sum(volumes, axis=0) / 10
        # average_temperatures = np.sum(temperatures,axis=0) / 10
        
        # y_data, edges = np.histogram(average_temperatures, weights=average_masses / np.sum(average_masses), bins=100)
        # plt.plot(0.5 * (edges[1:] + edges[:-1]), y_data, label='Mass-weighted')
        
        # Convert bin centres to log space and plot #
        l = np.sort(temperatures)
        print(l[0, :100])
        y_data, edges = np.histogram(temperatures, weights=masses / np.sum(masses), bins=100)
        print(np.sort(edges))
        bin_centres = 10 ** (0.5 * (np.log10(edges[1:]) + np.log10(edges[:-1])))
        plt.plot(bin_centres, y_data, label='Mass-weighted', color=colors[3])
        
        y_data, edges = np.histogram(temperatures, weights=volumes / np.sum(volumes), bins=100)
        bin_centres = 10 ** (0.5 * (np.log10(edges[1:]) + np.log10(edges[:-1])))
        plt.plot(bin_centres, y_data, label='Volume-weighted', color=colors[2])
    
    plt.legend(loc='upper left', fontsize=16, frameon=False, numpoints=1)
    pdf.savefig(figure, bbox_inches='tight')  # Save the figure.
    plt.close()
    return None


def gas_distance_temperature(pdf, data, redshift, read):
    """
    Temperature as a function of distance of gas cells.
    :param pdf: path to save the pdf from main.make_pdf
    :param data: data from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :param read: boolean to read new data.
    :return: None
    """
    # Read the data #
    if read is True:
        # Check if a folder to save the data exists, if not create one #
        path = '/u/di43/Auriga/plots/data/' + 'gdt/'
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
        particle_type = [0, 4]
        attributes = ['age', 'mass', 'ne', 'pos', 'rho', 'u', 'vol']
        data.select_haloes(default_level, redshift, loadonlyhalo=0, loadonlytype=particle_type, loadonly=attributes)
        
        # Loop over all available haloes #
        for s in data:
            # Check if any of the haloes' data already exists, if not then read and save it #
            names = glob.glob(path + '/name_*')
            names = [re.split('_|.npy', name)[1] for name in names]
            if str(s.haloname) in names:
                continue
            
            # Select the halo and rotate it based on its principal axes so galaxy's spin is aligned with the z-axis #
            s.calc_sf_indizes(s.subfind)
            s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)
            mask, = np.where(
                (s.data['type'] == 0) & (s.r() < s.subfind.data['frc2'][0]))  # Mask the data: select gas cells within the virial radius R200 #
            
            # Calculate the temperature of the gas cells #
            ne = s.data['ne'][mask]
            metallicity = s.data['gz'][mask]
            XH = s.data['gmet'][mask, element['H']]
            yhelium = (1 - XH - metallicity) / (4. * XH)
            mu = (1 + 4 * yhelium) / (1 + yhelium + ne)
            temperature = GAMMA_MINUS1 * s.data['u'][mask] * 1.0e10 * mu * PROTONMASS / BOLTZMANN
            
            # Save data for each halo in numpy arrays #
            np.save(path + 'name_' + str(s.haloname), s.haloname)
            np.save(path + 'temperature_' + str(s.haloname), temperature)
            np.save(path + 'spherical_distance_' + str(s.haloname), s.r()[mask])
    
    # Load and plot the data #
    names = glob.glob(path + '/name_18N*')
    names.sort()
    
    # Loop over all available haloes #
    for i in range(len(names)):
        # Generate the figure and set its parameters #
        figure, axis = plt.subplots(1, figsize=(10, 7.5))
        plt.grid(True, color='gray', linestyle='-')
        plt.xscale('log')
        plt.yscale('log')
        plt.ylim(1e1, 1e8)
        plt.xlim(1e-2, 2e2)
        cmap = matplotlib.cm.get_cmap('jet')
        axis.set_facecolor(cmap(0))
        plt.xlabel(r'$\mathrm{Radius\;[kpc]}$')
        plt.ylabel(r'$\mathrm{Temperature\;[K]}$')
        plt.tick_params(direction='out', which='both', top='on', right='on')
        figure.text(0.0, 1.01, 'Au-' + str(re.split('_|.npy', names[i])[1]), color='k', fontsize=16, transform=axis.transAxes)
        
        temperature = np.load(path + 'temperature_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        spherical_distance = np.load(path + 'spherical_distance_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        
        hb = plt.hexbin(spherical_distance * 1e3, temperature, bins='log', xscale='log', yscale='log')
        cb2 = plt.colorbar(hb)
        cb2.set_label(r'$\mathrm{Counts\;per\;hexbin}$', size=16)
        
        pdf.savefig(figure, bbox_inches='tight')  # Save the figure.
        plt.close()
    return None


def find_nearest(array, value):
    if len(value) == 1:
        idx = (np.abs(array - value)).argmin()
    else:
        idx = np.zeros(len(value))
        for i in range(len(value)):
            idx[i] = (np.abs(array - value[i])).argmin()
    
    return idx
