import os
import re
import glob
import calcGrid
import plot_tools
import numpy as np
import healpy as hlp
import astropy.units as u
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from const import *
from sfigure import *
from loadmodules import *
from matplotlib import gridspec
from scipy.special import gamma
from astropy_healpix import HEALPix
from scipy.optimize import curve_fit
from plot_tools import RotateCoordinates
from scripts.gigagalaxy.util import plot_helper
from parse_particledata import parse_particledata

default_level = 4
color_array = iter(cm.rainbow(np.linspace(0, 1, 9)))
photo_band = ['U', 'B', 'V', 'K', 'g', 'r', 'i', 'z']
colors = ['black', 'tab:red', 'tab:green', 'tab:blue']
Msunabs = [5.61, 5.48, 4.83, 3.30, 5.12, 4.68, 4.57, 4.60]
element = {'H':0, 'He':1, 'C':2, 'N':3, 'O':4, 'Ne':5, 'Mg':6, 'Si':7, 'Fe':8}


def circularity_distribution(pdf, data, redshift, read):
    """
    Plot the orbital circularity distribution for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param data: data from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :param read: boolean to read new data.
    :return: None
    """
    print("Invoking circularity_distribution")
    path = '/u/di43/Auriga/plots/data/' + 'cd/'

    # Read the data #
    if read is True:
        # Check if a folder to save the data exists, if not then create one #
        if not os.path.exists(path):
            os.makedirs(path)

        # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
        particle_type = [4]
        attributes = ['age', 'mass', 'pot', 'pos', 'type']
        data.select_haloes(default_level, redshift, loadonlyhalo=0, loadonlytype=particle_type, loadonly=attributes)

        # Loop over all available haloes #
        for s in data:
            # Check if halo's data already exists, if not then read it #
            names = glob.glob(path + '/name_*')
            names = [re.split('_|.npy', name)[1] for name in names]
            if str(s.haloname) in names:
                continue
            else:
                print("Analysing halo:", str(s.haloname))

            # Select the halo and rotate it based on its principal axes so galaxy's spin is aligned with the z-axis #
            s.calc_sf_indizes(s.subfind)
            s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)

            # Mask the data: select all stellar particles inside 0.1*R200c #
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

        # Plot the orbital circularity distribution #
        y_data, edges = np.histogram(epsilon, weights=stellar_masses / np.sum(stellar_masses), bins=100, range=[-1.7, 1.7])
        y_data /= edges[1:] - edges[:-1]
        axis.plot(0.5 * (edges[1:] + edges[:-1]), y_data, color=colors[0], label=r'$\mathrm{D/T = %.2f}$' % disc_fraction)

        # Create the legend, save and close the figure #
        axis.legend(loc='upper right', fontsize=16, frameon=False, numpoints=1)
        pdf.savefig(figure, bbox_inches='tight')
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
    stellar_mass = np.log10(mass_to_light * luminosity * 1e-10)  # In 1e10 Msun.
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
    stellar_mass = np.log10(mass_to_light * luminosity * 1e-10)  # In 1e10 Msun.
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
    stellar_mass = np.log10(mass_to_light * luminosity * 1e-10)  # In 1e10 Msun.
    stellar_mass = (stellar_mass - 0.230) / 0.922
    return 10 ** stellar_mass


def tully_fisher(pdf, data, redshift, read):
    """
    Plot the Tully-Fisher relation for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param data: data from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :param read: boolean to read new data.
    :return: None
    """
    print("Invoking tully_fisher")
    path = '/u/di43/Auriga/plots/data/' + 'tf/'

    # Read the data #
    if read is True:
        # Check if a folder to save the data exists, if not then create one #
        if not os.path.exists(path):
            os.makedirs(path)

        # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
        attributes = ['age', 'mass']
        data.select_haloes(default_level, redshift, loadonlyhalo=0, loadonly=attributes)

        # Loop over all available haloes #
        for s in data:
            # Check if halo's data already exists, if not then read it #
            names = glob.glob(path + '/name_*')
            names = [re.split('_|.npy', name)[1] for name in names]
            if str(s.haloname) in names:
                continue
            else:
                print("Analysing halo:", str(s.haloname))

            # Select the halo and do not rotate it #
            s.calc_sf_indizes(s.subfind)
            s.select_halo(s.subfind, rotate_disk=False)

            # Mask the data: select stellar particles inside 0.1*R200c #
            age = np.zeros(s.npartall)
            age[s.data['type'] == 4] = s.data['age']
            stellar_mask, = np.where((s.data['type'] == 4) & (age > 0.) & (s.r() < 0.1 * s.subfind.data['frc2'][0]))

            # Calculate the cumulative mass and circular velocity #
            mass, edges = np.histogram(s.r(), weights=s.data['mass'], bins=100, range=[0., 0.025])
            cumulative_mass = np.cumsum(mass)
            circular_velocity = np.sqrt(G * cumulative_mass * 1e10 * msol / (edges[1:] * 1e6 * parsec)) / 1e5

            # Calculate the cumulative stellar mass and total circular velocity #
            stellar_mass, edges = np.histogram(s.r()[stellar_mask], weights=s.data['mass'][stellar_mask], bins=100, range=[0., 0.025])
            cumulative_stellar_mass = np.cumsum(stellar_mass)
            total_circular_velocity = circular_velocity[np.abs(cumulative_stellar_mass - 0.8 * np.sum(s.data['mass'][stellar_mask])).argmin()]

            # Save data for each halo in numpy arrays #
            np.save(path + 'name_' + str(s.haloname), s.haloname)
            np.save(path + 'stellar_mass_' + str(s.haloname), np.sum(s.data['mass'][stellar_mask]))
            np.save(path + 'total_circular_velocity_' + str(s.haloname), total_circular_velocity)

    # Get the names and sort them #
    names = glob.glob(path + '/name_*')
    names.sort()

    # Loop over all available haloes #
    for i in range(len(names)):
        # Generate the figure and set its parameters #
        figure, axis = plt.subplots(1, figsize=(10, 7.5))
        plot_tools.set_axis(axis, xlim=[1e8, 1e12], ylim=[1.4, 2.6], xscale='log', xlabel=r'$\mathrm{M_{\bigstar}/M_{\odot}}$',
                            ylabel=r'$\mathrm{log_{10}(v_{circ}/(km\;s^{-1}))}$', aspect=None)
        figure.text(0.0, 0.9, r'$\mathrm{Au-%s}$''\n' r'$\mathrm{z=%s}$' % (str(re.split('_|.npy', names[i])[1]), str(redshift)), fontsize=16,
                    transform=axis.transAxes)

        # Load the data #
        stellar_mass = np.load(path + 'stellar_mass_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        total_circular_velocity = np.load(path + 'total_circular_velocity_' + str(re.split('_|.npy', names[i])[1]) + '.npy')

        plt.scatter(stellar_mass * 1e10, np.log10(total_circular_velocity), color=colors[0], s=200, zorder=5,
                    marker='*')  # Plot the Tully-Fisher relation.

        # Plot Pizagno et al. 2007 sample #
        table = "./data/pizagno.txt"
        rmag_p = np.genfromtxt(table, comments='#', usecols=3)
        vcirc_p = np.genfromtxt(table, comments='#', usecols=9)
        color_p = np.genfromtxt(table, comments='#', usecols=5)
        mass_p = pizagno_convert_color_to_mass(color_p, rmag_p)
        plt.scatter(1e10 * mass_p, np.log10(vcirc_p), color='lightgray', s=10, marker='^', label=r'$\mathrm{Pizagno+\;07}$')

        # Plot Verheijen 2001 sample #
        table = "./data/verheijen.txt"
        Bmag_v = np.genfromtxt(table, comments='#', usecols=1)
        Rmag_v = np.genfromtxt(table, comments='#', usecols=2)
        vcirc_v = np.genfromtxt(table, comments='#', usecols=7)
        color_v = Bmag_v - Rmag_v
        mass_v = verheijen_convert_color_to_mass(color_v, Bmag_v)
        plt.scatter(1e10 * mass_v, np.log10(vcirc_v), color='lightgray', s=10, marker='s', label=r'$\mathrm{Verheijen\;01}$')

        # Plot Courteau et al. 2007 sample #
        table = "./data/courteau.txt"
        loglum_c = np.genfromtxt(table, comments='#', usecols=6)
        vcirc_c = np.genfromtxt(table, comments='#', usecols=8)
        mass_c = courteau_convert_luminosity_to_mass(loglum_c)
        plt.scatter(1e10 * mass_c, vcirc_c, color='lightgray', s=10, marker='o', label=r'$\mathrm{Courteau+\;07}$')

        # Plot best fit from Dutton et al. 2011 #
        masses = np.arange(0.1, 50.0)
        plt.plot(1e10 * masses, np.log10(obs_tullyfisher_fit(masses)), color='darkgray', lw=0.8, ls='--', label=r'$\mathrm{Dutton+\;11}$')

        # Create the legend, save and close the figure #
        axis.legend(loc='lower right', fontsize=16, frameon=False, numpoints=1)
        pdf.savefig(figure, bbox_inches='tight')
        plt.close()
    return None


def guo_abundance_matching(mass):
    """
    # Abundance matching from equation 3 of Guo et al. 2010.
    :param mass: mass in 10^10 Msun
    :return: val
    """
    c = 0.129
    M_zero = 10 ** 1.4
    alpha = -0.926
    beta = 0.261
    gamma = -2.44

    val = c * mass * ((mass / M_zero) ** alpha + (mass / M_zero) ** beta) ** gamma
    return val


def stellar_vs_halo_mass(pdf, data, redshift, read):
    """
    Plot the abundance matching relation for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param data: data from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :param read: boolean to read new data.
    :return: None
    """
    print("Invoking stellar_vs_halo_mass")
    path = '/u/di43/Auriga/plots/data/' + 'svt/'

    # Read the data #
    if read is True:
        # Check if a folder to save the data exists, if not then create one #
        if not os.path.exists(path):
            os.makedirs(path)

        # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
        attributes = ['age', 'mass']
        data.select_haloes(default_level, redshift, loadonlyhalo=0, loadonly=attributes)

        # Loop over all available haloes #
        for s in data:
            # Check if halo's data already exists, if not then read it #
            names = glob.glob(path + '/name_*')
            names = [re.split('_|.npy', name)[1] for name in names]
            if str(s.haloname) in names:
                continue
            else:
                print("Analysing halo:", str(s.haloname))

            # Select the halo and do not rotate it #
            s.calc_sf_indizes(s.subfind)
            s.select_halo(s.subfind, rotate_disk=False)

            # Mask the data: select all stellar particles inside 0.1*R200c #
            age = np.zeros(s.npartall)
            age[s.data['type'] == 4] = s.data['age']
            stellar_mask, = np.where((s.data['type'] == 4) & (age > 0.) & (s.r() < 0.1 * s.subfind.data['frc2'][0]))
            halo_mask, = np.where((s.data['type'] == 1) & (s.r() < s.subfind.data['frc2'][0]))

            # Calculate the halo and stellar mass #
            stellar_mass = np.sum(s.data['mass'][stellar_mask])
            halo_mass = np.sum(s.data['mass'][halo_mask])

            # Save data for each halo in numpy arrays #
            np.save(path + 'name_' + str(s.haloname), s.haloname)
            np.save(path + 'halo_mass_' + str(s.haloname), halo_mass)
            np.save(path + 'stellar_mass_' + str(s.haloname), stellar_mass)

    # Get the names and sort them #
    names = glob.glob(path + '/name_*')
    names.sort()

    # Loop over all available haloes #
    for i in range(len(names)):
        # Generate the figure and set its parameters #
        figure, axis = plt.subplots(1, figsize=(10, 7.5))
        plot_tools.set_axis(axis, xlim=[1e11, 1e13], ylim=[1e9, 1e12], xscale='log', yscale='log', xlabel=r'$\mathrm{M_{halo}/M_{\odot}}$',
                            ylabel=r'$\mathrm{M_{\bigstar}/M_{\odot}}$', aspect=None)
        figure.text(0.0, 0.9, r'$\mathrm{Au-%s}$''\n' r'$\mathrm{z=%s}$' % (str(re.split('_|.npy', names[i])[1]), str(redshift)), fontsize=16,
                    transform=axis.transAxes)

        # Load the data #
        halo_mass = np.load(path + 'halo_mass_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        stellar_mass = np.load(path + 'stellar_mass_' + str(re.split('_|.npy', names[i])[1]) + '.npy')

        plt.scatter(halo_mass * 1e10, stellar_mass * 1e10, color=colors[0], s=200, zorder=5, marker='*')  # Plot the abundance matching relation.

        # Plot the cosmic baryon fraction relation #
        masses = np.arange(15., 300.)
        cosmic_baryon_frac = 0.048 / 0.307
        plt.plot(1e10 * masses, 1e10 * masses * cosmic_baryon_frac, color='k', ls='--', label=r'$\mathrm{M_{200}\;\Omega_b/\Omega_m}$')

        # Plot the Guo+10 relation #
        guo_high = guo_abundance_matching(masses) * 10 ** (+0.2)
        guo_low = guo_abundance_matching(masses) * 10 ** (-0.2)
        plt.fill_between(1e10 * masses, 1e10 * guo_low, 1e10 * guo_high, color='lightgray', edgecolor='None')
        plt.plot(1e10 * masses, 1e10 * guo_abundance_matching(masses), color='k', ls=':', label=r'$\mathrm{Guo+\;10}$')

        # Create the legend, save and close the figure #
        axis.legend(loc='upper right', fontsize=16, frameon=False)
        pdf.savefig(figure, bbox_inches='tight')
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


def gas_fraction_vs_magnitude(pdf, data, redshift, read):
    """
    Plot the gas fraction (gas to stellar plus gas mass ratio) as a function R-band magnitude for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param data: data from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :param read: boolean to read new data.
    :return: None
    """
    print("Invoking gas_fraction_vs_magnitude")
    path = '/u/di43/Auriga/plots/data/' + 'gfvm/'

    # Read the data #
    if read is True:
        # Check if a folder to save the data exists, if not then create one #
        if not os.path.exists(path):
            os.makedirs(path)

        # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
        particle_type = [0, 4]
        attributes = ['age', 'mass']
        data.select_haloes(default_level, redshift, loadonlyhalo=0, loadonlytype=particle_type, loadonly=attributes)

        # Loop over all available haloes #
        for s in data:
            # Check if halo's data already exists, if not then read it #
            names = glob.glob(path + '/name_*')
            names = [re.split('_|.npy', name)[1] for name in names]
            if str(s.haloname) in names:
                continue
            else:
                print("Analysing halo:", str(s.haloname))

            # Select the halo and do not rotate it #
            s.calc_sf_indizes(s.subfind)
            s.select_halo(s.subfind, rotate_disk=False)

            # Mask the data: stellar particles inside 0.1*R200c #
            age = np.zeros(s.npartall)
            age[s.data['type'] == 4] = s.data['age']
            stellar_mask, = np.where((s.data['type'] == 4) & (s.r() < 0.1 * s.subfind.data['frc2'][0]) & (age > 0.)) - s.nparticlesall[:4].sum()

            # Calculate the bolometric R-band magnitude in units of the bolometric magnitude of the Sun #
            Rband = convert_rband_to_Rband_mag(s.data['gsph'][stellar_mask, 5], s.data['gsph'][stellar_mask, 4])
            M_R = -2.5 * np.log10((10. ** (- 2.0 * Rband / 5.0)).sum())

            stellar_mask, = np.where((s.data['type'] == 4) & (s.r() < 0.1 * s.subfind.data['frc2'][0]) & (age > 0.))
            gas_mask, = np.where((s.data['type'] == 0) & (s.r() < 0.1 * s.subfind.data['frc2'][0]))
            gas_fraction = np.sum(s.data['mass'][gas_mask]) / (np.sum(s.data['mass'][gas_mask]) + np.sum(s.data['mass'][stellar_mask]))

            # Save data for each halo in numpy arrays #
            np.save(path + 'M_R_' + str(s.haloname), M_R)
            np.save(path + 'gas_fraction_' + str(s.haloname), gas_fraction)
            np.save(path + 'name_' + str(s.haloname), s.haloname)

    # Get the names and sort them #
    names = glob.glob(path + '/name_*')
    names.sort()

    # Loop over all available haloes #
    for i in range(len(names)):
        # Generate the figure and set its parameters #
        figure, axis = plt.subplots(1, figsize=(10, 7.5))
        plot_tools.set_axis(axis, xlim=[-23.2, -22], ylim=[0.1, 0.4], xlabel=r'$\mathrm{M_{R}/mag}$', ylabel=r'$\mathrm{f_{gas}}$', aspect=None)
        figure.text(0.0, 0.9, r'$\mathrm{Au-%s}$''\n' r'$\mathrm{z=%s}$' % (str(re.split('_|.npy', names[i])[1]), str(redshift)), fontsize=16,
                    transform=axis.transAxes)

        # Load the data #
        gas_fraction = np.load(path + 'gas_fraction_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        M_R = np.load(path + 'M_R_' + str(re.split('_|.npy', names[i])[1]) + '.npy')

        plt.scatter(M_R, gas_fraction, color=colors[0], s=200, zorder=5, marker='*')  # Plot the gas fraction as a function R-band magnitude.

        # Save and close the figure #
        pdf.savefig(figure, bbox_inches='tight')
        plt.close()
    return None


def bar_strength_profile(pdf, data, redshift, read):
    """
    Plot the bar strength radial profile from Fourier modes of surface density for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param data: data from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :param read: boolean to read new data.
    :return: None
    """
    print("Invoking bar_strength_profile")
    path = '/u/di43/Auriga/plots/data/' + 'bsp/'

    # Read the data #
    if read is True:
        # Check if a folder to save the data exists, if not then create one #
        if not os.path.exists(path):
            os.makedirs(path)

        # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
        particle_type = [4]
        attributes = ['age', 'mass', 'pos']
        data.select_haloes(default_level, redshift, loadonlyhalo=0, loadonlytype=particle_type, loadonly=attributes)

        # Loop over all available haloes #
        for s in data:
            # Check if halo's data already exists, if not then read it #
            names = glob.glob(path + '/name_*')
            names = [re.split('_|.npy', name)[1] for name in names]
            if str(s.haloname) in names:
                continue
            else:
                print("Analysing halo:", str(s.haloname))

            # Select the halo and rotate it based on its principal axes so galaxy's spin is aligned with the z-axis #
            s.calc_sf_indizes(s.subfind)
            s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)
            mask, = np.where(s.data['age'] > 0.)  # Mask the data: select stellar particles.

            z_rotated, y_rotated, x_rotated = plot_tools.rotate_bar(s.data['pos'][mask, 0] * 1e3, s.data['pos'][mask, 1] * 1e3,
                                                                    s.data['pos'][mask, 2] * 1e3)  # In Mpc.
            s.data['pos'] = np.vstack((z_rotated, y_rotated, x_rotated)).T  # Rebuild the s.data['pos'] attribute in kpc.
            x, y = s.data['pos'][:, 2] * 1e3, s.data['pos'][:, 1] * 1e3  # Load positions and convert from Mpc to Kpc.

            # Split up galaxy in radial bins and calculate the Fourier components #
            n_bins = 40  # Number of radial bins.
            r = np.sqrt(x[:] ** 2 + y[:] ** 2)  # Radius of each particle.

            # Initialise Fourier components #
            r_m, beta_2, alpha_0, alpha_2 = np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins)

            # Split up galaxy in radial bins and calculate Fourier components #
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
            ratio = np.divide(np.sqrt(alpha_2[:] ** 2 + beta_2[:] ** 2), alpha_0[:])

            # Save data for each halo in numpy arrays #
            np.save(path + 'r_m_' + str(s.haloname), r_m)
            np.save(path + 'ratio_' + str(s.haloname), ratio)
            np.save(path + 'name_' + str(s.haloname), s.haloname)

    # Get the names and sort them #
    names = glob.glob(path + '/name_*')
    names.sort()

    # Loop over all available haloes #
    for i in range(len(names)):
        # Generate the figure and set its parameters #
        figure, axis = plt.subplots(1, figsize=(10, 7.5))
        plot_tools.set_axis(axis, xlim=[0, 10], ylim=[-0.1, 1.1], xlabel=r'$\mathrm{R/kpc}$', ylabel=r'$\mathrm{\sqrt{a_{2}^{2}+b_{2}^{2}}/a_{0}}$',
                            aspect=None)
        figure.text(0.0, 0.9, r'$\mathrm{Au-%s}$''\n' r'$\mathrm{z=%s}$' % (str(re.split('_|.npy', names[i])[1]), str(redshift)), fontsize=16,
                    transform=axis.transAxes)

        # Load the data #
        ratio = np.load(path + 'ratio_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        r_m = np.load(path + 'r_m_' + str(re.split('_|.npy', names[i])[1]) + '.npy')

        # Plot the bar strength radial profile and get an estimate for the bar length from the maximum strength #
        A2 = max(ratio)
        plt.plot(r_m, ratio, color=colors[0], label=r'$\mathrm{A_{2}=%.2f}$' % A2)
        plt.plot([r_m[np.where(ratio == A2)], r_m[np.where(ratio == A2)]], [-0.0, A2], color=colors[0], linestyle='dashed',
                 label=r'$\mathrm{r_{A_{2}}=%.2fkpc}$' % r_m[np.where(ratio == A2)])

        # Create the legend, save and close the figure #
        axis.legend(loc='upper right', fontsize=16, frameon=False, numpoints=1)
        pdf.savefig(figure, bbox_inches='tight')
        plt.close()
    return None


def stellar_surface_density_profiles(pdf, data, redshift, read):
    """
    Plot the stellar surface density profiles for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param data: data from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :param read: boolean to read new data.
    :return: None
    """
    print("Invoking stellar_surface_density_profiles")
    path = '/u/di43/Auriga/plots/data/' + 'ssdp/'

    # Read the data #
    if read is True:
        # Check if a folder to save the data exists, if not then create one #
        if not os.path.exists(path):
            os.makedirs(path)

        # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
        particle_type = [4]
        attributes = ['pos', 'mass']
        data.select_haloes(default_level, redshift, loadonlyhalo=0, loadonlytype=particle_type, loadonly=attributes)

        # Loop over all available haloes #
        for s in data:
            # Check if halo's data already exists, if not then read it #
            names = glob.glob(path + '/name_*')
            names = [re.split('_|.npy', name)[1] for name in names]
            if str(s.haloname) in names:
                continue
            else:
                print("Analysing halo:", str(s.haloname))

            # Select the halo and rotate it based on its principal axes so galaxy's spin is aligned with the z-axis #
            s.calc_sf_indizes(s.subfind)
            s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)

            g = parse_particledata(s, s.subfind, attributes, radialcut=0.1 * s.subfind.data['frc2'][0])
            g.prep_data()
            sdata = g.sgdata['sdata']

            # Define the radial and vertical cuts and calculate the mass surface density #
            radial_cut = 0.1 * s.subfind.data['frc2'][0]  # Radial cut in Mpc.
            vertical_mask, = np.where((abs(sdata['pos'][:, 0]) < 0.005))  # Vertical cut in Mpc.
            r_xy = np.sqrt((sdata['pos'][:, 1:] ** 2).sum(axis=1))

            mass, edges = np.histogram(r_xy[vertical_mask], bins=50, range=(0., radial_cut), weights=sdata['mass'][vertical_mask])
            surface = np.zeros(len(edges) - 1)
            surface[:] = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
            surface_density = np.divide(mass, surface)

            # Get the bin centres and convert surface density to Msun pc^-2 and radius to pc #
            x = np.zeros(len(edges) - 1)
            x[:] = 0.5 * (edges[1:] + edges[:-1])
            surface_density *= 1e-6
            r = x * 1e3

            sdlim = 1.0
            indy = find_nearest(surface_density * 1e4, [sdlim]).astype('int64')
            rfit = x[indy] * 1e3
            sdfit = surface_density[:indy]
            r = r[:indy][sdfit > 0.0]
            sdfit = sdfit[sdfit > 0.0]
            p = plot_helper.plot_helper()  # Load the helper.

            # Fit a total (exponential plus Sersic) profile #
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

            # Save data for each halo in numpy arrays #
            np.save(path + 'r_' + str(s.haloname), r)
            np.save(path + 'rfit_' + str(s.haloname), rfit)
            np.save(path + 'sdfit_' + str(s.haloname), sdfit)
            np.save(path + 'popt0_' + str(s.haloname), popt[0])
            np.save(path + 'popt1_' + str(s.haloname), popt[1])
            np.save(path + 'popt2_' + str(s.haloname), popt[2])
            np.save(path + 'popt3_' + str(s.haloname), popt[3])
            np.save(path + 'popt4_' + str(s.haloname), popt[4])
            np.save(path + 'name_' + str(s.haloname), s.haloname)

    # Get the names and sort them #
    names = glob.glob(path + '/name_*')
    names.sort()

    # Loop over all available haloes #
    for i in range(len(names)):
        # Generate the figure and set its parameters #
        figure, axis = plt.subplots(1, figsize=(10, 7.5))
        plot_tools.set_axis(axis, xlim=[0, 30], ylim=[1e0, 1e6], yscale='log', xlabel=r'$\mathrm{R/kpc}$',
                            ylabel=r'$\mathrm{\Sigma_{\bigstar}/(M_{\odot}\;pc^{-2})}$', aspect=None)
        figure.text(0.0, 0.9, r'$\mathrm{Au-%s}$''\n' r'$\mathrm{z=%s}$' % (str(re.split('_|.npy', names[i])[1]), str(redshift)), fontsize=16,
                    transform=axis.transAxes)

        # Load the data #
        r = np.load(path + 'r_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        rfit = np.load(path + 'rfit_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        sdfit = np.load(path + 'sdfit_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        popt0 = np.load(path + 'popt0_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        popt1 = np.load(path + 'popt1_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        popt2 = np.load(path + 'popt2_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        popt3 = np.load(path + 'popt3_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        popt4 = np.load(path + 'popt4_' + str(re.split('_|.npy', names[i])[1]) + '.npy')

        # Plot the bar strength radial profile and get an estimate for the bar length from the maximum strength #
        p = plot_helper.plot_helper()  # Load the helper.
        plt.axvline(rfit, color='gray', linestyle='--')
        plt.scatter(r, 1e10 * sdfit * 1e-6, marker='o', s=15, color=colors[0], linewidth=0.0)
        plt.plot(r, 1e10 * p.exp_prof(r, popt0, popt1) * 1e-6, color=colors[3])
        plt.plot(r, 1e10 * p.sersic_prof1(r, popt2, popt3, popt4) * 1e-6, color=colors[1])
        plt.plot(r, 1e10 * p.total_profile(r, popt0, popt1, popt2, popt3, popt4) * 1e-6, color=colors[0])

        figure.text(0.8, 0.82, r'$\mathrm{n} = %.2f$' '\n' r'$\mathrm{R_{d}} = %.2f$' '\n' r'$\mathrm{R_{eff}} = %.2f$' '\n' % (
            1. / popt4, popt1, popt3 * p.sersic_b_param(1.0 / popt4) ** (1.0 / popt4)), fontsize=16, transform=axis.transAxes)

        # Save and close the figure #
        pdf.savefig(figure, bbox_inches='tight')
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


def circular_velocity_curves(pdf, data, redshift, read):
    """
    Plot the circular velocity curve for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param data: data from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :param read: boolean to read new data.
    :return: None
    """
    print("Invoking circular_velocity_curves")
    path = '/u/di43/Auriga/plots/data/' + 'cvc/'

    # Read the data #
    if read is True:
        # Check if a folder to save the data exists, if not then create one #
        if not os.path.exists(path):
            os.makedirs(path)

        # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
        particle_type = [0, 1, 4, 5]
        attributes = ['pos', 'vel', 'mass', 'age']
        data.select_haloes(default_level, redshift, loadonlyhalo=0, loadonlytype=particle_type, loadonly=attributes)

        # Loop over all available haloes #
        for s in data:
            # Check if halo's data already exists, if not then read it #
            names = glob.glob(path + '/name_*')
            names = [re.split('_|.npy', name)[1] for name in names]
            if str(s.haloname) in names:
                continue
            else:
                print("Analysing halo:", str(s.haloname))

            # Select the halo and rotate it based on its principal axes so galaxy's spin is aligned with the z-axis #
            s.calc_sf_indizes(s.subfind)
            s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)

            # Split up galaxy in radial bins #
            n_shells = 400
            radius = 0.04
            dr = radius / n_shells

            na = s.nparticlesall
            end = na.copy()
            for i in range(1, len(end)):
                end[i] += end[i - 1]

            start = np.zeros(len(na), dtype='int32')
            for i in range(1, len(start)):
                start[i] = end[i - 1]

            # Calculate the radial profiles #
            shell_mass = np.zeros((n_shells, 6))
            shell_velocity = np.zeros((n_shells, 6))
            for i in range(6):
                rp = calcGrid.calcRadialProfile(s.data['pos'][start[i]:end[i], :].astype('float64'),
                                                s.data['mass'][start[i]:end[i]].astype('float64'), 0, n_shells, dr, s.center[0], s.center[1],
                                                s.center[2])
                radius = rp[1, :]
                shell_mass[:, i] = rp[0, :]
                for j in range(1, n_shells):
                    shell_mass[j, i] += shell_mass[j - 1, i]
                shell_velocity[:, i] = np.sqrt(G * shell_mass[:, i] * 1e10 * msol / (radius * 1e6 * parsec)) / 1e5

            rp = calcGrid.calcRadialProfile(s.data['pos'].astype('float64'), s.data['mass'].astype('float64'), 0, n_shells, dr, s.center[0],
                                            s.center[1], s.center[2])
            radius = rp[1, :]
            total_mass = rp[0, :]

            # Calculate the total mass in all shells #
            for j in range(1, n_shells):
                total_mass[j] += total_mass[j - 1]

            # Save data for each halo in numpy arrays #
            np.save(path + 'radius_' + str(s.haloname), radius)
            np.save(path + 'name_' + str(s.haloname), s.haloname)
            np.save(path + 'total_mass_' + str(s.haloname), total_mass)
            np.save(path + 'shell_velocity_' + str(s.haloname), shell_velocity)

    # Get the names and sort them #
    names = glob.glob(path + '/name_*')
    names.sort()

    # Loop over all available haloes #
    for i in range(len(names)):
        # Generate the figure and set its parameters #
        figure, axis = plt.subplots(1, figsize=(10, 7.5))
        plot_tools.set_axis(axis, xlim=[0, 24], ylim=[0, 700], xlabel=r'$\mathrm{R/kpc}$', ylabel=r'$\mathrm{V_{circular}/(km\;s^{-1})}$',
                            aspect=None)
        figure.text(0.0, 0.9, r'$\mathrm{Au-%s}$''\n' r'$\mathrm{z=%s}$' % (str(re.split('_|.npy', names[i])[1]), str(redshift)), fontsize=16,
                    transform=axis.transAxes)

        # Load the data #
        radius = np.load(path + 'radius_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        total_mass = np.load(path + 'total_mass_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        shell_velocity = np.load(path + 'shell_velocity_' + str(re.split('_|.npy', names[i])[1]) + '.npy')

        # Plot the circular velocity curve #
        vtot = np.sqrt(G * total_mass * 1e10 * msol / (radius * 1e6 * parsec)) / 1e5
        plt.plot(radius * 1e3, vtot, color=colors[0], linewidth=4, label=r'$\mathrm{Total}$')
        plt.plot(radius * 1e3, shell_velocity[:, 0], color=colors[3], linestyle='--', linewidth=3, label=r'$\mathrm{Gas}$')
        plt.plot(radius * 1e3, shell_velocity[:, 4], color=colors[2], linestyle='--', linewidth=3, label=r'$\mathrm{Stars}$')
        plt.plot(radius * 1e3, shell_velocity[:, 1], color=colors[1], linestyle='--', linewidth=3, label=r'$\mathrm{Dark\;matter}$')

        # Create the legend, save and close the figure #
        axis.legend(loc='upper right', fontsize=16, frameon=False, numpoints=1)
        pdf.savefig(figure, bbox_inches='tight')
        plt.close()
    return None


def gas_temperature_vs_distance(pdf, data, redshift, read):
    """
    Plot the temperature as a function of distance of gas cells for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param data: data from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :param read: boolean to read new data.
    :return: None
    """
    print("Invoking gas_temperature_vs_distance")
    path = '/u/di43/Auriga/plots/data/' + 'gtd/'

    # Read the data #
    if read is True:
        # Check if a folder to save the data exists, if not then create one #
        if not os.path.exists(path):
            os.makedirs(path)

        # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
        particle_type = [0, 4]
        attributes = ['age', 'mass', 'ne', 'pos', 'rho', 'u', 'vol']
        data.select_haloes(default_level, redshift, loadonlyhalo=0, loadonlytype=particle_type, loadonly=attributes)

        # Loop over all available haloes #
        for s in data:
            # Check if halo's data already exists, if not then read it #
            names = glob.glob(path + '/name_*')
            names = [re.split('_|.npy', name)[1] for name in names]
            if str(s.haloname) in names:
                continue
            else:
                print("Analysing halo:", str(s.haloname))

            # Select the halo and rotate it based on its principal axes so galaxy's spin is aligned with the z-axis #
            s.calc_sf_indizes(s.subfind)
            s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)
            mask, = np.where(
                (s.data['type'] == 0) & (s.r() < s.subfind.data['frc2'][0]))  # Mask the data: select gas cells inside the virial radius R200c.

            # Calculate the temperature of the gas cells #
            ne = s.data['ne'][mask]
            metallicity = s.data['gz'][mask]
            XH = s.data['gmet'][mask, element['H']]
            yhelium = (1 - XH - metallicity) / (4. * XH)
            mu = (1 + 4 * yhelium) / (1 + yhelium + ne)
            temperature = GAMMA_MINUS1 * s.data['u'][mask] * 1e10 * mu * PROTONMASS / BOLTZMANN

            # Save data for each halo in numpy arrays #
            np.save(path + 'name_' + str(s.haloname), s.haloname)
            np.save(path + 'temperature_' + str(s.haloname), temperature)
            np.save(path + 'spherical_distance_' + str(s.haloname), s.r()[mask])

    # Load and plot the data #
    names = glob.glob(path + '/name_*')
    names.sort()

    # Loop over all available haloes #
    for i in range(len(names)):
        # Generate the figure and set its parameters #
        figure = plt.figure(figsize=(10, 7.5))
        gs = gridspec.GridSpec(1, 2, wspace=0.22, width_ratios=[1, 0.05])
        axis00, axis01 = plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1])

        plot_tools.set_axis(axis00, xlim=[1e-2, 2e2], ylim=[1e1, 1e8], xscale='log', yscale='log', xlabel=r'$\mathrm{R/kpc}$',
                            ylabel=r'$\mathrm{Temperature/K}$', aspect=None, which='major')
        figure.text(0.02, 0.92, r'$\mathrm{Au-%s}$''\n' r'$\mathrm{z=%s}$' % (str(re.split('_|.npy', names[i])[1]), str(redshift)), fontsize=16,
                    transform=axis00.transAxes)

        # Load the data #
        temperature = np.load(path + 'temperature_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        spherical_distance = np.load(path + 'spherical_distance_' + str(re.split('_|.npy', names[i])[1]) + '.npy')

        # Plot the temperature as a function of distance of gas cells #
        hb = axis00.hexbin(spherical_distance * 1e3, temperature, bins='log', xscale='log', yscale='log', cmap='gist_earth_r')
        plot_tools.create_colorbar(axis01, hb, label=r'$\mathrm{Counts\;per\;hexbin}$')

        # Save and close the figure #
        pdf.savefig(figure, bbox_inches='tight')
        plt.close()
    return None


def decomposition_IT20(date, data, redshift, read):
    """
    Plot the angular momentum maps and calculate D/T_IT20 for Auriga halo(es).
    :param date: date.
    :param data: data from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :param read: boolean to read new data.
    :return: None
    """
    print("Invoking decomposition_IT20")
    # Get the names and sort them #
    path = '/u/di43/Auriga/plots/data/' + 'di/' + str(redshift) + '/'

    # Read the data #
    if read is True:
        # Check if a folder to save the data exists, if not then create one #
        if not os.path.exists(path):
            os.makedirs(path)

        # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
        particle_type = [4]
        attributes = ['age', 'mass', 'pos', 'vel']
        data.select_haloes(default_level, redshift, loadonlyhalo=0, loadonlytype=particle_type, loadonly=attributes)

        # Loop over all available haloes #
        for s in data:
            # Check if halo's data already exists, if not then read it #
            names = glob.glob(path + '/name_*')
            names = [re.split('_|.npy', name)[1] for name in names]
            # if str(s.haloname) in names:
            #     continue
            # else:
            #     print("Analysing halo:", str(s.haloname))

            # Select the halo and rotate it based on its principal axes so galaxy's spin is aligned with the z-axis #
            s.calc_sf_indizes(s.subfind)
            s.select_halo(s.subfind)

            stellar_mask, = np.where((s.data['age'] > 0.0) & (s.r() * 1e3 < 30))  # Mask the data: select stellar particles inside a 30kpc sphere.

            # Calculate the angular momentum for each particle and for the galaxy and the unit vector parallel to the galactic angular momentum
            # vector #
            sp_mass = s.data['mass'][stellar_mask]
            prc_angular_momentum = s.data['mass'][stellar_mask, np.newaxis] * np.cross(s.data['pos'][stellar_mask] * 1e3,
                                                                                       s.data['vel'][stellar_mask])  # In Msun kpc km s^-1.

            glx_stellar_angular_momentum = np.sum(prc_angular_momentum, axis=0)
            glx_unit_vector = glx_stellar_angular_momentum / np.linalg.norm(glx_stellar_angular_momentum)

            # Rotate coordinates and velocities of stellar particles wrt galactic angular momentum #
            sp_am_unit_vector = RotateCoordinates.rotate_X(s.data, glx_unit_vector, stellar_mask)

            # Step (ii) in Section 2.2 Decomposition of IT20 #
            # Calculate the azimuth (alpha) and elevation (delta) angle of the angular momentum of all stellar particles #
            alpha = np.degrees(np.arctan2(sp_am_unit_vector[:, 1], sp_am_unit_vector[:, 2]))  # In degrees.
            delta = np.degrees(np.arcsin(sp_am_unit_vector[:, 0]))  # In degrees.

            # Step (ii) in Section 2.2 Decomposition of IT20 #
            # Generate the pixelisation of the angular momentum map #
            nside = 2 ** 4  # Define the resolution of the grid (number of divisions along the side of a base-resolution grid cell).
            hp = HEALPix(nside=nside)  # Initialise the HEALPix pixelisation class.
            indices = hp.lonlat_to_healpix(alpha * u.deg, delta * u.deg)  # Create a list of HEALPix indices from particles's alpha and delta.
            densities = np.bincount(indices, minlength=hp.npix)  # Count number of data points in each HEALPix grid cell.

            # Step (iii) in Section 2.2 Decomposition of IT20 #
            # Smooth the angular momentum map with a top-hat filter of angular radius 30 degrees #
            smoothed_densities = np.zeros(hp.npix)
            # Loop over all grid cells #
            for i in range(hp.npix):
                mask = hlp.query_disc(nside, hlp.pix2vec(nside, i), np.pi / 6.0)  # Do a 30 degree cone search around each grid cell.
                smoothed_densities[i] = np.mean(densities[mask])  # Average the densities of the ones inside and assign this value to the grid cell.

            # Step (iii) in Section 2.2 Decomposition of IT20 #
            # Find the location of the density maximum #
            index_densest = np.argmax(smoothed_densities)
            alpha_densest = (hp.healpix_to_lonlat([index_densest])[0].value + np.pi) % (2 * np.pi) - np.pi  # In radians.
            delta_densest = (hp.healpix_to_lonlat([index_densest])[1].value + np.pi / 2) % (2 * np.pi) - np.pi / 2  # In radians.

            # Step (iv) in Section 2.2 Decomposition of IT20 #
            # Calculate the angular separation of each stellar particle from the centre of the densest grid cell #
            Delta_theta = np.arccos(np.sin(delta_densest) * np.sin(np.radians(delta)) + np.cos(delta_densest) * np.cos(np.radians(delta)) * np.cos(
                alpha_densest - np.radians(alpha)))  # In radians.

            # Step (v) in Section 2.2 Decomposition of IT20 #
            # Calculate the disc mass fraction as the mass within 30 degrees from the densest grid cell #
            disc_mask_IT20, = np.where(Delta_theta < (np.pi / 6.0))
            spheroid_mask_IT20, = np.where(Delta_theta >= (np.pi / 6.0))
            disc_fraction_IT20 = np.sum(sp_mass[disc_mask_IT20]) / np.sum(sp_mass)

            # Step (vi) in Section 2.2 Decomposition of IT20 #
            # Normalise the disc fractions #
            chi = 0.5 * (1 - np.cos(np.pi / 6))
            disc_fraction_IT20 = np.divide(1, 1 - chi) * (disc_fraction_IT20 - chi)

            # Sample a 360x180 grid in sample_alpha and sample_delta #
            sample_alpha = np.linspace(-180.0, 180.0, num=360) * u.deg
            sample_delta = np.linspace(-90.0, 90.0, num=180) * u.deg
            alpha_grid, delta_grid = np.meshgrid(sample_alpha, sample_delta)

            # Find density at each coordinate position #
            coordinate_index = hp.lonlat_to_healpix(alpha_grid, delta_grid)
            density_map = densities[coordinate_index]

            # Save data for each halo in numpy arrays #
            np.save(path + 'name_' + str(s.haloname), s.haloname)
            np.save(path + 'density_map_' + str(s.haloname), density_map)
            np.save(path + 'disc_mask_IT20_' + str(s.haloname), disc_mask_IT20)
            np.save(path + 'spheroid_mask_IT20_' + str(s.haloname), spheroid_mask_IT20)
            np.save(path + 'disc_fraction_IT20_' + str(s.haloname), disc_fraction_IT20)

    # Load and plot the data #
    names = glob.glob(path + '/name_06.*')
    names.sort()

    # Loop over all available haloes #
    for i in range(len(names)):
        # Generate the figure and set its parameters #
        figure = plt.figure(figsize=(10, 7.5))
        gs = gridspec.GridSpec(1, 2, wspace=0.2, width_ratios=[1, 0.05])
        axis00, axis01 = plt.subplot(gs[0, 0], projection='mollweide'), plt.subplot(gs[0, 1])

        axis00.set_xlabel(r'$\mathrm{\alpha/\degree}$', size=16)
        axis00.set_ylabel(r'$\mathrm{\delta/\degree}$', size=16)
        axis00.set_yticklabels(['', '-60', '', '-30', '', '0', '', '30', '', '60', ''], size=16)
        axis00.set_xticklabels(['', '-120', '', '-60', '', '0', '', '60', '', '120', ''], size=16)
        figure.text(0.02, 1, r'$\mathrm{Au-%s}$''\n' r'$\mathrm{z=%s}$' % (str(re.split('_|.npy', names[i])[1]), str(redshift)), fontsize=16,
                    transform=axis00.transAxes)

        # Load the data #
        density_map = np.load(path + 'density_map_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        disc_fraction_IT20 = np.load(path + 'disc_fraction_IT20_' + str(re.split('_|.npy', names[i])[1]) + '.npy')

        # Plot the angular momentum maps and calculate D/T_IT20 #
        # Sample a 360x180 grid in sample_alpha and sample_delta #
        sample_alpha = np.linspace(-180.0, 180.0, num=360) * u.deg
        sample_delta = np.linspace(-90.0, 90.0, num=180) * u.deg

        # Display data on a 2D regular raster and create a pseudo-color plot #
        pcm = axis00.pcolormesh(np.radians(sample_alpha), np.radians(sample_delta), density_map, cmap='nipy_spectral_r')
        plot_tools.create_colorbar(axis01, pcm, label=r'$\mathrm{Particles\; per\; grid\; cell}$')

        figure.text(0.42, 1.1, r'$\mathrm{D/T=%.2f }$' % disc_fraction_IT20, fontsize=16, transform=axis00.transAxes)

        # Add save and close the figure #
        plt.savefig('/u/di43/Auriga/plots/' + 'di-' + date + '.png', bbox_inches='tight')
        plt.close()
    return None


def velocity_dispersion_profiles(pdf, data, redshift, read):
    """
    Plot the radial and vertical stellar and gas velocity dispersion profiles for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param data: data from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :param read: boolean to read new data.
    :return: None
    """
    print("Invoking velocity_dispersion_profiles")
    # Get the names and sort them #
    path = '/u/di43/Auriga/plots/data/' + 'vdp/' + str(redshift) + '/'

    # Read the data #
    if read is True:
        # Check if a folder to save the data exists, if not then create one #
        if not os.path.exists(path):
            os.makedirs(path)

        # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
        particle_type = [0, 4]
        attributes = ['age', 'mass', 'pos', 'vel']
        data.select_haloes(default_level, redshift, loadonlyhalo=0, loadonlytype=particle_type, loadonly=attributes)

        # Loop over all available haloes #
        for s in data:
            # Check if halo's data already exists, if not then read it #
            names = glob.glob(path + '/name_*')
            names = [re.split('_|.npy', name)[1] for name in names]
            # if str(s.haloname) in names:
            #     continue
            # else:
            #     print("Analysing halo:", str(s.haloname))

            # Select the halo and rotate it based on its principal axes so galaxy's spin is aligned with the z-axis #
            s.calc_sf_indizes(s.subfind)
            s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)

            # Calculate the z-direction and the distance on the xy plane #
            z = np.abs(s.data['pos'][:, 0]) * 1e3  # In kpc.
            rxy = np.sqrt((s.data['pos'][:, 1:] ** 2).sum(axis=1)) * 1e3  # In kpc.

            # Calculate the normalised (wrt CoM) radial velocities #
            # radial_mask, = np.where(s.r() < 0.1 * s.subfind.data['frc2'][0])
            # CoM_velocity = (s.data['vel'][radial_mask, :] * s.data['mass'][radial_mask][:, None]).sum(axis=0) / s.data['mass'][radial_mask].sum()
            # radial_velocities = np.divide(((s.data['vel'][:, 1:] - CoM_velocity[1:]) * s.data['pos'][:, 1:] * 1e3).sum(axis=1), rxy)
            radial_velocities = np.divide((s.data['pos'][:, 1] * s.data['vel'][:, 1] * 1e3 + s.data['pos'][:, 2] * s.data['vel'][:, 2] * 1e3), rxy)

            # Mask the data and calculate the mass distribution of stellar particles #
            age = np.zeros(s.npartall)
            age[s.data['type'] == 4] = s.data['age']
            stellar_mask, = np.where((s.data['type'] == 4) & (age > 0.) & (rxy < 30.0) & (
                z < 5))  # Mask the data: select all stellar particles inside a 30x30x1kpc region #
            stellar_mass, edges = np.histogram(rxy[stellar_mask], bins=60, range=[0, 30], weights=s.data['mass'][stellar_mask])
            stellar_centre = 0.5 * (edges[1:] + edges[:-1])

            # Calculate the distribution of the z and r components of the velocity #
            velocities_z, edges = np.histogram(rxy[stellar_mask], bins=60, range=[0, 30],
                                               weights=s.data['mass'][stellar_mask] * s.data['vel'][stellar_mask, 0])
            velocities_r, edges = np.histogram(rxy[stellar_mask], bins=60, range=[0, 30],
                                               weights=s.data['mass'][stellar_mask] * radial_velocities[stellar_mask])
            velocities_z /= stellar_mass
            velocities_r /= stellar_mass

            # Calculate the distribution of the z and r components of the velocity dispersion #
            bin_id = np.digitize(rxy[stellar_mask], edges) - 1
            sigma_z, edges = np.histogram(rxy[stellar_mask], bins=60, range=[0, 30],
                                          weights=s.data['mass'][stellar_mask] * (s.data['vel'][stellar_mask, 0] - velocities_z[bin_id]) ** 2)
            sigma_r, edges = np.histogram(rxy[stellar_mask], bins=60, range=[0, 30],
                                          weights=s.data['mass'][stellar_mask] * (radial_velocities[stellar_mask] - velocities_r[bin_id]) ** 2)
            stellar_sigma_z = np.sqrt(sigma_z / stellar_mass)
            stellar_sigma_r = np.sqrt(sigma_r / stellar_mass)

            # Calculate 3D stellar velocity dispersion fot the whole galaxy #
            CoM_velocity = np.sum(s.data['mass'][stellar_mask][:, None] * s.data['vel'][stellar_mask, :], axis=0) / np.sum(
                s.data['mass'][stellar_mask], axis=0)
            sigma = np.sqrt(np.sum(s.data['mass'][stellar_mask][:, None] * (s.data['vel'][stellar_mask, :] - CoM_velocity) ** 2.0) / np.sum(
                s.data['mass'][stellar_mask], axis=0))

            # Calculate the normalised (wrt CoM) radial velocities #
            radial_mask, = np.where(s.r() < 0.1 * s.subfind.data['frc2'][0])
            CoM_velocity = (s.data['vel'][radial_mask, :] * s.data['mass'][radial_mask][:, None]).sum(axis=0) / s.data['mass'][radial_mask].sum()
            radial_velocities = np.divide(((s.data['vel'][:, 1:] - CoM_velocity[1:]) * s.data['pos'][:, 1:] * 1e3).sum(axis=1), rxy)

            # Mask the data and calculate the mass distribution of gas cells #
            gas_mask, = np.where((s.data['type'] == 0) & (rxy < 30.0) & (z < 5))  # Mask the data: select all gas cells inside a 30x30x1kpc region #
            gas_mass, edges = np.histogram(rxy[gas_mask], bins=60, range=[0, 30], weights=s.data['mass'][gas_mask])
            gas_centre = 0.5 * (edges[1:] + edges[:-1])

            # Calculate the distribution of the z and r components of the velocity #
            velocities_z, edges = np.histogram(rxy[gas_mask], bins=60, range=[0, 30], weights=s.data['mass'][gas_mask] * s.data['vel'][gas_mask, 0])
            velocities_r, edges = np.histogram(rxy[gas_mask], bins=60, range=[0, 30], weights=s.data['mass'][gas_mask] * radial_velocities[gas_mask])
            velocities_z /= gas_mass
            velocities_r /= gas_mass

            # Calculate the distribution of the z and r components of the velocity dispersion #
            bin_id = np.digitize(rxy[gas_mask], edges) - 1
            sigma_z, edges = np.histogram(rxy[gas_mask], bins=60, range=[0, 30],
                                          weights=s.data['mass'][gas_mask] * (s.data['vel'][gas_mask, 0] - velocities_z[bin_id]) ** 2)
            sigma_r, edges = np.histogram(rxy[gas_mask], bins=60, range=[0, 30],
                                          weights=s.data['mass'][gas_mask] * (radial_velocities[gas_mask] - velocities_r[bin_id]) ** 2)
            gas_sigma_z = np.sqrt(sigma_z / gas_mass)
            gas_sigma_r = np.sqrt(sigma_r / gas_mass)

            # Save data for each halo in numpy arrays #
            np.save(path + 'sigma_' + str(s.haloname), sigma)
            np.save(path + 'name_' + str(s.haloname), s.haloname)
            np.save(path + 'gas_centre_' + str(s.haloname), gas_centre)
            np.save(path + 'gas_sigma_r_' + str(s.haloname), gas_sigma_r)
            np.save(path + 'gas_sigma_z_' + str(s.haloname), gas_sigma_z)
            np.save(path + 'stellar_centre_' + str(s.haloname), stellar_centre)
            np.save(path + 'stellar_sigma_r_' + str(s.haloname), stellar_sigma_r)
            np.save(path + 'stellar_sigma_z_' + str(s.haloname), stellar_sigma_z)

    # Load and plot the data #
    names = glob.glob(path + '/name_*')
    names.sort()

    # Loop over all available haloes #
    for i in range(len(names)):
        # Generate the figure and set its parameters #
        figure, axis = plt.subplots(1, figsize=(10, 7.5))

        plot_tools.set_axis(axis, xlim=[0, 30], ylim=[0, 160], xlabel=r'$\mathrm{R/kpc}$', ylabel=r'$\mathrm{\sigma/(km\;s^{-1})}$', aspect=None,
                            which='major')
        figure.text(0.02, 1, r'$\mathrm{Au-%s}$''\n' r'$\mathrm{z=%s}$' % (str(re.split('_|.npy', names[i])[1]), str(redshift)), fontsize=16,
                    transform=axis.transAxes)

        # Load the data #
        gas_centre = np.load(path + 'gas_centre_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        gas_sigma_r = np.load(path + 'gas_sigma_r_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        gas_sigma_z = np.load(path + 'gas_sigma_z_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        stellar_centre = np.load(path + 'stellar_centre_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        stellar_sigma_z = np.load(path + 'stellar_sigma_z_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        stellar_sigma_r = np.load(path + 'stellar_sigma_r_' + str(re.split('_|.npy', names[i])[1]) + '.npy')

        # Plot the radial and vertical stellar and gas velocity dispersion profiles #
        plt.plot(gas_centre, gas_sigma_r, 'r', label=r'$\mathrm{\sigma_{gas,r}}$')
        plt.plot(gas_centre, gas_sigma_z, 'k', label=r'$\mathrm{\sigma_{gas,z}}$')
        plt.plot(stellar_centre, stellar_sigma_r, 'g', label=r'$\mathrm{\sigma_{\bigstar,r}}$')
        plt.plot(stellar_centre, stellar_sigma_z, 'b', label=r'$\mathrm{\sigma_{\bigstar,z}}$')

        # Create the legend, save and close the figure #
        axis.legend(loc='upper right', fontsize=16, frameon=False)
        pdf.savefig(figure, bbox_inches='tight')
        plt.close()
    return None
