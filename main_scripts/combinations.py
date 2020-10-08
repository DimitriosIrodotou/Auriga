import os
import re
import glob
import galaxy
import matplotlib
import plot_tools

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

from const import *
from sfigure import *
from loadmodules import *
from scripts.gigagalaxy.util import plot_helper

res = 512
level = 4
boxsize = 0.06
default_level = 4
colors = ['black', 'tab:red', 'tab:green', 'tab:blue']
colors2 = ['black', 'tab:red', 'tab:green', 'black', 'tab:red', 'tab:green', 'black', 'tab:red', 'tab:green']
marker_array = iter(['o', 'o', 'o', '^', '^', '^', 's', 's', 's'])
element = {'H':0, 'He':1, 'C':2, 'N':3, 'O':4, 'Ne':5, 'Mg':6, 'Si':7, 'Fe':8}


def stellar_light_combination(pdf, redshift):
    """
    Plot a combination of the stellar light projections for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :return: None
    """
    print("Invoking stellar_light_combination")
    # Get the names and sort them #
    path = '/u/di43/Auriga/plots/data/''sl/' + str(redshift)
    names = glob.glob(path + '/name_*')
    names.sort()

    # Generate the figure and set its parameters #
    figure = plt.figure(figsize=(10, 20))
    axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22, axis30, axis31, axis32, axis40, axis41, axis42, axis50, axis51, \
    axis52 = plot_tools.create_axes_combinations(
        res=res, boxsize=boxsize * 1e3, multiple2=True)
    for axis in [axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22, axis30, axis31, axis32, axis40, axis41, axis42, axis50,
                 axis51, axis52]:
        axis.set_yticklabels([])
        axis.set_xticklabels([])
    plt.rcParams['savefig.facecolor'] = 'black'

    # Loop over all available haloes #
    axes_face_on = [axis00, axis20, axis40, axis01, axis21, axis41, axis02, axis22, axis42]
    axes_edge_on = [axis10, axis30, axis50, axis11, axis31, axis51, axis12, axis32, axis52]
    for i, axis_face_on, axis_edge_on in zip(range(len(names)), axes_face_on, axes_edge_on):
        # Load the data #
        face_on = np.load(path + '/face_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        edge_on = np.load(path + '/edge_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')

        # Plot the stellar light projections #
        axis_face_on.imshow(face_on, interpolation='nearest', aspect='equal')
        axis_edge_on.imshow(edge_on, interpolation='nearest', aspect='equal')

        figure.text(0.01, 0.92, r'$\mathrm{Au-%s}$' % str(re.split('_|.npy', names[i])[1]), color='w', fontsize=16, transform=axis_face_on.transAxes)

    # Save and close the figure #
    pdf.savefig(figure, bbox_inches='tight')
    plt.close()
    return None


def stellar_light_components_combination(pdf, redshift):
    """
    Plot a combination of the stellar light projections of the disc and spheroid component for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :return: None
    """
    print("Invoking stellar_light_components_combination")
    # Get the names and sort them #
    path = '/u/di43/Auriga/plots/data/''sl/' + str(redshift) + '/'
    path_components = '/u/di43/Auriga/plots/data/' + 'slc/' + str(redshift) + '/'
    names = glob.glob(path + '/name_06*')
    names.sort()

    # Generate the figure and set its parameters #
    figure = plt.figure(figsize=(10, 20))
    axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22, axis30, axis31, axis32, axis40, axis41, axis42, axis50, axis51, \
    axis52 = plot_tools.create_axes_combinations(
        res=res, boxsize=boxsize * 1e3, multiple8=True)
    for axis in [axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22, axis30, axis31, axis32, axis40, axis41, axis42, axis50, \
                 axis51, axis52]:
        axis.set_yticklabels([])
        axis.set_xticklabels([])
    plt.rcParams['savefig.facecolor'] = 'black'

    # Loop over all available haloes #
    axes_face_on = [axis00, axis20, axis40]
    axes_edge_on = [axis10, axis30, axis50]
    axes_disc_face_on = [axis01, axis21, axis41]
    axes_disc_edge_on = [axis11, axis31, axis51]
    axes_spheroid_face_on = [axis02, axis22, axis42]
    axes_spheroid_edge_on = [axis12, axis32, axis52]
    for i, axis_face_on, axis_edge_on, axis_disc_face_on, axis_disc_edge_on, axis_spheroid_face_on, axis_spheroid_edge_on in zip(range(len(names)),
                                                                                                                                 axes_face_on,
                                                                                                                                 axes_edge_on,
                                                                                                                                 axes_disc_face_on,
                                                                                                                                 axes_disc_edge_on,
                                                                                                                                 axes_spheroid_face_on,
                                                                                                                                 axes_spheroid_edge_on):
        # Load the data #
        face_on = np.load(path + 'face_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        edge_on = np.load(path + 'edge_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        disc_face_on = np.load(path_components + 'disc_face_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        disc_edge_on = np.load(path_components + 'disc_edge_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        spheroid_face_on = np.load(path_components + 'spheroid_face_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        spheroid_edge_on = np.load(path_components + 'spheroid_edge_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')

        # Plot the stellar light projections of the disc and spheroid component #
        axis_face_on.imshow(face_on, interpolation='nearest', aspect='equal')
        axis_edge_on.imshow(edge_on, interpolation='nearest', aspect='equal')
        axis_disc_face_on.imshow(disc_face_on, interpolation='nearest', aspect='equal')
        axis_disc_edge_on.imshow(disc_edge_on, interpolation='nearest', aspect='equal')
        axis_spheroid_face_on.imshow(spheroid_face_on, interpolation='nearest', aspect='equal')
        axis_spheroid_edge_on.imshow(spheroid_edge_on, interpolation='nearest', aspect='equal')

        figure.text(0.01, 0.92, r'$\mathrm{Au-%s}$' % str(re.split('_|.npy', names[i])[1]), color='w', fontsize=16, transform=axis_face_on.transAxes)
        figure.text(0.01, 0.92, r'$\mathrm{Disc}$', fontsize=16, color='w', transform=axis_disc_face_on.transAxes)
        figure.text(0.01, 0.92, r'$\mathrm{Spheroid}$', fontsize=16, color='w', transform=axis_spheroid_face_on.transAxes)
    # Save and close the figure #
    pdf.savefig(figure, bbox_inches='tight')
    plt.close()
    return None


def stellar_density_combination(pdf, redshift):
    """
    Plot a combination of the stellar density projections for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :return: None
    """
    print("Invoking stellar_density_combination")
    # Get the names and sort them #
    path = '/u/di43/Auriga/plots/data/' + 'sd/' + str(redshift)
    names = glob.glob(path + '/name_*')
    names.sort()

    # Generate the figure and set its parameters #
    figure = plt.figure(figsize=(10, 15))
    axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22, axis30, axis31, axis32, axis40, axis41, axis42, axis50, axis51, axis52,\
    axiscbar, x, y, y2, area = plot_tools.create_axes_combinations(
        res=res, boxsize=boxsize * 1e3, multiple3=True)
    cmap = matplotlib.cm.get_cmap('twilight')
    for axis in [axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22, axis30, axis31, axis32, axis40, axis41, axis42, axis50,
                 axis51, axis52]:
        axis.set_facecolor(cmap(0))
        plot_tools.set_axis(axis, xlim=[-30, 30], ylim=[-30, 30])
    for axis in [axis10, axis11, axis12, axis30, axis31, axis32, axis50, axis51, axis52]:
        axis.set_ylim([-15, 15])
    for axis in [axis01, axis02, axis11, axis12, axis21, axis22, axis31, axis32, axis41, axis42]:
        axis.set_xticklabels([])
        axis.set_yticklabels([])
    for axis in [axis00, axis10, axis20, axis30, axis40]:
        axis.set_xticklabels([])
    for axis in [axis51, axis52]:
        axis.set_yticklabels([])
    for axis in [axis50, axis51, axis52]:
        labels = ['', '-20', '', '0', '', '20', '']
        axis.set_xticklabels(labels)
        axis.set_xlabel(r'$\mathrm{x/kpc}$', size=16)
    for axis in [axis00, axis20, axis40]:
        axis.set_ylabel(r'$\mathrm{y/kpc}$', size=16)
    for axis in [axis10, axis30, axis50]:
        axis.set_ylabel(r'$\mathrm{z/kpc}$', size=16)

    # Loop over all available haloes #
    axes_face_on = [axis00, axis20, axis40, axis01, axis21, axis41, axis02, axis22, axis42]
    axes_edge_on = [axis10, axis30, axis50, axis11, axis31, axis51, axis12, axis32, axis52]
    for i, axis_face_on, axis_edge_on in zip(range(len(names)), axes_face_on, axes_edge_on):
        # Load the data #
        face_on = np.load(path + '/face_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        edge_on = np.load(path + '/edge_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')

        # Plot the stellar density projections #
        pcm = axis_face_on.pcolormesh(x, y, face_on, norm=matplotlib.colors.LogNorm(vmin=1e6, vmax=1e10), rasterized=True, cmap=cmap)
        axis_edge_on.pcolormesh(x, y2, edge_on, norm=matplotlib.colors.LogNorm(vmin=1e6, vmax=1e10), rasterized=True, cmap=cmap)
        plot_tools.create_colorbar(axiscbar, pcm, label=r'$\mathrm{\Sigma_{\bigstar}/(M_\odot\;kpc^{-2})}$')

        figure.text(0.01, 0.92, r'$\mathrm{Au-%s}$' % str(re.split('_|.npy', names[i])[1]), fontsize=16, transform=axis_face_on.transAxes)

    # Save and close the figure #
    pdf.savefig(figure, bbox_inches='tight')
    plt.close()
    return None


def gas_density_combination(pdf, redshift):
    """
    Plot a combination of the gas density projections for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :return: None
    """
    print("Invoking gas_density_combination")
    # Get the names and sort them #
    path = '/u/di43/Auriga/plots/data/' + 'gd/' + str(redshift)
    names = glob.glob(path + '/name_*')
    names.sort()

    # Generate the figure and set its parameters #
    figure = plt.figure(figsize=(10, 15))
    axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22, axis30, axis31, axis32, axis40, axis41, axis42, axis50, axis51, axis52,\
    axiscbar, x, y, y2, area = plot_tools.create_axes_combinations(
        res=res, boxsize=boxsize * 1e3, multiple3=True)
    for axis in [axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22, axis30, axis31, axis32, axis40, axis41, axis42, axis50,
                 axis51, axis52]:
        plot_tools.set_axis(axis, xlim=[-30, 30], ylim=[-30, 30], size=12)
    for axis in [axis10, axis11, axis12, axis30, axis31, axis32, axis50, axis51, axis52]:
        axis.set_ylim([-15, 15])
    for axis in [axis01, axis02, axis11, axis12, axis21, axis22, axis31, axis32, axis41, axis42]:
        axis.set_xticklabels([])
        axis.set_yticklabels([])
    for axis in [axis00, axis10, axis20, axis30, axis40]:
        axis.set_xticklabels([])
    for axis in [axis51, axis52]:
        axis.set_yticklabels([])
    for axis in [axis50, axis51, axis52]:
        axis.set_xlabel(r'$\mathrm{x/kpc}$', size=12)
    for axis in [axis00, axis20, axis40]:
        axis.set_ylabel(r'$\mathrm{y/kpc}$', size=12)
    for axis in [axis10, axis30, axis50]:
        axis.set_ylabel(r'$\mathrm{z/kpc}$', size=12)

    # Loop over all available haloes #
    axes_face_on = [axis00, axis20, axis40, axis01, axis21, axis41, axis02, axis22, axis42]
    axes_edge_on = [axis10, axis30, axis50, axis11, axis31, axis51, axis12, axis32, axis52]
    for i, axis_face_on, axis_edge_on in zip(range(len(names)), axes_face_on, axes_edge_on):
        # Load the data #
        face_on = np.load(path + '/face_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        edge_on = np.load(path + '/edge_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')

        # Plot the gas density projections #
        pcm = axis_face_on.pcolormesh(x, y, face_on.T, norm=matplotlib.colors.LogNorm(vmin=1e6, vmax=1e10), rasterized=True, cmap='magma')
        axis_edge_on.pcolormesh(x, 0.5 * y, edge_on.T, norm=matplotlib.colors.LogNorm(vmin=1e6, vmax=1e10), rasterized=True, cmap='magma')
        plot_tools.create_colorbar(axiscbar, pcm, label='$\mathrm{\Sigma_{gas}/(M_\odot\;kpc^{-2})}$')

        figure.text(0.01, 0.92, r'$\mathrm{Au-%s}$' % str(re.split('_|.npy', names[i])[1]), fontsize=16, transform=axis_face_on.transAxes)

    # Save and close the figure #
    pdf.savefig(figure, bbox_inches='tight')
    plt.close()
    return None


def gas_temperature_combination(pdf, redshift):
    """
    Plot a combination of the gas temperature projections for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :return: None
    """
    print("Invoking gas_temperature_combination")
    # Get the names and sort them #
    path = '/u/di43/Auriga/plots/data/' + 'gt/' + str(redshift)
    names = glob.glob(path + '/name_*')
    names.sort()

    # Generate the figure and set its parameters #
    figure = plt.figure(figsize=(10, 15))
    axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22, axis30, axis31, axis32, axis40, axis41, axis42, axis50, axis51, axis52,\
    axiscbar, x, y, y2, area = plot_tools.create_axes_combinations(
        res=res, boxsize=boxsize * 1e3, multiple3=True)
    for axis in [axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22, axis30, axis31, axis32, axis40, axis41, axis42, axis50,
                 axis51, axis52]:
        plot_tools.set_axis(axis, xlim=[-30, 30], ylim=[-30, 30], size=12)
    for axis in [axis10, axis11, axis12, axis30, axis31, axis32, axis50, axis51, axis52]:
        axis.set_ylim([-15, 15])
    for axis in [axis01, axis02, axis11, axis12, axis21, axis22, axis31, axis32, axis41, axis42]:
        axis.set_xticklabels([])
        axis.set_yticklabels([])
    for axis in [axis00, axis10, axis20, axis30, axis40]:
        axis.set_xticklabels([])
    for axis in [axis51, axis52]:
        axis.set_yticklabels([])
    for axis in [axis50, axis51, axis52]:
        axis.set_xlabel(r'$\mathrm{x/kpc}$', size=12)
    for axis in [axis00, axis20, axis40]:
        axis.set_ylabel(r'$\mathrm{y/kpc}$', size=12)
    for axis in [axis10, axis30, axis50]:
        axis.set_ylabel(r'$\mathrm{z/kpc}$', size=12)

    # Loop over all available haloes #
    axes_face_on = [axis00, axis20, axis40, axis01, axis21, axis41, axis02, axis22, axis42]
    axes_edge_on = [axis10, axis30, axis50, axis11, axis31, axis51, axis12, axis32, axis52]
    for i, axis_face_on, axis_edge_on in zip(range(len(names)), axes_face_on, axes_edge_on):
        # Load the data #
        face_on = np.load(path + '/face_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        edge_on = np.load(path + '/edge_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        face_on_rho = np.load(path + '/face_on_rho_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        edge_on_rho = np.load(path + '/edge_on_rho_' + str(re.split('_|.npy', names[i])[1]) + '.npy')

        # Plot the density-weighted gas temperature projections #
        pcm = axis_face_on.pcolormesh(x, y, (face_on / face_on_rho).T, norm=matplotlib.colors.LogNorm(vmin=1e3, vmax=1e7), rasterized=True,
                                      cmap='viridis')
        axis_edge_on.pcolormesh(x, 0.5 * y, (edge_on / edge_on_rho).T, norm=matplotlib.colors.LogNorm(vmin=1e3, vmax=1e7), rasterized=True,
                                cmap='viridis')
        plot_tools.create_colorbar(axiscbar, pcm, label=r'$\mathrm{T/K}$')

        figure.text(0.01, 0.92, r'$\mathrm{Au-%s}$' % str(re.split('_|.npy', names[i])[1]), fontsize=16, transform=axis_face_on.transAxes)

    # Save and close the figure #
    pdf.savefig(figure, bbox_inches='tight')
    plt.close()
    return None


def gas_metallicity_combination(pdf, redshift):
    """
    Plot a combination of the gas metallicity projections for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :return: None
    """
    print("Invoking gas_metallicity_combination")
    # Get the names and sort them #
    path = '/u/di43/Auriga/plots/data/' + 'gm/' + str(redshift)
    names = glob.glob(path + '/name_*')
    names.sort()

    # Generate the figure and set its parameters #
    figure = plt.figure(figsize=(10, 15))
    axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22, axis30, axis31, axis32, axis40, axis41, axis42, axis50, axis51, axis52,\
    axiscbar, x, y, y2, area = plot_tools.create_axes_combinations(
        res=res, boxsize=boxsize * 1e3, multiple3=True)
    for axis in [axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22, axis30, axis31, axis32, axis40, axis41, axis42, axis50,
                 axis51, axis52]:
        plot_tools.set_axis(axis, xlim=[-30, 30], ylim=[-30, 30], size=12)
    for axis in [axis10, axis11, axis12, axis30, axis31, axis32, axis50, axis51, axis52]:
        axis.set_ylim([-15, 15])
    for axis in [axis01, axis02, axis11, axis12, axis21, axis22, axis31, axis32, axis41, axis42]:
        axis.set_xticklabels([])
        axis.set_yticklabels([])
    for axis in [axis00, axis10, axis20, axis30, axis40]:
        axis.set_xticklabels([])
    for axis in [axis51, axis52]:
        axis.set_yticklabels([])
    for axis in [axis50, axis51, axis52]:
        axis.set_xlabel(r'$\mathrm{x/kpc}$', size=12)
    for axis in [axis00, axis20, axis40]:
        axis.set_ylabel(r'$\mathrm{y/kpc}$', size=12)
    for axis in [axis10, axis30, axis50]:
        axis.set_ylabel(r'$\mathrm{z/kpc}$', size=12)

    # Loop over all available haloes #
    axes_face_on = [axis00, axis20, axis40, axis01, axis21, axis41, axis02, axis22, axis42]
    axes_edge_on = [axis10, axis30, axis50, axis11, axis31, axis51, axis12, axis32, axis52]
    for i, axis_face_on, axis_edge_on in zip(range(len(names)), axes_face_on, axes_edge_on):
        # Load the data #
        face_on = np.load(path + '/face_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        edge_on = np.load(path + '/edge_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')

        # Plot the gas metallicity projections #
        pcm = axis_face_on.pcolormesh(x, y, face_on.T, norm=matplotlib.colors.LogNorm(vmin=0.3, vmax=3.), rasterized=True, cmap='viridis')
        axis_edge_on.pcolormesh(x, 0.5 * y, edge_on.T, norm=matplotlib.colors.LogNorm(vmin=0.3, vmax=3.), rasterized=True, cmap='viridis')
        plot_tools.create_colorbar(axiscbar, pcm, label=r'$\mathrm{Z/Z_\odot}$')

        figure.text(0.01, 0.92, r'$\mathrm{Au-%s}$' % str(re.split('_|.npy', names[i])[1]), fontsize=16, transform=axis_face_on.transAxes)

    # Save and close the figure #
    pdf.savefig(figure, bbox_inches='tight')
    plt.close()
    return None


def magnetic_field_combination(pdf, redshift):
    """
    Plot a combination of the magnetic field strength projection for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :return: None
    """
    print("Invoking magnetic_field_combination")
    # Get the names and sort them #
    path = '/u/di43/Auriga/plots/data/' + 'mf/' + str(redshift)
    names = glob.glob(path + '/name_*')
    names.sort()

    # Generate the figure and set its parameters #
    figure = plt.figure(figsize=(10, 15))
    axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22, axis30, axis31, axis32, axis40, axis41, axis42, axis50, axis51, axis52,\
    axiscbar, x, y, y2, area = plot_tools.create_axes_combinations(
        res=res, boxsize=boxsize * 1e3, multiple3=True)
    for axis in [axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22, axis30, axis31, axis32, axis40, axis41, axis42, axis50,
                 axis51, axis52]:
        plot_tools.set_axis(axis, xlim=[-30, 30], ylim=[-30, 30], size=12)
    for axis in [axis10, axis11, axis12, axis30, axis31, axis32, axis50, axis51, axis52]:
        axis.set_ylim([-15, 15])
    for axis in [axis01, axis02, axis11, axis12, axis21, axis22, axis31, axis32, axis41, axis42]:
        axis.set_xticklabels([])
        axis.set_yticklabels([])
    for axis in [axis00, axis10, axis20, axis30, axis40]:
        axis.set_xticklabels([])
    for axis in [axis51, axis52]:
        axis.set_yticklabels([])
    for axis in [axis50, axis51, axis52]:
        axis.set_xlabel(r'$\mathrm{x/kpc}$', size=12)
    for axis in [axis00, axis20, axis40]:
        axis.set_ylabel(r'$\mathrm{y/kpc}$', size=12)
    for axis in [axis10, axis30, axis50]:
        axis.set_ylabel(r'$\mathrm{z/kpc}$', size=12)

    # Loop over all available haloes #
    axes_face_on = [axis00, axis20, axis40, axis01, axis21, axis41, axis02, axis22, axis42]
    axes_edge_on = [axis10, axis30, axis50, axis11, axis31, axis51, axis12, axis32, axis52]
    for i, axis_face_on, axis_edge_on in zip(range(len(names)), axes_face_on, axes_edge_on):
        # Load the data #
        face_on = np.load(path + '/face_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        edge_on = np.load(path + '/edge_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')

        # Plot the magnetic field projections #
        pcm = axis_face_on.pcolormesh(x, y, face_on.T, norm=matplotlib.colors.LogNorm(vmin=1e-1, vmax=1e2), rasterized=True, cmap='CMRmap')
        axis_edge_on.pcolormesh(x, 0.5 * y, edge_on.T, norm=matplotlib.colors.LogNorm(vmin=1e-1, vmax=1e2), rasterized=True, cmap='CMRmap')
        plot_tools.create_colorbar(axiscbar, pcm, label='$\mathrm{B/\mu G}$')

        figure.text(0.01, 0.92, r'$\mathrm{Au-%s}$' % str(re.split('_|.npy', names[i])[1]), fontsize=16, transform=axis_face_on.transAxes)

    # Save and close the figure #
    pdf.savefig(figure, bbox_inches='tight')
    plt.close()
    return None


def circularity_distribution_combination(pdf):
    """
    Plot a combination of the orbital circularity distribution for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :return: None
    """
    print("Invoking circularity_distribution_combination")
    # Get the names and sort them #
    path = '/u/di43/Auriga/plots/data/' + 'cd/'
    names = glob.glob(path + '/name_*')
    names.sort()

    # Generate the figure and set its parameters #
    figure = plt.figure(figsize=(20, 7.5))
    axis00, axis01, axis02 = plot_tools.create_axes_combinations(res=res, boxsize=boxsize * 1e3, multiple4=True)
    plot_tools.set_axis(axis00, xlim=[-1.7, 1.7], ylim=[0, 5], xlabel=r'$\mathrm{\epsilon}$', ylabel=r'$\mathrm{f(\epsilon)}$', aspect=None)
    plot_tools.set_axis(axis01, xlim=[-1.7, 1.7], ylim=[0, 5], xlabel=r'$\mathrm{\epsilon}$', aspect=None)
    plot_tools.set_axis(axis02, xlim=[-1.7, 1.7], ylim=[0, 5], xlabel=r'$\mathrm{\epsilon}$', aspect=None)
    axis01.set_yticklabels([])
    axis02.set_yticklabels([])

    # Split the names into 3 groups and plot the three flavours of a halo together (i.e., original, NoR and NoRNoQ) #
    names_groups = np.array_split(names, 3)
    for i, axis in zip(range(len(names_groups)), [axis00, axis01, axis02]):
        names_flavours = names_groups[i]
        # Loop over all available flavours #
        for j in range(len(names_flavours)):
            # Load the data #
            epsilon = np.load(path + 'epsilon_' + str(re.split('_|.npy', names_flavours[j])[1]) + '.npy')
            disc_fraction = np.load(path + 'disc_fraction_' + str(re.split('_|.npy', names_flavours[j])[1]) + '.npy')
            stellar_masses = np.load(path + 'stellar_masses_' + str(re.split('_|.npy', names_flavours[j])[1]) + '.npy')

            # Plot the orbital circularity distribution #
            y_data, edges = np.histogram(epsilon, weights=stellar_masses / np.sum(stellar_masses), bins=100, range=[-1.7, 1.7])
            y_data /= edges[1:] - edges[:-1]
            axis.plot(0.5 * (edges[1:] + edges[:-1]), y_data, color=colors[j],
                      label=r'$\mathrm{Au-%s:D/T=%.2f}$' % (str(re.split('_|.npy', names_flavours[j])[1]), disc_fraction))

            axis.legend(loc='upper left', fontsize=16, frameon=False, numpoints=1)  # Create the legend.
    # Save and close the figure #
    pdf.savefig(figure, bbox_inches='tight')
    plt.close()
    return None


def tully_fisher_combination(pdf):
    """
    Plot a combination of the Tully-Fisher relation for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :return: None
    """
    print("Invoking tully_fisher_combination")
    # Get the names and sort them #
    path = '/u/di43/Auriga/plots/data/' + 'tf/'
    names = glob.glob(path + '/name_*')
    names.sort()

    # Generate the figure and set its parameters #
    figure, axis = plt.subplots(1, figsize=(10, 7.5))
    plot_tools.set_axis(axis, xlim=[1e8, 1e12], ylim=[1.4, 2.6], xscale='log', xlabel=r'$\mathrm{M_{\bigstar}/M_{\odot}}$',
                        ylabel=r'$\mathrm{log_{10}(v_{circ}/(km\;s^{-1}))}$', aspect=None)

    # Plot Pizagno et al. 2007 sample #
    table = "./data/pizagno.txt"
    rmag_p = np.genfromtxt(table, comments='#', usecols=3)
    vcirc_p = np.genfromtxt(table, comments='#', usecols=9)
    color_p = np.genfromtxt(table, comments='#', usecols=5)
    mass_p = galaxy.pizagno_convert_color_to_mass(color_p, rmag_p)
    Pizagno = plt.scatter(1e10 * mass_p, np.log10(vcirc_p), color='lightgray', s=10, marker='^')

    # Plot Verheijen 2001 sample #
    table = "./data/verheijen.txt"
    Bmag_v = np.genfromtxt(table, comments='#', usecols=1)
    Rmag_v = np.genfromtxt(table, comments='#', usecols=2)
    vcirc_v = np.genfromtxt(table, comments='#', usecols=7)
    color_v = Bmag_v - Rmag_v
    mass_v = galaxy.verheijen_convert_color_to_mass(color_v, Bmag_v)
    Verheijen = plt.scatter(1e10 * mass_v, np.log10(vcirc_v), color='lightgray', s=10, marker='s')

    # Plot Courteau et al. 2007 sample #
    table = "./data/courteau.txt"
    loglum_c = np.genfromtxt(table, comments='#', usecols=6)
    vcirc_c = np.genfromtxt(table, comments='#', usecols=8)
    mass_c = galaxy.courteau_convert_luminosity_to_mass(loglum_c)
    Courteau = plt.scatter(1e10 * mass_c, vcirc_c, color='lightgray', s=10, marker='o')

    # Plot best fit from Dutton et al. 2011 #
    masses = np.arange(0.1, 50.0)
    Dutton, = plt.plot(1e10 * masses, np.log10(galaxy.obs_tullyfisher_fit(masses)), color='darkgray', lw=0.8, ls='--')

    # Create the legend #
    legend = plt.legend([Pizagno, Verheijen, Courteau, Dutton],
                        [r'$\mathrm{Pizagno+\;07}$', r'$\mathrm{Verheijen\;01}$', r'$\mathrm{Courteau+\;07}$', r'$\mathrm{Dutton+\;11}$'],
                        loc='upper left', fontsize=12, frameon=False, numpoints=1)
    plt.gca().add_artist(legend)

    # Loop over all available haloes #
    for i in range(len(names)):
        # Load the data #
        sigma = np.load('/u/di43/Auriga/plots/data/vdp/0.0/' + 'sigma_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        stellar_mass = np.load(path + 'stellar_mass_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        total_circular_velocity = np.load(path + 'total_circular_velocity_' + str(re.split('_|.npy', names[i])[1]) + '.npy')

        # Plot the Tully-Fisher relation #
        plt.scatter(stellar_mass * 1e10, np.log10(sigma), color=colors2[i], s=100, zorder=5, marker=next(marker_array),
                    label=r'$\mathrm{Au-%s:v_{circ}/\sigma=%.2f}$' % (str(re.split('_|.npy', names[i])[1]), total_circular_velocity / sigma))

        plt.legend(loc='lower center', fontsize=16, frameon=False, numpoints=1, scatterpoints=1, ncol=3)  # Create the legend.

    # Save and close the figure #
    pdf.savefig(figure, bbox_inches='tight')
    plt.close()
    return None


def stellar_vs_halo_mass_combination(pdf):
    """
    Plot a combination of the abundance matching relation for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :return: None
    """
    print("Invoking stellar_vs_halo_mass_combination")
    # Get the names and sort them #
    path = '/u/di43/Auriga/plots/data/' + 'svt/'
    names = glob.glob(path + '/name_*')
    names.sort()

    # Generate the figure and set its parameters #
    figure, axis = plt.subplots(1, figsize=(10, 7.5))
    plot_tools.set_axis(axis, xlim=[1e11, 1e13], ylim=[1e9, 1e12], xscale='log', yscale='log', xlabel=r'$\mathrm{M_{halo}/M_{\odot}}$',
                        ylabel=r'$\mathrm{M_{\bigstar}/M_{\odot}}$', aspect=None)

    # Plot the cosmic baryon fraction relation #
    masses = np.arange(15., 300.)
    cosmic_baryon_frac = 0.048 / 0.307
    omega, = plt.plot(1e10 * masses, 1e10 * masses * cosmic_baryon_frac, color='k', ls='--')

    # Plot the Guo+10 relation #
    guo_high = galaxy.guo_abundance_matching(masses) * 10 ** (+0.2)
    guo_low = galaxy.guo_abundance_matching(masses) * 10 ** (-0.2)
    plt.fill_between(1e10 * masses, 1e10 * guo_low, 1e10 * guo_high, color='lightgray', edgecolor='None')
    Guo, = plt.plot(1e10 * masses, 1e10 * galaxy.guo_abundance_matching(masses), color='k', ls=':')

    # Create the legend #
    legend = plt.legend([omega, Guo], [r'$\mathrm{M_{200}\;\Omega_b/\Omega_m}$', r'$\mathrm{Guo+\;10}$'], loc='upper left', fontsize=12,
                        frameon=False, numpoints=1)
    plt.gca().add_artist(legend)

    # Loop over all available haloes #
    for i in range(len(names)):
        # Load the data #
        halo_mass = np.load(path + 'halo_mass_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        stellar_mass = np.load(path + 'stellar_mass_' + str(re.split('_|.npy', names[i])[1]) + '.npy')

        # Plot the abundance matching relation #
        plt.scatter(halo_mass * 1e10, stellar_mass * 1e10, color=colors2[i], s=100, zorder=5, marker=next(marker_array),
                    label=r'$\mathrm{Au-%s}$' % str(re.split('_|.npy', names[i])[1]))

        plt.legend(loc='lower center', fontsize=16, frameon=False, numpoints=1, scatterpoints=1, ncol=3)  # Create the legend.

    # Save and close the figure #
    pdf.savefig(figure, bbox_inches='tight')
    plt.close()
    return None


def gas_fraction_vs_magnitude_combination(pdf):
    """
    Plot a combination of the gas fraction (gas to stellar plus gas mass ratio) as a function R-band magnitude for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :return: None
    """
    print("Invoking gas_fraction_vs_magnitude_combination")
    # Get the names and sort them #
    path = '/u/di43/Auriga/plots/data/' + 'gfvm/'
    names = glob.glob(path + '/name_*')
    names.sort()

    # Generate the figure and set its parameters #
    figure, axis = plt.subplots(1, figsize=(10, 7.5))
    plot_tools.set_axis(axis, xlim=[-23.2, -22], ylim=[0.1, 0.4], xlabel=r'$\mathrm{M_{R}/mag}$', ylabel=r'$\mathrm{f_{gas}}$', aspect=None)

    # Loop over all available haloes #
    for i in range(len(names)):
        # Load the data #
        gas_fraction = np.load(path + 'gas_fraction_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        M_R = np.load(path + 'M_R_' + str(re.split('_|.npy', names[i])[1]) + '.npy')

        # Plot the gas fraction as a function R-band magnitude #
        plt.scatter(M_R, gas_fraction, color=colors2[i], s=100, zorder=5, marker=next(marker_array),
                    label=r'$\mathrm{Au-%s}$' % str(re.split('_|.npy', names[i])[1]))

        plt.legend(loc='upper center', fontsize=16, frameon=False, numpoints=1, scatterpoints=1, ncol=3)  # Create the legend.

    # Save and close the figure #
    pdf.savefig(figure, bbox_inches='tight')
    plt.close()
    return None


def bar_strength_profile_combination(pdf):
    """
    Plot a combination of the bar strength radial profile from Fourier modes of surface density for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :return: None
    """
    print("Invoking bar_strength_profile_combination")
    # Get the names and sort them #
    path = '/u/di43/Auriga/plots/data/' + 'bsp/'
    names = glob.glob(path + '/name_*')
    names.sort()

    # Generate the figure and set its parameters #
    figure = plt.figure(figsize=(20, 7.5))
    axis00, axis01, axis02 = plot_tools.create_axes_combinations(res=res, boxsize=boxsize * 1e3, multiple4=True)
    for axis in [axis00, axis01, axis02]:
        plot_tools.set_axis(axis, xlim=[0, 11], ylim=[-0.1, 1.1], xlabel=r'$\mathrm{R/kpc}$', ylabel=r'$\mathrm{\sqrt{a_{2}^{2}+b_{2}^{2}}/a_{0}}$',
                            aspect=None)
    for axis in [axis01, axis02]:
        axis.set_ylabel('')
        axis.set_yticklabels([])

    # Split the names into 3 groups and plot the three flavours of a halo together (i.e., original, NoR and NoRNoQ) #
    names_groups = np.array_split(names, 3)
    for i, axis in zip(range(len(names_groups)), [axis00, axis01, axis02]):
        names_flavours = names_groups[i]
        # Loop over all available flavours #
        for j in range(len(names_flavours)):
            # Load the data #
            ratio = np.load(path + 'ratio_' + str(re.split('_|.npy', names_flavours[j])[1]) + '.npy')
            r_m = np.load(path + 'r_m_' + str(re.split('_|.npy', names_flavours[j])[1]) + '.npy')

            # Plot the bar strength radial profile and get an estimate for the bar length from the maximum strength #
            A2 = max(ratio)
            axis.plot(r_m, ratio, color=colors[j], label=r'$\mathrm{Au-%s:A_{2}=%.2f}$' % (str(re.split('_|.npy', names_flavours[j])[1]), A2))
            axis.plot([r_m[np.where(ratio == A2)], r_m[np.where(ratio == A2)]], [-0.0, A2], color=colors[j], linestyle='dashed',
                      label=r'$\mathrm{r_{A_{2}}=%.2fkpc}$' % r_m[np.where(ratio == A2)])

            axis.legend(loc='upper left', fontsize=16, frameon=False, numpoints=1)  # Create the legend.
    # Save and close the figure #
    pdf.savefig(figure, bbox_inches='tight')
    plt.close()
    return None


def stellar_surface_density_profiles_combination(pdf):
    """
    Plot a combination of the stellar surface density profiles for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :return: None
    """
    print("Invoking stellar_surface_density_profiles_combination")
    # Get the names and sort them #
    path = '/u/di43/Auriga/plots/data/' + 'ssdp/'
    names = glob.glob(path + '/name_*')
    names.sort()

    # Generate the figure and set its parameters #
    figure = plt.figure(figsize=(20, 20))
    axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22 = plot_tools.create_axes_combinations(res=res, boxsize=boxsize * 1e3,
                                                                                                                 multiple5=True)
    for axis in [axis00, axis10, axis20]:
        plot_tools.set_axis(axis, xlim=[0, 29], ylim=[1e0, 1e6], yscale='log', xlabel=r'$\mathrm{R/kpc}$',
                            ylabel=r'$\mathrm{\Sigma_{\bigstar}/(M_{\odot}\;pc^{-2})}$', aspect=None, which='major', size=20)
    for axis in [axis01, axis02, axis11, axis12, axis21, axis22]:
        plot_tools.set_axis(axis, xlim=[0, 29], ylim=[1e0, 1e6], yscale='log', xlabel=r'$\mathrm{R/kpc}$', aspect=None, which='major', size=20)
        axis.set_yticklabels([])
    for axis in [axis00, axis01, axis02, axis10, axis11, axis12]:
        axis.set_xticklabels([])

    # Loop over all available haloes #
    for i, axis in zip(range(len(names)), [axis00, axis10, axis20, axis01, axis11, axis21, axis02, axis12, axis22]):
        # Load the data #
        r = np.load(path + 'r_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        rfit = np.load(path + 'rfit_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        sdfit = np.load(path + 'sdfit_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        popt0 = np.load(path + 'popt0_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        popt1 = np.load(path + 'popt1_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        popt2 = np.load(path + 'popt2_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        popt3 = np.load(path + 'popt3_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        popt4 = np.load(path + 'popt4_' + str(re.split('_|.npy', names[i])[1]) + '.npy')

        # Plot the stellar surface density profiles #
        p = plot_helper.plot_helper()  # Load the helper.
        axis.axvline(rfit, color='gray', linestyle='--')
        axis.scatter(r, 1e10 * sdfit * 1e-6, marker='o', s=15, color=colors[0], linewidth=0.0)
        axis.plot(r, 1e10 * p.exp_prof(r, popt0, popt1) * 1e-6, color=colors[3])
        axis.plot(r, 1e10 * p.sersic_prof1(r, popt2, popt3, popt4) * 1e-6, color=colors[1])
        axis.plot(r, 1e10 * p.total_profile(r, popt0, popt1, popt2, popt3, popt4) * 1e-6, color=colors[0])

        figure.text(0.01, 0.92, r'$\mathrm{Au-%s}$' % str(re.split('_|.npy', names[i])[1]), fontsize=20, transform=axis.transAxes)

    # Save and close the figure #
    pdf.savefig(figure, bbox_inches='tight')
    plt.close()
    return None


def circular_velocity_curves_combination(pdf):
    """
    Plot a combination of the circular velocity curve for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :return: None
    """
    print("Invoking circular_velocity_curves_combination")
    # Get the names and sort them #
    path = '/u/di43/Auriga/plots/data/' + 'cvc/'
    names = glob.glob(path + '/name_*')
    names.sort()

    # Generate the figure and set its parameters #
    figure = plt.figure(figsize=(20, 20))
    axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22 = plot_tools.create_axes_combinations(res=res, boxsize=boxsize * 1e3,
                                                                                                                 multiple5=True)
    for axis in [axis00, axis10, axis20]:
        plot_tools.set_axis(axis, xlim=[0, 24], ylim=[0, 700], xlabel=r'$\mathrm{R/kpc}$', ylabel=r'$\mathrm{V_{circular}/(km\;s^{-1})}$',
                            aspect=None, size=20)
    for axis in [axis01, axis02, axis11, axis12, axis21, axis22]:
        plot_tools.set_axis(axis, xlim=[0, 24], ylim=[0, 700], xlabel=r'$\mathrm{R/kpc}$', aspect=None, size=20)
        axis.set_yticklabels([])
    for axis in [axis00, axis01, axis02, axis10, axis11, axis12]:
        axis.set_xticklabels([])

    # Loop over all available haloes #
    for i, axis in zip(range(len(names)), [axis00, axis10, axis20, axis01, axis11, axis21, axis02, axis12, axis22]):
        # Load the data #
        radius = np.load(path + 'radius_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        total_mass = np.load(path + 'total_mass_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        shell_velocity = np.load(path + 'shell_velocity_' + str(re.split('_|.npy', names[i])[1]) + '.npy')

        # Plot the circular velocity curve #
        vtot = np.sqrt(G * total_mass * 1e10 * msol / (radius * 1e6 * parsec)) / 1e5
        axis.plot(radius * 1e3, vtot, color=colors[0], linewidth=4, label=r'$\mathrm{Total}$')
        axis.plot(radius * 1e3, shell_velocity[:, 0], color=colors[3], linestyle='--', linewidth=3, label=r'$\mathrm{Gas}$')
        axis.plot(radius * 1e3, shell_velocity[:, 4], color=colors[2], linestyle='--', linewidth=3, label=r'$\mathrm{Stars}$')
        axis.plot(radius * 1e3, shell_velocity[:, 1], color=colors[1], linestyle='--', linewidth=3, label=r'$\mathrm{Dark\;matter}$')

        figure.text(0.01, 0.92, r'$\mathrm{Au-%s}$' % str(re.split('_|.npy', names[i])[1]), fontsize=20, transform=axis.transAxes)

    # Save and close the figure #
    pdf.savefig(figure, bbox_inches='tight')
    plt.close()
    return None


def ssdp_cvc_combination(pdf):
    """
    Plot a combination of the circular velocity curve and the stellar surface density profiles for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :return: None
    """
    print("Invoking ssdp_cvc_combination")
    # Get the names and sort them #
    path_cvc = '/u/di43/Auriga/plots/data/' + 'cvc/'
    path_ssdp = '/u/di43/Auriga/plots/data/' + 'ssdp/'
    names = glob.glob(path_cvc + '/name_*')
    names.sort()

    # Generate the figure and set its parameters #
    figure = plt.figure(figsize=(20, 20))
    axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22, axis30, axis31, axis32, axis40, axis41, axis42, axis50, axis51, \
    axis52 = plot_tools.create_axes_combinations(
        res=res, boxsize=boxsize * 1e3, multiple9=True)

    for axis in [axis00, axis20, axis40]:
        plot_tools.set_axis(axis, xlim=[0, 29], ylim=[1e0, 1e6], yscale='log', ylabel=r'$\mathrm{\Sigma_{\bigstar}/(M_{\odot}\;pc^{-2})}$',
                            aspect=None, which='major', size=20)
    for axis in [axis01, axis02, axis21, axis22, axis41, axis42]:
        plot_tools.set_axis(axis, xlim=[0, 29], ylim=[1e0, 1e6], yscale='log', aspect=None, which='major', size=20)
        axis.set_yticklabels([])

    for axis in [axis10, axis30, axis50]:
        plot_tools.set_axis(axis, xlim=[0, 29], ylim=[1, 800], xlabel=r'$\mathrm{R/kpc}$', ylabel=r'$\mathrm{V_{circular}/(km\;s^{-1})}$',
                            aspect=None, size=20)
        axis.set_yticklabels(['', '', '', '300', '', '', '600', '', ''])
    for axis in [axis11, axis12, axis31, axis32, axis51, axis52]:
        plot_tools.set_axis(axis, xlim=[0, 29], ylim=[1, 800], xlabel=r'$\mathrm{R/kpc}$', aspect=None, size=20)
        axis.set_yticklabels([])

    for axis in [axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22, axis30, axis31, axis32, axis40, axis41, axis42]:
        axis.set_xticklabels([])

    # Loop over all available haloes #
    axes = [axis00, axis20, axis40, axis01, axis21, axis41, axis02, axis22, axis42]
    axes_cvc = [axis10, axis30, axis50, axis11, axis31, axis51, axis12, axis32, axis52]

    for i, axis, axis_cvc in zip(range(len(names)), axes, axes_cvc):
        # Load the data #
        r = np.load(path_ssdp + 'r_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        rfit = np.load(path_ssdp + 'rfit_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        sdfit = np.load(path_ssdp + 'sdfit_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        popt0 = np.load(path_ssdp + 'popt0_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        popt1 = np.load(path_ssdp + 'popt1_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        popt2 = np.load(path_ssdp + 'popt2_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        popt3 = np.load(path_ssdp + 'popt3_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        popt4 = np.load(path_ssdp + 'popt4_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        radius = np.load(path_cvc + 'radius_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        total_mass = np.load(path_cvc + 'total_mass_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        shell_velocity = np.load(path_cvc + 'shell_velocity_' + str(re.split('_|.npy', names[i])[1]) + '.npy')

        # Plot the circular velocity curve #
        vtot = np.sqrt(G * total_mass * 1e10 * msol / (radius * 1e6 * parsec)) / 1e5
        axis_cvc.plot(radius * 1e3, vtot, color=colors[0], linewidth=4, label=r'$\mathrm{Total}$')
        axis_cvc.plot(radius * 1e3, shell_velocity[:, 0], color=colors[3], linestyle='--', linewidth=3, label=r'$\mathrm{Gas}$')
        axis_cvc.plot(radius * 1e3, shell_velocity[:, 4], color=colors[2], linestyle='--', linewidth=3, label=r'$\mathrm{Stars}$')
        axis_cvc.plot(radius * 1e3, shell_velocity[:, 1], color=colors[1], linestyle='--', linewidth=3, label=r'$\mathrm{Dark\;matter}$')

        # Plot the stellar surface density profiles #
        p = plot_helper.plot_helper()  # Load the helper.
        axis.axvline(rfit, color='gray', linestyle='--')
        axis.scatter(r, 1e10 * sdfit * 1e-6, marker='o', s=15, color=colors[0], linewidth=0.0)
        axis.plot(r, 1e10 * p.exp_prof(r, popt0, popt1) * 1e-6, color=colors[3], label=r'$\mathrm{Total}$')
        axis.plot(r, 1e10 * p.sersic_prof1(r, popt2, popt3, popt4) * 1e-6, color=colors[1], label=r'$\mathrm{Sersic}$')
        axis.plot(r, 1e10 * p.total_profile(r, popt0, popt1, popt2, popt3, popt4) * 1e-6, color=colors[0], label=r'$\mathrm{Exponential}$')

        figure.text(0.01, 0.92, r'$\mathrm{Au-%s}$' % str(re.split('_|.npy', names[i])[1]), fontsize=20, transform=axis.transAxes)
        # figure.text(0.8, 0.75, r'$\mathrm{n} = %.2f$' '\n' r'$\mathrm{R_{d}} = %.2f$' '\n' r'$\mathrm{R_{eff}} = %.2f$' '\n' % (
        #     1. / popt4, popt1, popt3 * p.sersic_b_param(1.0 / popt4) ** (1.0 / popt4)), fontsize=16, transform=axis.transAxes)
        axis.legend(loc='upper right', fontsize=16, frameon=False, numpoints=1)
        axis_cvc.legend(loc='upper center', fontsize=16, frameon=False, numpoints=1, ncol=2)

    # Save and close the figure #
    pdf.savefig(figure, bbox_inches='tight')
    plt.close()
    return None


def gas_temperature_vs_distance_combination(date):
    """
    Plot a combination of the temperature as a function of distance of gas cells for Auriga halo(es).
    :param date: date.
    :return: None
    """
    print("Invoking gas_temperature_vs_distance_combination")
    # Get the names and sort them #
    path = '/u/di43/Auriga/plots/data/' + 'gtd/'
    names = glob.glob(path + '/name_*')
    names.sort()

    # Generate the figure and set its parameters #
    figure = plt.figure(figsize=(20, 20))
    axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22, axiscbar = plot_tools.create_axes_combinations(res=res,
                                                                                                                           boxsize=boxsize * 1e3,
                                                                                                                           multiple6=True)
    for axis in [axis00, axis10, axis20]:
        plot_tools.set_axis(axis, xlim=[2e-2, 2e2], ylim=[1e3, 2e8], xscale='log', yscale='log', xlabel=r'$\mathrm{R/kpc}$',
                            ylabel=r'$\mathrm{Temperature/K}$', aspect=None, which='major', size=20)
    for axis in [axis01, axis02, axis11, axis12, axis21, axis22]:
        plot_tools.set_axis(axis, xlim=[2e-2, 2e2], ylim=[1e3, 2e8], xscale='log', yscale='log', xlabel=r'$\mathrm{R/kpc}$', aspect=None,
                            which='major', size=20)

    # Loop over all available haloes #
    for i, axis in zip(range(len(names)), [axis00, axis10, axis20, axis01, axis11, axis21, axis02, axis12, axis22]):
        # Load the data #
        temperature = np.load(path + 'temperature_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        spherical_distance = np.load(path + 'spherical_distance_' + str(re.split('_|.npy', names[i])[1]) + '.npy')

        # Plot the temperature as a function of distance of gas cells #
        hb = axis.hexbin(spherical_distance * 1e3, temperature, bins='log', xscale='log', yscale='log', cmap='gist_earth_r')

        figure.text(0.01, 0.92, r'$\mathrm{Au-%s}$' % str(re.split('_|.npy', names[i])[1]), fontsize=20, transform=axis.transAxes)

    for axis in [axis01, axis02, axis11, axis12, axis21, axis22]:
        axis.set_yticklabels([])
    for axis in [axis00, axis01, axis02, axis10, axis11, axis12]:
        axis.set_xticklabels([])

    # Add a colorbar, save and close the figure #
    plot_tools.create_colorbar(axiscbar, hb, label=r'$\mathrm{Counts\;per\;hexbin}$', size=20)
    plt.savefig('/u/di43/Auriga/plots/' + 'gtd-' + date + '.png', bbox_inches='tight')
    plt.close()
    return None


def decomposition_IT20_combination(date, redshift):
    """
    Plot the angular momentum maps and calculate D/T_IT20 for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param data: data from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :return: None
    """
    print("Invoking decomposition_IT20_combination")
    # Get the names and sort them #
    path = '/u/di43/Auriga/plots/data/' + 'di/' + str(redshift) + '/'

    # Generate the figure and set its parameters #
    figure = plt.figure(figsize=(20, 15))
    axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22, axiscbar = plot_tools.create_axes_combinations(res=res,
                                                                                                                           boxsize=boxsize * 1e3,
                                                                                                                           mollweide=True)
    axes = [axis00, axis10, axis20, axis01, axis11, axis21, axis02, axis12, axis22]
    for axis in axes:
        axis.set_yticklabels([])
        axis.set_xlabel(r'$\mathrm{\alpha/\degree}$', size=20)
        axis.set_xticklabels(['', '-120', '', '-60', '', '0', '', '60', '', '120', ''], size=20)
    for axis in [axis00, axis10, axis20]:
        axis.set_ylabel(r'$\mathrm{\delta/\degree}$', size=20)
        axis.set_yticklabels(['', '-60', '', '-30', '', '0', '', '30', '', '60', ''], size=20)

    # Load and plot the data #
    names = glob.glob(path + '/name_*')
    names.sort()

    # Loop over all available haloes #
    for i, axis in zip(range(len(names)), axes):
        # Generate the figure and set its parameters #

        # Load the data #
        density_map = np.load(path + 'density_map_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        disc_fraction_IT20 = np.load(path + 'disc_fraction_IT20_' + str(re.split('_|.npy', names[i])[1]) + '.npy')

        # Plot the angular momentum maps and calculate D/T_IT20 #
        # Sample a 360x180 grid in sample_alpha and sample_delta #
        sample_alpha = np.linspace(-180.0, 180.0, num=360) * u.deg
        sample_delta = np.linspace(-90.0, 90.0, num=180) * u.deg

        # Display data on a 2D regular raster and create a pseudo-color plot #
        pcm = axis.pcolormesh(np.radians(sample_alpha), np.radians(sample_delta), density_map, cmap='nipy_spectral_r')

        figure.text(0.01, 1.05, r'$\mathrm{Au-%s:D/T=%.2f}$' % (str(re.split('_|.npy', names[i])[1]), disc_fraction_IT20), fontsize=20,
                    transform=axis.transAxes)

    # Add a colorbar, save and close the figure #
    plot_tools.create_colorbar(axiscbar, pcm, label=r'$\mathrm{Particles\;per\;grid\;cell}$', size=20)
    plt.savefig('/u/di43/Auriga/plots/' + 'dic-' + date + '.png', bbox_inches='tight')
    plt.close()
    return None


def bar_strength_combination(pdf):
    """
    Plot a combination of the evolution of bar strength from Fourier modes of surface density for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :return: None
    """
    print("Invoking bar_strength_combination")
    # Get the names and sort them #
    path = '/u/di43/Auriga/plots/data/' + 'bse/'
    names = glob.glob(path + '/name_*')
    names.sort()

    # Generate the figure and set its parameters #
    figure = plt.figure(figsize=(20, 7.5))
    axis00, axis01, axis02 = plot_tools.create_axes_combinations(res=res, boxsize=boxsize * 1e3, multiple4=True)
    axis002 = axis00.twiny()
    plot_tools.set_axes_evolution(axis00, axis002, ylim=[-0.1, 1.1], ylabel=r'$\mathrm{A_{2}}$', aspect=None)
    for axis in [axis01, axis02]:
        axis2 = axis.twiny()
        plot_tools.set_axes_evolution(axis, axis2, ylim=[-0.1, 1.1], aspect=None)
        axis.set_yticklabels([])

    # Split the names into 3 groups and plot the three flavours of a halo together (i.e., original, NoR and NoRNoQ) #
    names_groups = np.array_split(names, 3)
    for i, axis in zip(range(len(names_groups)), [axis00, axis01, axis02]):
        names_flavours = names_groups[i]
        # Loop over all available flavours #
        for j in range(len(names_flavours)):
            # Load the data #
            max_A2s = np.load(path + 'max_A2s_' + str(re.split('_|.npy', names_flavours[j])[1]) + '.npy')
            lookback_times = np.load(path + 'lookback_times_' + str(re.split('_|.npy', names_flavours[j])[1]) + '.npy')

            # Plot the evolution of bar strength #
            axis.plot(lookback_times, max_A2s, color=colors[j], label=r'$\mathrm{Au-%s}$' % str(re.split('_|.npy', names_flavours[j])[1]))

            axis.legend(loc='upper left', fontsize=16, frameon=False, numpoints=1)  # Create the legend.
    # Save and close the figure #
    pdf.savefig(figure, bbox_inches='tight')
    plt.close()
    return None


def gas_temperature_regimes_combination(pdf):
    """
    Plot a combination of the evolution of gas fractions in different temperature regimes for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :return: None
    """
    print("Invoking gas_temperature_regimes_combination")
    # Get the names and sort them #
    path = '/u/di43/Auriga/plots/data/' + 'gtr/'
    names = glob.glob(path + '/name_*')
    names.sort()

    # Generate the figure and set its parameters #
    figure = plt.figure(figsize=(20, 20))
    axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22 = plot_tools.create_axes_combinations(res=res, boxsize=boxsize * 1e3,
                                                                                                                 multiple5=True)
    for axis in [axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22]:
        axis2 = axis.twiny()
        plot_tools.set_axes_evolution(axis, axis2, ylim=[-0.1, 1.1], ylabel=r'$\mathrm{Gas\;fraction}$', aspect=None, size=20)
        if axis in [axis10, axis11, axis12, axis20, axis21, axis22]:
            axis2.set_xlabel('')
            axis2.set_xticklabels([])
    for axis in [axis00, axis01, axis02, axis10, axis11, axis12]:
        axis.set_xlabel('')
        axis.set_xticklabels([])
    for axis in [axis01, axis02, axis11, axis12, axis21, axis22]:
        axis.set_ylabel('')
        axis.set_yticklabels([])

    # Loop over all available haloes #
    for i, axis in zip(range(len(names)), [axis00, axis10, axis20, axis01, axis11, axis21, axis02, axis12, axis22]):
        # Load the data #
        wg_ratios = np.load(path + 'wg_ratios_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        hg_ratios = np.load(path + 'hg_ratios_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        sfg_ratios = np.load(path + 'sfg_ratios_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        lookback_times = np.load(path + 'lookback_times_' + str(re.split('_|.npy', names[i])[1]) + '.npy')

        # Plot the evolution of gas fraction in different temperature regimes #
        axis.plot(lookback_times, hg_ratios, color='red', label=r'$\mathrm{Hot\;gas}$')
        axis.plot(lookback_times, sfg_ratios, color='blue', label=r'$\mathrm{Cold\;gas}$')
        axis.plot(lookback_times, wg_ratios, color='green', label=r'$\mathrm{Warm\;gas}$')

        figure.text(0.01, 0.92, r'$\mathrm{Au-%s}$' % str(re.split('_|.npy', names[i])[1]), fontsize=20, transform=axis.transAxes)
        axis.legend(loc='upper right', fontsize=16, frameon=False, numpoints=1)  # Create the legend.
    # Save and close the figure #
    pdf.savefig(figure, bbox_inches='tight')
    plt.close()
    return None


def AGN_modes_distribution_combination(date):
    """
    Plot a combination of the energy of different black hole feedback modes from log files and plot its evolution for Auriga halo(es).
    :param date: date.
    :return: None
    """
    print("Invoking AGN_modes_distribution_combination")
    # Get the names and sort them #
    path = '/u/di43/Auriga/plots/data/' + 'AGNmd/'
    names = glob.glob(path + '/name_*')
    names.sort()
    n_bins = 130
    time_bin_width = (13 - 0) / n_bins  # In Gyr.

    # Generate the figure and set its parameters #
    figure = plt.figure(figsize=(20, 20))
    axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22, axis30, axis31, axis32, axis40, axis41, axis42, axis50, axis51, \
    axis52 = plot_tools.create_axes_combinations(
        res=res, boxsize=boxsize * 1e3, multiple7=True)

    for axis in [axis10, axis30, axis50, axis11, axis31, axis51, axis12, axis32, axis52]:
        axis2 = axis.twiny()
        plot_tools.set_axes_evolution(axis, axis2, ylim=[1e51, 2e61], yscale='log', aspect=None, which='major', size=20)
    for axis in [axis10, axis11, axis12, axis30, axis31, axis32]:
        axis.set_xlabel('')
        axis.set_xticklabels([])
    axis10.set_ylabel(r'$\mathrm{(Mechanical\;feedback\;energy)/ergs}$', size=20)
    axis30.set_ylabel(r'$\mathrm{(Thermal\;feedback\;energy)/ergs}$', size=20)
    axis50.set_ylabel(r'$\mathrm{(Thermal\;feedback\;energy)/ergs}$', size=20)

    # Get the names and sort them #
    names = glob.glob(path + '/name_*')
    names.sort()

    # Split the names into 3 groups and plot the three flavours of a halo together (i.e., original, NoR and NoRNoQ) #
    names_groups = np.array_split(names, 3)
    for i in range(len(names_groups)):
        names_flavours = names_groups[i]
        # Loop over all available flavours #
        for j in range(len(names_flavours)):
            # Load the data #
            if j == 0:
                mechanicals = np.load(path + 'mechanicals_' + str(re.split('_|.npy', names_flavours[j])[1]) + '.npy')
                # Transform the arrays to comma separated strings and convert each element to float #
                mechanicals = ','.join(mechanicals)
                mechanicals = np.fromstring(mechanicals, dtype=np.float, sep=',')
            thermals = np.load(path + 'thermals_' + str(re.split('_|.npy', names_flavours[j])[1]) + '.npy')
            lookback_times = np.load(path + 'lookback_times_' + str(re.split('_|.npy', names_flavours[j])[1]) + '.npy')
            # Transform the arrays to comma separated strings and convert each element to float #
            thermals = ','.join(thermals)
            thermals = np.fromstring(thermals, dtype=np.float, sep=',')

            # Define the plot and colorbar axes #
            if j == 0:
                if i == 0:
                    axes = [axis10, axis30]
                    axescbar = [axis00, axis20]
                    modes = [mechanicals, thermals]
                if i == 1:
                    axes = [axis11, axis31]
                    axescbar = [axis01, axis21]
                    modes = [mechanicals, thermals]
                if i == 2:
                    axes = [axis12, axis32]
                    axescbar = [axis02, axis22]
                    modes = [mechanicals, thermals]
            else:
                if i == 0:
                    axes = [axis50]
                    modes = [thermals]
                    axescbar = [axis40]
                if i == 1:
                    axes = [axis51]
                    modes = [thermals]
                    axescbar = [axis41]
                if i == 2:
                    axes = [axis52]
                    modes = [thermals]
                    axescbar = [axis42]

            # Plot 2D distribution of the modes and their binned sum line #
            for axis, axiscbar, mode in zip(axes, axescbar, modes):
                hb = axis.hexbin(lookback_times[np.where(mode > 0)], mode[np.where(mode > 0)], bins='log', yscale='log', cmap='hot_r')
                plot_tools.create_colorbar(axiscbar, hb, label=r'$\mathrm{Counts\;per\;hexbin}$', orientation='horizontal', size=20)
                x_value, sum = plot_tools.binned_sum(lookback_times[np.where(mode > 0)], mode[np.where(mode > 0)], n_bins=n_bins)
                axis.plot(x_value, sum / time_bin_width, color=colors[0], label=r'$\mathrm{Sum}$')
                figure.text(0.01, 0.92, 'Au-' + str(re.split('_|.npy', names_flavours[j])[1]), fontsize=16, transform=axis.transAxes)

            for axis in [axis11, axis12, axis31, axis32, axis51, axis52]:
                axis.set_yticklabels([])

    # Create the legends, save and close the figure #
    axis10.legend(loc='upper right', fontsize=16, frameon=False, numpoints=1)
    axis11.legend(loc='upper right', fontsize=16, frameon=False, numpoints=1)
    axis12.legend(loc='upper right', fontsize=16, frameon=False, numpoints=1)
    plt.savefig('/u/di43/Auriga/plots/' + 'AGNmdc-' + date + '.png', bbox_inches='tight')

    return None


def AGN_feedback_kernel_combination(pdf):
    """
    Plot a combination of the evolution of black hole radius and the volume of gas cells within that for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :return: None
    """
    print("Invoking AGN_feedback_kernel_combination")
    # Get the names and sort them #
    path = '/u/di43/Auriga/plots/data/' + 'AGNfk/'
    names = glob.glob(path + '/name_*')
    names.sort()

    # Remove the NoRNoQ flavours #
    names.remove('/u/di43/Auriga/plots/data/AGNfk/name_06NoRNoQ.npy')
    names.remove('/u/di43/Auriga/plots/data/AGNfk/name_17NoRNoQ.npy')
    names.remove('/u/di43/Auriga/plots/data/AGNfk/name_18NoRNoQ.npy')

    # Generate the figure and set its parameters #
    figure = plt.figure(figsize=(16, 9))
    axis00, axis01, axis02, axis10, axis11, axis12 = plot_tools.create_axes_combinations(res=res, boxsize=boxsize * 1e3, multiple10=True)
    for axis in [axis00, axis01, axis02, axis10, axis11, axis12]:
        axis2 = axis.twiny()
        axis3 = axis.twinx()
        axis3.yaxis.label.set_color('red')
        axis3.spines['right'].set_color('red')
        plot_tools.set_axis(axis3, ylim=[-0.1, 1.1], xlabel=r'$\mathrm{t_{look}/Gyr}$', ylabel=r'$\mathrm{BH_{sml}/kpc}$', aspect=None)
        plot_tools.set_axes_evolution(axis, axis2, ylim=[-0.1, 1.1], ylabel=r'$\mathrm{V_{nSFR}(r<BH_{sml})/V_{all}(r<BH_{sml})}$', aspect=None)
        axis3.tick_params(axis='y', direction='out', left='off', colors='red')
        if axis in [axis10, axis11, axis12]:
            axis2.set_xlabel('')
            axis2.set_xticklabels([])
        if axis in [axis00, axis01, axis10, axis11]:
            axis3.set_ylabel('')
            axis3.set_yticklabels([])
    for axis in [axis00, axis01, axis02, axis10, axis11, axis12]:
        axis.set_xlabel('')
        axis.set_xticklabels([])
    for axis in [axis01, axis02, axis11, axis12]:
        axis.set_ylabel('')
        axis.set_yticklabels([])

    # Loop over all available haloes #
    for i, axis in zip(range(len(names)), [axis00, axis10, axis01, axis11, axis02, axis12]):
        # Load the data #
        gas_volumes = np.load(path + 'gas_volumes_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        lookback_times = np.load(path + 'lookback_times_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        nsf_gas_volumes = np.load(path + 'nsf_gas_volumes_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        blackhole_hsmls = np.load(path + 'blackhole_hsmls_' + str(re.split('_|.npy', names[i])[1]) + '.npy')

        # Plot median and 1-sigma lines #
        x_value, median, shigh, slow = plot_tools.binned_median_1sigma(lookback_times, nsf_gas_volumes / gas_volumes, bin_type='equal_number',
                                                                       n_bins=len(lookback_times) / 5, log=False)
        axis.plot(x_value[x_value > 0], median[x_value > 0], color=colors[0], linewidth=3)
        axis.fill_between(x_value[x_value > 0], shigh[x_value > 0], slow[x_value > 0], color=colors[0], alpha='0.3')

        x_value, median, shigh, slow = plot_tools.binned_median_1sigma(lookback_times, blackhole_hsmls * 1e3, bin_type='equal_number',
                                                                       n_bins=len(lookback_times) / 5, log=False)
        axis.plot(x_value[x_value > 0], median[x_value > 0], color=colors[1], linewidth=3)
        axis.fill_between(x_value[x_value > 0], shigh[x_value > 0], slow[x_value > 0], color=colors[1], alpha='0.3')

        figure.text(0.01, 0.92, r'$\mathrm{Au-%s}$' % str(re.split('_|.npy', names[i])[1]), fontsize=20, transform=axis.transAxes)

    # Save and close the figure #
    pdf.savefig(figure, bbox_inches='tight')
    plt.close()
    return None


def central_combination(pdf, data, redshift, read):
    """
    Plot a combination of the projections for Auriga halo(es) for the central 2kpc.
    :param pdf: path to save the pdf from main.make_pdf
    :param data: data from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :param read: boolean to read new data.
    :return: None
    """
    print("Invoking central_combination")
    boxsize = 0.004  # Decrease the boxsize.

    # Check if a folder to save the data exists, if not create one #
    path = '/u/di43/Auriga/plots/data/' + 'm/' + str(redshift) + '/'
    if not os.path.exists(path):
        os.makedirs(path)

    # Read the data #
    if read is True:
        # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
        particle_type = [0, 4]
        attributes = ['mass', 'ne', 'pos', 'rho', 'u', 'bfld', 'sfr']
        data.select_haloes(level, redshift, loadonlytype=particle_type, loadonlyhalo=0, loadonly=attributes)

        # Loop over all available haloes #
        for s in data:
            # Check if any of the haloes' data already exists, if not then create it #
            names = glob.glob(path + '/name_*')
            names = [re.split('_|.npy', name)[1] for name in names]
            if str(s.haloname) in names:
                continue

            # Select the halo and rotate it based on its principal axes so galaxy's spin is aligned with the z-axis #
            s.calc_sf_indizes(s.subfind)
            s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)

            # Get the gas density projections #
            density_face_on = s.get_Aslice("rho", res=res, axes=[1, 2], box=[boxsize, boxsize], proj=True, proj_fact=0.125, numthreads=8)[
                                  "grid"] * boxsize * 1e3
            density_edge_on = s.get_Aslice("rho", res=res, axes=[1, 0], box=[boxsize, boxsize], proj=True, proj_fact=0.125, numthreads=8)[
                                  "grid"] * boxsize * 1e3

            # Get the gas temperature projections #
            meanweight = 4.0 / (1.0 + 3.0 * 0.76 + 4.0 * 0.76 * s.data['ne']) * 1.67262178e-24
            temperature = (5.0 / 3.0 - 1.0) * s.data['u'] / KB * (1e6 * parsec) ** 2.0 / (1e6 * parsec / 1e5) ** 2 * meanweight
            s.data['temprho'] = s.rho * temperature

            temperature_face_on = s.get_Aslice("temprho", res=res, axes=[1, 2], box=[boxsize, boxsize], proj=True, proj_fact=0.125, numthreads=8)[
                "grid"]
            temperature_face_on_rho = s.get_Aslice("rho", res=res, axes=[1, 2], box=[boxsize, boxsize], proj=True, proj_fact=0.125, numthreads=8)[
                "grid"]
            temperature_edge_on = s.get_Aslice("temprho", res=res, axes=[1, 0], box=[boxsize, boxsize], proj=True, proj_fact=0.125, numthreads=8)[
                "grid"]
            temperature_edge_on_rho = s.get_Aslice("rho", res=res, axes=[1, 0], box=[boxsize, boxsize], proj=True, proj_fact=0.125, numthreads=8)[
                "grid"]

            # Get the magnetic field projections #
            s.data['b2'] = (s.data['bfld'] ** 2.).sum(axis=1)
            bfld_face_on = np.sqrt(
                s.get_Aslice("b2", res=res, axes=[1, 2], box=[boxsize, boxsize], proj=True, proj_fact=0.125, numthreads=8)["grid"] / res) * bfac * 1e6
            bfld_edge_on = np.sqrt(
                s.get_Aslice("b2", res=res, axes=[1, 0], box=[boxsize, boxsize], proj=True, proj_fact=0.125, numthreads=8)["grid"] / res) * bfac * 1e6

            # Get the gas sfr projections #
            sfr_face_on = s.get_Aslice("sfr", res=res, axes=[1, 2], box=[boxsize, boxsize], proj=True, proj_fact=0.125, numthreads=8)["grid"]
            sfr_edge_on = s.get_Aslice("sfr", res=res, axes=[1, 0], box=[boxsize, boxsize], proj=True, proj_fact=0.125, numthreads=8)["grid"]

            # Get the gas total pressure projections #
            elements_mass = [1.01, 4.00, 12.01, 14.01, 16.00, 20.18, 24.30, 28.08, 55.85, 88.91, 87.62, 91.22, 137.33]
            meanweight = np.sum(s.gmet[s.data['type'] == 0, 0:9], axis=1) / (
                np.sum(s.gmet[s.data['type'] == 0, 0:9] / elements_mass[0:9], axis=1) + s.data['ne'] * s.gmet[s.data['type'] == 0, 0])
            Tfac = 1. / meanweight * (1.0 / (5. / 3. - 1.)) * KB / PROTONMASS * 1e10 * msol / 1.989e53

            # Un megabars (10**12dyne/cm**2)
            s.data['T'] = s.u / Tfac
            s.data['dens'] = s.rho / (1e6 * parsec) ** 3. * msol * 1e10
            s.data['Ptherm'] = s.data['dens'] * s.data['T'] / (meanweight * PROTONMASS)

            pressure_face_on = s.get_Aslice("Ptherm", res=res, axes=[1, 2], box=[boxsize, boxsize], proj=True, proj_fact=0.125, numthreads=8)["grid"]
            pressure_edge_on = s.get_Aslice("Ptherm", res=res, axes=[1, 0], box=[boxsize, boxsize], proj=True, proj_fact=0.125, numthreads=8)["grid"]

            # Get the radial velocity projections #
            gas_mask, = np.where(s.data['type'] == 0)
            spherical_radius = np.sqrt(np.sum(s.data['pos'][gas_mask, :] ** 2, axis=1))
            CoM_velocity = np.sum(s.data['vel'][gas_mask, :] * s.data['mass'][gas_mask][:, None], axis=0) / np.sum(s.data['mass'][gas_mask])
            s.data['vrad'] = np.sum((s.data['vel'][gas_mask] - CoM_velocity) * s.data['pos'][gas_mask], axis=1) / spherical_radius

            vrad_face_on = s.get_Aslice("vrad", res=res, axes=[1, 2], box=[boxsize, boxsize], proj=True, proj_fact=0.125, numthreads=8)["grid"]
            vrad_edge_on = s.get_Aslice("vrad", res=res, axes=[1, 0], box=[boxsize, boxsize], proj=True, proj_fact=0.125, numthreads=8)["grid"]

            # Save data for each halo in numpy arrays #
            np.save(path + 'name_' + str(s.haloname), s.haloname)
            np.save(path + 'density_face_on_' + str(s.haloname), density_face_on)
            np.save(path + 'density_edge_on_' + str(s.haloname), density_edge_on)
            np.save(path + 'temperature_face_on_' + str(s.haloname), temperature_face_on)
            np.save(path + 'temperature_edge_on_' + str(s.haloname), temperature_edge_on)
            np.save(path + 'temperature_face_on_rho_' + str(s.haloname), temperature_face_on_rho)
            np.save(path + 'temperature_edge_on_rho_' + str(s.haloname), temperature_edge_on_rho)
            np.save(path + 'bfld_face_on_' + str(s.haloname), bfld_face_on)
            np.save(path + 'bfld_edge_on_' + str(s.haloname), bfld_edge_on)
            np.save(path + 'sfr_face_on_' + str(s.haloname), sfr_face_on)
            np.save(path + 'sfr_edge_on_' + str(s.haloname), sfr_edge_on)
            np.save(path + 'pressure_face_on_' + str(s.haloname), pressure_face_on)
            np.save(path + 'pressure_edge_on_' + str(s.haloname), pressure_edge_on)
            np.save(path + 'vrad_face_on_' + str(s.haloname), vrad_face_on)
            np.save(path + 'vrad_edge_on_' + str(s.haloname), vrad_edge_on)

    # Get the names and sort them #
    names = glob.glob(path + '/name_*')
    names.sort()

    # Loop over all available haloes #
    for i in range(len(names)):
        # Generate the figure and set its parameters #
        figure = plt.figure(figsize=(16, 9))
        axis00, axis01, axis02, axis03, axis04, axis05, axis10, axis11, axis12, axis13, axis14, axis15, axis20, axis21, axis22, axis23, axis24, \
        axis25, x, y, area = plot_tools.create_axes_combinations(
            res=res, boxsize=boxsize * 1e3, multiple=True)
        tick_labels = np.array(['', '-1.5', '', '', '0', '', '', '1.5', ''])
        for axis in [axis00, axis01, axis02, axis03, axis04, axis05]:
            axis.set_xlim(-2, 2)
            axis.set_ylim(-2, 2)
            axis.set_aspect('equal')
            axis.set_xticklabels([])
            axis.tick_params(direction='out', which='both', top='on', right='on')
        for axis in [axis20, axis21, axis22, axis23, axis24, axis25]:
            axis.set_xlim(-2, 2)
            axis.set_ylim(-2, 2)
            axis.set_aspect('equal')
            axis.set_xticklabels(tick_labels)
            axis.set_xlabel(r'$\mathrm{x/kpc}$', size=12)
            axis.tick_params(direction='out', which='both', top='on', right='on')
            for label in axis.xaxis.get_ticklabels():
                label.set_size(12)
        for axis in [axis01, axis21, axis02, axis22, axis03, axis23, axis04, axis24, axis05, axis25]:
            axis.set_yticklabels([])

        axis00.set_ylabel(r'$\mathrm{y/kpc}$', size=12)
        axis20.set_ylabel(r'$\mathrm{z/kpc}$', size=12)
        for axis in [axis00, axis20]:
            for label in axis.yaxis.get_ticklabels():
                label.set_size(12)
                axis.set_yticklabels(tick_labels)

        # Load and plot the data #
        density_face_on = np.load(path + 'density_face_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        density_edge_on = np.load(path + 'density_edge_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        temperature_face_on = np.load(path + 'temperature_face_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        temperature_edge_on = np.load(path + 'temperature_edge_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        temperature_face_on_rho = np.load(path + 'temperature_face_on_rho_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        temperature_edge_on_rho = np.load(path + 'temperature_edge_on_rho_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        bfld_face_on = np.load(path + 'bfld_face_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        bfld_edge_on = np.load(path + 'bfld_edge_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        sfr_face_on = np.load(path + 'sfr_face_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        sfr_edge_on = np.load(path + 'sfr_edge_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        pressure_face_on = np.load(path + 'pressure_face_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        pressure_edge_on = np.load(path + 'pressure_edge_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        vrad_face_on = np.load(path + 'vrad_face_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        vrad_edge_on = np.load(path + 'vrad_edge_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')

        # Plot the gas density projections #
        pcm = axis00.pcolormesh(x, y, density_face_on.T, norm=matplotlib.colors.LogNorm(), cmap='magma', rasterized=True)
        axis20.pcolormesh(x, y, density_edge_on.T, norm=matplotlib.colors.LogNorm(), cmap='magma', rasterized=True)
        plot_tools.create_colorbar(axis10, pcm, "$\mathrm{\Sigma_{gas}/(M_\odot\;kpc^{-2})}$", orientation='horizontal')

        # Plot the gas temperature projections #
        pcm = axis01.pcolormesh(x, y, (temperature_face_on / temperature_face_on_rho).T, norm=matplotlib.colors.LogNorm(), cmap='viridis',
                                rasterized=True)
        axis21.pcolormesh(x, y, (temperature_edge_on / temperature_edge_on_rho).T, norm=matplotlib.colors.LogNorm(), cmap='viridis', rasterized=True)
        plot_tools.create_colorbar(axis11, pcm, "$\mathrm{T/K}$", orientation='horizontal')

        # Plot the magnetic field projections #
        pcm = axis02.pcolormesh(x, y, bfld_face_on.T, norm=matplotlib.colors.LogNorm(), cmap='CMRmap', rasterized=True)
        axis22.pcolormesh(x, y, bfld_edge_on.T, norm=matplotlib.colors.LogNorm(), cmap='CMRmap', rasterized=True)
        plot_tools.create_colorbar(axis12, pcm, "$\mathrm{B/\mu G}$", orientation='horizontal')

        # Plot the sfr projections #
        pcm = axis03.pcolormesh(x, y, sfr_face_on.T, norm=matplotlib.colors.LogNorm(), cmap='gist_heat', rasterized=True)
        axis23.pcolormesh(x, y, sfr_edge_on.T, norm=matplotlib.colors.LogNorm(), cmap='gist_heat', rasterized=True)
        plot_tools.create_colorbar(axis13, pcm, "$\mathrm{SFR/(M_\odot\;yr^{-1})}$", orientation='horizontal')

        # Plot the gas total pressure projections #
        pcm = axis04.pcolormesh(x, y, pressure_face_on.T, norm=matplotlib.colors.LogNorm(), cmap='cividis', rasterized=True)
        axis24.pcolormesh(x, y, pressure_edge_on.T, norm=matplotlib.colors.LogNorm(), cmap='cividis', rasterized=True)
        plot_tools.create_colorbar(axis14, pcm, "$\mathrm{P/(K\;cm^{-3})}$", orientation='horizontal')

        pcm = axis05.pcolormesh(x, y, vrad_face_on.T, cmap='coolwarm', vmin=-7000, vmax=7000, rasterized=True)
        axis25.pcolormesh(x, y, vrad_edge_on.T, cmap='coolwarm', vmin=-7000, vmax=7000, rasterized=True)
        plot_tools.create_colorbar(axis15, pcm, "$\mathrm{Velocity/km\;s^{-1})}$", orientation='horizontal')

        for axis in [axis10, axis11, axis12, axis13, axis14, axis15]:
            axis.xaxis.tick_top()
            for label in axis.xaxis.get_ticklabels():
                label.set_size(12)
            axis.tick_params(direction='out', which='both', top='on', right='on')

        figure.text(0.0, 1.01, 'Au-' + str(re.split('_|.npy', names[i])[1]) + ' redshift = ' + str(redshift), fontsize=12, transform=axis00.transAxes)

        # Save and close the figure #
        pdf.savefig(figure, bbox_inches='tight')
        plt.close()
    return None
