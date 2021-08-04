import os
import re
import glob
import galaxy
import matplotlib
import plot_tools

import numpy as np
import cmasher as cmr
import astropy.units as u
import matplotlib.pyplot as plt

import matplotlib.collections as collections

from const import *
from sfigure import *
from loadmodules import *
from matplotlib.lines import Line2D
from scripts.gigagalaxy.util import plot_helper
from matplotlib.legend_handler import HandlerPatch
from matplotlib.legend_handler import HandlerTuple

style.use("classic")
plt.rcParams.update({'font.family':'serif'})

res = 512
level = 4
boxsize = 0.06
default_level = 4
colors = ['black', 'tab:red', 'tab:green']
colors3 = ['black', 'tab:red', 'tab:green', 'tab:blue']
marker_array = iter(['o', 'o', 'o', '^', '^', '^', 's', 's', 's'])
element = {'H':0, 'He':1, 'C':2, 'N':3, 'O':4, 'Ne':5, 'Mg':6, 'Si':7, 'Fe':8}
colors2 = ['black', 'tab:red', 'tab:green', 'black', 'tab:red', 'tab:green', 'black', 'tab:red', 'tab:green']
haloes_text = [r'$\mathrm{Au-06,\;Au-06NoR,\;Au-06NoRNoQ}$', r'$\mathrm{Au-17,\;Au-17NoR,\;Au-17NoRNoQ}$',
               r'$\mathrm{Au-18,\;Au-18NoR,\;Au-18NoRNoQ}$']
flavours_text = [(r'$\mathrm{Au-06:quasar,\;quasar_{eff}}$', r'$\mathrm{Au-06NoR:quasar,\;quasar_{eff}}$'),
                 (r'$\mathrm{Au-17:quasar,\;quasar_{eff}}$', r'$\mathrm{Au-17NoR:quasar,\;quasar_{eff}}$'),
                 (r'$\mathrm{Au-18:quasar,\;quasar_{eff}}$', r'$\mathrm{Au-18NoR:quasar,\;quasar_{eff}}$')]

# Create symbols for each halo #
circles = collections.CircleCollection([10], facecolors=colors, edgecolors='none')
squares = collections.RegularPolyCollection(numsides=4, rotation=np.pi / 4, sizes=(20,), facecolors=colors,
                                            edgecolors='none')
triangles = collections.RegularPolyCollection(numsides=3, sizes=(20,), facecolors=colors, edgecolors='none')
custom_lines = [(Line2D([0], [0], color=colors[0], lw=3, ls=':'), Line2D([0], [0], color=colors[0], lw=3, ls='-')),
                (Line2D([0], [0], color=colors[1], lw=3, ls=':'), Line2D([0], [0], color=colors[1], lw=3, ls='-'))]


def stellar_light_combination(pdf, redshift):
    """
    Plot a combination of the stellar light projections for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :return: None
    """
    print("Invoking stellar_light_combination")
    # Get the names and sort them #
    path = '/u/di43/Auriga/plots/data/' + 'sl/' + str(redshift) + '/'
    names = glob.glob(path + 'name_*')
    names.sort()

    # Generate the figure and set its parameters #
    figure = plt.figure(figsize=(10, 15))
    axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22, axis30, axis31, axis32, axis40, axis41, \
    axis42, axis50, axis51, axis52 = plot_tools.create_axes_combinations(
        res=res, boxsize=boxsize * 1e3, multiple2=True)
    for axis in [axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22, axis30, axis31, axis32, axis40,
                 axis41, axis42, axis50, axis51, axis52]:
        axis.set_yticklabels([])
        axis.set_xticklabels([])
    plt.rcParams['savefig.facecolor'] = 'black'

    # Loop over all available haloes #
    axes_face_on = [axis00, axis01, axis02, axis20, axis21, axis22, axis40, axis41, axis42]
    axes_edge_on = [axis10, axis11, axis12, axis30, axis31, axis32, axis50, axis51, axis52]
    for i, axis_face_on, axis_edge_on in zip(range(len(names)), axes_face_on, axes_edge_on):
        print("Plotting data for halo:", str(re.split('_|.npy', names[i])[1]))
        # Load the data #
        face_on = np.load(path + 'face_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        edge_on = np.load(path + 'edge_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')

        # Plot the stellar light projections #
        axis_face_on.imshow(face_on, interpolation='nearest', aspect='equal')
        axis_edge_on.imshow(edge_on, interpolation='nearest', aspect='equal')

        figure.text(0.01, 0.95, r'$\mathrm{Au-%s}$' % str(re.split('_|.npy', names[i])[1]), color='w', fontsize=20,
                    transform=axis_face_on.transAxes)

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
    path = '/u/di43/Auriga/plots/data/' + 'sl/' + str(redshift) + '/'
    path_components = '/u/di43/Auriga/plots/data/' + 'slc/' + str(redshift) + '/'
    names = glob.glob(path + 'name_06*')
    names.sort()

    # Generate the figure and set its parameters #
    figure = plt.figure(figsize=(10, 20))
    axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22, axis30, axis31, axis32, axis40, axis41, \
    axis42, axis50, axis51, axis52 = plot_tools.create_axes_combinations(
        res=res, boxsize=boxsize * 1e3, multiple8=True)
    for axis in [axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22, axis30, axis31, axis32, axis40,
                 axis41, axis42, axis50, axis51, axis52]:
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
    for i, axis_face_on, axis_edge_on, axis_disc_face_on, axis_disc_edge_on, axis_spheroid_face_on, \
        axis_spheroid_edge_on in zip(
        range(len(names)), axes_face_on, axes_edge_on, axes_disc_face_on, axes_disc_edge_on, axes_spheroid_face_on,
        axes_spheroid_edge_on):
        print("Plotting data for halo:", str(re.split('_|.npy', names[i])[1]))
        # Load the data #
        face_on = np.load(path + 'face_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        edge_on = np.load(path + 'edge_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        disc_face_on = np.load(path_components + 'disc_face_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        disc_edge_on = np.load(path_components + 'disc_edge_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        spheroid_face_on = np.load(
            path_components + 'spheroid_face_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        spheroid_edge_on = np.load(
            path_components + 'spheroid_edge_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')

        # Plot the stellar light projections of the disc and spheroid component #
        axis_face_on.imshow(face_on, interpolation='nearest', aspect='equal')
        axis_edge_on.imshow(edge_on, interpolation='nearest', aspect='equal')
        axis_disc_face_on.imshow(disc_face_on, interpolation='nearest', aspect='equal')
        axis_disc_edge_on.imshow(disc_edge_on, interpolation='nearest', aspect='equal')
        axis_spheroid_face_on.imshow(spheroid_face_on, interpolation='nearest', aspect='equal')
        axis_spheroid_edge_on.imshow(spheroid_edge_on, interpolation='nearest', aspect='equal')

        figure.text(0.01, 0.95, r'$\mathrm{Au-%s}$' % str(re.split('_|.npy', names[i])[1]), color='w', fontsize=20,
                    transform=axis_face_on.transAxes)
        figure.text(0.01, 0.95, r'$\mathrm{Disc}$', fontsize=20, color='w', transform=axis_disc_face_on.transAxes)
        figure.text(0.01, 0.95, r'$\mathrm{Spheroid}$', fontsize=20, color='w',
                    transform=axis_spheroid_face_on.transAxes)
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
    path = '/u/di43/Auriga/plots/data/' + 'sd/' + str(redshift) + '/'
    names = glob.glob(path + 'name_*')
    names.sort()

    # Generate the figure and set its parameters #
    figure = plt.figure(figsize=(15, 20))
    axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22, axis30, axis31, axis32, axis40, axis41, \
    axis42, axis50, axis51, axis52, axiscbar, x, y, y2, area = plot_tools.create_axes_combinations(
        res=res, boxsize=boxsize * 1e3, multiple3=True)
    cmap = matplotlib.cm.get_cmap('cmr.eclipse')
    for axis in [axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22, axis30, axis31, axis32, axis40,
                 axis41, axis42, axis50, axis51, axis52]:
        axis.set_facecolor(cmap(0))
        plot_tools.set_axes(axis, xlim=[-30, 30], ylim=[-30, 30], aspect=None, size=25)
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
        axis.set_xticklabels(['', '-20', '', '0', '', '20', ''])
        axis.set_xlabel(r'$\mathrm{x/kpc}$', size=25)
    for axis in [axis00, axis20, axis40]:
        axis.set_yticklabels(['', '-20', '', '0', '', '20', ''])
        axis.set_ylabel(r'$\mathrm{y/kpc}$', size=25)
    for axis in [axis10, axis30, axis50]:
        axis.set_yticklabels(['', '-10', '', '0', '', '10', ''])
        axis.set_ylabel(r'$\mathrm{z/kpc}$', size=25)

    # Loop over all available haloes #
    axes_face_on = [axis00, axis01, axis02, axis20, axis21, axis22, axis40, axis41, axis42]
    axes_edge_on = [axis10, axis11, axis12, axis30, axis31, axis32, axis50, axis51, axis52]
    for i, axis_face_on, axis_edge_on in zip(range(len(names)), axes_face_on, axes_edge_on):
        print("Plotting data for halo:", str(re.split('_|.npy', names[i])[1]))
        # Load the data #
        face_on = np.load(path + 'face_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        edge_on = np.load(path + 'edge_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')

        # Plot the stellar density projections #
        pcm = axis_face_on.pcolormesh(x, y, face_on, norm=matplotlib.colors.LogNorm(vmin=1e6, vmax=1e10),
                                      rasterized=True, cmap=cmap)
        axis_edge_on.pcolormesh(x, y2, edge_on, norm=matplotlib.colors.LogNorm(vmin=1e6, vmax=1e10), rasterized=True,
                                cmap=cmap)
        plot_tools.create_colorbar(axiscbar, pcm, label=r'$\mathrm{\Sigma_{\bigstar}/(M_\odot\;kpc^{-2})}$', size=25)

        figure.text(0.01, 0.9, r'$\mathrm{Au-%s}$' % str(re.split('_|.npy', names[i])[1]), fontsize=25, color='w',
                    transform=axis_face_on.transAxes)

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
    path = '/u/di43/Auriga/plots/data/' + 'gd/' + str(redshift) + '/'
    names = glob.glob(path + 'name_*')
    names.sort()

    # Generate the figure and set its parameters #
    figure = plt.figure(figsize=(15, 15))
    axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22, axiscbar, x, y, y2, \
    area = plot_tools.create_axes_combinations(
        res=res, boxsize=boxsize * 1e3, multiple12=True)
    for axis in [axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22]:
        plot_tools.set_axes(axis, xlim=[-15, 15], ylim=[-15, 15], aspect=None, size=30)
    for axis in [axis00, axis01, axis02, axis10, axis11, axis12]:
        axis.set_xticklabels([])
    for axis in [axis01, axis02, axis11, axis12, axis21, axis22]:
        axis.set_yticklabels([])
    for axis in [axis20, axis21, axis22]:
        axis.set_xticklabels(['', '-10', '', '0', '', '10', ''])
        axis.set_xlabel(r'$\mathrm{x/kpc}$', size=30)
    for axis in [axis00, axis10, axis20]:
        axis.set_yticklabels(['', '-10', '', '0', '', '10', ''])
        axis.set_ylabel(r'$\mathrm{y/kpc}$', size=30)

    # Loop over all available haloes #
    axes = [axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22]
    for i, axis in zip(range(len(names)), axes):
        print("Plotting data for halo:", str(re.split('_|.npy', names[i])[1]))
        # Load the data #
        face_on = np.load(path + 'face_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')

        # Plot the gas density projections #
        pcm = axis.pcolormesh(x, y, face_on.T, norm=matplotlib.colors.LogNorm(vmin=1e7, vmax=1e10), rasterized=True,
                              cmap='inferno')
        plot_tools.create_colorbar(axiscbar, pcm, label='$\mathrm{\Sigma_{gas}/(M_\odot\;kpc^{-2})}$', size=30)

        figure.text(0.01, 0.9, r'$\mathrm{Au-%s}$' % str(re.split('_|.npy', names[i])[1]), fontsize=30, color='w',
                    transform=axis.transAxes)

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
    path = '/u/di43/Auriga/plots/data/' + 'gt/' + str(redshift) + '/'
    names = glob.glob(path + 'name_*')
    names.sort()

    # Generate the figure and set its parameters #
    figure = plt.figure(figsize=(10, 15))
    axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22, axis30, axis31, axis32, axis40, axis41, \
    axis42, axis50, axis51, axis52, axiscbar, x, y, y2, area = plot_tools.create_axes_combinations(
        res=res, boxsize=boxsize * 1e3, multiple3=True)
    for axis in [axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22, axis30, axis31, axis32, axis40,
                 axis41, axis42, axis50, axis51, axis52]:
        plot_tools.set_axes(axis, xlim=[-30, 30], ylim=[-30, 30], size=20)
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
        axis.set_xlabel(r'$\mathrm{x/kpc}$', size=20)
    for axis in [axis00, axis20, axis40]:
        axis.set_ylabel(r'$\mathrm{y/kpc}$', size=20)
    for axis in [axis10, axis30, axis50]:
        axis.set_ylabel(r'$\mathrm{z/kpc}$', size=20)

    # Loop over all available haloes #
    axes_face_on = [axis00, axis20, axis40, axis01, axis21, axis41, axis02, axis22, axis42]
    axes_edge_on = [axis10, axis30, axis50, axis11, axis31, axis51, axis12, axis32, axis52]
    for i, axis_face_on, axis_edge_on in zip(range(len(names)), axes_face_on, axes_edge_on):
        print("Plotting data for halo:", str(re.split('_|.npy', names[i])[1]))
        # Load the data #
        face_on = np.load(path + 'face_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        edge_on = np.load(path + 'edge_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        face_on_rho = np.load(path + 'face_on_rho_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        edge_on_rho = np.load(path + 'edge_on_rho_' + str(re.split('_|.npy', names[i])[1]) + '.npy')

        # Plot the density-weighted gas temperature projections #
        pcm = axis_face_on.pcolormesh(x, y, (face_on / face_on_rho).T,
                                      norm=matplotlib.colors.LogNorm(vmin=1e3, vmax=1e7), rasterized=True,
                                      cmap='viridis')
        axis_edge_on.pcolormesh(x, 0.5 * y, (edge_on / edge_on_rho).T,
                                norm=matplotlib.colors.LogNorm(vmin=1e3, vmax=1e7), rasterized=True, cmap='viridis')
        plot_tools.create_colorbar(axiscbar, pcm, label=r'$\mathrm{T/K}$')

        figure.text(0.01, 0.95, r'$\mathrm{Au-%s}$' % str(re.split('_|.npy', names[i])[1]), fontsize=20,
                    transform=axis_face_on.transAxes)

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
    path = '/u/di43/Auriga/plots/data/' + 'gm/' + str(redshift) + '/'
    names = glob.glob(path + 'name_*')
    names.sort()

    # Generate the figure and set its parameters #
    figure = plt.figure(figsize=(10, 15))
    axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22, axis30, axis31, axis32, axis40, axis41, \
axis42, axis50, axis51, axis52, axiscbar, x, y, y2, area = plot_tools.create_axes_combinations(
        res=res, boxsize=boxsize * 1e3, multiple3=True)
    for axis in [axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22, axis30, axis31, axis32, axis40,
                 axis41, axis42, axis50, axis51, axis52]:
        plot_tools.set_axes(axis, xlim=[-30, 30], ylim=[-30, 30], size=20)
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
        axis.set_xlabel(r'$\mathrm{x/kpc}$', size=20)
    for axis in [axis00, axis20, axis40]:
        axis.set_ylabel(r'$\mathrm{y/kpc}$', size=20)
    for axis in [axis10, axis30, axis50]:
        axis.set_ylabel(r'$\mathrm{z/kpc}$', size=20)

    # Loop over all available haloes #
    axes_face_on = [axis00, axis20, axis40, axis01, axis21, axis41, axis02, axis22, axis42]
    axes_edge_on = [axis10, axis30, axis50, axis11, axis31, axis51, axis12, axis32, axis52]
    for i, axis_face_on, axis_edge_on in zip(range(len(names)), axes_face_on, axes_edge_on):
        print("Plotting data for halo:", str(re.split('_|.npy', names[i])[1]))
        # Load the data #
        face_on = np.load(path + 'face_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        edge_on = np.load(path + 'edge_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')

        # Plot the gas metallicity projections #
        pcm = axis_face_on.pcolormesh(x, y, face_on.T, norm=matplotlib.colors.LogNorm(vmin=0.3, vmax=3.),
                                      rasterized=True, cmap='viridis')
        axis_edge_on.pcolormesh(x, 0.5 * y, edge_on.T, norm=matplotlib.colors.LogNorm(vmin=0.3, vmax=3.),
                                rasterized=True, cmap='viridis')
        plot_tools.create_colorbar(axiscbar, pcm, label=r'$\mathrm{Z/Z_\odot}$')

        figure.text(0.01, 0.95, r'$\mathrm{Au-%s}$' % str(re.split('_|.npy', names[i])[1]), fontsize=20,
                    transform=axis_face_on.transAxes)

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
    path = '/u/di43/Auriga/plots/data/' + 'mf/' + str(redshift) + '/'
    names = glob.glob(path + 'name_*')
    names.sort()

    # Generate the figure and set its parameters #
    figure = plt.figure(figsize=(10, 15))
    axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22, axis30, axis31, axis32, axis40, axis41, \
    axis42, axis50, axis51, axis52, axiscbar, x, y, y2, area = plot_tools.create_axes_combinations(
        res=res, boxsize=boxsize * 1e3, multiple3=True)
    for axis in [axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22, axis30, axis31, axis32, axis40,
                 axis41, axis42, axis50, axis51, axis52]:
        plot_tools.set_axes(axis, xlim=[-30, 30], ylim=[-30, 30], size=20)
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
        axis.set_xlabel(r'$\mathrm{x/kpc}$', size=20)
    for axis in [axis00, axis20, axis40]:
        axis.set_ylabel(r'$\mathrm{y/kpc}$', size=20)
    for axis in [axis10, axis30, axis50]:
        axis.set_ylabel(r'$\mathrm{z/kpc}$', size=20)

    # Loop over all available haloes #
    axes_face_on = [axis00, axis20, axis40, axis01, axis21, axis41, axis02, axis22, axis42]
    axes_edge_on = [axis10, axis30, axis50, axis11, axis31, axis51, axis12, axis32, axis52]
    for i, axis_face_on, axis_edge_on in zip(range(len(names)), axes_face_on, axes_edge_on):
        print("Plotting data for halo:", str(re.split('_|.npy', names[i])[1]))
        # Load the data #
        face_on = np.load(path + 'face_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        edge_on = np.load(path + 'edge_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')

        # Plot the magnetic field projections #
        pcm = axis_face_on.pcolormesh(x, y, face_on.T, norm=matplotlib.colors.LogNorm(vmin=1e-1, vmax=1e2),
                                      rasterized=True, cmap='CMRmap')
        axis_edge_on.pcolormesh(x, 0.5 * y, edge_on.T, norm=matplotlib.colors.LogNorm(vmin=1e-1, vmax=1e2),
                                rasterized=True, cmap='CMRmap')
        plot_tools.create_colorbar(axiscbar, pcm, label='$\mathrm{B/\mu G}$')

        figure.text(0.01, 0.95, r'$\mathrm{Au-%s}$' % str(re.split('_|.npy', names[i])[1]), fontsize=20,
                    transform=axis_face_on.transAxes)

    # Save and close the figure #
    pdf.savefig(figure, bbox_inches='tight')
    plt.close()
    return None


def circularity_distribution_combination(pdf, redshift):
    """
    Plot a combination of the orbital circularity distribution for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :return: None
    """
    print("Invoking circularity_distribution_combination")
    # Get the names and sort them #
    path = '/u/di43/Auriga/plots/data/' + 'cd/' + str(redshift) + '/'
    names = glob.glob(path + 'name_*')
    names.sort()

    # Generate the figure and set its parameters #
    figure = plt.figure(figsize=(20, 7.5))
    axis00, axis01, axis02 = plot_tools.create_axes_combinations(res=res, boxsize=boxsize * 1e3, multiple4=True)
    plot_tools.set_axes(axis00, xlim=[-1.7, 1.7], ylim=[0, 5], xlabel=r'$\mathrm{\epsilon}$',
                        ylabel=r'$\mathrm{f(\epsilon)}$', aspect=None)
    plot_tools.set_axes(axis01, xlim=[-1.7, 1.7], ylim=[0, 5], xlabel=r'$\mathrm{\epsilon}$', aspect=None)
    plot_tools.set_axes(axis02, xlim=[-1.7, 1.7], ylim=[0, 5], xlabel=r'$\mathrm{\epsilon}$', aspect=None)
    axis01.set_yticklabels([])
    axis02.set_yticklabels([])

    # Split the names into 3 groups and plot the three flavours of a halo together (i.e. original, NoR and NoRNoQ) #
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
            y_data, edges = np.histogram(epsilon, weights=stellar_masses / np.sum(stellar_masses), bins=100,
                                         range=[-1.7, 1.7])
            y_data /= edges[1:] - edges[:-1]
            axis.plot(0.5 * (edges[1:] + edges[:-1]), y_data, color=colors[j], label=r'$\mathrm{Au-%s:D/T=%.2f}$' % (
                str(re.split('_|.npy', names_flavours[j])[1]), disc_fraction))

            axis.legend(loc='upper left', fontsize=16, frameon=False)  # Create the legend.
    # Save and close the figure #
    pdf.savefig(figure, bbox_inches='tight')
    plt.close()
    return None


def tully_fisher_combination(pdf, redshift):
    """
    Plot a combination of the Tully-Fisher relation for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :return: None
    """
    print("Invoking tully_fisher_combination")
    # Get the names and sort them #
    path = '/u/di43/Auriga/plots/data/' + 'tf/' + str(redshift) + '/'
    names = glob.glob(path + 'name_*')
    names.sort()

    # Generate the figure and set its parameters #
    figure, axis = plt.subplots(1, figsize=(10, 10))
    plot_tools.set_axes(axis, xlim=[1e10, 1e12], ylim=[2.0, 2.7], xscale='log',
                        xlabel=r'$\mathrm{M_{\bigstar}/M_{\odot}}$',
                        ylabel=r'$\mathrm{log_{10}(v_{circ}/(km\;s^{-1}))}$', aspect=None)

    # Plot Pizagno et al. 2007 sample #
    table = "./data/pizagno.txt"
    rmag_p = np.genfromtxt(table, comments='#', usecols=3)
    vcirc_p = np.genfromtxt(table, comments='#', usecols=9)
    color_p = np.genfromtxt(table, comments='#', usecols=5)
    mass_p = galaxy.pizagno_convert_color_to_mass(color_p, rmag_p)
    Pizagno = plt.scatter(1e10 * mass_p, np.log10(vcirc_p), color='grey', s=50, marker='X')

    # Plot Verheijen 2001 sample #
    table = "./data/verheijen.txt"
    Bmag_v = np.genfromtxt(table, comments='#', usecols=1)
    Rmag_v = np.genfromtxt(table, comments='#', usecols=2)
    vcirc_v = np.genfromtxt(table, comments='#', usecols=7)
    color_v = Bmag_v - Rmag_v
    mass_v = galaxy.verheijen_convert_color_to_mass(color_v, Bmag_v)
    Verheijen = plt.scatter(1e10 * mass_v, np.log10(vcirc_v), color='grey', s=50, marker='P')

    # Plot Courteau et al. 2007 sample #
    table = "./data/courteau.txt"
    loglum_c = np.genfromtxt(table, comments='#', usecols=6)
    vcirc_c = np.genfromtxt(table, comments='#', usecols=8)
    mass_c = galaxy.courteau_convert_luminosity_to_mass(loglum_c)
    Courteau = plt.scatter(1e10 * mass_c, vcirc_c, color='grey', s=25, marker='D')

    # Plot best fit from Dutton et al. 2011 #
    masses = np.arange(0.1, 50.0)
    Dutton, = plt.plot(1e10 * masses, np.log10(galaxy.obs_tullyfisher_fit(masses)), color=colors[0], lw=3, ls='-')

    # Create the legend #
    legend = plt.legend([Pizagno, Verheijen, Courteau, Dutton],
                        [r'$\mathrm{Pizagno+\;07}$', r'$\mathrm{Verheijen\;01}$', r'$\mathrm{Courteau+\;07}$',
                         r'$\mathrm{Dutton+\;11}$'], loc='lower right', fontsize=20, markerscale=1, scatterpoints=1,
                        frameon=False, scatteryoffsets=[0.5])
    axis.add_artist(legend)

    legend = plt.legend([circles, squares, triangles], haloes_text, loc='upper left', fontsize=20, markerscale=3,
                        frameon=False, scatteryoffsets=[0.5], handlelength=len(haloes_text))
    axis.add_artist(legend)

    # Loop over all available haloes #
    for i in range(len(names)):
        print("Plotting data for halo:", str(re.split('_|.npy', names[i])[1]))
        # Load the data #
        stellar_mass = np.load(path + 'stellar_mass_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        total_circular_velocity = np.load(
            path + 'total_circular_velocity_' + str(re.split('_|.npy', names[i])[1]) + '.npy')

        # Plot the Tully-Fisher relation #
        plt.scatter(stellar_mass * 1e10, np.log10(total_circular_velocity), color=colors2[i], s=100,
                    marker=next(marker_array))

    # Save and close the figure #
    pdf.savefig(figure, bbox_inches='tight')
    plt.close()
    return None


def stellar_vs_halo_mass_combination(pdf, redshift):
    """
    Plot a combination of the abundance matching relation for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :return: None
    """
    print("Invoking stellar_vs_halo_mass_combination")
    # Get the names and sort them #
    path = '/u/di43/Auriga/plots/data/' + 'svt/' + str(redshift) + '/'
    names = glob.glob(path + 'name_*')
    names.sort()

    # Generate the figure and set its parameters #
    figure, axis = plt.subplots(1, figsize=(10, 10))
    plot_tools.set_axes(axis, xlim=[1.5e11, 2e12], ylim=[2e10, 5e11], xscale='log', yscale='log',
                        xlabel=r'$\mathrm{M_{halo}/M_{\odot}}$', ylabel=r'$\mathrm{M_{\bigstar}/M_{\odot}}$',
                        aspect=None, which='major', size=35)

    # Plot the cosmic baryon fraction relation #
    masses = np.arange(15., 300.)
    cosmic_baryon_frac = 0.048 / 0.307
    omega, = plt.plot(1e10 * masses, 1e10 * masses * cosmic_baryon_frac, color='k', lw=3, ls='--')

    # Plot the Guo+10 relation #
    guo_high = galaxy.guo_abundance_matching(masses) * 10 ** (+0.2)
    guo_low = galaxy.guo_abundance_matching(masses) * 10 ** (-0.2)
    plt.fill_between(1e10 * masses, 1e10 * guo_low, 1e10 * guo_high, color='lightgray', edgecolor='None')
    Guo, = plt.plot(1e10 * masses, 1e10 * galaxy.guo_abundance_matching(masses), color='k', lw=3, ls=':')

    # Create the legend #
    legend = plt.legend([omega, Guo], [r'$\mathrm{M_{200}\;\Omega_b/\Omega_m}$', r'$\mathrm{Guo+\;10}$'], fontsize=25,
                        loc='center left', frameon=False)
    axis.add_artist(legend)

    legend = plt.legend([circles, squares, triangles], haloes_text, fontsize=25, loc='upper left', markerscale=4,
                        frameon=False, scatteryoffsets=[0.5], handlelength=len(haloes_text))
    axis.add_artist(legend)

    # Loop over all available haloes #
    for i in range(len(names)):
        print("Plotting data for halo:", str(re.split('_|.npy', names[i])[1]))
        # Load the data #
        halo_mass = np.load(path + 'halo_mass_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        stellar_mass = np.load(path + 'stellar_mass_' + str(re.split('_|.npy', names[i])[1]) + '.npy')

        # Plot the abundance matching relation #
        plt.scatter(halo_mass * 1e10, stellar_mass * 1e10, color=colors2[i], s=250, marker=next(marker_array))

    # Save and close the figure #
    pdf.savefig(figure, bbox_inches='tight')
    plt.close()
    return None


def gas_fraction_vs_magnitude_combination(pdf, redshift):
    """
    Plot a combination of the gas fraction (gas to stellar plus gas mass ratio) as a function R-band magnitude for
    Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :return: None
    """
    print("Invoking gas_fraction_vs_magnitude_combination")
    # Get the names and sort them #
    path = '/u/di43/Auriga/plots/data/' + 'gfvm/' + str(redshift) + '/'
    names = glob.glob(path + 'name_*')
    names.sort()

    # Generate the figure and set its parameters #
    figure, axis = plt.subplots(1, figsize=(10, 7.5))
    plot_tools.set_axes(axis, xlim=[-23.2, -22], ylim=[0.1, 0.4], xlabel=r'$\mathrm{M_{R}/mag}$',
                        ylabel=r'$\mathrm{f_{gas}}$', aspect=None)

    # Loop over all available haloes #
    for i in range(len(names)):
        print("Plotting data for halo:", str(re.split('_|.npy', names[i])[1]))
        # Load the data #
        gas_fraction = np.load(path + 'gas_fraction_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        M_R = np.load(path + 'M_R_' + str(re.split('_|.npy', names[i])[1]) + '.npy')

        # Plot the gas fraction as a function R-band magnitude #
        plt.scatter(M_R, gas_fraction, color=colors2[i], s=100, marker=next(marker_array),
                    label=r'$\mathrm{Au-%s}$' % str(re.split('_|.npy', names[i])[1]))

        plt.legend(loc='upper center', fontsize=16, frameon=False, scatterpoints=1, ncol=3)  # Create the legend.

    # Save and close the figure #
    pdf.savefig(figure, bbox_inches='tight')
    plt.close()
    return None


def bar_strength_profile_combination(pdf, redshift):
    """
    Plot a combination of the bar strength radial profile from Fourier modes of surface density for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :return: None
    """
    print("Invoking bar_strength_profile_combination")
    # Get the names and sort them #
    path = '/u/di43/Auriga/plots/data/' + 'bsp/' + str(redshift) + '/'
    names = glob.glob(path + 'name_*')
    names.sort()

    # Generate the figure and set its parameters #
    figure = plt.figure(figsize=(15, 5))
    axis00, axis01, axis02 = plot_tools.create_axes_combinations(res=res, boxsize=boxsize * 1e3, multiple4=True)
    for axis in [axis00, axis01, axis02]:
        plot_tools.set_axes(axis, xlim=[0, 11], ylim=[0, 1.1], xlabel=r'$\mathrm{R/kpc}$',
                            ylabel=r'$\mathrm{\sqrt{a_{2}^{2}+b_{2}^{2}}/a_{0}}$', aspect=None)
    for axis in [axis01, axis02]:
        axis.set_ylabel('')
        axis.set_yticklabels([])

    # Split the names into 3 groups and plot the three flavours of a halo together (i.e. original, NoR and NoRNoQ) #
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
            axis.plot(r_m, ratio, color=colors[j], lw=3, label=r'$\mathrm{Au-%s:r_{A_{2}}=%.2fkpc}$' % (
                str(re.split('_|.npy', names_flavours[j])[1]), r_m[np.where(ratio == A2)]))
            axis.plot([r_m[np.where(ratio == A2)], r_m[np.where(ratio == A2)]], [-0.0, A2], color=colors[j], lw=3,
                      linestyle='dashed')

            axis.legend(loc='upper left', fontsize=15, frameon=False)  # Create the legend.
    # Save and close the figure #
    pdf.savefig(figure, bbox_inches='tight')
    plt.close()
    return None


def stellar_surface_density_profiles_combination(pdf, redshift):
    """
    Plot a combination of the stellar surface density profiles for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :return: None
    """
    print("Invoking stellar_surface_density_profiles_combination")
    # Get the names and sort them #
    path = '/u/di43/Auriga/plots/data/' + 'ssdp/' + str(redshift) + '/'
    names = glob.glob(path + 'name_*')
    names.sort()

    # Generate the figure and set its parameters #
    figure = plt.figure(figsize=(15, 15))
    axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22 = plot_tools.create_axes_combinations(
        res=res, boxsize=boxsize * 1e3, multiple5=True)
    for axis in [axis00, axis10, axis20]:
        plot_tools.set_axes(axis, xlim=[0, 24], ylim=[1e0, 9e4], yscale='log', xlabel=r'$\mathrm{R/kpc}$',
                            ylabel=r'$\mathrm{\Sigma_{\bigstar}/(M_{\odot}\;pc^{-2})}$', aspect=None, which='major',
                            size=20)
    for axis in [axis01, axis02, axis11, axis12, axis21, axis22]:
        plot_tools.set_axes(axis, xlim=[0, 24], ylim=[1e0, 9e4], yscale='log', xlabel=r'$\mathrm{R/kpc}$', aspect=None,
                            which='major', size=20)
        axis.set_yticklabels([])
    for axis in [axis00, axis01, axis02, axis10, axis11, axis12]:
        axis.set_xticklabels([])

    # Loop over all available haloes #
    for i, axis in zip(range(len(names)), [axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22]):
        print("Plotting data for halo:", str(re.split('_|.npy', names[i])[1]))
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
        axis.axvline(rfit, color='gray', linestyle='--', lw=3)
        axis.scatter(r, 1e10 * sdfit * 1e-6, marker='o', s=15, color=colors3[0], linewidth=0.0)
        axis.plot(r, 1e10 * p.exp_prof(r, popt0, popt1) * 1e-6, color=colors3[3], lw=3)
        axis.plot(r, 1e10 * p.sersic_prof1(r, popt2, popt3, popt4) * 1e-6, color=colors3[1], lw=3)
        axis.plot(r, 1e10 * p.total_profile(r, popt0, popt1, popt2, popt3, popt4) * 1e-6, color=colors3[0], lw=3)

        figure.text(0.01, 0.9, r'$\mathrm{Au-%s}$' % str(re.split('_|.npy', names[i])[1]), fontsize=20,
                    transform=axis.transAxes)
        figure.text(0.3, 0.5, r'$\mathrm{R_{eff.}} = %.2f$ kpc' '\n' % (
        popt3 * p.sersic_b_param(1.0 / popt4) ** (1.0 / popt4)), fontsize=20, transform=axis.transAxes)

    # Save and close the figure #
    pdf.savefig(figure, bbox_inches='tight')
    plt.close()
    return None


def circular_velocity_curves_combination(pdf, redshift):
    """
    Plot a combination of the circular velocity curve for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :return: None
    """
    print("Invoking circular_velocity_curves_combination")
    # Get the names and sort them #
    path = '/u/di43/Auriga/plots/data/' + 'cvc/' + str(redshift) + '/'
    names = glob.glob(path + 'name_*')
    names.sort()

    # Generate the figure and set its parameters #
    figure = plt.figure(figsize=(20, 20))
    axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22 = plot_tools.create_axes_combinations(
        res=res, boxsize=boxsize * 1e3, multiple5=True)
    for axis in [axis00, axis10, axis20]:
        plot_tools.set_axes(axis, xlim=[0, 24], ylim=[0, 700], xlabel=r'$\mathrm{R/kpc}$',
                            ylabel=r'$\mathrm{V_{circular}/(km\;s^{-1})}$', aspect=None)
    for axis in [axis01, axis02, axis11, axis12, axis21, axis22]:
        plot_tools.set_axes(axis, xlim=[0, 24], ylim=[0, 700], xlabel=r'$\mathrm{R/kpc}$', aspect=None)
        axis.set_yticklabels([])
    for axis in [axis00, axis01, axis02, axis10, axis11, axis12]:
        axis.set_xticklabels([])

    # Loop over all available haloes #
    for i, axis in zip(range(len(names)), [axis00, axis10, axis20, axis01, axis11, axis21, axis02, axis12, axis22]):
        print("Plotting data for halo:", str(re.split('_|.npy', names[i])[1]))
        # Load the data #
        radius = np.load(path + 'radius_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        total_mass = np.load(path + 'total_mass_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        shell_velocity = np.load(path + 'shell_velocity_' + str(re.split('_|.npy', names[i])[1]) + '.npy')

        # Plot the circular velocity curve #
        vtot = np.sqrt(G * total_mass * 1e10 * msol / (radius * 1e6 * parsec)) / 1e5
        axis.plot(radius * 1e3, vtot, color=colors3[0], lw=3, label=r'$\mathrm{Total}$')
        axis.plot(radius * 1e3, shell_velocity[:, 0], color=colors3[3], linestyle='--', lw=3, label=r'$\mathrm{Gas}$')
        axis.plot(radius * 1e3, shell_velocity[:, 4], color=colors3[2], linestyle='--', lw=3, label=r'$\mathrm{Stars}$')
        axis.plot(radius * 1e3, shell_velocity[:, 1], color=colors3[1], linestyle='--', lw=3,
                  label=r'$\mathrm{Dark\;matter}$')

        figure.text(0.01, 0.95, r'$\mathrm{Au-%s}$' % str(re.split('_|.npy', names[i])[1]), fontsize=20,
                    transform=axis.transAxes)

    # Save and close the figure #
    pdf.savefig(figure, bbox_inches='tight')
    plt.close()
    return None


def ssdp_cvc_combination(pdf, redshift):
    """
    Plot a combination of the circular velocity curve and the stellar surface density profiles for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :return: None
    """
    print("Invoking ssdp_cvc_combination")
    # Get the names and sort them #
    path_cvc = '/u/di43/Auriga/plots/data/' + 'cvc/' + str(redshift) + '/'
    path_ssdp = '/u/di43/Auriga/plots/data/' + 'ssdp/' + str(redshift) + '/'
    names = glob.glob(path_cvc + 'name_*')
    names.sort()

    # Generate the figure and set its parameters #
    figure = plt.figure(figsize=(20, 20))
    axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22, axis30, axis31, axis32, axis40, axis41, \
    axis42, axis50, axis51, axis52 = plot_tools.create_axes_combinations(
        res=res, boxsize=boxsize * 1e3, multiple9=True)

    for axis in [axis00, axis20, axis40]:
        plot_tools.set_axes(axis, xlim=[0, 24], ylim=[0, 6],
                            ylabel=r'$\mathrm{log_{10}(\Sigma_{\bigstar}/(M_{\odot}\;pc^{-2}))}$', aspect=None,
                            which='major', size=25)
    for axis in [axis01, axis02, axis21, axis22, axis41, axis42]:
        plot_tools.set_axes(axis, xlim=[0, 24], ylim=[0, 6], aspect=None, which='major', size=25)
        axis.set_yticklabels([])

    for axis in [axis10, axis30, axis50]:
        plot_tools.set_axes(axis, xlim=[0, 24], ylim=[0, 580], xlabel=r'$\mathrm{R/kpc}$',
                            ylabel=r'$\mathrm{V_{c}/(km\;s^{-1})}$', aspect=None, size=25)
    for axis in [axis11, axis12, axis31, axis32, axis51, axis52]:
        plot_tools.set_axes(axis, xlim=[0, 24], ylim=[0, 580], xlabel=r'$\mathrm{R/kpc}$', aspect=None, size=25)
        axis.set_yticklabels([])
    for axis in [axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22, axis30, axis31, axis32, axis40,
                 axis41, axis42]:
        axis.set_xlabel('')
        axis.set_xticklabels([])
    for axis in [axis00, axis01, axis02, axis20, axis21, axis22, axis40, axis41, axis42]:
        axis.set_yticks([1, 3, 5])
    for axis in [axis10, axis11, axis12, axis30, axis31, axis32, axis50, axis51, axis52]:
        axis.set_yticks([100, 300, 500])

    # Loop over all available haloes #
    axes = [axis00, axis01, axis02, axis20, axis21, axis22, axis40, axis41, axis42]
    axes_cvc = [axis10, axis11, axis12, axis30, axis31, axis32, axis50, axis51, axis52]

    for i, axis, axis_cvc in zip(range(len(names)), axes, axes_cvc):
        print("Plotting data for halo:", str(re.split('_|.npy', names[i])[1]))
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
        axis_cvc.plot(radius * 1e3, vtot, color=colors3[0], lw=3, label=r'$\mathrm{Total}$')
        axis_cvc.plot(radius * 1e3, shell_velocity[:, 0], color=colors3[3], linestyle='--', lw=3,
                      label=r'$\mathrm{Gas}$')
        axis_cvc.plot(radius * 1e3, shell_velocity[:, 4], color=colors3[2], linestyle='--', lw=3,
                      label=r'$\mathrm{Stars}$')
        axis_cvc.plot(radius * 1e3, shell_velocity[:, 1], color=colors3[1], linestyle='--', lw=3,
                      label=r'$\mathrm{DM}$')

        # Plot the stellar surface density profiles #
        p = plot_helper.plot_helper()  # Load the helper.
        axis.axvline(rfit, color='gray', linestyle='--')
        axis.scatter(r, np.log10(1e10 * sdfit * 1e-6), marker='o', s=25, color=colors3[0], linewidth=0.0, zorder=5)
        axis.plot(r, np.log10(1e10 * p.total_profile(r, popt0, popt1, popt2, popt3, popt4) * 1e-6), color=colors3[0],
                  lw=3, label=r'$\mathrm{Total}$')
        axis.plot(r, np.log10(1e10 * p.sersic_prof1(r, popt2, popt3, popt4) * 1e-6), color=colors3[1], lw=3,
                  label=r'$\mathrm{Sersic}$')
        axis.plot(r, np.log10(1e10 * p.exp_prof(r, popt0, popt1) * 1e-6), color=colors3[3], lw=3,
                  label=r'$\mathrm{Exponential}$')

        figure.text(0.01, 0.9, r'$\mathrm{Au-%s}$' % str(re.split('_|.npy', names[i])[1]), fontsize=25,
                    transform=axis.transAxes)  # figure.text(0.3, 0.7,  #     r'$\mathrm{n} = %.2f$' '\n' r'$\mathrm{
        # R_{d}} =  # %.2f$' '\n' r'$\mathrm{R_{eff}} = %.2f$' '\n' % (  #     1. / popt4, popt1,
        # popt3 *   #  #  # p.sersic_b_param(1.0 /  # popt4) ** (1.0 / popt4)), fontsize=16,
                    #     transform=axis.transAxes)

        # Compute component masses from the fit #  # disc_mass = 2.0 * np.pi * popt0 * popt1 * popt1  # bulge_mass =
        # np.pi * popt2 * popt3 * popt3 * gamma(2.0 / popt4 + 1)  # print(disc_mass, bulge_mass)

    axis00.legend(loc='upper right', fontsize=25, frameon=False)
    axis10.legend(loc='upper center', fontsize=25, frameon=False, ncol=2)

    # Save and close the figure #
    pdf.savefig(figure, bbox_inches='tight')
    plt.close()
    return None


def gas_temperature_vs_distance_combination(date, redshift):
    """
    Plot a combination of the temperature as a function of distance of gas cells for Auriga halo(es).
    :param date: date.
    :param redshift: redshift from main.make_pdf
    :return: None
    """
    print("Invoking gas_temperature_vs_distance_combination")
    # Get the names and sort them #
    path = '/u/di43/Auriga/plots/data/' + 'gtd/' + str(redshift) + '/'
    names = glob.glob(path + 'name_*')
    names.sort()

    # Generate the figure and set its parameters #
    figure = plt.figure(figsize=(20, 20))
    axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22, axiscbar = \
    plot_tools.create_axes_combinations(
        res=res, boxsize=boxsize * 1e3, multiple6=True)
    for axis in [axis00, axis10, axis20]:
        plot_tools.set_axes(axis, xlim=[2e-2, 2e2], ylim=[1e3, 2e8], xscale='log', yscale='log',
                            xlabel=r'$\mathrm{R/kpc}$', ylabel=r'$\mathrm{Temperature/K}$', aspect=None, which='major',
                            size=25)
    for axis in [axis01, axis02, axis11, axis12, axis21, axis22]:
        plot_tools.set_axes(axis, xlim=[2e-2, 2e2], ylim=[1e3, 2e8], xscale='log', yscale='log',
                            xlabel=r'$\mathrm{R/kpc}$', aspect=None, which='major', size=25)

    # Loop over all available haloes #
    for i, axis in zip(range(len(names)), [axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22]):
        print("Plotting data for halo:", str(re.split('_|.npy', names[i])[1]))
        # Load the data #
        sfr = np.load(path + 'sfr_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        temperature = np.load(path + 'temperature_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        spherical_distance = np.load(path + 'spherical_distance_' + str(re.split('_|.npy', names[i])[1]) + '.npy')

        # Plot the temperature as a function of distance of gas cells #
        sfr_mask, = np.where(sfr > 0)
        no_sfr_mask, = np.where(sfr == 0)
        axis.scatter(spherical_distance[no_sfr_mask] * 1e3, temperature[no_sfr_mask], s=5, edgecolor='none', c='gray',
                     zorder=5)
        hb = axis.scatter(spherical_distance[sfr_mask] * 1e3, temperature[sfr_mask], s=5, edgecolor='none',
                          c=sfr[sfr_mask] * 1e6, cmap='plasma_r', norm=matplotlib.colors.LogNorm(vmin=4, vmax=650),
                          zorder=5)
        figure.text(0.01, 0.9, r'$\mathrm{Au-%s}$' % str(re.split('_|.npy', names[i])[1]), fontsize=25,
                    transform=axis.transAxes)

    for axis in [axis01, axis02, axis11, axis12, axis21, axis22]:
        axis.set_yticklabels([])
    for axis in [axis00, axis01, axis02, axis10, axis11, axis12]:
        axis.set_xlabel('')
        axis.set_xticklabels([])

    # Add a colorbar, save and close the figure #
    plot_tools.create_colorbar(axiscbar, hb, label="$\mathrm{SFR/(M_\odot\;Myr^{-1})}$", size=25)
    plt.savefig('/u/di43/Auriga/plots/' + 'gtdc-' + date + '.png', bbox_inches='tight')
    plt.close()
    return None


def decomposition_IT20_combination(date, redshift):
    """
    Plot the angular momentum maps and calculate D/T_IT20 for Auriga halo(es).
    :param date: date
    :param redshift: redshift from main.make_pdf
    :return: None
    """
    print("Invoking decomposition_IT20_combination")
    # Get the names and sort them #
    path = '/u/di43/Auriga/plots/data/' + 'di/' + str(redshift) + '/'

    # Generate the figure and set its parameters #
    figure = plt.figure(figsize=(20, 15))
    axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22, axiscbar = \
        plot_tools.create_axes_combinations(
        res=res, boxsize=boxsize * 1e3, mollweide=True)
    axes = [axis00, axis10, axis20, axis01, axis11, axis21, axis02, axis12, axis22]
    for axis in axes:
        axis.set_yticklabels([])
        axis.set_xlabel(r'$\mathrm{\alpha/\degree}$', size=20)
        axis.set_xticklabels(['', '-120', '', '-60', '', '0', '', '60', '', '120', ''], size=20)
    for axis in [axis00, axis10, axis20]:
        axis.set_ylabel(r'$\mathrm{\delta/\degree}$', size=20)
        axis.set_yticklabels(['', '-60', '', '-30', '', '0', '', '30', '', '60', ''], size=20)

    # Load and plot the data #
    names = glob.glob(path + 'name_*')
    names.sort()

    # Loop over all available haloes #
    for i, axis in zip(range(len(names)), axes):
        print("Plotting data for halo:", str(re.split('_|.npy', names[i])[1]))
        # Load the data #
        density_map = np.load(path + 'density_map_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        disc_fraction_IT20 = np.load(path + 'disc_fraction_IT20_' + str(re.split('_|.npy', names[i])[1]) + '.npy')

        # Plot the angular momentum maps and calculate D/T_IT20 #
        # Sample a 360x180 grid in sample_alpha and sample_delta #
        sample_alpha = np.linspace(-180.0, 180.0, num=360) * u.deg
        sample_delta = np.linspace(-90.0, 90.0, num=180) * u.deg

        # Display data on a 2D regular raster and create a pseudo-color plot #
        pcm = axis.pcolormesh(np.radians(sample_alpha), np.radians(sample_delta), density_map, cmap='nipy_spectral_r')

        figure.text(0.01, 1.05,
                    r'$\mathrm{Au-%s:D/T=%.2f}$' % (str(re.split('_|.npy', names[i])[1]), disc_fraction_IT20),
                    fontsize=20, transform=axis.transAxes)

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
    names = glob.glob(path + 'name_*')
    names.sort()

    # Generate the figure and set its parameters #
    figure = plt.figure(figsize=(15, 5))
    axis00, axis01, axis02 = plot_tools.create_axes_combinations(res=res, boxsize=boxsize * 1e3, multiple4=True)
    axis002 = axis00.twiny()
    plot_tools.set_axes_evolution(axis00, axis002, ylim=[0, 1.1], ylabel=r'$\mathrm{A_2}$', aspect=None)
    for axis in [axis01, axis02]:
        axis2 = axis.twiny()
        plot_tools.set_axes_evolution(axis, axis2, ylim=[0, 1.1], aspect=None)
        axis.set_yticklabels([])

    # Split the names into 3 groups and plot the three flavours of a halo together (i.e. original, NoR and NoRNoQ) #
    names_groups = np.array_split(names, 3)
    for i, axis in zip(range(len(names_groups)), [axis00, axis01, axis02]):
        names_flavours = names_groups[i]
        # Loop over all available flavours #
        for j in range(len(names_flavours)):
            # Load the data #
            max_A2s = np.load(path + 'max_A2s_' + str(re.split('_|.npy', names_flavours[j])[1]) + '.npy')
            lookback_times = np.load(path + 'lookback_times_' + str(re.split('_|.npy', names_flavours[j])[1]) + '.npy')

            # Downsample the runs which have more snapshots #
            if j == 0:
                original_lookback_times = lookback_times
            else:
                max_A2s = plot_tools.linear_resample(max_A2s, len(original_lookback_times))
                lookback_times = plot_tools.linear_resample(lookback_times, len(original_lookback_times))

            # Plot the evolution of bar strength #
            axis.plot(lookback_times, max_A2s, color=colors[j], lw=3,
                      label=r'$\mathrm{Au-%s}$' % str(re.split('_|.npy', names_flavours[j])[1]))

            axis.legend(loc='upper left', fontsize=15, frameon=False)  # Create the legend.
    # Save and close the figure #
    pdf.savefig(figure, bbox_inches='tight')
    plt.close()
    return None


def gas_temperature_regimes_combination(pdf):
    """
    Plot a combination of the evolution of mass- and volume-weighted gas fractions in different temperature regimes
    for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :return: None
    """
    print("Invoking gas_temperature_regimes_combination")
    # Get the names and sort them #
    path = '/u/di43/Auriga/plots/data/' + 'gtr/'
    names = glob.glob(path + 'name_*')
    names.sort()
    k = 0

    # Generate the figure and set its parameters #
    figure = plt.figure(figsize=(15, 15))
    axis00, axis10 = plot_tools.create_axes_combinations(res=res, boxsize=boxsize * 1e3, multiple14=True)
    plot_tools.set_axes(axis00, xlim=[-0.1, 0.9], ylim=[0, 1.19], ylabel=r'$\mathrm{Volume\; fraction}$', aspect=None,
                        size=30)
    plot_tools.set_axes(axis10, xlim=[-0.1, 0.9], ylim=[0, 1.19], ylabel=r'$\mathrm{Mass\; fraction}$', aspect=None,
                        size=30)
    axis00.set_xticklabels([])
    axis00.set_xticks(np.arange(-0.1, 0.9, 0.1))
    axis10.set_xticks(np.arange(-0.1, 0.9, 0.1))
    axis10.set_xticklabels(np.append('', [r'$\mathrm{Au-%s}$' % re.split('_|.npy', halo)[1] for halo in names]), '',
                           rotation=25, ha="right")

    # Split the names into 3 groups and plot the three flavours of a halo together (i.e. original, NoR and NoRNoQ) #
    names_groups = np.array_split(names, 3)

    for i in range(len(names_groups)):
        names_flavours = names_groups[i]
        # Loop over all available flavours #
        for j in range(len(names_flavours)):
            print("Plotting data for halo:", str(re.split('_|.npy', names_flavours[j])[1]))
            # Load the data #
            sfg_mass_ratios = np.load(
                path + 'sfg_mass_ratios_' + str(re.split('_|.npy', names_flavours[j])[1]) + '.npy')
            wg_mass_ratios = np.load(path + 'wg_mass_ratios_' + str(re.split('_|.npy', names_flavours[j])[1]) + '.npy')
            hg_mass_ratios = np.load(path + 'hg_mass_ratios_' + str(re.split('_|.npy', names_flavours[j])[1]) + '.npy')
            sfg_volume_ratios = np.load(
                path + 'sfg_volume_ratios_' + str(re.split('_|.npy', names_flavours[j])[1]) + '.npy')
            wg_volume_ratios = np.load(
                path + 'wg_volume_ratios_' + str(re.split('_|.npy', names_flavours[j])[1]) + '.npy')
            hg_volume_ratios = np.load(
                path + 'hg_volume_ratios_' + str(re.split('_|.npy', names_flavours[j])[1]) + '.npy')
            lookback_times = np.load(path + 'lookback_times_' + str(re.split('_|.npy', names_flavours[j])[1]) + '.npy')

            # Downsample the runs which have more snapshots #
            if j == 0:
                original_lookback_times = lookback_times
            else:
                sfg_mass_ratios = plot_tools.linear_resample(sfg_mass_ratios, len(original_lookback_times))
                wg_mass_ratios = plot_tools.linear_resample(wg_mass_ratios, len(original_lookback_times))
                hg_mass_ratios = plot_tools.linear_resample(hg_mass_ratios, len(original_lookback_times))
                lookback_times = plot_tools.linear_resample(lookback_times, len(original_lookback_times))

            # Plot the mass- and volume-weighted gas fractions in different temperature regimes #
            time_mask, = np.where(lookback_times < 1)  # In Gyr.
            b1, = axis00.bar(k / 10, np.average(sfg_volume_ratios[time_mask]), width=0.09, color=colors3[3], alpha=0.85,
                             edgecolor='none')
            b2, = axis00.bar(k / 10, np.average(wg_volume_ratios[time_mask]),
                             bottom=np.average(sfg_volume_ratios[time_mask]), width=0.09, color=colors3[2], alpha=0.85,
                             edgecolor='none')
            b3, = axis00.bar(k / 10, np.average(hg_volume_ratios[time_mask]), bottom=np.sum(
                np.vstack([np.average(sfg_volume_ratios[time_mask]), np.average(wg_volume_ratios[time_mask])]).T),
                             width=0.09, color=colors3[1], alpha=0.85, edgecolor='none')
            b4, = axis10.bar(k / 10, np.average(sfg_mass_ratios[time_mask]), width=0.09, color=colors3[3], alpha=0.85,
                             edgecolor='none')
            b5, = axis10.bar(k / 10, np.average(wg_mass_ratios[time_mask]),
                             bottom=np.average(sfg_mass_ratios[time_mask]), width=0.09, color=colors3[2], alpha=0.85,
                             edgecolor='none')
            b6, = axis10.bar(k / 10, np.average(hg_mass_ratios[time_mask]), bottom=np.sum(
                np.vstack([np.average(sfg_mass_ratios[time_mask]), np.average(wg_mass_ratios[time_mask])]).T),
                             width=0.09, color=colors3[1], alpha=0.85, edgecolor='none')
            k += 1

    # Plot vertical lines to separate the haloes #
    for axis in [axis00, axis10]:
        axis.axvline(0.25, ymax=0.996 / axis.get_ylim()[1], color=colors[0], lw=3)
        axis.axvline(0.55, ymax=0.996 / axis.get_ylim()[1], color=colors[0], lw=3)

    # Create the legends #
    axis00.legend([b1, b2, b3], [r'$\mathrm{Cold\;gas}$', r'$\mathrm{Warm\;gas}$', r'$\mathrm{Hot\;gas}$'],
                  loc='upper center', fontsize=30, frameon=False, ncol=3)

    # Save and close the figure #
    pdf.savefig(figure, bbox_inches='tight')
    plt.close()
    return None


def quasar_mode_distribution_combination(pdf):
    """
    Plot a combination of the evolution of quasar mode energy feedback from log files for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :return: None
    """
    print("Invoking quasar_mode_distribution_combination")
    # Get the names and sort them #
    path = '/u/di43/Auriga/plots/data/' + 'AGNmd/'
    path_kernel = '/u/di43/Auriga/plots/data/' + 'AGNfk/'
    names = glob.glob(path + 'name_*')
    names.sort()
    n_bins = 44  # np.int(np.floor(len(gas_volumes) / 2))
    time_bin_width = ((13 - 0) / n_bins) * u.Gyr.to(u.second)  # In second

    # Generate the figure and set its parameters #
    figure = plt.figure(figsize=(15, 5))
    axis00, axis01, axis02 = plot_tools.create_axes_combinations(res=res, boxsize=boxsize * 1e3, multiple4=True)
    axis002 = axis00.twiny()
    plot_tools.set_axes_evolution(axis00, axis002, ylim=[1e41, 1e46], yscale='log',
                                  ylabel=r'$\mathrm{Energy\;rate/(erg\;s^{-1}})$', which='major', aspect=None)
    for axis in [axis01, axis02]:
        axis2 = axis.twiny()
        plot_tools.set_axes_evolution(axis, axis2, ylim=[1e41, 1e46], yscale='log', which='major', aspect=None)
        axis.set_yticklabels([])

    # Get the names and sort them #
    names = glob.glob(path + 'name_*')
    names.sort()

    # Split the names into 3 groups and plot the three flavours of a halo together (i.e. original, NoR and NoRNoQ) #
    names_groups = np.array_split(names, 3)
    for i, axis in zip(range(len(names_groups)), [axis00, axis01, axis02]):
        names_flavours = names_groups[i]
        # Loop over all available flavours #
        for j in range(len(names_flavours)):
            print("Plotting data for halo:", str(re.split('_|.npy', names_flavours[j])[1]))
            thermals = np.load(path + 'thermals_' + str(re.split('_|.npy', names_flavours[j])[1]) + '.npy')
            lookback_times = np.load(path + 'lookback_times_' + str(re.split('_|.npy', names_flavours[j])[1]) + '.npy')
            # Transform the arrays to comma separated strings and convert each element to float #
            thermals = ','.join(thermals)
            thermals = np.fromstring(thermals, dtype=np.float, sep=',')

            gas_volumes = np.load(path_kernel + 'gas_volumes_' + str(re.split('_|.npy', names_flavours[j])[1]) + '.npy')
            nsf_gas_volumes = np.load(
                path_kernel + 'nsf_gas_volumes_' + str(re.split('_|.npy', names_flavours[j])[1]) + '.npy')

            # Downsample the runs which have more snapshots #
            gas_volumes = plot_tools.linear_resample(gas_volumes, n_bins)
            nsf_gas_volumes = plot_tools.linear_resample(nsf_gas_volumes, n_bins)

            # Plot 2D distribution of the modes and their binned sum line #
            x_values, sum = plot_tools.binned_sum(lookback_times[np.where(thermals > 0)],
                                                  thermals[np.where(thermals > 0)], n_bins=n_bins)
            efficiency = np.flip(nsf_gas_volumes / gas_volumes)  # binned_sum flips the x_values
            mask, = np.where(efficiency > 0)
            axis.plot(x_values[mask], np.divide(efficiency[mask] * sum[mask], time_bin_width), color=colors[j], lw=3,
                      linestyle='-')

            axis.plot(x_values[mask], sum[mask] / time_bin_width, color=colors[j], lw=3, linestyle=':')

        # Create the legend #
        axis.legend(custom_lines, flavours_text[i], handler_map={tuple:HandlerTuple(ndivide=None)}, numpoints=1,
                    frameon=False, fontsize=15, loc='upper left')

        # Save and close the figure #
    pdf.savefig(figure, bbox_inches='tight')
    plt.close()
    return None


def radio_mode_distribution_combination(pdf):
    """
    Plot a combination of the evolution of radio mode energy feedback from log files for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :return: None
    """
    print("Invoking radio_mode_distribution_combination")
    # Get the names and sort them #
    path = '/u/di43/Auriga/plots/data/' + 'AGNmd/'
    names = glob.glob(path + 'name_*')
    names.sort()
    n_bins = 44  # np.int(np.floor(len(gas_volumes) / 2))
    time_bin_width = ((13 - 0) / n_bins) * u.Gyr.to(u.second)  # In second

    # Generate the figure and set its parameters #
    figure = plt.figure(figsize=(15, 5))
    axis00, axis01, axis02 = plot_tools.create_axes_combinations(res=res, boxsize=boxsize * 1e3, multiple4=True)
    axis002 = axis00.twiny()
    plot_tools.set_axes_evolution(axis00, axis002, ylim=[1e40, 1e44], yscale='log',
                                  ylabel=r'$\mathrm{Energy\;rate/(erg\;s^{-1}})$', which='major', aspect=None)
    for axis in [axis01, axis02]:
        axis2 = axis.twiny()
        plot_tools.set_axes_evolution(axis, axis2, ylim=[1e40, 1e44], yscale='log', which='major', aspect=None)
        axis.set_yticklabels([])

    # Get the names and sort them #
    names = glob.glob(path + 'name_*')
    names.sort()

    # Split the names into 3 groups and plot the three flavours of a halo together (i.e. original, NoR and NoRNoQ) #
    names_groups = np.array_split(names, 3)
    for i, axis in zip(range(len(names_groups)), [axis00, axis01, axis02]):
        names_flavours = names_groups[i]
        # Loop over all available flavours #
        print("Plotting data for halo:", str(re.split('_|.npy', names_flavours[0])[1]))
        mechanicals = np.load(path + 'mechanicals_' + str(re.split('_|.npy', names_flavours[0])[1]) + '.npy')
        lookback_times = np.load(path + 'lookback_times_' + str(re.split('_|.npy', names_flavours[0])[1]) + '.npy')

        # Transform the arrays to comma separated strings and convert each element to float #
        mechanicals = ','.join(mechanicals)
        mechanicals = np.fromstring(mechanicals, dtype=np.float, sep=',')

        # Plot 2D distribution of the modes and their binned sum line #
        x_values, sum = plot_tools.binned_sum(lookback_times[np.where(mechanicals > 0)],
                                              mechanicals[np.where(mechanicals > 0)], n_bins=n_bins)
        # Plot the effective thermal energies #
        axis.plot(x_values, sum / time_bin_width, color=colors[0], lw=3, linestyle='--',
                  label=r'$\mathrm{Au-%s}$' % str(re.split('_|.npy', names_flavours[0])[1]))

        # Create the legend #
        axis.legend(numpoints=1, frameon=False, fontsize=15, loc='upper left')

        # Save and close the figure #
    pdf.savefig(figure, bbox_inches='tight')
    plt.close()
    return None


def AGN_feedback_kernel_combination(pdf):
    """
    Plot a combination of the evolution of black hole radius and the volume of gas cells within that for Auriga halo(
    es).
    :param pdf: path to save the pdf from main.make_pdf
    :return: None
    """
    print("Invoking AGN_feedback_kernel_combination")
    # Get the names and sort them #
    path = '/u/di43/Auriga/plots/data/' + 'AGNfk/'
    names = glob.glob(path + 'name_*')
    names.sort()

    # Remove the NoRNoQ flavours #
    names.remove('/u/di43/Auriga/plots/data/AGNfk/name_06NoRNoQ.npy')
    names.remove('/u/di43/Auriga/plots/data/AGNfk/name_17NoRNoQ.npy')
    names.remove('/u/di43/Auriga/plots/data/AGNfk/name_18NoRNoQ.npy')

    # Generate the figure and set its parameters #
    figure = plt.figure(figsize=(15, 10))
    axis00, axis01, axis02, axis10, axis11, axis12 = plot_tools.create_axes_combinations(res=res, boxsize=boxsize * 1e3,
                                                                                         multiple13=True)
    for axis in [axis00, axis01, axis02, axis10, axis11, axis12]:
        axis2 = axis.twiny()
        axis3 = axis.twinx()
        axis3.yaxis.label.set_color('tab:red')
        axis3.spines['right'].set_color('tab:red')
        plot_tools.set_axes(axis3, ylim=[-0.1, 2.1], xlabel=r'$\mathrm{t_{look}/Gyr}$', ylabel=r'$\mathrm{r_{BH}/kpc}$',
                            aspect=None)
        plot_tools.set_axes_evolution(axis, axis2, ylim=[-0.1, 2.1],
                                      ylabel=r'$\mathrm{V_{nSFR}(r<r_{BH})/V_{all}(r<r_{BH})}$', aspect=None)
        axis3.tick_params(axis='y', direction='out', left='off', colors='tab:red')
        if axis in [axis10, axis11, axis12]:
            axis2.set_xlabel('')
            axis2.set_xticklabels([])
            axis2.tick_params(top=False)
        if axis in [axis00, axis01, axis10, axis11]:
            axis3.set_ylabel('')
            axis3.set_yticklabels([])
    for axis in [axis01, axis02, axis11, axis12]:
        axis.set_ylabel('')
        axis.set_yticklabels([])
    for axis in [axis00, axis01, axis02]:
        axis.set_xlabel('')
        axis.set_xticklabels([])

    # Loop over all available haloes #
    for i, axis in zip(range(len(names)), [axis00, axis10, axis01, axis11, axis02, axis12]):
        print("Plotting data for halo:", str(re.split('_|.npy', names[i])[1]))
        # Load the data #
        gas_volumes = np.load(path + 'gas_volumes_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        lookback_times = np.load(path + 'lookback_times_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        nsf_gas_volumes = np.load(path + 'nsf_gas_volumes_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        blackhole_hsmls = np.load(path + 'blackhole_hsmls_' + str(re.split('_|.npy', names[i])[1]) + '.npy')

        # Downsample the runs which have more snapshots #
        if i == 0:
            original_lookback_times = lookback_times
        else:
            gas_volumes = plot_tools.linear_resample(gas_volumes, len(original_lookback_times))
            nsf_gas_volumes = plot_tools.linear_resample(nsf_gas_volumes, len(original_lookback_times))
            blackhole_hsmls = plot_tools.linear_resample(blackhole_hsmls, len(original_lookback_times))
            lookback_times = plot_tools.linear_resample(lookback_times, len(original_lookback_times))

        # Plot median and 1-sigma lines #
        n_bins = np.int(np.floor(len(original_lookback_times) / 2))
        x_values, median, shigh, slow = plot_tools.binned_median_1sigma(lookback_times, nsf_gas_volumes / gas_volumes,
                                                                        bin_type='equal_width', n_bins=n_bins,
                                                                        log=False)
        median_volumes, = axis.plot(x_values, median, color=colors[0], lw=3)
        axis.fill_between(x_values, shigh, slow, color=colors[0], alpha='0.3')
        fill_volumes, = plt.fill(np.NaN, np.NaN, color=colors[0], alpha=0.3)

        x_values, median, shigh, slow = plot_tools.binned_median_1sigma(lookback_times, blackhole_hsmls * 1e3,
                                                                        bin_type='equal_width', n_bins=n_bins,
                                                                        log=False)
        median_bh, = axis.plot(x_values, median, color=colors[1], lw=3)
        axis.fill_between(x_values, shigh, slow, color=colors[1], alpha='0.3')
        fill_bh, = plt.fill(np.NaN, np.NaN, color=colors[1], alpha=0.3)

        figure.text(0.01, 0.92, r'$\mathrm{Au-%s}$' % str(re.split('_|.npy', names[i])[1]), fontsize=20,
                    transform=axis.transAxes)

    # Create the legend #
    axis00.legend([median_volumes, fill_volumes, median_bh, fill_bh],
                  [r'$\mathrm{Median}$', r'$\mathrm{16^{th}-84^{th}\;\%ile}$', r'$\mathrm{Median}$',
                   r'$\mathrm{16^{th}-84^{th}\;\%ile}$'], frameon=False, fontsize=20, loc='upper right')

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

    # Check if a folder to save the data exists, if not then create one #
    path = '/u/di43/Auriga/plots/data/' + 'cc/' + str(redshift) + '/'
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
            # Check if halo's data already exists, if not then read it #
            names = glob.glob(path + 'name_*')
            names = [re.split('_|.npy', name)[1] for name in names]
            if str(s.haloname) in names:
                continue

            # Select the halo and rotate it based on its principal axes so galaxy's spin is aligned with the z-axis #
            s.calc_sf_indizes(s.subfind)
            s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)

            # Get the gas density projections #
            density_face_on = \
                s.get_Aslice("rho", res=res, axes=[1, 2], box=[boxsize, boxsize], proj=True, proj_fact=0.125,
                             numthreads=8)["grid"] * boxsize * 1e3
            density_edge_on = \
                s.get_Aslice("rho", res=res, axes=[1, 0], box=[boxsize, boxsize], proj=True, proj_fact=0.125,
                             numthreads=8)["grid"] * boxsize * 1e3

            # Get the gas temperature projections #
            meanweight = 4.0 / (1.0 + 3.0 * 0.76 + 4.0 * 0.76 * s.data['ne']) * 1.67262178e-24
            temperature = (5.0 / 3.0 - 1.0) * s.data['u'] / KB * (1e6 * parsec) ** 2.0 / (
                1e6 * parsec / 1e5) ** 2 * meanweight
            s.data['temprho'] = s.rho * temperature

            temperature_face_on = \
                s.get_Aslice("temprho", res=res, axes=[1, 2], box=[boxsize, boxsize], proj=True, proj_fact=0.125,
                             numthreads=8)["grid"]
            temperature_face_on_rho = \
                s.get_Aslice("rho", res=res, axes=[1, 2], box=[boxsize, boxsize], proj=True, proj_fact=0.125,
                             numthreads=8)["grid"]
            temperature_edge_on = \
                s.get_Aslice("temprho", res=res, axes=[1, 0], box=[boxsize, boxsize], proj=True, proj_fact=0.125,
                             numthreads=8)["grid"]
            temperature_edge_on_rho = \
                s.get_Aslice("rho", res=res, axes=[1, 0], box=[boxsize, boxsize], proj=True, proj_fact=0.125,
                             numthreads=8)["grid"]

            # Get the magnetic field projections #
            s.data['b2'] = (s.data['bfld'] ** 2.).sum(axis=1)
            bfld_face_on = np.sqrt(
                s.get_Aslice("b2", res=res, axes=[1, 2], box=[boxsize, boxsize], proj=True, proj_fact=0.125,
                             numthreads=8)["grid"] / res) * bfac * 1e6
            bfld_edge_on = np.sqrt(
                s.get_Aslice("b2", res=res, axes=[1, 0], box=[boxsize, boxsize], proj=True, proj_fact=0.125,
                             numthreads=8)["grid"] / res) * bfac * 1e6

            # Get the gas sfr projections #
            sfr_face_on = s.get_Aslice("sfr", res=res, axes=[1, 2], box=[boxsize, boxsize], proj=True, proj_fact=0.125,
                                       numthreads=8)["grid"]
            sfr_edge_on = s.get_Aslice("sfr", res=res, axes=[1, 0], box=[boxsize, boxsize], proj=True, proj_fact=0.125,
                                       numthreads=8)["grid"]

            # Get the gas total pressure projections #
            elements_mass = [1.01, 4.00, 12.01, 14.01, 16.00, 20.18, 24.30, 28.08, 55.85, 88.91, 87.62, 91.22, 137.33]
            meanweight = np.sum(s.gmet[s.data['type'] == 0, 0:9], axis=1) / (
                np.sum(s.gmet[s.data['type'] == 0, 0:9] / elements_mass[0:9], axis=1) + s.data['ne'] * s.gmet[
                s.data['type'] == 0, 0])
            Tfac = 1. / meanweight * (1.0 / (5. / 3. - 1.)) * KB / PROTONMASS * 1e10 * msol / 1.989e53

            # Un megabars (10**12dyne/cm**2)
            s.data['T'] = s.u / Tfac
            s.data['dens'] = s.rho / (1e6 * parsec) ** 3. * msol * 1e10
            s.data['Ptherm'] = s.data['dens'] * s.data['T'] / (meanweight * PROTONMASS)

            pressure_face_on = \
                s.get_Aslice("Ptherm", res=res, axes=[1, 2], box=[boxsize, boxsize], proj=True, proj_fact=0.125,
                             numthreads=8)["grid"]
            pressure_edge_on = \
                s.get_Aslice("Ptherm", res=res, axes=[1, 0], box=[boxsize, boxsize], proj=True, proj_fact=0.125,
                             numthreads=8)["grid"]

            # Get the radial velocity projections #
            gas_mask, = np.where(s.data['type'] == 0)
            spherical_radius = np.sqrt(np.sum(s.data['pos'][gas_mask, :] ** 2, axis=1))
            CoM_velocity = np.sum(s.data['vel'][gas_mask, :] * s.data['mass'][gas_mask][:, None], axis=0) / np.sum(
                s.data['mass'][gas_mask])
            s.data['vrad'] = np.sum((s.data['vel'][gas_mask] - CoM_velocity) * s.data['pos'][gas_mask],
                                    axis=1) / spherical_radius

            vrad_face_on = \
                s.get_Aslice("vrad", res=res, axes=[1, 2], box=[boxsize, boxsize], proj=True, proj_fact=0.125,
                             numthreads=8)["grid"]
            vrad_edge_on = \
                s.get_Aslice("vrad", res=res, axes=[1, 0], box=[boxsize, boxsize], proj=True, proj_fact=0.125,
                             numthreads=8)["grid"]

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
    names = glob.glob(path + 'name_*')
    names.sort()

    # Loop over all available haloes #
    for i in range(len(names)):
        # Generate the figure and set its parameters #
        figure = plt.figure(figsize=(16, 9))
        axis00, axis01, axis02, axis03, axis04, axis05, axis10, axis11, axis12, axis13, axis14, axis15, axis20, \
        axis21, axis22, axis23, axis24, axis25, x, y, area = plot_tools.create_axes_combinations(
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
            axis.set_xlabel(r'$\mathrm{x/kpc}$', size=20)
            axis.tick_params(direction='out', which='both', top='on', right='on')
            for label in axis.xaxis.get_ticklabels():
                label.set_size(12)
        for axis in [axis01, axis21, axis02, axis22, axis03, axis23, axis04, axis24, axis05, axis25]:
            axis.set_yticklabels([])

        axis00.set_ylabel(r'$\mathrm{y/kpc}$', size=20)
        axis20.set_ylabel(r'$\mathrm{z/kpc}$', size=20)
        for axis in [axis00, axis20]:
            for label in axis.yaxis.get_ticklabels():
                label.set_size(12)
                axis.set_yticklabels(tick_labels)

        # Load and plot the data #
        density_face_on = np.load(path + 'density_face_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        density_edge_on = np.load(path + 'density_edge_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        temperature_face_on = np.load(path + 'temperature_face_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        temperature_edge_on = np.load(path + 'temperature_edge_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        temperature_face_on_rho = np.load(
            path + 'temperature_face_on_rho_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        temperature_edge_on_rho = np.load(
            path + 'temperature_edge_on_rho_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        bfld_face_on = np.load(path + 'bfld_face_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        bfld_edge_on = np.load(path + 'bfld_edge_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        sfr_face_on = np.load(path + 'sfr_face_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        sfr_edge_on = np.load(path + 'sfr_edge_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        pressure_face_on = np.load(path + 'pressure_face_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        pressure_edge_on = np.load(path + 'pressure_edge_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        vrad_face_on = np.load(path + 'vrad_face_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        vrad_edge_on = np.load(path + 'vrad_edge_on_' + str(re.split('_|.npy', names[i])[1]) + '.npy')

        # Plot the gas density projections #
        pcm = axis00.pcolormesh(x, y, density_face_on.T, norm=matplotlib.colors.LogNorm(), cmap='magma',
                                rasterized=True)
        axis20.pcolormesh(x, y, density_edge_on.T, norm=matplotlib.colors.LogNorm(), cmap='magma', rasterized=True)
        plot_tools.create_colorbar(axis10, pcm, "$\mathrm{\Sigma_{gas}/(M_\odot\;kpc^{-2})}$", orientation='horizontal')

        # Plot the gas temperature projections #
        pcm = axis01.pcolormesh(x, y, (temperature_face_on / temperature_face_on_rho).T,
                                norm=matplotlib.colors.LogNorm(), cmap='viridis', rasterized=True)
        axis21.pcolormesh(x, y, (temperature_edge_on / temperature_edge_on_rho).T, norm=matplotlib.colors.LogNorm(),
                          cmap='viridis', rasterized=True)
        plot_tools.create_colorbar(axis11, pcm, "$\mathrm{T/K}$", orientation='horizontal')

        # Plot the magnetic field projections #
        pcm = axis02.pcolormesh(x, y, bfld_face_on.T, norm=matplotlib.colors.LogNorm(), cmap='CMRmap', rasterized=True)
        axis22.pcolormesh(x, y, bfld_edge_on.T, norm=matplotlib.colors.LogNorm(), cmap='CMRmap', rasterized=True)
        plot_tools.create_colorbar(axis12, pcm, "$\mathrm{B/\mu G}$", orientation='horizontal')

        # Plot the sfr projections #
        pcm = axis03.pcolormesh(x, y, sfr_face_on.T, norm=matplotlib.colors.LogNorm(), cmap='gist_heat',
                                rasterized=True)
        axis23.pcolormesh(x, y, sfr_edge_on.T, norm=matplotlib.colors.LogNorm(), cmap='gist_heat', rasterized=True)
        plot_tools.create_colorbar(axis13, pcm, "$\mathrm{SFR/(M_\odot\;yr^{-1})}$", orientation='horizontal')

        # Plot the gas total pressure projections #
        pcm = axis04.pcolormesh(x, y, pressure_face_on.T, norm=matplotlib.colors.LogNorm(), cmap='cividis',
                                rasterized=True)
        axis24.pcolormesh(x, y, pressure_edge_on.T, norm=matplotlib.colors.LogNorm(), cmap='cividis', rasterized=True)
        plot_tools.create_colorbar(axis14, pcm, "$\mathrm{P/k_{B}/(K\;cm^{-3})}$", orientation='horizontal')

        pcm = axis05.pcolormesh(x, y, vrad_face_on.T, cmap='coolwarm', vmin=-7000, vmax=7000, rasterized=True)
        axis25.pcolormesh(x, y, vrad_edge_on.T, cmap='coolwarm', vmin=-7000, vmax=7000, rasterized=True)
        plot_tools.create_colorbar(axis15, pcm, "$\mathrm{Velocity/km\;s^{-1})}$", orientation='horizontal')

        for axis in [axis10, axis11, axis12, axis13, axis14, axis15]:
            axis.xaxis.tick_top()
            for label in axis.xaxis.get_ticklabels():
                label.set_size(12)
            axis.tick_params(direction='out', which='both', top='on', right='on')

        figure.text(0.0, 1.01, 'Au-' + str(re.split('_|.npy', names[i])[1]) + ' redshift = ' + str(redshift),
                    fontsize=20, transform=axis00.transAxes)

        # Save and close the figure #
        pdf.savefig(figure, bbox_inches='tight')
        plt.close()
    return None


def mass_loading_combination(pdf, method):
    """
    Plot a combination of the evolution of mass loading for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param method: method to calculate flows.
    :return: None
    """
    print("Invoking mass_loading_combination")
    dT = 250  # In Myr.
    # Get the names and sort them #
    path = '/u/di43/Auriga/plots/data/' + 'gfml/'
    names = glob.glob(path + 'name_*')
    names.sort()

    # Generate the figure and set its parameters #
    figure = plt.figure(figsize=(20, 7.5))
    axis00, axis01, axis02 = plot_tools.create_axes_combinations(res=res, boxsize=boxsize * 1e3, multiple4=True)
    axis002 = axis00.twiny()
    plot_tools.set_axes_evolution(axis00, axis002, ylim=[1e-1, 1e2], yscale='log', ylabel=r'$\mathrm{Mass\;loading}$',
                                  aspect=None)
    for axis in [axis01, axis02]:
        axis2 = axis.twiny()
        plot_tools.set_axes_evolution(axis, axis2, ylim=[1e-1, 1e2], yscale='log', aspect=None)
        axis.set_yticklabels([])

    # Split the names into 3 groups and plot the three flavours of a halo together (i.e. original, NoR and NoRNoQ) #
    names_groups = np.array_split(names, 3)
    for i, axis in zip(range(len(names_groups)), [axis00, axis01, axis02]):
        names_flavours = names_groups[i]
        # Loop over all available flavours #
        for j in range(len(names_flavours)):
            # Load the data #
            sfrs = np.load(path + 'sfrs_' + str(re.split('_|.npy', names_flavours[j])[1]) + '.npy', allow_pickle=True)
            gas_masses = np.load(path + 'gas_masses_' + str(re.split('_|.npy', names_flavours[j])[1]) + '.npy',
                                 allow_pickle=True)
            Rvirs = np.load(path + 'Rvirs_' + str(re.split('_|.npy', names[j])[1]) + '.npy', allow_pickle=True)
            lookback_times = np.load(path + 'lookback_times_' + str(re.split('_|.npy', names_flavours[j])[1]) + '.npy',
                                     allow_pickle=True)
            spherical_radii = np.load(
                path + 'spherical_radii_' + str(re.split('_|.npy', names_flavours[j])[1]) + '.npy', allow_pickle=True)
            radial_velocities = np.load(
                path + 'radial_velocities_' + str(re.split('_|.npy', names_flavours[j])[1]) + '.npy', allow_pickle=True)

            # Declare arrays to store the data #
            mass_outflows, mass_inflows, mass_loading = np.zeros(len(lookback_times)), np.zeros(
                len(lookback_times)), np.zeros(len(lookback_times))

            if method == 'time_interval':
                # Loop over all radial limits #
                radial_cut = 0.5
                for l in range(len(lookback_times)):
                    outflow_mask, = np.where((spherical_radii[l] < radial_cut * Rvirs[l]) & (spherical_radii[l] + (
                        radial_velocities[l] * u.km.to(u.Mpc) / u.second.to(u.Myr)) * dT > radial_cut * Rvirs[l]))
                    inflow_mask, = np.where((spherical_radii[l] > radial_cut * Rvirs[l]) & (spherical_radii[l] + (
                        radial_velocities[l] * u.km.to(u.Mpc) / u.second.to(u.Myr)) * dT < radial_cut * Rvirs[l]))
                    gas_mask, = np.where(spherical_radii[l] < radial_cut * Rvirs[l])
                    mass_outflows[l] = np.divide(np.sum(gas_masses[l][outflow_mask]) * 1e10, dT * 1e6)
                    mass_inflows[l] = np.divide(np.sum(gas_masses[l][inflow_mask]) * 1e10, dT * 1e6)
                    mass_loading[l] = mass_outflows[l] / np.sum(sfrs[l][gas_mask])

            elif method == 'shell':
                # Loop over all lookback times #
                radial_cut = 0.5
                for l in range(len(lookback_times)):
                    outflow_mask, = np.where((spherical_radii[l] > radial_cut * Rvirs[l]) & (
                        spherical_radii[l] < 1e-3 + radial_cut * Rvirs[l]) & (radial_velocities[l] > 0))
                    inflow_mask, = np.where((spherical_radii[l] > radial_cut * Rvirs[l]) & (
                        spherical_radii[l] < 1e-3 + radial_cut * Rvirs[l]) & (radial_velocities[l] < 0))
                    gas_mask, = np.where(spherical_radii[l] < radial_cut * Rvirs[l])
                    mass_outflows[l] = np.divide(np.sum(gas_masses[l][outflow_mask] * (
                        radial_velocities[l][outflow_mask] * u.km.to(u.Mpc) / u.second.to(u.yr))), 1e-3) * 1e10
                    mass_inflows[l] = np.divide(np.sum(gas_masses[l][inflow_mask] * (
                        radial_velocities[l][inflow_mask] * u.km.to(u.Mpc) / u.second.to(u.yr))), 1e-3) * 1e10
                    mass_loading[l] = mass_outflows[l] / np.sum(sfrs[l][gas_mask])

            # Downsample the runs which have more snapshots #
            if j == 0:
                original_mass_loading = mass_loading
            else:
                mass_loading = plot_tools.linear_resample(mass_loading, len(original_mass_loading))
                lookback_times = plot_tools.linear_resample(lookback_times, len(original_mass_loading))

            # Plot the evolution of mass loading #
            axis.plot(lookback_times, mass_loading, color=colors[j],
                      label=r'$\mathrm{Au-%s}$' % str(re.split('_|.npy', names_flavours[j])[1]))
            axis.legend(loc='upper left', fontsize=16, frameon=False)  # Create the legend.
    # Save and close the figure #
    pdf.savefig(figure, bbox_inches='tight')
    plt.close()
    return None


def gas_temperature_edge_on_combination(pdf, redshift):
    """
    Plot a combination of the gas temperature projections for Auriga halo(es).
    :param pdf: path to save the pdf from main.make_pdf
    :param redshift: redshift from main.make_pdf
    :return: None
    """
    print("Invoking gas_temperature_edge_on_combination")
    # Get the names and sort them #
    path = '/u/di43/Auriga/plots/data/' + 'gteo/' + str(redshift) + '/'
    names = glob.glob(path + 'name_*')
    names.sort()

    # Generate the figure and set its parameters #
    figure = plt.figure(figsize=(15, 15))
    axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22, axiscbar, x, y, y2, \
    area = plot_tools.create_axes_combinations(
        res=res, boxsize=0.2 * 1e3, multiple12=True)
    for axis in [axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22]:
        plot_tools.set_axes(axis, xlim=[-240, 240], ylim=[-240, 240], aspect=None, size=30)
    for axis in [axis00, axis01, axis02, axis10, axis11, axis12]:
        axis.set_xticklabels([])
    for axis in [axis01, axis02, axis11, axis12, axis21, axis22]:
        axis.set_yticklabels([])
    for axis in [axis20, axis21, axis22]:
        # axis.set_xticklabels(['', '-50', '0', '50', ''])
        axis.set_xlabel(r'$\mathrm{x/kpc}$', size=30)
    for axis in [axis00, axis10, axis20]:
        # axis.set_yticklabels(['', '-50', '0', '50', ''])
        axis.set_ylabel(r'$\mathrm{z/kpc}$', size=30)

    # Loop over all available haloes #
    axes = [axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22]
    for i, axis in zip(range(len(names)), axes):
        print("Plotting data for halo:", str(re.split('_|.npy', names[i])[1]))
        # Load the data #
        boxsize = np.load(path + 'boxsize_4_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        edge_on = np.load(path + 'edge_on_4_' + str(re.split('_|.npy', names[i])[1]) + '.npy')
        edge_on_rho = np.load(path + 'edge_on_rho_4_' + str(re.split('_|.npy', names[i])[1]) + '.npy')

        # Plot R200 #
        circle = plt.Circle((0, 0), 500 * boxsize, color='k', fill=False)
        axis.add_patch(circle)

        # Plot the density-weighted gas temperature projections #
        x = np.linspace(-0.5 * boxsize, +0.5 * boxsize, res)
        z = np.linspace(-0.5 * boxsize, +0.5 * boxsize, res)
        pcm = axis.pcolormesh(x * 1e3, z * 1e3, (edge_on / edge_on_rho).T,
                              norm=matplotlib.colors.LogNorm(vmin=1e4, vmax=1e7), rasterized=True, shading='gouraud',
                              cmap='Spectral_r')

        plot_tools.create_colorbar(axiscbar, pcm, label='$\mathrm{T/K}$', size=30)

        figure.text(0.01, 0.9, r'$\mathrm{Au-%s}$' % str(re.split('_|.npy', names[i])[1]), fontsize=30,
                    transform=axis.transAxes)

    # Save and close the figure #
    pdf.savefig(figure, bbox_inches='tight')
    plt.close()
    return None


def delta_sfr_regimes_combination(pdf):
    """
    Plot the evolution of star formation rate for different spatial regimes
    and the difference between Auriga haloes.
    :param pdf: path to save the pdf from main.make_pdf
    :return: None
    """
    print("Invoking delta_sfr_regimes_combination")
    n_bins = 130

    # Get limits based on the region #
    radial_cuts_min, radial_cuts_max = (0.0, 1e-3, 5e-3), (1e-3, 5e-3, 15e-3)

    # Generate the figure and set its parameters #
    figure = plt.figure(figsize=(20, 20))
    axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22, axis30, axis31, axis32, axis40, axis41, \
    axis42, axis50, axis51, axis52 = plot_tools.create_axes_combinations(
        res=res, boxsize=boxsize * 1e3, multiple9=True)

    for axis in [axis00, axis01, axis02, axis20, axis21, axis22, axis40, axis41, axis42]:
        axis2 = axis.twiny()
        plot_tools.set_axes_evolution(axis, axis2, ylim=[0, 23], ylabel=r'$\mathrm{SFR/(M_\odot\;yr^{-1})}$',
                                      aspect=None, size=25)
        if axis in [axis20, axis21, axis22, axis40, axis41, axis42]:
            axis2.set_xlabel('')
            axis2.set_xticklabels([])
            axis2.tick_params(top=False)

    for axis in [axis10, axis11, axis12, axis30, axis31, axis32, axis50, axis51, axis52]:
        axis2 = axis.twiny()
        plot_tools.set_axes_evolution(axis, axis2, ylim=[-1.1, 13], ylabel=r'$\mathrm{(\delta SFR)_{norm}}$',
                                      aspect=None, size=25)
        axis.set_yticks([0, 5, 10])
        axis2.set_xlabel('')
        axis2.set_xticklabels([])
        axis2.tick_params(top=False)

    for axis in [axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22, axis30, axis31, axis32, axis40,
                 axis41, axis42]:
        axis.set_xlabel('')
        axis.set_xticklabels([])
        axis.tick_params(top=False)
    for axis in [axis01, axis02, axis11, axis12, axis21, axis22, axis31, axis32, axis41, axis42, axis51, axis52]:
        axis.set_ylabel('')
        axis.set_yticklabels([])

    # Loop over all radial limits #
    SFR_axes, delta_SFR_axes = [[axis00, axis20, axis40], [axis01, axis21, axis41], [axis02, axis22, axis42]], [
        [axis10, axis30, axis50], [axis11, axis31, axis51], [axis12, axis32, axis52]]
    for radial_cut_min, radial_cut_max, top_axes, bottom_axes in zip(radial_cuts_min, radial_cuts_max, SFR_axes,
                                                                     delta_SFR_axes):
        # Get the names and sort them #
        path = '/u/di43/Auriga/plots/data/' + 'dsr/' + str(radial_cut_max) + '/'
        names = glob.glob(path + 'name_*')
        names.sort()

        # Split the names into 3 groups and plot the three flavours of a halo together (i.e. original, NoR and NoRNoQ) #
        names_groups = np.array_split(names, 3)
        labels = [r'$\mathrm{Original}$', r'$\mathrm{NoR}$', r'$\mathrm{NoRNoQ}$']
        for i, top_axis, bottom_axis in zip(range(len(names_groups)), top_axes, bottom_axes):
            names_flavours = names_groups[i]
            # Loop over all available flavours #
            for j in range(len(names_flavours)):
                print("Plotting data for halo:", str(re.split('_|.npy', names_flavours[j])[1]))
                # Load the data #
                weights = np.load(path + 'weights_' + str(re.split('_|.npy', names_flavours[j])[1]) + '.npy')
                lookback_times = np.load(
                    path + 'lookback_times_' + str(re.split('_|.npy', names_flavours[j])[1]) + '.npy')

                # Plot the evolution of SFR and the normalised delta SFR #
                counts, bins, bars = top_axis.hist(lookback_times, weights=weights, histtype='step', bins=n_bins,
                                                   range=[0, 13], color=colors2[j], lw=3, label=labels[j])
                if j == 0:
                    original_bins, original_counts = bins, counts
                else:
                    bottom_axis.plot(original_bins[:-1], np.divide(counts - original_counts, original_counts),
                                     color=colors2[j], lw=3, label=labels[j])

            # Add the text #
            if top_axis in [axis00, axis20, axis40]:
                figure.text(0.01, 0.9, r'$\mathrm{Au-%s}$' % str(re.split('_|.npy', names_flavours[0])[1]), fontsize=25,
                            transform=top_axis.transAxes)
        figure.text(0.5, 0.9, r'$\mathrm{%.0f<r/kpc\leq%.0f}$' % (
            (np.float(radial_cut_min) * 1e3), (np.float(radial_cut_max) * 1e3)), fontsize=25,
                    transform=top_axes[0].transAxes)

        # Create the legend #
        axis00.legend(loc='center left', fontsize=20, frameon=False, numpoints=1)
        axis10.legend(loc='upper left', fontsize=20, frameon=False, numpoints=1)

    # Save and close the figure #
    pdf.savefig(figure, bbox_inches='tight')
    plt.close()
    return None
