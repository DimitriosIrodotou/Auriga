import os
import re
import glob
import matplotlib

import numpy as np
import matplotlib.pyplot as plt

from const import *
from sfigure import *
from loadmodules import *
from matplotlib import gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

res = 512
level = 4
boxsize = 0.2
element = {'H':0, 'He':1, 'C':2, 'N':3, 'O':4, 'Ne':5, 'Mg':6, 'Si':7, 'Fe':8}


def gas_movie(data, read):
    """
    Plot stellar light, gas temperature, peculiar velocity and density projections for Auriga halo(es).
    :param data: data from main.make_pdf
    :param read: boolean to read new data.
    :return: None
    """
    print("Invoking gas_movie")

    # Check if a folder to save the data exists, if not create one #
    path = '/u/di43/Auriga/plots/data/' + 'gm/' + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Read the data #
    if read is True:
        
        # Get all available redshifts #
        haloes = data.get_haloes(level)
        for name, halo in haloes.items():
            redshifts = halo.get_redshifts()

        # Loop over all desired redshifts #
        redshift_cut = 5e-2
        # redshift_cut = 1e-14
        for redshift in np.flip(redshifts[np.where(redshifts <= redshift_cut)]):
            print(redshift)
            print(np.flip(redshifts[np.where(redshifts <= redshift_cut)]))
            
            # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
            particle_type = [0, 4]
            attributes = ['mass', 'ne', 'pos', 'rho', 'u']
            data.select_haloes(level, redshift, loadonlytype=particle_type, loadonlyhalo=0, loadonly=attributes)
            
            # Loop over all available haloes #
            for s in data:
                # Check if any of the haloes' data already exists, if not then read and save it #
                names = glob.glob(path + '/name_3000*')
                names = [re.split('_|.npy', name)[2] for name in names]
                if str(redshift) in names:
                    continue
                
                # Select the halo and rotate it based on its principal axes so galaxy's spin is aligned with the z-axis #
                s.calc_sf_indizes(s.subfind)
                if redshift < 1e-15:
                    rot = s.select_halo(s.subfind, rotate_disk=True, use_principal_axis=True, do_rotation=True)
                else:
                    s.select_halo(s.subfind, rotate_disk=False, use_principal_axis=False, do_rotation=False)
                    s.data['pos'] = rotate_value(s.data['pos'], rot)
                    s.data['vel'] = rotate_value(s.data['vel'], rot)
                
                gas_mask, = np.where(s.data['type'] == 0)
                
                # Get the density-weighted temperature projections #
                mean_weight = 4.0 / (1.0 + 3.0 * 0.76 + 4.0 * 0.76 * s.data['ne']) * 1.67262178e-24
                temperature = (5.0 / 3.0 - 1.0) * s.data['u'] / KB * (1e6 * parsec) ** 2.0 / (1e6 * parsec / 1e5) ** 2 * mean_weight
                s.data['temprho'] = s.data['rho'] * temperature
                # s.data['temprho'] = s.data['mass'] * temperature
                
                # ne = s.data['ne'][gas_mask]
                # metallicity = s.data['gz'][gas_mask]
                # XH = s.data['gmet'][gas_mask, element['H']]
                # yhelium = (1 - XH - metallicity) / (4. * XH)
                # mu = (1 + 4 * yhelium) / (1 + yhelium + ne)
                # temperature = GAMMA_MINUS1 * s.data['u'][gas_mask] * 1.0e10 * mu * PROTONMASS / BOLTZMANN
                # s.data['temprho'] = temperature
                
                temperature_face_on = s.get_Aslice("temprho", res=res, axes=[1, 2], box=[boxsize, boxsize], proj=True, proj_fact=0.05, numthreads=8)[
                    "grid"]
                rho_face_on = s.get_Aslice("rho", res=res, axes=[1, 2], box=[boxsize, boxsize], proj=True, proj_fact=0.05, numthreads=8)["grid"]
                temperature_edge_on = s.get_Aslice("temprho", res=res, axes=[1, 0], box=[boxsize, boxsize], proj=True, proj_fact=0.05, numthreads=8)[
                    "grid"]
                rho_edge_on = s.get_Aslice("rho", res=res, axes=[1, 0], box=[boxsize, boxsize], proj=True, proj_fact=0.05, numthreads=8)["grid"]
                
                # Get the radial velocity projections #
                cylindrical_radius = np.sqrt(np.sum(s.data['pos'][gas_mask, 1:] ** 2, axis=1))  # In Mpc.
                s.data['vrad'] = np.divide(np.sum(s.data['vel'][gas_mask, 1:] * s.data['pos'][gas_mask, 1:], axis=1), cylindrical_radius)
                vrad_face_on = s.get_Aslice("vrad", res=res, axes=[1, 2], box=[boxsize, boxsize], proj=True, proj_fact=0.05, numthreads=8)["grid"]
                
                # Re-normalise velocities #
                s.data['vz'] = s.data['vel'][gas_mask, 0]
                negative_mask, = np.where(s.data['pos'][gas_mask, 0] < 0)
                s.data['vz'][negative_mask] *= -1
                
                vrad_edge_on = s.get_Aslice("vz", res=res, axes=[1, 0], box=[boxsize, boxsize], proj=True, proj_fact=0.05, numthreads=8)["grid"]
                
                # Get the gas total pressure projections #
                elements_mass = [1.01, 4.00, 12.01, 14.01, 16.00, 20.18, 24.30, 28.08, 55.85, 88.91, 87.62, 91.22, 137.33]
                meanweight = np.sum(s.gmet[s.data['type'] == 0, 0:9], axis=1) / (
                    np.sum(s.gmet[s.data['type'] == 0, 0:9] / elements_mass[0:9], axis=1) + s.data['ne'] * s.gmet[s.data['type'] == 0, 0])
                Tfac = 1. / meanweight * (1.0 / (5. / 3. - 1.)) * KB / PROTONMASS * 1e10 * msol / 1.989e53
                
                # Un megabars (10**12dyne/cm**2)
                s.data['T'] = s.u / Tfac
                s.data['dens'] = s.rho / (1e6 * parsec) ** 3. * msol * 1e10
                s.data['Ptherm'] = s.data['dens'] * s.data['T'] / (meanweight * PROTONMASS)
                
                pressure_face_on = s.get_Aslice("Ptherm", res=res, axes=[1, 2], box=[boxsize, boxsize], proj=True, proj_fact=0.05, numthreads=8)[
                    "grid"]
                pressure_edge_on = s.get_Aslice("Ptherm", res=res, axes=[1, 0], box=[boxsize, boxsize], proj=True, proj_fact=0.05, numthreads=8)[
                    "grid"]
                
                # Save data for each halo in numpy arrays #
                np.save(path + 'name_' + str(s.haloname) + '_' + str(redshift), s.haloname)
                np.save(path + 'redshift_' + str(s.haloname) + '_' + str(redshift), redshift)
                np.save(path + 'rho_face_on_' + str(s.haloname) + '_' + str(redshift), rho_face_on)
                np.save(path + 'rho_edge_on_' + str(s.haloname) + '_' + str(redshift), rho_edge_on)
                np.save(path + 'vrad_face_on_' + str(s.haloname) + '_' + str(redshift), vrad_face_on)
                np.save(path + 'vrad_edge_on_' + str(s.haloname) + '_' + str(redshift), vrad_edge_on)
                np.save(path + 'pressure_face_on_' + str(s.haloname) + '_' + str(redshift), pressure_face_on)
                np.save(path + 'pressure_edge_on_' + str(s.haloname) + '_' + str(redshift), pressure_edge_on)
                np.save(path + 'temperature_face_on_' + str(s.haloname) + '_' + str(redshift), temperature_face_on)
                np.save(path + 'temperature_edge_on_' + str(s.haloname) + '_' + str(redshift), temperature_edge_on)
    
    # Get the redshifts and sort them #
    names = glob.glob(path + '/name_3000*')
    names.sort(reverse=True)
    names.append(names.pop(0))  # Move redshift zero to the end.
    
    # Loop over all available haloes #
    for i in range(len(names)):
        print(names[i])
        # Load the data #
        redshift = np.load(path + 'redshift_3000_' + str(re.split('_|.npy', names[i])[2]) + '.npy')
        rho_face_on = np.load(path + 'rho_face_on_3000_' + str(re.split('_|.npy', names[i])[2]) + '.npy')
        rho_edge_on = np.load(path + 'rho_edge_on_3000_' + str(re.split('_|.npy', names[i])[2]) + '.npy')
        vrad_face_on = np.load(path + 'vrad_face_on_3000_' + str(re.split('_|.npy', names[i])[2]) + '.npy')
        vrad_edge_on = np.load(path + 'vrad_edge_on_3000_' + str(re.split('_|.npy', names[i])[2]) + '.npy')
        pressure_face_on = np.load(path + 'pressure_face_on_3000_' + str(re.split('_|.npy', names[i])[2]) + '.npy')
        pressure_edge_on = np.load(path + 'pressure_edge_on_3000_' + str(re.split('_|.npy', names[i])[2]) + '.npy')
        temperature_face_on = np.load(path + 'temperature_face_on_3000_' + str(re.split('_|.npy', names[i])[2]) + '.npy')
        temperature_edge_on = np.load(path + 'temperature_edge_on_3000_' + str(re.split('_|.npy', names[i])[2]) + '.npy')
        
        # Generate the figure and set its parameters #
        plt.rcParams['savefig.facecolor'] = 'black'
        figure, axis = plt.subplots(1, figsize=(20, 20))
        gs = gridspec.GridSpec(2, 2, hspace=0, wspace=0)
        axis00 = plt.subplot(gs[0, 0])
        axis01 = plt.subplot(gs[0, 1])
        axis10 = plt.subplot(gs[1, 0])
        axis11 = plt.subplot(gs[1, 1])
        for axis in [axis00, axis01, axis10, axis11]:
            axis.axis('off')
            axis.set_xlim(-50, 50)
            axis.set_ylim(-50, 50)
            axis.set_aspect('auto')
        
        # Plot the projections #
        x = np.linspace(-0.5 * boxsize * 1e3, +0.5 * boxsize * 1e3, res + 1)
        y = np.linspace(-0.5 * boxsize * 1e3, +0.5 * boxsize * 1e3, res + 1)
        
        temperature = axis00.pcolormesh(x, y, (temperature_face_on / rho_face_on).T, norm=matplotlib.colors.LogNorm(vmin=3e3, vmax=5e7), cmap='ocean',
                                      rasterized=True)
        vrad = axis01.pcolormesh(x, y, vrad_face_on.T, cmap='bwr', vmin=-5e3, vmax=5e3, rasterized=True)
        rho = axis10.pcolormesh(x, y, (rho_face_on * boxsize * 1e3).T, norm=matplotlib.colors.LogNorm(vmin=1e6, vmax=1e10), cmap='magma',
                              rasterized=True)
        pressure = axis11.pcolormesh(x, y, pressure_face_on.T, norm=matplotlib.colors.LogNorm(vmin=1e3, vmax=1e8), cmap='cubehelix', rasterized=True)
        
        # Add colorbars in each panel #
        axes = [axis00, axis01, axis10, axis11]
        labels = [r'$\mathrm{T\;/K}$', r'$\mathrm{v_{rad}/(km\,s^{-1})}$', r'$\mathrm{\Sigma_{gas}\;/(M_\odot\;kpc^{-2})}$',
                  r'$\mathrm{P\;/(K\;cm^{-3})}$']
        attributes = [temperature, vrad, rho, pressure]
        ticks = [[5e3, 5e5, 5e7], [-1e4, 0, 1e4], [1e6, 1e8, 1e10], [1e4, 1e6, 1e8]]
        for axis, attribute, label, tick in zip(axes, attributes, labels, ticks):
            cbaxis = inset_axes(axis, width='30%', height='3%', loc=3)
            cb = plt.colorbar(attribute, cax=cbaxis, ticks=tick, orientation='horizontal', extend='both')
            cb.set_label(label, size=18, color='white')
            cbaxis.xaxis.tick_top()
            cbaxis.xaxis.set_label_position("top")
            cbaxis.xaxis.set_tick_params(color='white')
            plt.setp(plt.getp(cbaxis.axes, 'xticklabels'), color='white')
            cbaxis.tick_params(direction='out', which='both', top='on')
        
        figure.tight_layout()
        figure.text(0.1, 0.97, 'z = %.3f' % float(redshift), color='w', fontsize=18, transform=axis10.transAxes)
        plt.savefig('/u/di43/Auriga/plots/gm/' + 'gmf_%04d.png' % i, bbox_inches='tight')  # Save the figure.
        plt.close()
        
        # Generate the figure and set its parameters #
        figure, axis = plt.subplots(1, figsize=(20, 20))
        gs = gridspec.GridSpec(2, 2, hspace=0, wspace=0)
        axis00 = plt.subplot(gs[0, 0])
        axis01 = plt.subplot(gs[0, 1])
        axis10 = plt.subplot(gs[1, 0])
        axis11 = plt.subplot(gs[1, 1])
        for axis in [axis00, axis01, axis10, axis11]:
            axis.axis('off')
            axis.set_xlim(-100, 100)
            axis.set_ylim(-100, 100)
            axis.set_aspect('auto')
        
        # Plot the projections #
        temperature = axis00.pcolormesh(x, y, (temperature_edge_on / rho_edge_on).T, norm=matplotlib.colors.LogNorm(vmin=3e3, vmax=5e7), cmap='ocean',
                                      rasterized=False)
        vrad = axis01.pcolormesh(x, y, vrad_edge_on.T, cmap='bwr', vmin=-3.5e4, vmax=3.5e4, rasterized=True)
        rho = axis10.pcolormesh(x, y, (rho_edge_on * boxsize * 1e3).T, norm=matplotlib.colors.LogNorm(vmin=1e6, vmax=1e10), cmap='magma',
                              rasterized=True)
        pressure = axis11.pcolormesh(x, y, pressure_edge_on.T, norm=matplotlib.colors.LogNorm(vmin=1e3, vmax=1e8), cmap='cubehelix', rasterized=True)
        
        # Add colorbars in each panel #
        axes = [axis00, axis01, axis10, axis11]
        labels = [r'$\mathrm{T\;/K}$', r'$\mathrm{v_{rad}/(km\,s^{-1})}$', r'$\mathrm{\Sigma_{gas}\;/(M_\odot\;kpc^{-2})}$',
                  r'$\mathrm{P\;/(K\;cm^{-3})}$']
        attributes = [temperature, vrad, rho, pressure]
        ticks = [[5e3, 5e5, 5e7], [-3.5e4, 0, 3.5e4], [1e6, 1e8, 1e10], [1e4, 1e6, 1e8]]
        for axis, attribute, label, tick in zip(axes, attributes, labels, ticks):
            cbaxis = inset_axes(axis, width='30%', height='3%', loc=3)
            cb = plt.colorbar(attribute, cax=cbaxis, ticks=tick, orientation='horizontal', extend='both')
            cb.set_label(label, size=18, color='white')
            cbaxis.xaxis.tick_top()
            cbaxis.xaxis.set_label_position("top")
            cbaxis.xaxis.set_tick_params(color='white')
            plt.setp(plt.getp(cbaxis.axes, 'xticklabels'), color='white')
            cbaxis.tick_params(direction='out', which='major', top='on')
        
        figure.tight_layout()
        figure.text(0.1, 0.97, 'z = %.3f' % float(redshift), color='w', fontsize=18, transform=axis10.transAxes)
        plt.savefig('/u/di43/Auriga/plots/gm/' + 'gme_%04d.png' % i, bbox_inches='tight')  # Save the figure.
        plt.close()
    
    return None


def rotate_value(value, matrix):
    new_value = np.zeros(np.shape(value))
    for i in range(3):
        new_value[:, i] = (value * matrix[i, :][None, :]).sum(axis=1)
    return new_value
