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

res = 512
level = 4
boxsize = 0.1


def gas_movie(data, read):
    """
    Plot stellar light, gas temperature, peculiar velocity and density projections for Auriga halo(es).
    :param data: data from main.make_pdf
    :param read: boolean to read new data.
    :return: None
    """
    
    # Check if a folder to save the data exists, if not create one #
    path = '/u/di43/Auriga/plots/data/' + 'gm/' + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Read the data #
    if read is True:
        # redshift_cut = 5e-2
        redshift_cut = 1e-14
        
        # Get all available redshifts #
        haloes = data.get_haloes(level)
        for name, halo in haloes.items():
            redshifts = halo.get_redshifts()
        
        for redshift in np.flip(redshifts[np.where(redshifts <= redshift_cut)]):
            print(redshift)
            print(np.flip(redshifts[np.where(redshifts <= redshift_cut)]))
            
            # Read desired galactic property(ies) for specific particle type(s) for Auriga halo(es) #
            particle_type = [0, 4]
            attributes = ['mass', 'ne', 'pos', 'rho', 'u']
            data.select_haloes(level, redshift, loadonlytype=particle_type, loadonlyhalo=0, loadonly=attributes)
            
            # Loop over all haloes #
            for s in data:
                # Check if any of the haloes' data already exists, if not then read and save it #
                names = glob.glob(path + '/name_18_*')
                names = [re.split('_|.npy', name)[2] for name in names]
                # if str(redshift) in names:
                #     continue
                
                # Select the halo and rotate it based on its principal axes so galaxy's spin is aligned with the z-axis #
                s.calc_sf_indizes(s.subfind)
                if redshift < 1e-15:
                    rot = s.select_halo(s.subfind, rotate_disk=True, use_principal_axis=True, do_rotation=True)
                else:
                    s.select_halo(s.subfind, rotate_disk=False, use_principal_axis=False, do_rotation=False)
                    s.data['pos'] = rotate_value(s.data['pos'], rot)
                    s.data['vel'] = rotate_value(s.data['vel'], rot)
                
                # Get the density-weighted temperature projections #
                mean_weight = 4.0 / (1.0 + 3.0 * 0.76 + 4.0 * 0.76 * s.data['ne']) * 1.67262178e-24
                temperature = (5.0 / 3.0 - 1.0) * s.data['u'] / KB * (1e6 * parsec) ** 2.0 / (1e6 * parsec / 1e5) ** 2 * mean_weight
                s.data['temprho'] = s.rho * temperature
                
                face_on = s.get_Aslice("temprho", res=res, axes=[1, 2], box=[boxsize, boxsize], proj=True, proj_fact=0.05, numthreads=8)["grid"]
                face_on_rho = s.get_Aslice("rho", res=res, axes=[1, 2], box=[boxsize, boxsize], proj=True, proj_fact=0.05, numthreads=8)["grid"]
                edge_on = s.get_Aslice("temprho", res=res, axes=[1, 0], box=[boxsize, boxsize], proj=True, proj_fact=0.05, numthreads=8)["grid"]
                edge_on_rho = s.get_Aslice("rho", res=res, axes=[1, 0], box=[boxsize, boxsize], proj=True, proj_fact=0.05, numthreads=8)["grid"]
                
                # Get the radial velocity projections #
                gas_mask, = np.where(s.type == 0)
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
                meanweight = np.sum(s.gmet[s.type == 0, 0:9], axis=1) / (
                    np.sum(s.gmet[s.type == 0, 0:9] / elements_mass[0:9], axis=1) + s.data['ne'] * s.gmet[s.type == 0, 0])
                Tfac = 1. / meanweight * (1.0 / (5. / 3. - 1.)) * KB / PROTONMASS * 1e10 * msol / 1.989e53

                # Un megabars (10**12dyne/cm**2)
                s.data['T'] = s.u / Tfac
                s.data['dens'] = s.rho / (1e6 * parsec) ** 3. * msol * 1e10
                s.data['Ptherm'] = s.data['dens'] * s.data['T'] / (meanweight * PROTONMASS)

                pressure_face_on = s.get_Aslice("Ptherm", res=res, axes=[1, 2], box=[boxsize, boxsize], proj=True, proj_fact=0.125, numthreads=8)[
                    "grid"]
                pressure_edge_on = s.get_Aslice("Ptherm", res=res, axes=[1, 0], box=[boxsize, boxsize], proj=True, proj_fact=0.125, numthreads=8)[
                    "grid"]
                
                # Save data for each halo in numpy arrays #
                np.save(path + 'name_' + str(s.haloname) + '_' + str(redshift), s.haloname)
                np.save(path + 'face_on_' + str(s.haloname) + '_' + str(redshift), face_on)
                np.save(path + 'edge_on_' + str(s.haloname) + '_' + str(redshift), edge_on)
                np.save(path + 'redshift_' + str(s.haloname) + '_' + str(redshift), redshift)
                np.save(path + 'face_on_rho_' + str(s.haloname) + '_' + str(redshift), face_on_rho)
                np.save(path + 'edge_on_rho_' + str(s.haloname) + '_' + str(redshift), edge_on_rho)
                np.save(path + 'vrad_face_on_' + str(s.haloname) + '_' + str(redshift), vrad_face_on)
                np.save(path + 'vrad_edge_on_' + str(s.haloname) + '_' + str(redshift), vrad_edge_on)
                np.save(path + 'pressure_face_on_' + str(s.haloname) + '_' + str(redshift), pressure_face_on)
                np.save(path + 'pressure_edge_on_' + str(s.haloname) + '_' + str(redshift), pressure_edge_on)
    
    # Get the redshifts and sort them #
    names = glob.glob(path + '/name_18_*')
    names.sort(reverse=True)
    names.append(names.pop(0))  # Move redshift zero to the end.
    
    # Loop over all available haloes #
    for i in range(len(names)):
        print(names[i])
        # Load and plot the data #
        face_on = np.load(path + 'face_on_18_' + str(re.split('_|.npy', names[i])[2]) + '.npy')
        edge_on = np.load(path + 'edge_on_18_' + str(re.split('_|.npy', names[i])[2]) + '.npy')
        redshift = np.load(path + 'redshift_18_' + str(re.split('_|.npy', names[i])[2]) + '.npy')
        face_on_rho = np.load(path + 'face_on_rho_18_' + str(re.split('_|.npy', names[i])[2]) + '.npy')
        edge_on_rho = np.load(path + 'edge_on_rho_18_' + str(re.split('_|.npy', names[i])[2]) + '.npy')
        vrad_face_on = np.load(path + 'vrad_face_on_18_' + str(re.split('_|.npy', names[i])[2]) + '.npy')
        vrad_edge_on = np.load(path + 'vrad_edge_on_18_' + str(re.split('_|.npy', names[i])[2]) + '.npy')
        pressure_face_on = np.load(path + 'pressure_face_on_18_' + str(re.split('_|.npy', names[i])[2]) + '.npy')
        pressure_edge_on = np.load(path + 'pressure_edge_on_18_' + str(re.split('_|.npy', names[i])[2]) + '.npy')
        
        # Generate the figure and define its parameters #
        figure, ax = plt.subplots(1, figsize=(20, 20))
        gs = gridspec.GridSpec(2, 2)
        ax00 = plt.subplot(gs[0, 0])
        ax01 = plt.subplot(gs[0, 1])
        ax10 = plt.subplot(gs[1, 0])
        ax11 = plt.subplot(gs[1, 1])
        for a in [ax00, ax01, ax10, ax11]:
            a.axis('off')
            a.set_xlim(-50, 50)
            a.set_ylim(-50, 50)
            a.set_aspect('auto')
        
        # Plot the projections #
        x = np.linspace(-0.5 * boxsize * 1e3, +0.5 * boxsize * 1e3, res + 1)
        y = np.linspace(-0.5 * boxsize * 1e3, +0.5 * boxsize * 1e3, res + 1)
        ax00.pcolormesh(x, y, (face_on / face_on_rho).T, norm=matplotlib.colors.LogNorm(vmin=1e3, vmax=1e7), cmap='viridis', rasterized=True)
        ax01.pcolormesh(x, y, vrad_face_on.T, cmap='coolwarm', vmin=-8e3, vmax=8e3, rasterized=True)
        ax10.pcolormesh(x, y, (face_on_rho * boxsize * 1e3).T, norm=matplotlib.colors.LogNorm(vmin=1e6, vmax=1e10), cmap='magma', rasterized=True)
        ax11.pcolormesh(x, y, pressure_face_on.T, norm=matplotlib.colors.LogNorm(), cmap='cividis', rasterized=True)
        
        figure.tight_layout()
        figure.text(0.0, 0.97, 'z = %.3f' % float(redshift), color='k', fontsize=16, transform=ax00.transAxes)
        plt.savefig('/u/di43/Auriga/plots/gm/' + 'gmf_%04d.png' % i, bbox_inches='tight')  # Save the figure.
        plt.close()
        
        # Generate the figure and define its parameters #
        figure, ax = plt.subplots(1, figsize=(20, 20))
        gs = gridspec.GridSpec(2, 2)
        ax00 = plt.subplot(gs[0, 0])
        ax01 = plt.subplot(gs[0, 1])
        ax10 = plt.subplot(gs[1, 0])
        ax11 = plt.subplot(gs[1, 1])
        for a in [ax00, ax01, ax10, ax11]:
            a.axis('off')
            a.set_xlim(-50, 50)
            a.set_ylim(-50, 50)
            a.set_aspect('auto')
        
        # Plot the projections #
        ax00.pcolormesh(x, y, (edge_on / edge_on_rho).T, norm=matplotlib.colors.LogNorm(vmin=1e3, vmax=1e7), cmap='viridis', rasterized=True)
        ax01.pcolormesh(x, y, vrad_edge_on.T, cmap='coolwarm', vmin=-3.5e4, vmax=3.5e4, rasterized=True)
        ax10.pcolormesh(x, y, (edge_on_rho * boxsize * 1e3).T, norm=matplotlib.colors.LogNorm(vmin=1e6, vmax=1e10), cmap='magma', rasterized=True)
        ax11.pcolormesh(x, y, pressure_edge_on.T, norm=matplotlib.colors.LogNorm(), cmap='cividis', rasterized=True)
        
        figure.tight_layout()
        figure.text(0.0, 0.97, 'z = %.3f' % float(redshift), color='k', fontsize=16, transform=ax00.transAxes)
        plt.savefig('/u/di43/Auriga/plots/gm/' + 'gme_%04d.png' % i, bbox_inches='tight')  # Save the figure.
        plt.close()
    
    return None


def rotate_value(value, matrix):
    new_value = np.zeros(np.shape(value))
    for i in range(3):
        new_value[:, i] = (value * matrix[i, :][None, :]).sum(axis=1)
    return new_value
