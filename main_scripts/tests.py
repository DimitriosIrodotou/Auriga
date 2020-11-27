import re
import glob
import plot_tools

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.style as style

from const import *
from sfigure import *
from loadmodules import *

style.use("classic")
plt.rcParams.update({'font.family':'serif'})

res = 512
boxsize = 0.06
default_level = 4
default_redshift = 0.0
colors = ['black', 'tab:red', 'tab:green', 'tab:blue', 'tab:orange', 'tab:purple']
element = {'H':0, 'He':1, 'C':2, 'N':3, 'O':4, 'Ne':5, 'Mg':6, 'Si':7, 'Fe':8}


def test_gas_flow(date):
    """
    Test the evolution of gas flow and mass loading for Auriga halo(es).
    :param date: date.
    :return: None
    """
    print("Invoking test_gas_flow")
    path = '/u/di43/Auriga/plots/data/' + 'gf/'

    # Get the names and sort them #
    names = glob.glob(path + '/name_06NoRNoQ.*')
    names.sort()

    # Loop over all available haloes #
    for i in range(len(names)):
        # Generate the figure and set its parameters #
        figure, axis = plt.subplots(1, figsize=(10, 10))

        # Load and plot the data #
        positions = np.load(path + 'positions_' + str(re.split('_|.npy', names[i])[1]) + '.npy', allow_pickle=True)
        radial_velocities = np.load(path + 'radial_velocities_' + str(re.split('_|.npy', names[i])[1]) + '.npy',
            allow_pickle=True)
        Rvirs = np.load(path + 'Rvirs_' + str(re.split('_|.npy', names[i])[1]) + '.npy', allow_pickle=True)

        # Plot the evolution of gas flow and mass loading #
        draw_circle = plt.Circle((0, 0), 0.5 * Rvirs[-1], fill=False)
        draw_circle2 = plt.Circle((0, 0), 0.5 * Rvirs[-1] + 1, fill=False)
        axis.add_artist(draw_circle)
        axis.add_artist(draw_circle2)
        pc = axis.scatter(positions[-1][:, 2], positions[-1][:, 0], edgecolor='none', c=radial_velocities[-1],
            cmap='seismic', vmin=-400, vmax=400)
        plt.colorbar(pc, orientation='horizontal', ticks=[-400, 0, 400])
        plt.savefig('/u/di43/Auriga/plots/test-' + str(date) + '.png', bbox_inches='tight')  # Save the figure.
        plt.close()
    return None
