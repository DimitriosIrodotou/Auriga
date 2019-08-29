import matplotlib.pyplot as plt
import numpy as np
from loadmodules import *
from matplotlib import gridspec
from numpy import pi, cos, sin
from pylab import *

import eat

# Start the time #
start_time = time.time()
date = time.strftime("%d\%m\%y\%H%M")


def Plot_sdens(x_pos, y_pos, z_pos, halo_number, outdir, title, xmax=30, extend=".png"):
    """
    Stellar surface density. Create an XY an XZ map of all stars.

    :param x_pos: the x position of the galaxy
    :param y_pos: the y position of the galaxy
    :param z_pos: the z position of the galaxy
    :param halo_number: number of the halo.
    :param outdir: path to save the plot.
    :param xmax: the maximum x-pos
    :param extend: file type of the plot.
    :return: saves the plot
    """

    numbins = 300
    count, xedges, yedges = np.histogram2d(x_pos, y_pos, bins=numbins, range=[[-xmax, xmax], [-xmax, xmax]])
    countlog = np.ma.log10(count)
    vmin = 0.  # countlog.min()*1.1
    vmax = 5.  # countlog.max()*0.9
    if "DM" in halo_number:
        vmax = 2.
    extent = [-xmax, xmax, -xmax, xmax]
    levels = np.arange(vmin, vmax + 0.5, 0.25)

    # Generate initial figure
    plt.close()
    fig = plt.figure(figsize=(9, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1], width_ratios=(20, 1))
    gs.update(hspace=0.0)

    # Generate an XY map of all stars
    ax0 = plt.subplot(gs[0, 0])
    plt.yticks(fontsize=16)
    ax0.set_ylabel("$y$ (kpc)", fontsize=16)
    ax0.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')

    pl1 = plt.imshow(countlog.T, extent=extent, origin='lower', cmap='magma', interpolation='bicubic')
    # pl1 = plt.imshow(countlog.T,extent=extent,origin='lower',cmap='Spectral_r',vmin=vmin,vmax=vmax,interpolation='bicubic')

    # Plot the contours
    numbins = 70
    count, xedges, yedges = np.histogram2d(x_pos, y_pos, bins=numbins, range=[[-xmax, xmax], [-xmax, xmax]])
    countlog = np.ma.log10(count)

    cont = plt.contour(countlog.T, colors="k", linewidth=2, linestyle="-", extent=extent, levels=levels)
    # plt.clabel(cont, fmt = '%2.1f', colors = 'w', fontsize=14)

    # Generate an XZ map of all stars
    ax1 = plt.subplot(gs[1, 0], sharex=ax0)
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    ax1.set_xlabel("$x$ (kpc)", fontsize=16)
    ax1.set_ylabel("$z$ (kpc)", fontsize=16)

    # Plot the contours
    numbins = 300
    count, xedges, yedges = np.histogram2d(x_pos, z_pos, bins=numbins, range=[[-xmax, xmax], [-xmax, xmax]])
    countlog = np.ma.log10(count)

    extent = [-xmax, xmax, -xmax, xmax]
    levels = np.arange(vmin, vmax + 0.5, 0.5)

    plt.imshow(countlog.T, extent=extent, origin='lower', cmap='magma', interpolation='bicubic')
    # plt.imshow(countlog.T,extent=extent,origin='lower',cmap='Spectral_r',vmin=vmin,vmax=vmax,interpolation='bicubic')

    numbins = 70
    count, xedges, yedges = np.histogram2d(x_pos, z_pos, bins=numbins, range=[[-xmax, xmax], [-xmax, xmax]])
    countlog = np.ma.log10(count)
    cont = plt.contour(countlog.T, colors="k", linewidth=2, linestyle="-", extent=extent, levels=levels)
    # plt.clabel(cont, fmt = '%2.1f', colors = 'w', fontsize=14)

    # Create the colorbar
    ax2 = plt.subplot(gs[:, 1])
    cbar = plt.colorbar(pl1, cax=ax2)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_ticks(levels)

    plt.title(title)

    plt.savefig(outdir + "sden_Au" + halo_number + '-' + date + extend, bbox_inches='tight')


def Rotate_vector(xbar, ybar, zbar, tangle):
    """
    Transform bar back to horizontal position
    
    :param xbar: the x-position of the bar
    :param ybar: the y-position of the bar
    :param zbar: the z-position of the bar
    :param tangle: the rotation angle
    :return: the rotated x, y and z position
    """
    x_pos = (xbar) * cos(tangle) - (ybar) * sin(tangle)
    y_pos = (xbar) * sin(tangle) + (ybar) * cos(tangle)
    z_pos = zbar

    return x_pos, y_pos, z_pos


datadir = "/virgo/simulations/Auriga/level4_MHD/halo_22"  # Path to get the data from
title = "AGN"  # Title of the plot
outdir = "/u/di43/Auriga/plots/"  # Where do you want to save the plots?
halo_number = 22  # Which Auriga halo do you want to use? Integer in range(1, 31)
level = 4  # Which level of the Auriga simulation do you want to use? 3 = high, 4 = 'normal' and 5 = low.
startsnap = 127  # Which snapshot do you want to use? Snapshot number - 127 = redshift
endsnap = 1  # How many snapshot you want to read and make plots for?

phase_array = []
phase_inst = []

for ifile in range(0, endsnap):  # Loop over snapshots

    halo_snap = startsnap + ifile
    # s, sf = eat_snap_and_fof(level, halo_number, halo_snap, datadir + str(level) + "_MHD/halo_" + str(halo_number) + "/output/")
    s, sf = eat.eat_snap_and_fof(level, halo_number, halo_snap, datadir + "/output/")
    times = s.cosmology_get_lookback_time_from_a(s.time.astype('float64'), is_flat=True)  # get lookback time

    ages = s.cosmology_get_lookback_time_from_a(s.data['age'].astype('f8'), is_flat=True)  # get ages of stars
    istars, = numpy.where((s.type == 4) & (ages > 0.) & (s.halo == 0) & (s.subhalo == 0) & (s.r() * 1000. < 30) & (
            abs(s.pos[:, 0] * 1000.) < 5.))  # select which stars we want
    x_now, y_now, z_now = s.pos[istars, 2] * 1000, s.pos[istars, 1] * 1000, s.pos[istars, 0] * 1000  # Load positions and convert from Mpc to Kpc
    vx_now, vy_now, vz_now = s.vel[istars, 2], s.vel[istars, 1], s.vel[istars, 0]  # Load velocities
    Plot_sdens(x_now, y_now, z_now, str(halo_number), outdir, title, xmax=25)  # Make a stellar surface density plot (face-on and edge-on)

    # Calculate bar strength from Fourier modes of surface density (see e.g. sec 2.3.2 from Athanassoula et al. 2013)
    # Number of radial bins
    nr1 = 50

    # Radius of each particle
    rnow = np.sqrt(x_now[:] ** 2 + y_now[:] ** 2)

    # Initialise fourier components
    r_m = np.zeros(nr1)
    alpha_0 = np.zeros(nr1)
    alpha_2 = np.zeros(nr1)
    beta_2 = np.zeros(nr1)
    phase_bar = np.zeros(nr1)

    # Split up in radius bins
    for i in range(0, nr1):

        r_s = float(i) * 0.25
        r_b = float(i) * 0.25 + 0.25
        r_m[i] = float(i) * 0.25 + 0.125

        xfit = x_now[(rnow < r_b) & (rnow > r_s) & (abs(z_now) < 0.8)]
        yfit = y_now[(rnow < r_b) & (rnow > r_s) & (abs(z_now) < 0.8)]

        l = len(xfit)

        for k in range(0, l):
            th_i = math.atan2(yfit[k], xfit[k])
            alpha_0[i] = alpha_0[i] + 1
            alpha_2[i] = alpha_2[i] + cos(2 * th_i)
            beta_2[i] = beta_2[i] + sin(2 * th_i)
            phase_bar[i] = 0.5 * math.atan2(beta_2[i], alpha_2[i])

    # Calculate the bar angle (between 0.5 and 4kpc)
    r_b = 4.0
    r_s = 0.5

    k = 0
    phase_in = 0.
    test_phase = []
    rtp = []
    for i in range(0, nr1):
        if ((r_m[i] < r_b) & (r_m[i] > r_s)):
            test_phase.append(0.5 * math.atan2(beta_2[i], alpha_2[i]))
            rtp.append(r_m[i])
            if k > 0:
                if ((test_phase[k] - test_phase[k - 1]) > 0.6) or ((test_phase[k] - test_phase[k - 1]) < -0.6):
                    print("Have to break ")
                    print("(test_phase[k-1]-test_phase[k-2])", (test_phase[k] - test_phase[k - 1]))
                    break
            phase_in = phase_in + 0.5 * math.atan2(beta_2[i], alpha_2[i])
            k = k + 1
    phase_in = phase_in / float(k)

    phase_inst.append(phase_in)

    if ifile == 0:
        phase_array.append(phase_in)
        print("\nFirst snapshot phase ", phase_in)

    else:
        print("\nSnapshot number ", ifile, " phase in ", phase_in)

        dtheta = phase_inst[ifile] - phase_inst[ifile - 1]
        print("\ndtheta ", dtheta)

        if dtheta > 0.0:
            print("\n delta theta ", dtheta)
            print("\n phase_array[ifile-1] ", phase_array[ifile - 1])

            new_phase = phase_array[ifile - 1] + dtheta

            print("\n new phase phase_array[ifile-1] + dtheta ", new_phase)

        if dtheta < 0.0:
            print("\n !!!!!!!!!!!!!!")
            print("there is a jump")
            # value is proper delta theta = delta_theta(with jump) + 180 degrees (which is the jump)
            value = dtheta + pi  # If dtheta < 0 then value = dtheta - pi?
            print("\n dtheta + pi", value)

            new_phase = phase_array[ifile - 1] + value
            print("\n new_phase = phase_array[ifile-1] + value", new_phase)

        phase_array.append(new_phase)

    # Transform bar back to horizontal position
    x_pos, y_pos, z_pos = Rotate_vector(x_now, y_now, z_now, -phase_array[ifile])
    vx_pos, vy_pos, vz_pos = Rotate_vector(vx_now, vy_now, vz_now, -phase_array[ifile])

    # Plot again
    Plot_sdens(x_pos, y_pos, z_pos, str(halo_number) + '_Rot', outdir, title, xmax=25)

# Print the total time #
print("--- %s seconds ---" % (time.time() - start_time))