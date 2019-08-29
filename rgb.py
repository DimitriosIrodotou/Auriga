import matplotlib.pyplot as plt
from loadmodules import *
from matplotlib import gridspec
from pylab import *

import eat
from build_stellar_projection import get_stellar_projection

# Start the time #
start_time = time.time()
date = time.strftime('%d\%m\%y\%H%M')


def rgb_plot(s, mask, halo_number, outdir, title, extend='.png'):
    """
    Face-on projected stellar density. The image is synthesised from a projection of the K-, B- and U-band luminosity of stars, which are shown by
    the red, green and blue colour channels, in logarithmic intervals, respectively. Younger (older) star particles are therefore represented by
    bluer (redder) colours

    :param s: an instance of the gadget_snapshot class.
    :param mask: select which stars you want.
    :param halo_number: number of the halo.
    :param outdir: path to save the plot.
    :param extend: file type of the plot.
    :return: saves the plot
    """

    plt.close()
    plt.rcParams.update({'font.size': 18})

    XYplane = get_stellar_projection(s, mask, 0)
    XZplane = get_stellar_projection(s, mask, 1)
    ZYplane = get_stellar_projection(s, mask, 2)

    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(1, 1)
    # axxz = fig.add_subplot(gs.new_subplotspec((0,0)))
    axxy = fig.add_subplot(gs.new_subplotspec((0, 0)))

    # gs.update(wspace=0, hspace=0)

    axxy.imshow(XYplane, interpolation='nearest', origin='lower', extent=[-25, 25, -25, 25])
    axxy.set_xlabel(r'$x \rm (kpc)$')
    axxy.set_ylabel(r'$y \rm (kpc)$')
    # axxy.set_xlim(0, 2*res)
    # axxy.set_ylim(0, 2*res)

    # axxz.imshow( XZplane, interpolation='nearest', origin='lower' )
    # axxz.set_ylabel('z [kpc]')
    # fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

    plt.title(title)

    plt.savefig(outdir + 'rgb_Au' + halo_number + '-' + date + extend, bbox_inches='tight')


# Set these parameters #
title = 'AGN'  # Title of the plot
datadir = '/virgo/simulations/Auriga/level4_MHD/halo_22'  # Path to get the data from
level, halo_number = (re.split('level|_MHD/halo_', datadir)[1:3])  # Get level of the Auriga simulation and halo
outdir = '/u/di43/Auriga/plots/'  # Where do you want to save the plots?
startsnap = 127  # Which snapshot do you want to use? Snapshot number - 127 = redshift
endsnap = 1  # How many snapshot you want to read and make plots for?

for ifile in range(0, endsnap):  # Loop over snapshots

    halo_snap = startsnap + ifile  # snapshot number - 127 = redshift 0
# s, sf = eat_snap_and_fof(level, halo_number, halo_snap, datadir + str(level) + '_MHD/halo_' + str(halo_number) + '/output/')
s, sf = eat.eat_snap_and_fof(int(level), int(halo_number), startsnap, datadir + '/output/')
times = s.cosmology_get_lookback_time_from_a(s.time.astype('float64'), is_flat=True)  # get lookback time

# Mask your galaxies and select the particles you want to plot (istars).
ages = s.cosmology_get_lookback_time_from_a(s.data['age'].astype('f8'), is_flat=True)  # Get ages of stars
istars, = np.where((s.type == 4) & (ages > 0.) & (s.halo == 0) & (s.subhalo == 0) & (s.r() * 1000. < 30) & (abs(s.pos[:, 0] * 1000.) < 5.))
rgb_plot(s, istars, str(halo_number), outdir, title)  # Make a RGB plot

# Print the total time #
print('--- %s seconds ---' % (time.time() - start_time))