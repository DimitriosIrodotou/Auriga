import numpy as np
import matplotlib.pyplot as plt

from matplotlib import gridspec
import satellite_utilities

res = 512
boxsize = 0.06


def binned_median_1sigma(x_data, y_data, bin_type, n_bins, log=False):
    """
    Calculate the binned median and 1-sigma lines in either equal number of width bins.
    :param x_data: x-axis data.
    :param y_data: y-axis data.
    :param bin_type: equal number or width type of the bin.
    :param n_bins: number of the bin.
    :param log: boolean.
    :return: x_value, median, shigh, slow
    """
    if bin_type == 'equal_number':
        if log is True:
            x = np.log10(x_data)
        else:
            x = x_data

        # Declare arrays to store the data #
        n_bins = np.quantile(np.sort(x), np.linspace(0, 1, n_bins + 1))
        slow = np.zeros(len(n_bins))
        shigh = np.zeros(len(n_bins))
        median = np.zeros(len(n_bins))
        x_value = np.zeros(len(n_bins))

        # Loop over all bins and calculate the median and 1-sigma lines #
        for i in range(len(n_bins) - 1):
            index, = np.where((x >= n_bins[i]) & (x < n_bins[i + 1]))
            x_value[i] = np.mean(x_data[index])
            if len(index) > 0:
                median[i] = np.nanmedian(y_data[index])
                slow[i] = np.nanpercentile(y_data[index], 15.87)
                shigh[i] = np.nanpercentile(y_data[index], 84.13)

        return x_value, median, shigh, slow

    elif bin_type == 'equal_width':
        if log is True:
            x = np.log10(x_data)
        else:
            x = x_data
        x_low = min(x)

        # Declare arrays to store the data #
        bin_width = (max(x) - min(x)) / n_bins
        slow = np.zeros(n_bins)
        shigh = np.zeros(n_bins)
        median = np.zeros(n_bins)
        x_value = np.zeros(n_bins)

        # Loop over all bins and calculate the median and 1-sigma lines #
        for i in range(n_bins):
            index, = np.where((x >= x_low) & (x < x_low + bin_width))
            x_value[i] = np.mean(x_data[index])
            if len(index) > 0:
                median[i] = np.nanmedian(y_data[index])
                slow[i] = np.nanpercentile(y_data[index], 15.87)
                shigh[i] = np.nanpercentile(y_data[index], 84.13)
            x_low += bin_width

        return x_value, median, shigh, slow


def binned_sum(x_data, y_data, n_bins, log=False):
    """
    Calculate the binned sum line.
    :param x_data: x-axis data.
    :param y_data: y-axis data.
    :param n_bins: number of the bin.
    :param log: boolean.
    :return: x_value, sum
    """
    if log is True:
        x = np.log10(x_data)
    else:
        x = x_data
    x_low = min(x)

    # Declare arrays to store the data #
    bin_width = (max(x) - min(x)) / n_bins
    sum = np.zeros(n_bins)
    x_value = np.zeros(n_bins)

    # Loop over all bins and calculate the sum line #
    for i in range(n_bins):
        index, = np.where((x >= x_low) & (x < x_low + bin_width))
        x_value[i] = np.mean(x_data[index])
        if len(index) > 0:
            sum[i] = np.sum(y_data[index])
        x_low += bin_width

    return x_value, sum


def create_colorbar(axis, plot, label, orientation='vertical', ticks=None, size=16):
    """
    Generate a colorbar.
    :param axis: colorbar axis.
    :param plot: corresponding plot.
    :param label: colorbar label.
    :param orientation: colorbar orientation.
    :param ticks: array of ticks.
    :param size: text size.
    :return: None
    """
    cbar = plt.colorbar(plot, cax=axis, ticks=ticks, orientation=orientation)
    cbar.set_label(label, size=size)
    axis.tick_params(direction='out', which='both', right='on', labelsize=size)

    if orientation == 'horizontal':
        axis.xaxis.tick_top()
        axis.xaxis.set_label_position("top")
        axis.tick_params(direction='out', which='both', top='on', labelsize=size)
    return None


def set_axis(axis, xlim=None, ylim=None, xscale=None, yscale=None, xlabel=None, ylabel=None, aspect='equal', which='both', size=16):
    """
    Set axis parameters.
    :param axis: name of the axis.
    :param xlim: x axis limits.
    :param ylim: y axis limits.
    :param xscale: x axis scale.
    :param yscale: y axis scale.
    :param xlabel: x axis label.
    :param ylabel: y axis label.
    :param aspect: aspect of the axis scaling.
    :param which: major, minor or both for grid and ticks.
    :param size: text size.
    :return: None
    """
    # Set axis limits #
    if xlim:
        axis.set_xlim(xlim)
    if ylim:
        axis.set_ylim(ylim)

    # Set axis labels #
    if xlabel:
        axis.set_xlabel(xlabel, size=size)
    if ylabel:
        axis.set_ylabel(ylabel, size=size)

    # Set axis scales #
    if xscale:
        axis.set_xscale(xscale)
    if yscale:
        axis.set_yscale(yscale)

    if not xlim and not xlabel:
        axis.set_xticks([])
        axis.set_xticklabels([])
    if not ylim and not ylabel:
        axis.set_yticks([])
        axis.set_yticklabels([])

    # Set grid and tick parameters #
    if aspect is not None:
        axis.set_aspect('equal')
    axis.grid(True, which=which, axis='both', color='gray', linestyle='-')
    axis.tick_params(direction='out', which=which, top='on', bottom='on', left='on', right='on', labelsize=size)

    return None


def set_axes_evolution(axis, axis2, ylim=None, yscale=None, ylabel=None, aspect='equal', which='both', size=16):
    """
    Set axes parameters for evolution plots.
    :param axis: name of the axis.
    :param axis2: name of the twin axis.
    :param ylim: y axis limit.
    :param yscale: y axis scale.
    :param ylabel: y axis label.
    :param aspect: aspect of the axis scaling.
    :param which: major, minor or both for grid and ticks.
    :param size: text size.
    :return: None
    """
    z = np.array([5.0, 3.0, 2.0, 1.0, 0.5, 0.2, 0.0])
    times = satellite_utilities.return_lookbacktime_from_a((z + 1.0) ** (-1.0))  # In Gyr.

    lb = []
    for v in z:
        if v >= 1.0:
            lb += ["%.0f" % v]
        else:
            if v:
                lb += ["%.1f" % v]
            else:
                lb += ["%.0f" % v]

    # Set axis limits #
    axis2.set_xticks(times)
    axis2.set_xticklabels(lb)
    if ylim:
        axis.set_ylim(ylim)
    axis.set_xlim(13, 0)
    axis2.set_xlim(axis.get_xlim())

    # Set axis scales #
    if yscale:
        axis.set_yscale(yscale)

    # Set axis labels #
    if ylabel:
        axis.set_ylabel(ylabel, size=size)

    axis.set_xlabel(r'$\mathrm{t_{look}/Gyr}$', size=size)
    axis2.set_xlabel(r'$\mathrm{z}$', size=size)

    # Set grid and tick parameters #
    if aspect is not None:
        axis.set_aspect('equal')
    axis.grid(True, which=which, axis='both', color='gray', linestyle='-')
    axis.tick_params(direction='out', which=which, top='on', bottom='on', left='on', right='on', labelsize=size)
    axis2.tick_params(direction='out', which=which, top='on', left='on', right='on', labelsize=size)

    return None


def create_axes_combinations(res=res, boxsize=boxsize, contour=False, colorbar=False, velocity_vectors=False, multiple=False, multiple2=False,
                             multiple3=False, multiple4=False, multiple5=False, multiple6=False, multiple7=False, mollweide=False, multiple8=False,
                             multiple9=False, multiple10=False):
    """
    Generate plot axes.
    :param res: resolution
    :param boxsize: boxsize
    :param contour: 2x2 matrix plus colorbar
    :param colorbar: 2x1 matrix plus colorbar
    :param velocity_vectors: 2x1 matrix
    :param multiple: 2x6 matrix plus 6 colorbars
    :param multiple2: 6x3 matrix
    :param multiple3: 5x4 matrix plus colorbar
    :param multiple4: 1x3 matrix
    :param multiple5: 3x3 matrix
    :param multiple6: 3x3 matrix plus colorbar
    :param multiple7: 3x3 matrix plus 9 colorbars
    :param mollweide: 3x3 mollweide projection
    :param multiple8: 6x3 matrix
    :param multiple9: 6x3 matrix
    :param multiple10: 2x3 matrix

    :return: axes
    """

    # Set the axis values #
    x = np.linspace(-0.5 * boxsize, +0.5 * boxsize, res + 1)
    y = np.linspace(-0.5 * boxsize, +0.5 * boxsize, res + 1)
    y2 = np.linspace(-0.5 * boxsize, +0.5 * boxsize, res / 2 + 1)

    area = (boxsize / res) ** 2  # Calculate the area.

    # Generate the panels #
    if contour is True:
        gs = gridspec.GridSpec(2, 3, hspace=0.05, wspace=0.05, height_ratios=[1, 0.5], width_ratios=[1, 1, 0.05])
        axis00 = plt.subplot(gs[0, 0])
        axis01 = plt.subplot(gs[0, 1])
        axis10 = plt.subplot(gs[1, 0])
        axis11 = plt.subplot(gs[1, 1])
        axiscbar = plt.subplot(gs[:, 2])

        return axis00, axis01, axis10, axis11, axiscbar, x, y, y2, area

    elif colorbar is True:
        gs = gridspec.GridSpec(2, 2, hspace=0.05, wspace=0.05, height_ratios=[1, 0.5], width_ratios=[1, 0.05])
        axis00 = plt.subplot(gs[0, 0])
        axis10 = plt.subplot(gs[1, 0])
        axiscbar = plt.subplot(gs[:, 1])

        return axis00, axis10, axiscbar, x, y, y2, area

    elif velocity_vectors is True:
        gs = gridspec.GridSpec(2, 1, hspace=0.05)
        axis00 = plt.subplot(gs[0, 0])
        axis10 = plt.subplot(gs[1, 0])

        return axis00, axis10, x, y, y2, area

    elif multiple is True:
        gs = gridspec.GridSpec(3, 6, hspace=0.0, wspace=0.05, height_ratios=[1, 0.05, 1])
        axis00, axis01, axis02, axis03, axis04, axis05 = plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[0, 2]), plt.subplot(
            gs[0, 3]), plt.subplot(gs[0, 4]), plt.subplot(gs[0, 5])
        axis10, axis11, axis12, axis13, axis14, axis15 = plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1]), plt.subplot(gs[1, 2]), plt.subplot(
            gs[1, 3]), plt.subplot(gs[1, 4]), plt.subplot(gs[1, 5])
        axis20, axis21, axis22, axis23, axis24, axis25 = plt.subplot(gs[2, 0]), plt.subplot(gs[2, 1]), plt.subplot(gs[2, 2]), plt.subplot(
            gs[2, 3]), plt.subplot(gs[2, 4]), plt.subplot(gs[2, 5])

        return axis00, axis01, axis02, axis03, axis04, axis05, axis10, axis11, axis12, axis13, axis14, axis15, axis20, axis21, axis22, axis23, \
               axis24, axis25, x, y, area

    elif multiple2 is True:
        gs = gridspec.GridSpec(6, 3, hspace=0, wspace=0)
        axis00, axis01, axis02 = plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[0, 2])
        axis10, axis11, axis12 = plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1]), plt.subplot(gs[1, 2])
        axis20, axis21, axis22 = plt.subplot(gs[2, 0]), plt.subplot(gs[2, 1]), plt.subplot(gs[2, 2])
        axis30, axis31, axis32 = plt.subplot(gs[3, 0]), plt.subplot(gs[3, 1]), plt.subplot(gs[3, 2])
        axis40, axis41, axis42 = plt.subplot(gs[4, 0]), plt.subplot(gs[4, 1]), plt.subplot(gs[4, 2])
        axis50, axis51, axis52 = plt.subplot(gs[5, 0]), plt.subplot(gs[5, 1]), plt.subplot(gs[5, 2])
        return axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22, axis30, axis31, axis32, axis40, axis41, axis42, axis50, \
               axis51, axis52

    elif multiple3 is True:
        gs = gridspec.GridSpec(6, 4, hspace=0.1, wspace=0.1, height_ratios=[1, 0.5, 1, 0.5, 1, 0.5], width_ratios=[1, 1, 1, 0.1])
        axis00, axis01, axis02 = plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[0, 2])
        axis10, axis11, axis12 = plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1]), plt.subplot(gs[1, 2])
        axis20, axis21, axis22 = plt.subplot(gs[2, 0]), plt.subplot(gs[2, 1]), plt.subplot(gs[2, 2])
        axis30, axis31, axis32 = plt.subplot(gs[3, 0]), plt.subplot(gs[3, 1]), plt.subplot(gs[3, 2])
        axis40, axis41, axis42 = plt.subplot(gs[4, 0]), plt.subplot(gs[4, 1]), plt.subplot(gs[4, 2])
        axis50, axis51, axis52 = plt.subplot(gs[5, 0]), plt.subplot(gs[5, 1]), plt.subplot(gs[5, 2])
        axiscbar = plt.subplot(gs[:, 3])
        return axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22, axis30, axis31, axis32, axis40, axis41, axis42, axis50, \
               axis51, axis52, axiscbar, x, y, y2, area

    elif multiple4 is True:
        gs = gridspec.GridSpec(1, 3, hspace=0, wspace=0)
        axis00, axis01, axis02 = plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[0, 2])
        return axis00, axis01, axis02

    elif multiple5 is True:
        gs = gridspec.GridSpec(3, 3, hspace=0, wspace=0)
        axis00, axis01, axis02 = plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[0, 2])
        axis10, axis11, axis12 = plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1]), plt.subplot(gs[1, 2])
        axis20, axis21, axis22 = plt.subplot(gs[2, 0]), plt.subplot(gs[2, 1]), plt.subplot(gs[2, 2])
        return axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22

    elif multiple6 is True:
        gs = gridspec.GridSpec(3, 4, hspace=0, wspace=0, width_ratios=[1, 1, 1, 0.1])
        axis00, axis01, axis02 = plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[0, 2])
        axis10, axis11, axis12 = plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1]), plt.subplot(gs[1, 2])
        axis20, axis21, axis22 = plt.subplot(gs[2, 0]), plt.subplot(gs[2, 1]), plt.subplot(gs[2, 2])
        axiscbar = plt.subplot(gs[:, 3])
        return axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22, axiscbar

    elif multiple7 is True:
        gs = gridspec.GridSpec(6, 3, hspace=0.4, wspace=0, height_ratios=[0.05, 1, 0.05, 1, 0.05, 1])
        axis00, axis01, axis02 = plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[0, 2])
        axis10, axis11, axis12 = plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1]), plt.subplot(gs[1, 2])
        axis20, axis21, axis22 = plt.subplot(gs[2, 0]), plt.subplot(gs[2, 1]), plt.subplot(gs[2, 2])
        axis30, axis31, axis32 = plt.subplot(gs[3, 0]), plt.subplot(gs[3, 1]), plt.subplot(gs[3, 2])
        axis40, axis41, axis42 = plt.subplot(gs[4, 0]), plt.subplot(gs[4, 1]), plt.subplot(gs[4, 2])
        axis50, axis51, axis52 = plt.subplot(gs[5, 0]), plt.subplot(gs[5, 1]), plt.subplot(gs[5, 2])
        return axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22, axis30, axis31, axis32, axis40, axis41, axis42, axis50, \
               axis51, axis52
    elif mollweide is True:
        gs = gridspec.GridSpec(3, 4, hspace=0, wspace=0, width_ratios=[1, 1, 1, 0.1])
        axis00, axis01, axis02 = plt.subplot(gs[0, 0], projection='mollweide'), plt.subplot(gs[0, 1], projection='mollweide'), plt.subplot(gs[0, 2],
                                                                                                                                           projection='mollweide')
        axis10, axis11, axis12 = plt.subplot(gs[1, 0], projection='mollweide'), plt.subplot(gs[1, 1], projection='mollweide'), plt.subplot(gs[1, 2],
                                                                                                                                           projection='mollweide')
        axis20, axis21, axis22 = plt.subplot(gs[2, 0], projection='mollweide'), plt.subplot(gs[2, 1], projection='mollweide'), plt.subplot(gs[2, 2],
                                                                                                                                           projection='mollweide')
        axiscbar = plt.subplot(gs[:, 3])
        return axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22, axiscbar

    elif multiple8 is True:
        gs = gridspec.GridSpec(6, 3, hspace=0, wspace=0)
        axis00, axis01, axis02 = plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[0, 2])
        axis10, axis11, axis12 = plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1]), plt.subplot(gs[1, 2])
        axis20, axis21, axis22 = plt.subplot(gs[2, 0]), plt.subplot(gs[2, 1]), plt.subplot(gs[2, 2])
        axis30, axis31, axis32 = plt.subplot(gs[3, 0]), plt.subplot(gs[3, 1]), plt.subplot(gs[3, 2])
        axis40, axis41, axis42 = plt.subplot(gs[4, 0]), plt.subplot(gs[4, 1]), plt.subplot(gs[4, 2])
        axis50, axis51, axis52 = plt.subplot(gs[5, 0]), plt.subplot(gs[5, 1]), plt.subplot(gs[5, 2])

        return axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22, axis30, axis31, axis32, axis40, axis41, axis42, axis50, \
               axis51, axis52

    elif multiple9 is True:
        gs = gridspec.GridSpec(6, 3, hspace=0, wspace=0, height_ratios=[1, 0.5, 1, 0.5, 1, 0.5])
        axis00, axis01, axis02 = plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[0, 2])
        axis10, axis11, axis12 = plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1]), plt.subplot(gs[1, 2])
        axis20, axis21, axis22 = plt.subplot(gs[2, 0]), plt.subplot(gs[2, 1]), plt.subplot(gs[2, 2])
        axis30, axis31, axis32 = plt.subplot(gs[3, 0]), plt.subplot(gs[3, 1]), plt.subplot(gs[3, 2])
        axis40, axis41, axis42 = plt.subplot(gs[4, 0]), plt.subplot(gs[4, 1]), plt.subplot(gs[4, 2])
        axis50, axis51, axis52 = plt.subplot(gs[5, 0]), plt.subplot(gs[5, 1]), plt.subplot(gs[5, 2])

        return axis00, axis01, axis02, axis10, axis11, axis12, axis20, axis21, axis22, axis30, axis31, axis32, axis40, axis41, axis42, axis50, \
               axis51, axis52

    elif multiple10 is True:
        gs = gridspec.GridSpec(2, 3, hspace=0, wspace=0)
        axis00, axis01, axis02 = plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[0, 2])
        axis10, axis11, axis12 = plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1]), plt.subplot(gs[1, 2])

        return axis00, axis01, axis02, axis10, axis11, axis12

    else:
        gs = gridspec.GridSpec(2, 1, hspace=0.05, height_ratios=[1, 0.5])
        axis00 = plt.subplot(gs[0, 0])
        axis10 = plt.subplot(gs[1, 0])

        return axis00, axis10, x, y, y2, area


def rotate_bar(z, y, x):
    """
    Calculate bar strength and rotate bar to horizontal position.
    :param z: the z-position of the particles.
    :param y: the y-position of the particles.
    :param x: the x-position of the particles.
    :return:
    """
    # Declare arrays to store the data #
    n_bins = 40  # Number of radial bins.
    r_m = np.zeros(n_bins)
    beta_2 = np.zeros(n_bins)
    alpha_0 = np.zeros(n_bins)
    alpha_2 = np.zeros(n_bins)

    # Split disc in radial bins and calculate Fourier components #
    r = np.sqrt(x[:] ** 2 + y[:] ** 2)  # Radius of each particle.
    for i in range(0, n_bins):
        r_s = float(i) * 0.25
        r_b = float(i) * 0.25 + 0.25
        r_m[i] = float(i) * 0.25 + 0.125
        xfit = x[(r < r_b) & (r > r_s)]
        yfit = y[(r < r_b) & (r > r_s)]
        l = len(xfit)
        for k in range(0, l):
            th_i = np.arctan2(yfit[k], xfit[k])
            alpha_0[i] = alpha_0[i] + 1
            alpha_2[i] = alpha_2[i] + np.cos(2 * th_i)
            beta_2[i] = beta_2[i] + np.sin(2 * th_i)

    # Calculate bar rotation angle for each time by averaging over radii between 1 and 5 kpc #
    r_b = 5  # In kpc.
    r_s = 1  # In kpc.
    k = 0.0
    phase_in = 0.0
    for i in range(0, n_bins):
        if (r_m[i] < r_b) & (r_m[i] > r_s):
            k = k + 1.
            phase_in = phase_in + 0.5 * np.arctan2(beta_2[i], alpha_2[i])
    phase_in = phase_in / k

    # Transform back -tangle to horizontal position #
    z_pos = z[:]
    y_pos = np.cos(-phase_in) * (y[:]) + np.sin(-phase_in) * (x[:])
    x_pos = np.cos(-phase_in) * (x[:]) - np.sin(-phase_in) * (y[:])

    return z_pos / 1e3, y_pos / 1e3, x_pos / 1e3  # In kpc.


def create_axes_projections(res=res, boxsize=boxsize, contour=False, colorbar=False, velocity_vectors=False, multiple=False, multiple2=False,
                            multiple3=False, multiple4=False):
    """
    Generate plot axes.
    :param res: resolution
    :param boxsize: boxsize
    :param contour: contour
    :param colorbar: colorbar
    :param velocity_vectors: velocity_vectors
    :param multiple: multiple
    :return: axes
    """

    # Set the axis values #
    x = np.linspace(-0.5 * boxsize, +0.5 * boxsize, res + 1)
    y = np.linspace(-0.5 * boxsize, +0.5 * boxsize, res + 1)
    y2 = np.linspace(-0.5 * boxsize, +0.5 * boxsize, res / 2 + 1)

    area = (boxsize / res) ** 2  # Calculate the area.

    # Generate the panels #
    if contour is True:
        gs = gridspec.GridSpec(2, 3, hspace=0.05, wspace=0.0, height_ratios=[1, 0.5], width_ratios=[1, 1, 0.05])
        axis00, axis01 = plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1])
        axis10, axis11 = plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1])
        axiscbar = plt.subplot(gs[:, 2])

        return axis00, axis01, axis10, axis11, axiscbar, x, y, y2, area

    elif colorbar is True:
        gs = gridspec.GridSpec(2, 2, hspace=0.05, wspace=0.0, height_ratios=[1, 0.5], width_ratios=[1, 0.05])
        axis00 = plt.subplot(gs[0, 0])
        axis10 = plt.subplot(gs[1, 0])
        axiscbar = plt.subplot(gs[:, 1])

        return axis00, axis10, axiscbar, x, y, y2, area

    elif velocity_vectors is True:
        gs = gridspec.GridSpec(2, 1, hspace=0.05)
        axis00 = plt.subplot(gs[0, 0])
        axis10 = plt.subplot(gs[1, 0])

        return axis00, axis10, x, y, y2, area

    elif multiple is True:
        gs = gridspec.GridSpec(3, 6, hspace=0.0, wspace=0.05, height_ratios=[1, 0.05, 1])
        axis00, axis01, axis02, axis03, axis04, axis05 = plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[0, 2]), plt.subplot(
            gs[0, 3]), plt.subplot(gs[0, 4]), plt.subplot(gs[0, 5])
        axis10, axis11, axis12, axis13, axis14, axis15 = plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1]), plt.subplot(gs[1, 2]), plt.subplot(
            gs[1, 3]), plt.subplot(gs[1, 4]), plt.subplot(gs[1, 5])
        axis20, axis21, axis22, axis23, axis24, axis25 = plt.subplot(gs[2, 0]), plt.subplot(gs[2, 1]), plt.subplot(gs[2, 2]), plt.subplot(
            gs[2, 3]), plt.subplot(gs[2, 4]), plt.subplot(gs[2, 5])

        return axis00, axis10, axis20, axis01, axis11, axis21, axis02, axis12, axis22, axis03, axis13, axis23, axis04, axis14, axis24, axis05, \
               axis15, axis25, x, y, area

    elif multiple2 is True:
        gs = gridspec.GridSpec(4, 3, hspace=0, wspace=0, height_ratios=[1, 0.5, 1, 0.5])
        axis00, axis01, axis02 = plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[0, 2])
        axis10, axis11, axis12 = plt.subplot(gs[1, 0]), plt.subplot(gs[1, 2]), plt.subplot(gs[1, 2])
        axis20, axis21, axis22 = plt.subplot(gs[2, 0]), plt.subplot(gs[2, 1]), plt.subplot(gs[2, 2])
        axis30, axis31, axis32 = plt.subplot(gs[3, 0]), plt.subplot(gs[3, 1]), plt.subplot(gs[3, 2])

        return axis00, axis10, axis20, axis30, axis01, axis11, axis21, axis31, axis02, axis12, axis22, axis32, x, y, y2, area

    elif multiple3 is True:
        gs = gridspec.GridSpec(4, 4, hspace=0.05, wspace=0, height_ratios=[1, 0.5, 1, 0.5], width_ratios=[1, 1, 1, 0.1])
        axis00, axis01, axis02 = plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[0, 2])
        axis10, axis11, axis12 = plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1]), plt.subplot(gs[1, 2])
        axis20, axis21, axis22 = plt.subplot(gs[2, 0]), plt.subplot(gs[2, 1]), plt.subplot(gs[2, 2])
        axis30, axis31, axis32 = plt.subplot(gs[3, 0]), plt.subplot(gs[3, 1]), plt.subplot(gs[3, 2])
        axiscbar = plt.subplot(gs[:, 3])

        return axis00, axis10, axis20, axis30, axis01, axis11, axis21, axis31, axis02, axis12, axis22, axis32, axiscbar, x, y, y2, area

    if multiple4 is True:
        gs = gridspec.GridSpec(2, 3, hspace=0, wspace=0)
        axis00, axis01 = plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1])
        axis10, axis11 = plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1])

        return axis00, axis01, axis10, axis11, x, y, y2, area


    else:
        gs = gridspec.GridSpec(2, 1, hspace=0.05)
        axis00 = plt.subplot(gs[0, 0])
        axis10 = plt.subplot(gs[1, 0])

        return axis00, axis10, x, y, y2, area


class RotateCoordinates:
    """
    Rotate coordinates and velocities wrt different quantities.
    """


    @staticmethod
    def rotate_X(data, glx_unit_vector, stellar_mask):
        """
        Rotate first about z-axis to set y=0 and then about the y-axis to set z=0
        :param data: halo data.
        :param glx_unit_vector: halo unit vector of stellar angular momentum.
        :param stellar_mask: stellar particles mask.
        :return: prc_unit_vector
        """
        # Calculate the rotation matrices and combine them #
        ra = np.arctan2(glx_unit_vector[1], glx_unit_vector[2])
        el = np.arcsin(glx_unit_vector[0])

        Rz = np.array([[np.cos(ra), np.sin(ra), 0], [-np.sin(ra), np.cos(ra), 0], [0, 0, 1]])
        Ry = np.array([[np.cos(el), 0, np.sin(el)], [0, 1, 0], [-np.sin(el), 0, np.cos(el)]])
        Ryz = np.matmul(Ry, Rz)

        # Flip, rotate and flip back the coordinates and velocities of stellar particles #
        pos = np.fliplr(data['pos'][stellar_mask])
        vel = np.fliplr(data['vel'][stellar_mask])
        pos = np.matmul(Ryz, pos[..., None]).squeeze()
        vel = np.matmul(Ryz, vel[..., None]).squeeze()
        pos = np.fliplr(pos)
        vel = np.fliplr(vel)

        # Recalculate the angular momentum for each particle and for the galaxy and the unit vector parallel to the galactic angular momentum vector #
        prc_angular_momentum = data['mass'][stellar_mask, np.newaxis] * np.cross(pos * 1e3, vel)  # In Msun kpc km s^-1.
        vector_mask, = np.where(np.linalg.norm(prc_angular_momentum, axis=1) > 0)
        prc_unit_vector = prc_angular_momentum[vector_mask] / np.linalg.norm(prc_angular_momentum[vector_mask], axis=1)[:, np.newaxis]

        return prc_unit_vector
