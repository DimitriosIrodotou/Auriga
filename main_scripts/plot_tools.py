import numpy as np
import matplotlib.pyplot as plt


def median_1sigma(x_data, y_data, delta, log):
    """
    Calculate the median and 1-sigma lines.
    :param x_data: x-axis data.
    :param y_data: y-axis data.
    :param delta: step.
    :param log: boolean.
    :return: x_value, median, shigh, slow
    """
    # Initialise arrays #
    if log is True:
        x = np.log10(x_data)
    else:
        x = x_data
    n_bins = int((max(x) - min(x)) / delta)
    x_value = np.empty(n_bins)
    median = np.empty(n_bins)
    slow = np.empty(n_bins)
    shigh = np.empty(n_bins)
    x_low = min(x)
    
    # Loop over all bins and calculate the median and 1-sigma lines #
    for i in range(n_bins):
        index, = np.where((x >= x_low) & (x < x_low + delta))
        x_value[i] = np.mean(x_data[index])
        if len(index) > 0:
            median[i] = np.nanmedian(y_data[index])
        slow[i] = np.nanpercentile(y_data[index], 15.87)
        shigh[i] = np.nanpercentile(y_data[index], 84.13)
        x_low += delta
    
    return x_value, median, shigh, slow


def create_colorbar(axis, plot, label, orientation='vertical', size=16):
    """
    Generate a colorbar.
    :param axis: colorbar axis.
    :param plot: corresponding plot.
    :param label: colorbar label.
    :param orientation: colorbar orientation.
    :return: None
    """
    cbar = plt.colorbar(plot, cax=axis, orientation=orientation)
    cbar.set_label(label, size=size)
    axis.tick_params(direction='out', which='both', right='on', labelsize=size)
    
    if orientation == 'horizontal':
        axis.xaxis.tick_top()
        axis.xaxis.set_label_position("top")
        axis.tick_params(direction='out', which='both', top='on', labelsize=size)
    return None


def set_axis(axis, xlim=None, ylim=None, xscale=None, yscale=None, xlabel=None, ylabel=None, size=16):
    """
    Set axis parameters.
    :param axis: name of the axis.
    :param xlim: x axis limits.
    :param ylim: y axis limits.
    :param xscale: x axis scale.
    :param yscale: y axis scale.
    :param xlabel: x axis label.
    :param ylabel: y axis label.
    :param size: text size.
    :return:
    """
    # Set axis limits #
    if xlim:
        axis.set_xlim(xlim)
    if ylim:
        axis.set_ylim(ylim)
    
    # Set axis labels #
    if xlabel:
        axis.set_xlabel(xlabel, size=size)
    else:
        axis.set_xticklabels([])
    if ylabel:
        axis.set_ylabel(ylabel, size=size)
    else:
        axis.set_yticklabels([])
    
    # Set axis scales #
    if xscale:
        axis.set_xscale(xscale)
    if yscale:
        axis.set_xticklabels([])
    
    # Set grid and tick parameters #
    axis.set_aspect('equal')
    axis.grid(True, which='both', axis='both', color='gray', linestyle='-')
    axis.tick_params(direction='out', which='both', top='on', right='on', labelsize=size)
    
    return None
