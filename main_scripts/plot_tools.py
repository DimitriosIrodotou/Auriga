import numpy as np
import matplotlib.pyplot as plt
from scripts.gigagalaxy.util import satellite_utilities


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
        # Initialise arrays #
        if log is True:
            x = np.log10(x_data)
        else:
            x = x_data
        
        n_bins = np.quantile(np.sort(x), np.linspace(0, 1, n_bins + 1))
        slow = np.empty(len(n_bins))
        shigh = np.empty(len(n_bins))
        median = np.empty(len(n_bins))
        x_value = np.empty(len(n_bins))
        
        # Loop over all bins and calculate the median and 1-sigma lines #
        for i in range(len(n_bins) - 1):
            index, = np.where((x >= n_bins[i]) & (x < n_bins[i + 1]))
            x_value[i] = np.mean(x_data[index])
            if len(index) > 0:
                median[i] = np.nanmedian(y_data[index])
                slow[i] = np.nanpercentile(y_data[index], 15.87)
                shigh[i] = np.nanpercentile(y_data[index], 84.13)
        
        return x_value, median
    
    elif bin_type == 'equal_width':
        # Initialise arrays #
        if log is True:
            x = np.log10(x_data)
        else:
            x = x_data
        x_low = min(x)
        
        bin_width = (max(x) - min(x)) / n_bins
        slow = np.empty(n_bins)
        shigh = np.empty(n_bins)
        median = np.empty(n_bins)
        x_value = np.empty(n_bins)
        
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


def binned_sum(x_data, y_data, bin_width, log=False):
    """
    Calculate the binned sum line.
    :param x_data: x-axis data.
    :param y_data: y-axis data.
    :param bin_width: width of the bin.
    :param log: boolean.
    :return: x_value, sum
    """
    # Initialise arrays #
    if log is True:
        x = np.log10(x_data)
    else:
        x = x_data
    x_low = min(x)
    
    n_bins = int((max(x) - min(x)) / bin_width)
    sum = np.empty(n_bins)
    x_value = np.empty(n_bins)
    
    # Loop over all bins and calculate the sum line #
    for i in range(n_bins):
        index, = np.where((x >= x_low) & (x < x_low + bin_width))
        x_value[i] = np.mean(x_data[index])
        if len(index) > 0:
            sum[i] = np.sum(y_data[index])
        x_low += bin_width
    
    return x_value, sum


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


def set_axis(axis, xlim=None, ylim=None, xscale=None, yscale=None, xlabel=None, ylabel=None, aspect='equal', size=16):
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
        axis.set_yscale(yscale)
    
    # Set grid and tick parameters #
    if aspect is not None:
        axis.set_aspect('equal')
    axis.grid(True, which='both', axis='both', color='gray', linestyle='-')
    axis.tick_params(direction='out', which='both', top='on', right='on', labelsize=size)
    
    return None


def set_axis_evo(axis, axis2, ylabel=None):
    z = np.array([5.0, 3.0, 2.0, 1.0, 0.5, 0.2, 0.0])
    times = satellite_utilities.return_lookbacktime_from_a((z + 1.0) ** (-1.0))  # In Gyr.
    
    lb = []
    for v in z:
        if v >= 1.0:
            lb += ["%.0f" % v]
        else:
            if v != 0:
                lb += ["%.1f" % v]
            else:
                lb += ["%.0f" % v]
    
    axis.set_xlim(13, 0)
    axis.set_ylabel(ylabel, size=16)
    axis.set_xlabel(r'$\mathrm{t_{look}/Gyr}$', size=16)
    axis.tick_params(direction='out', which='both', right='on')
    
    axis2.set_xticks(times)
    axis2.set_xticklabels(lb)
    axis2.set_xlim(axis.get_xlim())
    axis2.set_xlabel(r'$\mathrm{z}$', size=16)
    axis2.tick_params(direction='out', which='both', top='on', right='on')
    
    return None
