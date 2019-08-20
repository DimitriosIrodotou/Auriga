import os.path

from mpl_toolkits.axes_grid1 import make_axes_locatable
from pylab import *
from util import multipanel_layout
from util.subhalos_utilities import subhalos_properties

# from cosmological_factors import CosmologicalFactors

toinch = 0.393700787
# base = '/output/fof_subhalo_tab_'
base = '/fof_subhalo_tab_'


def set_colorbar(axb, vval, cblabel, cticks, cmap):
    # cmap = mpl.cm.jet
    bounds = cticks
    # norm = mpl.colors.Normalize(vmin=vval[0],vmax=vval[1])
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    cb = mpl.colorbar.ColorbarBase(axb, cmap=cmap, norm=norm, boundaries=bounds, ticks=bounds, orientation='vertical')
    cb.set_label(cblabel, fontsize=7)


def get_scalar_mappable_for_colorbar(parameters, cmap):
    norm = mpl.colors.Normalize(vmin=np.min(parameters), vmax=np.max(parameters))
    print
    "norm=", norm
    # c_m = mpl.cm.jet
    s_m = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    s_m.set_array([])
    
    return s_m


def plot_mtotvsdist(runs, dirs, outpath, snaps, redshift, suffix, mpanel=False):
    if mpanel == True:
        nrows = 4
        ncols = 4
        panels = 16
        figure = multipanel_layout.multipanel_layout(nrows=nrows, ncols=ncols, npanels=panels)
        figure.set_figure_layout()
        xlim, ylim = [10., 0.1], [8., 500.]
        xlabel, ylabel = "$\\rm{t_{lookback}\,[Gyr]}$", "$\\rm{d\\,[kpc]}$"
        # fontsize = 7
        figure.set_axis_limits_and_aspect(xlim=xlim, ylim=ylim)
        figure.set_axis_labels(xlabel=xlabel, ylabel=ylabel)
        figure.set_fontsize()
    
    elif mpanel == False:
        fig, axarr = plt.subplots(2, sharex=True)
        ax = axarr[0]
        ax2 = axarr[1]
        
        ax.set_ylim([1., 500.])
        ax.set_xlim([7.2, 0.])
        ax.set_yscale('log')
        ax.set_xlabel("$\\rm{z}$")
        ax.set_ylabel("$\\rm{d\\,[kpc]}$")
        
        ax2.set_ylim([1., 500.])
        ax2.set_xlim([7.2, 0.])
        ax2.set_yscale('log')
        ax2.set_xlabel("$\\rm{z}$")
        ax2.set_ylabel("$\\rm{v\\,[km \; s^{-1}]}$")
    
    cmap = plt.get_cmap('CMRmap_r')
    params = np.linspace(8, 11, 25)
    s_m = get_scalar_mappable_for_colorbar(params, cmap)
    
    for d in range(len(runs)):
        wpath = outpath + runs[d] + '/satellites/'
        if not os.path.exists(wpath):
            os.makedirs(wpath)
        wpath = str(wpath)
        
        print
        'doing halo', runs[d]
        dd = dirs[d] + runs[d] + '/output/'
        
        if mpanel == True:
            ax = figure.axes[d]
            ax.set_yscale('log')
        
        sintp = np.zeros(len(snaps))
        mm = np.zeros(len(snaps))
        dis = np.zeros(len(snaps))
        for i, snap in enumerate(snaps):
            subhalos = subhalos_properties(snap, base=base, directory=dd, hdf5=True)
            subhalos.set_subhalos_properties(parent_group=0)
            totalmass, dist, vcirc = subhalos.get_totalmass_vs_position_vcirc()
            
            msort = np.argsort(totalmass)
            totalmass = totalmass[msort]
            dist = dist[msort]
            vcirc = vcirc[msort]
            
            m20 = log10(totalmass[-6:-1] * 1e10)
            d20 = dist[-6:-1]
            v20 = vcirc[-6:-1]
            
            if i == 0:
                a = 1. / (1. + np.array(redshift))
                time = subhalos.cosmology_get_lookback_time_from_a(a, is_flat=True)
            
            print
            "time=", time[i]
            ax.scatter([time[i]] * 5, d20 * 1e3, edgecolor=s_m.to_rgba(m20), color=s_m.to_rgba(m20), marker='o', s=0.7)
            
            mm[i] = m20[-1]
            dis[i] = d20[-1]
        
        if mpanel == True:
            figure.set_colorbar([8., 10.5], '$\\rm{log_{10} (M/M_{\odot})}$', [8., 8.25, 8.5, 8.75, 9., 9.25, 9.5, 9.75, 10., 10.25, 10.5], cmap,
                                bounds=True)
            figure.set_panel_title(panel=d, title="$\\rm{%s\,{%s}}$" % ("Au", runs[d].split('_')[1]), position='bottom right')
        elif mpanel == False:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            set_colorbar(cax, [8., 10.5], "$\\rm{log_{10} M}$", [8., 8.25, 8.5, 8.75, 9., 9.25, 9.5, 9.75, 10., 10.25, 10.5], cmap)
        
        if mpanel == True:
            figure.fig.savefig("%smass-dist_history_all.%s" % (outpath, suffix))
        elif mpanel == False:
            savefig("%smass-dist_history_%s.%s" % (wpath, runs[d], suffix))