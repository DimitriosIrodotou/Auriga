from const import *
from loadmodules import *
from pylab import *
from util import *


def plot_vertical_density_evol(runs, dirs, outpath, outputlistfile, zlist, suffix, nrows, ncols, radii=[0.0025, 0.0075, 0.0125]):
    panels = len(runs)
    figure = multipanel_layout.multipanel_layout(nrows=nrows, ncols=ncols, npanels=panels)
    figure.set_figure_layout()
    figure.set_axis_locators(xminloc=1.0, xmajloc=5.0, yminloc=0.1, ymajloc=0.5)
    figure.set_axis_labels(xlabel=r"$\rm{z [kpc]}$", ylabel=r"$\rm{\rho [M_{\odot} kpc^{-3}]}$")
    figure.set_fontsize()
    figure.set_axis_limits_and_aspect(xlim=[-3.0, 3.0], ylim=[5e5, 6e8], logyaxis=True)
    
    clist = ['r', 'b', 'c', 'g', 'k']
    
    for d in range(len(runs)):
        dd = dirs[d] + runs[d]
        ax = figure.fig.axes[d]
        figure.set_panel_title(panel=d, title="$\\rm{%s%s}$" % ("Au", runs[d].split('_')[1]), position='top right')
        
        nshells = 35
        snaps = np.int_(select_snapshot_number.select_snapshot_number(outputlistfile[d], zlist))
        if isinstance(snaps, int):
            snaps = [snaps]
        
        time = pylab.zeros(len(snaps))
        for isnap in range(len(snaps)):
            snap = snaps[isnap]
            print("Doing dir %s snap %d." % (dd, snap))
            rd = radii
            mnow = np.zeros(len(rd))
            
            attrs = ['pos', 'vel', 'mass', 'age']
            s = gadget_readsnap(snap, snappath=dd + '/output/', hdf5=True, loadonlytype=[4], loadonly=attrs)
            sf = load_subfind(snap, dir=dd + '/output/', hdf5=True, loadonly=['fpos', 'frc2', 'svel', 'flty', 'fnsh', 'slty', 'fmc2'])
            s.calc_sf_indizes(sf)
            s.select_halo(sf, 3., use_principal_axis=True, use_cold_gas_spin=False, do_rotation=True)
            time[isnap] = s.cosmology_get_lookback_time_from_a(s.time.astype('float64'), is_flat=True)
            g = parse_particledata(s, sf, attrs, radialcut=0.1 * sf.data['frc2'][0])
            g.prep_data()
            
            sdata = g.sgdata['sdata']
            smass = sdata['mass']
            pos = sdata['pos']
            
            parameters = np.linspace(0., time[0], len(snaps))
            s_m = figure.get_scalar_mappable_for_colorbar(parameters, plt.get_cmap('magma'))
            
            galrad = 0.1 * sf.data['frc2'][0]
            zmax = 0.003
            
            star_radius = np.sqrt((pos[:, 1:] ** 2).sum(axis=1))
            star_height = pos[:, 0]
            
            nz = 30
            dr = 0.0025
            dv = np.zeros(nz)
            rho = np.zeros(nz)
            zbin = np.zeros(nz)
            
            dashes = [(None, None), [1, 1], [4, 2], [4, 2, 1, 2]]
            for j in range(len(rd)):
                jj, = np.where((star_radius < (rd[j] + dr)) & (star_radius > (rd[j] - dr)))
                pn, edges = np.histogram(star_height[jj], bins=nz, weights=smass[jj], range=(-zmax, zmax))
                zbin[:] = (edges[:-1] + edges[1:]) * 0.5
                dv[:] = np.pi * ((rd[j] + dr) ** 2 - (rd[j] - dr) ** 2) * abs(edges[1:] - edges[:-1]) * 1e9
                rho[:] = pn[:] * 1e10 / dv[:]
                
                ax.semilogy(zbin * 1e3, rho, color=s_m.to_rgba(time[isnap]), lw=0.5, dashes=dashes[j])
        
        rtime = [round(elem, 1) for elem in time]
        figure.set_colorbar([time[0], time[-1]], '$\\rm{ t_{lookback} \; [Gyr]}$', rtime)
        
        ax.xaxis.set_ticks([-3, -2, -1, 0, 1, 2, 3])
        ax.yaxis.set_ticks([1000000, 10000000, 100000000])
        
        sol = matplotlib.lines.Line2D([0], [0], mfc='black', color='k', dashes=dashes[0])
        dot = matplotlib.lines.Line2D([0], [0], mfc='black', color='k', dashes=dashes[1])
        das = matplotlib.lines.Line2D([0], [0], mfc='black', color='k', dashes=[4, 2])
        labels = [r'0<R<5', r'5<R<10', r'10<R<15']
        handle = [sol, dot, das]
        ax.legend(handle, labels, loc='upper left', frameon=False, prop={'size': 4}, handlelength=4, markerscale=0.5)
    
    figure.reset_axis_limits()
    figure.fig.savefig('%s/rho_z_prof_evol%03d.%s' % (outpath, snap, suffix))