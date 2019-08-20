from const import *
from loadmodules import *
from pylab import *
from util import *


def plot_angularmomentum_profile(runs, dirs, outpath, outputlistfile, zlist, suffix, nrows, ncols, types=[4]):
    panels = nrows * ncols
    
    figure = multipanel_layout.multipanel_layout(nrows=nrows, ncols=ncols, npanels=panels)
    figure.set_figure_layout()
    figure.set_axis_limits_and_aspect(xlim=[0.0, 24.0], ylim=[0.0, 3.])
    figure.set_axis_locators(xminloc=1., xmajloc=5., yminloc=0.5, ymajloc=0.4)
    figure.set_fontsize(6)
    figure.set_axis_labels(xlabel="$\\rm{R [kpc]}$", ylabel="$\\rm{ l_z(<R)\,[10^9\;km\,s^{-1}]}$")
    
    figure2 = multipanel_layout.multipanel_layout(nrows=nrows, ncols=ncols, npanels=panels)
    figure2.set_figure_layout()
    figure2.set_axis_limits_and_aspect(xlim=[0.0, 24.0], ylim=[0.0, 10.0])
    figure2.set_axis_locators(xminloc=1., xmajloc=5.0, yminloc=1., ymajloc=2.)
    figure2.set_axis_labels(xlabel="$\\rm{ R \, [kpc]}$", ylabel="$\\rm{M(<R)\,[10^{10}M_{\odot}]}$")
    figure2.set_fontsize()
    
    lstyle = ['-', '--', ':']
    
    for d in range(len(runs)):
        dd = dirs[d] + runs[d]
        ax = figure.axes[d]
        ax2 = figure2.axes[d]
        
        snaps = np.int_(select_snapshot_number.select_snapshot_number(outputlistfile[d], zlist))
        
        for isnap in range(len(snaps)):
            snap = snaps[isnap]
            print("Doing dir %s snap %d." % (dd, snap))
            attrs = ['pos', 'vel', 'mass', 'age', 'id']
            
            s = gadget_readsnap(snap, snappath=dd + '/output/', hdf5=True, loadonlytype=[4], loadonly=attrs, forcesingleprec=True)
            sf = load_subfind(snap, dir=dd + '/output/', hdf5=True, loadonly=['fpos', 'frc2', 'svel', 'flty', 'fnsh', 'slty'])
            s.calc_sf_indizes(sf)
            s.select_halo(sf, 3., use_principal_axis=True, use_cold_gas_spin=False, do_rotation=True)
            galrad = 0.1 * sf.data['frc2'][0]
            g = parse_particledata(s, sf, attrs, radialcut=galrad)
            g.prep_data()
            
            time = s.cosmology_get_lookback_time_from_a(s.time.astype('float64'))
            t = str(time)
            ts = 't=' + t[:4] + ' Gyr'  ##
            rtime = round(time, 1)
            
            for ival, i in enumerate(types):
                dstr = g.datatypes[i]
                
                data = g.sgdata[dstr]
                mass = data['mass']
                if dstr == 'sdata':
                    age = data['age']
                
                rr = np.sqrt((data['pos'] ** 2).sum(axis=1))
                vel = data['vel']
                pos = data['pos']
                
                jz = np.squeeze(np.cross(pos, vel))[:, 0]
                
                rsort = rr.argsort()
                rr = rr[rsort]
                jz = jz[rsort]
                mass = mass[rsort]
                
                jzcum = np.cumsum(jz)
                mcum = np.cumsum(mass)
                
                nshells = 35
                
                n, edges = np.histogram(rr, bins=nshells, range=(0., galrad))
                jzbin, edges = np.histogram(rr, bins=nshells, range=(0., galrad), weights=jzcum)
                jzbin /= n
                
                mbin, edges = np.histogram(rr, bins=nshells, range=(0., galrad), weights=mcum)
                mbin /= n
                
                rbin = 0.5 * (edges[1:] + edges[:-1])
                
                params = np.linspace(zlist[0], zlist[-1], 20)
                cmap = plt.get_cmap('magma')
                s_m = figure.get_scalar_mappable_for_colorbar(params, cmap)
                
                ax.plot(rbin * 1e3, jzbin * 1e-6, linestyle=lstyle[ival], lw=0.7, color=s_m.to_rgba(zlist[isnap]))
                ax2.plot(rbin * 1e3, mbin, linestyle=lstyle[ival], lw=0.7, color=s_m.to_rgba(zlist[isnap]))
        
        if d == 0:
            ax.legend(loc='upper left', frameon=False, prop={'size': 4}, ncol=2)
            ax2.legend(loc='upper left', frameon=False, prop={'size': 4}, ncol=2)
        
        figure.set_panel_title(panel=d, title="$\\rm{%s\,{%s}}$" % ("Au", runs[d].split('_')[1]), position='bottom right')
        figure2.set_panel_title(panel=d, title="$\\rm{%s\,{%s}}$" % ("Au", runs[d].split('_')[1]), position='bottom right')
    
    pticks = np.linspace(zlist[0], zlist[-1], 5)
    pticks = [round(elem, 1) for elem in pticks]
    figure.set_colorbar([zlist[0], zlist[-1]], '$\\rm{z}$', pticks, cmap=cmap)
    figure2.set_colorbar([zlist[0], zlist[-1]], '$\\rm{z}$', pticks, cmap=cmap)
    figure.fig.savefig('%s/lzprof_test.%s' % (outpath, suffix))
    figure2.fig.savefig('%s/mass_cumprof_test.%s' % (outpath, suffix))