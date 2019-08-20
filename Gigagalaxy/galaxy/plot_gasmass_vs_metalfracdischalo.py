import numpy as np
from const import *
from loadmodules import *
from pylab import *
from util import *

colors = ['g', 'b', 'r', 'k', 'c', 'y', 'm', 'purple']
markers = ['o', '^', 'd', 's']
lencol = len(colors)


def plot_gasmass_vs_metalfracdischalo(runs, dirs, outpath, nrows, ncols, outputlistfile, zlist, suffix, subhalo=0, restest=[5, 4], zcut=0.005,
                                      colorbymass=False):
    panels = nrows * ncols
    figure = multipanel_layout.multipanel_layout(nrows=nrows, ncols=ncols, npanels=panels)
    figure.set_fontsize(8)
    figure.set_figure_layout()
    figure.set_fontsize(8)
    figure.set_axis_labels(xlabel="$\\rm{M_{gas-metal}\\,[M_\\odot]}$", ylabel="$\\rm{f_{disc,halo}}$")
    figure.set_axis_limits_and_aspect(xlim=[0., 0.1], ylim=[0., 1.], logxaxis=False)
    
    wpath = outpath + '/plotall/'
    if not os.path.exists(wpath):
        os.makedirs(wpath)
    
    nsnaps = len(zlist)
    
    for isnap in range(nsnaps):
        ax = figure.axes[isnap]
        
        for d in range(len(runs)):
            dd = dirs[d] + runs[d]
            
            snaps = np.int_(select_snapshot_number.select_snapshot_number(outputlistfile[d], zlist))
            if isinstance(snaps, int):
                snaps = [snaps]
            
            print("Doing dir %s" % (dd))
            
            attrs = ['pos', 'vel', 'mass', 'age', 'gmet', 'gz']
            
            s = gadget_readsnap(snaps[isnap], snappath=dd + '/output/', hdf5=True, loadonlytype=[0, 1, 4], loadonly=attrs)
            sf = load_subfind(snaps[isnap], dir=dd + '/output/', hdf5=True, loadonly=['fpos', 'frc2', 'svel', 'flty', 'fnsh', 'slty', 'fmc2'])
            s.calc_sf_indizes(sf)
            s.center = sf.data['fpos'][subhalo, :]
            s.select_halo(sf, 3., use_principal_axis=True, use_cold_gas_spin=False, do_rotation=True)
            
            g = parse_particledata(s, sf, attrs, radialcut=sf.data['frc2'][subhalo])
            g.prep_data()
            sdata = g.sgdata['sdata']
            gdata = g.sgdata['gdata']
            
            mgas = gdata['mass'].sum()
            
            metal_mass_tot = (gdata['gz'] * gdata['mass']).sum()
            
            # define disc-halo geometrically
            idisc, = np.where(((np.sqrt((gdata['pos'][:, 1:] ** 2).sum(axis=1))) < 0.1 * sf.data['frc2'][subhalo]) & (gdata['pos'][:, 0] < zcut))
            
            iall = arange(size(gdata['mass']))
            
            ihalo, = np.where(np.in1d(iall, idisc) == False)
            # ihalo, = np.where( ( (np.sqrt(gdata['pos'][:,1:]**2).sum(axis=1)) > 0.1 * sf.data['frc2'][subhalo] ) & ( gdata['pos'][:,
            # 0] < zcut ) )
            
            metal_mass_disc = (gdata['gz'][idisc] * gdata['mass'][idisc]).sum()
            metal_mass_halo = (gdata['gz'][ihalo] * gdata['mass'][ihalo]).sum()
            
            if isnap == nsnaps - 1 and (restest or not colorbymass):
                label = "$\\rm{%s\, %s}$" % ('Au', runs[d].split('_')[1])
            else:
                label = ''
            
            if restest:
                for i, level in enumerate(restest):
                    lst = 'level%01d' % level
                    if lst in dirs[d]:
                        marker = markers[i]
                        label += " $\\rm{lvl %01d}$" % level
                
                color = colors[d % lencol]
            else:
                marker = 'o'
                color = colors[d]
            
            if colorbymass:
                pmin = 0.1;
                pmax = 10.
                params = np.linspace(pmin, pmax, 20)
                cmap = plt.get_cmap('magma')
                s_m = figure.get_scalar_mappable_for_colorbar(params, cmap)
                color = s_m.to_rgba(mstars)
            
            ax.plot(metal_mass_tot / mgas, metal_mass_disc / metal_mass_tot, marker='^', mfc='none', mec=color, ms=5., mew=1., linestyle='None',
                    label=label)
            ax.plot(metal_mass_tot / mgas, metal_mass_halo / metal_mass_tot, marker='o', mfc='none', mec=color, ms=5., mew=1., linestyle='None')
        
        ax.legend(loc='upper left', prop={'size': 5}, frameon=False, numpoints=1, ncol=1)
        
        ax.text(0.7, 0.1, "$\\rm{z=%01d}$" % zlist[isnap], transform=ax.transAxes, fontsize=10)
    
    if colorbymass:
        pticks = np.linspace(np.log10(pmin * 1e10), np.log10(pmax * 1e10), 5)
        pticks = [round(elem, 1) for elem in pticks]
        figure.set_colorbar([np.log10(pmin * 1e10), np.log10(pmax * 1e10)], '$\\rm{ log_{10}(M_{*} \, [M_{\odot}])}$', pticks, cmap=cmap)
    
    if restest:
        figure.fig.savefig("%s/gasmass_metalfracdischalo%03d_restest.pdf" % (wpath, snaps[-1]), dpi=300)
    else:
        figure.fig.savefig("%s/gasmass_metalfracdischalo%03d.pdf" % (wpath, snaps[-1]), dpi=300)