import numpy as np
from const import *
from loadmodules import *
from parse_particledata import parse_particledata
from pylab import *
from util import *

colors = ['g', 'b', 'r', 'k', 'c', 'y', 'm', 'purple']
markers = ['o', '^', 'd', 's']
lencol = len(colors)


def plot_stellarmass_vs_metallicity(runs, dirs, outpath, nrows, ncols, outputlistfile, zlist, suffix, subhalo=0, restest=[5, 4], colorbymass=False,
                                    dat=False):
    panels = nrows * ncols
    figure = multipanel_layout.multipanel_layout(nrows=nrows, ncols=ncols, npanels=panels)
    figure.set_fontsize(8)
    figure.set_figure_layout()
    figure.set_fontsize(8)
    figure.set_axis_labels(xlabel="$\\rm{M_{stars}\\,[M_\\odot]}$", ylabel="$\\rm{Z\\,[dex]}$")
    figure.set_axis_limits_and_aspect(xlim=[5.e8, 5.e11], ylim=[-1.2, 0.6], logxaxis=True)
    
    wpath = outpath + '/plotall/'
    if not os.path.exists(wpath):
        os.makedirs(wpath)
    
    nsnaps = len(zlist)
    
    for isnap in range(nsnaps):
        ax = figure.axes[isnap]
        
        if zlist[isnap] < 0.1:
            # plot Gallazzi+ (2003) table
            table = './data/gallazzi.txt'
            masses = genfromtxt(table, comments='#', usecols=0)
            medianfe = genfromtxt(table, comments='#', usecols=1)
            sigmadownfe = genfromtxt(table, comments='#', usecols=2)
            sigmaupfe = genfromtxt(table, comments='#', usecols=3)
            # put data in correct units
            masses = 10 ** masses
            
            ax.fill_between(masses, sigmadownfe, sigmaupfe, color='royalblue', alpha=0.2, edgecolor='None')
            ax.semilogx(masses, medianfe, color='royalblue', lw=1.)
            label = ["Gallazzi+ 05"]
            l1 = ax.legend(label, loc='upper left', fontsize=6, frameon=False, numpoints=1)
        
        for d in range(len(runs)):
            dd = dirs[d] + runs[d]
            color = colors[d]
            marker = markers[d]
            
            snaps = np.int_(select_snapshot_number.select_snapshot_number(outputlistfile[d], zlist))
            if isinstance(snaps, int):
                snaps = [snaps]
            
            print("Doing dir %s" % (dd))
            
            attrs = ['pos', 'vel', 'mass', 'age', 'gmet', 'gz']
            
            s = gadget_readsnap(snaps[isnap], snappath=dd + '/output/', hdf5=True, loadonlytype=[1, 4], loadonly=attrs)
            sf = load_subfind(snaps[isnap], dir=dd + '/output/', hdf5=True, loadonly=['fpos', 'frc2'])
            s.center = sf.data['fpos'][subhalo, :]
            
            g = parse_particledata(s, sf, attrs, radialcut=sf.data['frc2'][subhalo])
            g.prep_data()
            sdata = g.sgdata['sdata']
            gdata = g.sgdata['gdata']
            
            mstars = sdata['mass'].sum()
            
            mass_weighted_metallicity = np.log10(((10 ** sdata['Fe']) * sdata['mass']).sum() / mstars)
            
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
            
            if colorbymass:
                pmin = 0.1;
                pmax = 10.
                params = np.linspace(pmin, pmax, 20)
                cmap = plt.get_cmap('magma')
                s_m = figure.get_scalar_mappable_for_colorbar(params, cmap)
                color = s_m.to_rgba(mstars)
            
            ax.semilogx(1.0e10 * mstars, mass_weighted_metallicity, marker=marker, mfc='none', mec=color, ms=5., mew=1., linestyle='None',
                        label=label)
        
        ax.legend(loc='upper left', prop={'size': 5}, frameon=False, numpoints=1, ncol=1)
        
        ax.text(0.7, 0.1, "$\\rm{z=%01d}$" % zlist[isnap], transform=ax.transAxes, fontsize=10)
    
    if colorbymass:
        pticks = np.linspace(np.log10(pmin * 1e10), np.log10(pmax * 1e10), 5)
        pticks = [round(elem, 1) for elem in pticks]
        figure.set_colorbar([np.log10(pmin * 1e10), np.log10(pmax * 1e10)], '$\\rm{ log_{10}(M_{*} \, [M_{\odot}])}$', pticks, cmap=cmap)
    
    if restest:
        figure.fig.savefig("%s/stellarmass_vs_metallicity%03d_restest.pdf" % (wpath, snaps[-1]), dpi=300)
    else:
        figure.fig.savefig("%s/stellarmass_vs_metallicity%03d.pdf" % (wpath, snaps[-1]), dpi=300)