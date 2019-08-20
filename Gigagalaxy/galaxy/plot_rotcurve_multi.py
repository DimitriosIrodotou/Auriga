import numpy as np
from const import *
from loadmodules import *
from pylab import *
from util import *

colors = ['g', 'b', 'r', 'k', 'c', 'y', 'm', 'purple']
markers = ['o', '^', 'd', 's']
lencol = len(colors)
lenmrk = len(markers)


def plot_rotcurve_multi(runs, dirs, outpath, nrows, ncols, outputlistfile, z, suffix, subhalo=0, restest=[5, 4], dat=None):
    if restest:
        panels = len(runs) / len(restest)
    else:
        panels = nrows * ncols
    
    figure = multipanel_layout.multipanel_layout(nrows=nrows, ncols=ncols, npanels=panels)
    figure.set_fontsize(8)
    figure.set_figure_layout()
    figure.set_fontsize(8)
    figure.set_axis_locators(xminloc=1.0, xmajloc=5.0, yminloc=10.0, ymajloc=50.0)
    figure.set_axis_labels(xlabel="$\\rm{R\\,[kpc]}$", ylabel="$\\rm{V_c\\,[km\\,s^{-1}]}$")
    figure.set_axis_limits_and_aspect(xlim=[0., 24.], ylim=[0., 340.])
    
    wpath = outpath + '/plotall/'
    if not os.path.exists(wpath):
        os.makedirs(wpath)
    
    if dat:
        table = './data/huang16.txt'
        rad_tab = genfromtxt(table, comments='#', usecols=0)
        vrot_tab = genfromtxt(table, comments='#', usecols=1)
        err_tab = genfromtxt(table, comments='#', usecols=2)
    
    for d in range(len(runs)):
        dd = dirs[d] + runs[d]
        
        snap = np.int_(select_snapshot_number.select_snapshot_number(outputlistfile[d], z))
        
        color = colors[d]
        marker = markers[d]
        
        if not restest:
            ax = figure.axes[d]
        else:
            ax = figure.axes[int(d / len(restest))]
        
        attrs = ['pos', 'vel', 'mass', 'age']
        
        s = gadget_readsnap(snap, snappath=dd + '/output/', hdf5=True, loadonlytype=[0, 1, 4, 5], loadonly=attrs)
        sf = load_subfind(snap, dir=dd + '/output/', hdf5=True, loadonly=['fpos', 'frc2'])
        s.center = sf.data['fpos'][subhalo, :]
        
        pos = s.data['pos'].astype('float64')
        mass = s.data['mass'].astype('float64')
        
        nshells = 100
        radius = 0.04
        dr = radius / nshells
        
        na = s.nparticlesall
        end = na.copy()
        
        for i in range(1, len(end)):
            end[i] += end[i - 1]
        
        start = zeros(len(na), dtype='int32')
        for i in range(1, len(start)):
            start[i] = end[i - 1]
        
        shmass = pylab.zeros((nshells, 6))
        shvel = pylab.zeros((nshells, 6))
        # vtot = pylab.zeros(nshells)
        for i in range(6):
            rp = calcGrid.calcRadialProfile(s.pos[start[i]:end[i], :].astype('float64'), s.data['mass'][start[i]:end[i]].astype('float64'), 0,
                                            nshells, dr, s.center[0], s.center[1], s.center[2])
            
            radius = rp[1, :]
            shmass[:, i] = rp[0, :]
            for j in range(1, nshells):
                shmass[j, i] += shmass[j - 1, i]
            # shvel[:,i] = np.sqrt( 43.0071 * shmass[:,i] / (radius) )
            # vtot[:] += (shvel[:,i]**2)
            shvel[:, i] = pylab.sqrt(G * shmass[:, i] * 1e10 * msol / (radius * 1e6 * parsec)) / 1e5
        
        rp = calcGrid.calcRadialProfile(s.pos.astype('float64'), s.data['mass'].astype('float64'), 0, nshells, dr, s.center[0], s.center[1],
                                        s.center[2])
        
        radius = rp[1, :]
        mtot = rp[0, :]
        
        for j in range(1, nshells):
            mtot[j] += mtot[j - 1]
        
        vtot = pylab.sqrt(G * mtot * 1e10 * msol / (radius * 1e6 * parsec)) / 1e5
        
        if dat:
            ax.errorbar(rad_tab[:9], vrot_tab[:9], yerr=err_tab[:9], fmt='+', color='gray', markersize=2, linewidth=0.7, elinewidth=0.7)
            ax.errorbar(rad_tab[9:20], vrot_tab[9:20], yerr=err_tab[9:20], fmt='x', color='m', markersize=2, linewidth=0.7, elinewidth=0.7)
            ax.errorbar(rad_tab[20:], vrot_tab[20:], yerr=err_tab[20:], fmt='o', color='c', mec='c', markersize=2, linewidth=0.7, elinewidth=0.7)
        
        if not restest:
            ax.plot(radius * 1e3, vtot, '-', color='k', label='total')
            ax.plot(radius * 1e3, shvel[:, 0], '--', color='b', label='gas')
            ax.plot(radius * 1e3, shvel[:, 1], '-.', color='r', label='halo')
            ax.plot(radius * 1e3, shvel[:, 4], ':', color='g', label='stars')
            figure.set_panel_title(panel=d, title="$\\rm{%s\,%s}$" % ('Au', runs[d].split('_')[1]), position='top right')
        else:
            label = "$\\rm{%s\, %s}$" % ('Au', runs[d].split('_')[1])
            for i, level in enumerate(restest):
                lst = 'level%01d' % level
                if lst in dirs[d]:
                    label += '_%01d' % level
            color = colors[d % lencol]
            
            ax.plot(radius * 1e3, vtot, '-', lw=1., color=color, label=label)
        
        if restest:
            ax.legend(loc='upper left', frameon=False, prop={'size': 5}, numpoints=1, ncol=2)
        elif d == 0:
            ax.legend(loc='upper left', frameon=False, prop={'size': 5}, numpoints=1, ncol=2)
    
    if restest:
        figure.fig.savefig("%s/rotcurve_multi%03d_restest.%s" % (wpath, snap, suffix), dpi=300)
    else:
        figure.fig.savefig("%s/rotcurve_multi%03d.%s" % (wpath, snap, suffix), dpi=300)