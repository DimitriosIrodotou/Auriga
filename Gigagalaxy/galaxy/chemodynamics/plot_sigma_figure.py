from const import *
from loadmodules import *
from pylab import *
from util import *

colors = ['b', 'gray', 'r', 'k', 'g', 'c', 'y', 'm', 'purple']
lws = [0.7, 1., 1.3]


def plot_sigma_figure(runs, dirs, outpath, outputlistfile, zlist, suffix, nrows, ncols, agecut=[[0., 1., 5., 0.], [1., 5., 13., 13.]], ecut=0.7,
                      restest=False):
    if restest:
        panels = len(runs) / len(restest)
    else:
        panels = nrows * ncols
    
    figure = multipanel_layout.multipanel_layout(nrows=nrows, ncols=ncols, npanels=panels, left=0.22, right=0.98, bottom=0.22, top=0.98)
    figure.set_figure_layout()
    figure.set_axis_limits_and_aspect(xlim=[0.0, 16.], ylim=[0.0, 90.])
    figure.set_axis_locators(xminloc=1.0, xmajloc=5., yminloc=5.0, ymajloc=20.0)
    
    figure.set_axis_labels(xlabel="$\\rm{R\\,[kpc]}$", ylabel="$\\rm{\sigma _z\\,[km\\,s^{-1}]}$")
    figure.set_fontsize()
    
    figure2 = multipanel_layout.multipanel_layout(nrows=nrows, ncols=ncols, npanels=panels, left=0.22, right=0.98, bottom=0.22, top=0.98)
    figure2.set_figure_layout()
    figure2.set_axis_limits_and_aspect(xlim=[0.0, 16.], ylim=[0.0, 4.5])
    figure2.set_axis_locators(xminloc=1.0, xmajloc=5.0, yminloc=0.5, ymajloc=1.0)
    
    figure2.set_axis_labels(xlabel="$\\rm{R\\,[kpc]}$", ylabel="$\\rm{<z^{2}>^{1/2}\\,[kpc]}$")
    figure2.set_fontsize()
    
    for d in range(len(runs)):
        dd = dirs[d] + runs[d]
        
        if not restest:
            ax = figure.axes[d]
            ax2 = figure2.axes[d]
        else:
            ax = figure.axes[int(d / len(restest))]
            ax2 = figure2.axes[int(d / len(restest))]
        
        snap = np.int_(select_snapshot_number.select_snapshot_number(outputlistfile[d], zlist))
        
        attrs = ['pos', 'vel', 'mass', 'age', 'pot']
        
        print("Doing dir %s." % dd)
        loadtype = [4]
        s = gadget_readsnap(snap, snappath=dd + '/output/', hdf5=True, loadonly=attrs, loadonlytype=loadtype, forcesingleprec=True)
        sf = load_subfind(snap, dir=dd + '/output/', hdf5=True, loadonly=['fpos', 'frc2', 'flty', 'fnsh', 'slty', 'svel'], forcesingleprec=True)
        s.calc_sf_indizes(sf)
        
        s.select_halo(sf, 3., use_principal_axis=True, use_cold_gas_spin=False, do_rotation=True)
        
        g = parse_particledata(s, sf, attrs, radialcut=0.1 * sf.data['frc2'][0])
        g.prep_data()
        
        sdata = g.sgdata['sdata']
        
        eps2 = sdata['eps2']
        smass = sdata['mass']
        star_age = sdata['age']
        svel = sdata['vel']
        spos = sdata['pos']
        srxy = np.sqrt((spos[:, 1:] ** 2).sum(axis=1))
        
        radius = 0.016
        nshells = 50
        for i in range(len(agecut[0])):
            if d == 0:
                label = r"$\rm{%2.0f < \tau < %2.0f}$" % (agecut[0][i], agecut[1][i])
            else:
                label = ''
            
            if ecut:
                jj, = np.where((eps2 > ecut) & (star_age >= agecut[0][i]) & (star_age < agecut[1][i]) & (abs(spos[:, 0]) < 0.001))
            else:
                jj, = np.where((eps2 > -1.1) & (star_age >= agecut[0][i]) & (star_age < agecut[1][i]) & (abs(spos[:, 0]) < 0.001))
            
            mass = smass[jj]
            vel = svel[jj]
            pos = spos[jj]
            rxy = srxy[jj]
            
            avgvel, edge = np.histogram(rxy, bins=nshells, range=(0.0, radius), weights=(mass[:] * vel[:, 0]))
            avgz, edge = np.histogram(rxy, bins=nshells, range=(0.0, radius), weights=(mass[:] * pos[:, 0]))
            masstot, edge = np.histogram(rxy, bins=nshells, range=(0.0, radius), weights=mass[:])
            
            xbin = np.zeros(len(edge) - 1)
            xbin[:] = 0.5 * (edge[:-1] + edge[1:])
            
            avgvel /= masstot
            avgz /= masstot
            
            binind = np.digitize(rxy, xbin) - 1
            sigma, edge = np.histogram(rxy, bins=nshells, range=(0.0, radius), weights=(mass[:] * (vel[:, 0] - avgvel[binind]) ** 2))
            sigmaz, edge = np.histogram(rxy, bins=nshells, range=(0.0, radius), weights=(mass[:] * (pos[:, 0] - avgz[binind]) ** 2))
            
            sigma /= masstot
            sigmaz /= masstot
            sigma = np.sqrt(sigma)
            sigmaz = np.sqrt(sigmaz)
            
            if restest:
                for j, level in enumerate(restest):
                    lst = 'level%01d' % level
                    if lst in dirs[d]:
                        lw = lws[j]  # rlab = " $\\rm{Au %s lvl %01d}$" % (runs[d].split('_')[1], level )
            else:
                lw = 0.7
            rlab = "$\\rm{%s\,%s}$" % ("Au", runs[d].split('_')[1])
            
            ax.plot(xbin * 1.e3, sigma, marker=None, linestyle='-', color=colors[i], lw=lw, label=label)
            ax2.plot(xbin * 1.e3, 1.e3 * sigmaz, marker=None, linestyle='-', color=colors[i], lw=lw, label=label)
        
        ax.legend(loc='upper right', frameon=False, prop={'size': 6})
        ax2.legend(loc='upper right', frameon=False, prop={'size': 6})
        
        if restest:
            if d % len(restest):
                ax.text(0.1, 0.85, rlab, color='k', transform=ax.transAxes, fontsize=8)
                ax2.text(0.1, 0.85, rlab, color='k', transform=ax2.transAxes, fontsize=8)
    
    if restest:
        figure.fig.savefig("%s/%s%03d_restest.%s" % (outpath, "sigma", snap, suffix))
        figure2.fig.savefig("%s/%s%03d_restest.%s" % (outpath, "height", snap, suffix))
    else:
        figure.fig.savefig("%s/%s%03d.%s" % (outpath, "sigma", snap, suffix))
        figure2.fig.savefig("%s/%s%03d.%s" % (outpath, "height", snap, suffix))