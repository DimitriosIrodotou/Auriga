import numpy as np
from const import *
from loadmodules import *
from parse_particledata import parse_particledata
from pylab import *
from util import *

colors = ['g', 'b', 'r', 'k', 'c', 'y', 'm', 'purple']
markers = ['o', '^', 'd', 's']
lencol = len(colors)
lenmrk = len(markers)


def plot_tullyfisher_mag(runs, dirs, outpath, nrows, ncols, outputlistfile, zlist, suffix, subhalo=0, restest=[5, 4], colorbymass=True,
                         plot_max_val=False, dat=None):
    panels = nrows * ncols
    figure = multipanel_layout.multipanel_layout(nrows=nrows, ncols=ncols, npanels=panels)
    figure.set_fontsize(8)
    figure.set_figure_layout()
    figure.set_fontsize(8)
    figure.set_axis_labels(xlabel="$\\rm{M_{stars}\\,[M_\\odot]}$", ylabel="$\\rm{log_{10}\\,\\,v\\,[km\\,\\,s^{-1}]}$")
    figure.set_axis_limits_and_aspect(xlim=[2.e9, 5.e11], ylim=[1.7, 2.6], logxaxis=True)
    
    wpath = outpath + '/plotall/'
    if not os.path.exists(wpath):
        os.makedirs(wpath)
    
    # initialise plot helper class
    ph = plot_helper.plot_helper()
    
    if plot_max_val == False:  # take vale at optical radius
        file1 = outpath + '/plotall/r_optical.txt'
        if os.path.isfile(file1):
            f = open(file1, 'r')
            data = np.loadtxt(f, delimiter=None, skiprows=1, usecols=(1, 2))
            f.close()
            ropt = data[:, 0]
    
    for d in range(len(runs)):
        dd = dirs[d] + runs[d]
        
        snaps = np.int_(select_snapshot_number.select_snapshot_number(outputlistfile[d], zlist))
        if isinstance(snaps, int):
            snaps = [snaps]
        nsnaps = len(snaps)
        
        color = colors[d]
        marker = markers[d]
        for isnap in range(nsnaps):
            ax = figure.axes[isnap]
            
            if zlist[isnap] == 0. and d == 0:
                if 'pizagno' in dat:
                    mass_p, vcirc_p = ph.get_pizagno_data()
                    print("mass_p, vcirc_p=", mass_p, vcirc_p)
                    ax.semilogx(1.0e10 * mass_p, np.log10(vcirc_p), '^', mfc='hotpink', alpha=0.5, ms=2.5, mec='None', label='$\\rm{Pizagno+ 07}$')
                
                if 'verheijen' in dat:
                    # plot Verheijen (2001) sample
                    mass_v, vcirc_v = ph.get_verheijen_data()
                    ax.semilogx(1.0e10 * mass_v, np.log10(vcirc_v), 's', mfc='hotpink', alpha=0.5, ms=2.5, mec='None', label='$\\rm{Verheijen 01}$')
                
                if 'courteau' in dat:
                    # plot Courteau+ (2007) sample
                    mass_c, vcirc_c = ph.get_courteau_data()
                    ax.semilogx(1.0e10 * mass_c, vcirc_c, 'o', mfc='hotpink', alpha=0.5, ms=2.5, mec='None', label='$\\rm{Courteau+ 07}$')
                
                if 'dutton' in dat:
                    # plot best fit from Dutton et al. (2011)
                    masses = arange(0.1, 50.)
                    tfd = ph.get_dutton_bestfit(masses)
                    ax.semilogx(1.0e10 * masses, tfd, ls='--', dashes=(3, 3), color='red', lw=1., label='$\\rm{Dutton+ 11}$')
                if 'tiley' in dat:
                    # plot best fit from Tiley
                    logm = np.linspace(2.e9, 5.e11, 30)
                    logv, logvlo, logvhi = ph.tiley_bestfit(np.log10(logm))
                    ax.semilogx(logm, logvlo, linestyle='-', color='royalblue', alpha=0.5)
                    ax.semilogx(logm, logvhi, linestyle='-', color='royalblue', alpha=0.5)
                    ax.semilogx(logm, logv, linestyle='-', lw=0.7, color='royalblue', label='$\\rm{McGaugh+ 15}$')
                
                ax.legend(loc='upper left', fontsize=6, frameon=False, numpoints=1)
            
            attrs = ['pos', 'vel', 'mass', 'age']
            s = gadget_readsnap(snaps[isnap], snappath=dd + '/output/', hdf5=True, loadonlytype=[0, 1, 4], loadonly=attrs)
            sf = load_subfind(snaps[isnap], dir=dd + '/output/', hdf5=True, loadonly=['fpos', 'frc2'])
            s.center = sf.data['fpos'][subhalo, :]
            g = parse_particledata(s, sf, attrs, radialcut=sf.data['frc2'][subhalo])
            g.prep_data()
            
            if not plot_max_val and os.path.isfile(file1):
                rindex = int(runs[d].split('_')[1]) - 1
                rmeas = ropt[rindex]
            else:
                rmeas = 0.1 * sf.data['frc2'][subhalo] * 1e3
            
            sdata = g.sgdata['sdata']
            gdata = g.sgdata['gdata']
            
            mstars = sdata['mass'].sum() + gdata['mass'].sum()
            
            pos = s.data['pos'].astype('float64')
            mass = s.data['mass'].astype('float64')
            
            spos = sdata['pos'].astype('float64')
            smass = sdata['mass'].astype('float64')
            
            nshells = 100
            radius = 0.04
            dr = radius / nshells
            
            rp = calcGrid.calcRadialProfile(pos, mass, 0, nshells, dr, g.s.center[0], g.s.center[1], g.s.center[2])
            
            totradius = rp[1, :]
            totmass = rp[0, :]
            
            for j in range(1, nshells):
                totmass[j] += totmass[j - 1]
            
            vtot = pylab.sqrt(G * totmass * 1e10 * msol / (totradius * 1e6 * parsec)) / 1e5
            
            rp = calcGrid.calcRadialProfile(spos, smass, 0, nshells, dr, g.s.center[0], g.s.center[1], g.s.center[2])
            
            sradius = rp[1, :]
            smass = rp[0, :]
            
            if isnap == nsnaps - 1 and (restest or not colorbymass):
                label = "$\\rm{%s\, %s}$" % ('Au', runs[d].split('_')[1])
            else:
                label = ''
            
            if restest:
                for i, level in enumerate(restest):
                    lst = 'level%01d' % level
                    if lst in dirs[d]:
                        marker = markers[i]
                        label += '_%01d' % level
                
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
            
            if plot_max_val:
                ax.semilogx(1.0e10 * smass[:ph.find_nearest(vtot, [vtot.max()])].sum(), log10(vtot[ph.find_nearest(vtot, [vtot.max()])]),
                            marker=marker, ms=5.0, mfc='none', mec=color, mew=1., linestyle='None', label=label)
            else:
                ax.semilogx(1.0e10 * smass[:ph.find_nearest(sradius, [rmeas * 1e-3])].sum(), log10(vtot[ph.find_nearest(sradius, [rmeas * 1e-3])]),
                            marker=marker, ms=5.0, mfc='none', mec=color, mew=1., linestyle='None', label=label)
            
            if len(zlist) > 1:
                ax.text(0.7, 0.1, "$\\rm{z=%01d}$" % zlist[isnap], transform=ax.transAxes, fontsize=10)
            ax.legend(loc='upper left', frameon=False, prop={'size': 4}, numpoints=1, ncol=2)
    
    if colorbymass:
        pticks = np.linspace(log10(pmin * 1e10), log10(pmax * 1e10), 5)
        pticks = [round(elem, 1) for elem in pticks]
        figure.set_colorbar([log10(pmin * 1e10), log10(pmax * 1e10)], '$\\rm{ log_{10}(M_{*} \, [M_{\odot}])}$', pticks, cmap=cmap)
    
    if restest:
        figure.fig.savefig("%s/tullyfisher_mag%03d_ropt_restest.%s" % (wpath, snaps[-1], suffix), dpi=300)
    else:
        figure.fig.savefig("%s/tullyfisher_mag%03d_ropt.%s" % (wpath, snaps[-1], suffix), dpi=300)