import numpy as np
from const import *
from loadmodules import *
from parse_particledata import parse_particledata
from pylab import *
from util import *

colors = ['g', 'b', 'r', 'k', 'c', 'y', 'm', 'purple']
toinch = 0.393700787
markers = ['o', '^', 'd', 's']
lencol = len(colors)


def plot_sfr_vs_stellarmass(runs, dirs, outpath, nrows, ncols, outputlistfile, zlist, suffix, subhalo=0, time_lim=0.5, restest=[5, 4],
                            colorbymass=False, dat=True):
    panels = nrows * ncols
    figure = multipanel_layout.multipanel_layout(nrows=nrows, ncols=ncols, npanels=panels)
    figure.set_fontsize(8)
    figure.set_figure_layout()
    figure.set_fontsize(8)
    figure.set_axis_labels(xlabel="$\\rm{M_{stars}\\,[M_\\odot]}$", ylabel="$\\rm{SFR\,[M_\odot\\,yr^{-1}]}$")
    figure.set_axis_limits_and_aspect(xlim=[5.e8, 5.e11], ylim=[1.0e-1, 100.], logaxis=True)
    
    wpath = outpath + '/plotall/'
    if not os.path.exists(wpath):
        os.makedirs(wpath)
    
    p = plot_helper.plot_helper()
    
    if dat:
        redshift = p.get_fits_value(fitsfile='./data/gal_info_dr7_v5_2.fits', fitsfield='z', fitsflag=['z_warning', 0.1])
        sfrfits = p.get_fits_value(fitsfile='./data/gal_totsfr_dr7_v5_2.fits', fitsfield='avg')
        
        mstar = p.get_fits_value(fitsfile='./data/gal_kcorrect_dr7_v5_2.fits', fitsfield='model_mass')  # , index=0 )
        u = p.get_fits_value(fitsfile='./data/gal_kcorrect_dr7_v5_2.fits', fitsfield='model_absmag', index=0)
        g = p.get_fits_value(fitsfile='./data/gal_kcorrect_dr7_v5_2.fits', fitsfield='model_absmag', index=1)
        r = p.get_fits_value(fitsfile='./data/gal_kcorrect_dr7_v5_2.fits', fitsfield='model_absmag', index=2)
        
        i, = np.where((mstar > 0.0) & (sfrfits > -99.) & (g < 0.0) & (r < 0.0))
        
        grcolor = g[i] - r[i]
        urcolor = u[i] - r[i]
        u = u[i]
        r = r[i]
        sfrfits = sfrfits[i]
        mstar = mstar[i]
        blu, = np.where((grcolor < (0.59 + 0.052 * log10(mstar * 1.0e-10))))
        red, = np.where((grcolor >= (0.59 + 0.052 * log10(mstar * 1.0e-10))))
        
        sfr_blu = 10 ** sfrfits[blu]
        mstar_blu = mstar[blu]
        gr_blu = grcolor[blu]
        ur_blu = urcolor[blu]
        umag_blu = u[blu]
        rmag_blu = r[blu]
        
        sfr_red = 10 ** sfrfits[red]
        mstar_red = mstar[red]
        gr_red = grcolor[red]
        ur_red = urcolor[red]
        umag_red = u[red]
        rmag_red = r[red]
        
        xlimit = [5.0e8, 1.0e12]
        ylimit = [1.0e-2, 100.]
        nx = 40
        xlimit = log10(xlimit)
        ylimit = log10(ylimit)
        dx = (xlimit[1] - xlimit[0]) / nx
        dy = (ylimit[1] - ylimit[0]) / nx
        xlimit[0] -= dx
        xlimit[1] += dx
        ylimit[0] -= dy
        ylimit[1] += dy
        
        x = np.logspace(xlimit[0], xlimit[1], nx)
        y = np.logspace(ylimit[0], ylimit[1], nx)
        
        zred, xedges, yedges = np.histogram2d(mstar_red, sfr_red, bins=[x, y], range=[xlimit, ylimit])
        zblue, xedges, yedges = np.histogram2d(mstar_blu, sfr_blu, bins=[x, y], range=[xlimit, ylimit])
        
        xbin = np.zeros(len(xedges) - 1)
        xbin[:] = 0.5 * (xedges[1:] + xedges[:-1])
        ybin = np.zeros(len(yedges) - 1)
        ybin[:] = 0.5 * (yedges[1:] + yedges[:-1])
        levels = np.linspace(1.5, 4., 20)
    
    nsnaps = len(zlist)
    
    for isnap in range(nsnaps):
        ax = figure.axes[isnap]
        
        masses = np.linspace(5e8, 2e11, 20)
        masses_karim = masses
        
        if zlist[isnap] < 0.1:
            ax.contourf(xbin, ybin, log10(zred.T), cmap=cm.get_cmap('Reds'), alpha=0.5, levels=levels)
            ax.contourf(xbin, ybin, log10(zblue.T), cmap=cm.get_cmap('Blues'), alpha=0.5, levels=levels)
        
        if dat and zlist[isnap] <= 0.1:
            sfr_elbaz11 = p.elbaz11(masses)
            ax.loglog(masses, sfr_elbaz11, linestyle='-', color='seagreen', label='$\\rm{Elbaz+ \, 11 \, (z \sim 0.05)}$')
            sfr_elbaz07 = p.elbaz07(masses / 1e11, 0)
            ax.loglog(masses, sfr_elbaz07, linestyle='-.', color='gray', label='$\\rm{Elbaz+ \, 07 \, (z \sim 0.06)}$')
        if dat and (zlist[isnap] >= 0.8) & (zlist[isnap] < 1.1):
            sfr_karim11 = p.karim11(masses, 10 ** (-0.48), -0.38)
            ax.loglog(masses_karim, sfr_karim11, '-', color='hotpink', label='$\\rm{Karim+ \, 11 \, (z \sim 0.9)}$')
            sfr_elbaz07 = p.elbaz07(masses / 1e10, 1)
            ax.loglog(masses, sfr_elbaz07, linestyle='-.', color='gray', label='$\\rm{Elbaz+ \, 07 \, (z \sim 1)}$')
        if dat and (zlist[isnap] >= 1.7) & (zlist[isnap] < 2.2):
            sfr_karim11 = p.karim11(masses, 10 ** (0.1), -0.41)
            ax.loglog(masses_karim, sfr_karim11, '-', color='hotpink', label='$\\rm{Karim+ \, 11 \, (z \sim 1.8)}$')
            sfr_daddi07 = p.daddi07(masses / 1e11)
            ax.loglog(masses, sfr_daddi07, linestyle='-.', color='cornflowerblue', label='$\\rm{Daddi+ \, 07 \, (z \sim 1.95)}$')
        if dat and (zlist[isnap] >= 2.5) & (zlist[isnap] < 3.5):
            sfr_karim11 = p.karim11(masses, 10 ** (0.22), -0.42)
            ax.loglog(masses_karim, sfr_karim11, '-', color='hotpink', label='$\\rm{Karim+ \, 11 \, (z \sim 2.75)}$')
        if dat and (zlist[isnap] >= 2.5) & (zlist[isnap] < 3.5):
            sfr_magdis10 = p.magdis10(masses / 1e11)
            ax.loglog(masses, sfr_magdis10, '-', color='black', label='$\\rm{Magdis+ \, 10 \, (z \sim 3)}$')
        
        for d in range(len(runs)):
            dd = dirs[d] + runs[d]
            color = colors[d]
            marker = markers[d]
            
            snaps = np.int_(select_snapshot_number.select_snapshot_number(outputlistfile[d], zlist))
            if isinstance(snaps, int):
                snaps = [snaps]
            
            print("Doing dir %s" % (dd))
            
            attrs = ['pos', 'vel', 'mass', 'age', 'sfr', 'gima']
            s = gadget_readsnap(snaps[isnap], snappath=dd + '/output/', hdf5=True, loadonlytype=[0, 4], loadonly=attrs)
            sf = load_subfind(snaps[isnap], dir=dd + '/output/', hdf5=True, loadonly=['fpos', 'frc2'])
            s.center = sf.data['fpos'][subhalo, :]
            s.centerat(s.center)
            g = parse_particledata(s, sf, attrs, radialcut=sf.data['frc2'][subhalo])
            g.prep_data()
            sdata = g.sgdata['sdata']
            gdata = g.sgdata['gdata']
            
            mstars = sdata['mass'].sum()
            sfr = sdata['gima'][(sdata['age'] - s.time) < time_lim].sum() / time_lim * 10.
            
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
            
            ax.loglog(1.0e10 * mstars, sfr, marker=marker, mfc='none', mec=color, ms=5., mew=1., linestyle='None', label=label)
        
        ax.legend(loc='upper left', prop={'size': 5}, frameon=False, numpoints=1, ncol=1)
        
        ax.text(0.7, 0.1, "$\\rm{z=%01d}$" % zlist[isnap], transform=ax.transAxes, fontsize=10)
    
    if colorbymass:
        pticks = np.linspace(log10(pmin * 1e10), log10(pmax * 1e10), 5)
        pticks = [round(elem, 1) for elem in pticks]
        figure.set_colorbar([log10(pmin * 1e10), log10(pmax * 1e10)], '$\\rm{ log_{10}(M_{*} \, [M_{\odot}])}$', pticks, cmap=cmap)
    
    if restest:
        figure.fig.savefig("%s/sfr_vs_starmass_evo%03d_restest.pdf" % (wpath, snaps[-1]), dpi=300)
    else:
        figure.fig.savefig("%s/sfr_vs_starmass_evo%03d.pdf" % (wpath, snaps[-1]), dpi=300)