from const import *
from loadmodules import *
from matplotlib.colors import LogNorm
from pylab import *
from util import *


def plot_metal_gradient_star(runs, dirs, outpath, outputlistfile, redshift, suffix, nrows, ncols, slicedir='z', pptype=4,
                             loage=[0., 8., 6., 4., 2., 0.], upage=[14., 10., 8., 6., 4., 2.], lolim=[0., 0.5, 1.], uplim=[0.5, 1., 2.],
                             birthdatafile=None, atbirth=False, disc_stars=False):
    colors = ['gray', 'r', 'k', 'm', 'c']
    
    if slicedir == 'R':
        bra = [0., 3.]
        ylim = [-0.3, 0.1]
        lab = 'z'
        xlabel = "$\\rm{z\,[kpc]}$"
    elif slicedir == 'z':
        bra = [0., 20.]
        ylim = [-0.1, 0.02]
        lab = 'R'
        xlabel = "$\\rm{R\,[kpc]}$"
    
    ylim = [-1., 1.5]
    # ylim = [0., 5.]
    fig2 = False
    if fig2:
        figure2 = multipanel_layout.multipanel_layout(nrows=1, ncols=ncols, npanels=ncols)
        figure2.set_figure_layout()
        figure2.set_axis_limits_and_aspect(xlim=[0., 12.], ylim=ylim, logaxis=False)
        figure2.set_axis_locators(xminloc=1., xmajloc=2., ymajloc=0.1, yminloc=0.05)
        figure2.set_axis_labels(xlabel="$\\rm{age}\\,[Gyr]$", ylabel="$\\rm{\Delta _%s}\\,[dex\,kpc^{-1}]}$" % lab)
        figure2.set_fontsize(5)
    
    meanage = 0.5 * (np.array(upage) + np.array(loage))
    
    meanlim = 0.5 * (np.array(uplim) + np.array(lolim))
    nrows = len(meanage)
    ncols = len(uplim)
    
    dfe = np.zeros(len(runs) * nrows * ncols)
    
    for d in range(len(runs)):
        tpath = '%s/%s/tracers/' % (outpath, runs[d])
        fname = '%s/nrecycle_tracers.txt' % (tpath)
        fin = open(fname, 'r')
        data = np.loadtxt(fin)
        star_rad = data[:, 0]
        star_age = data[:, 1]
        star_met = data[:, 2]
        star_prid = data[:, 3]
        nrec = data[:, 4]
        fin.close()
        
        fstr = ''
        
        panels = nrows * ncols
        
        figure = multipanel_layout.multipanel_layout(nrows=nrows, ncols=ncols, npanels=panels)
        figure.set_figure_layout()  # [-2., 1.]
        figure.set_axis_limits_and_aspect(xlim=bra, ylim=ylim, logaxis=False)
        # figure.set_axis_locators( xminloc=10., ymajloc=0.5, yminloc=0.05 )
        figure.set_axis_labels(xlabel=xlabel, ylabel="$\\rm{[Fe/H]}$")
        figure.set_fontsize(5)
        
        figure3 = multipanel_layout.multipanel_layout(nrows=1, ncols=ncols, npanels=ncols)
        figure3.set_figure_layout()
        figure3.set_axis_limits_and_aspect(xlim=[0., 12.], ylim=ylim, logaxis=False)
        figure3.set_axis_locators(xminloc=1., xmajloc=2., ymajloc=0.1, yminloc=0.05)
        figure3.set_axis_labels(xlabel="$\\rm{age}\\,[Gyr]$", ylabel="$\\rm{\Delta _%s}\\,[dex\,kpc^{-1}]}$" % lab)
        figure3.set_fontsize(5)
        
        snap = np.int_(select_snapshot_number.select_snapshot_number(outputlistfile[d], redshift))
        dd = dirs[d] + runs[d]
        wpath = outpath + runs[d] + '/metals/'
        print("doing %s, snap=%03d" % (dd, snap))
        
        if atbirth:
            if birthdatafile:  # read birth data from post-processed file
                stardatafile = outpath + runs[d] + '/rm4/' + birthdatafile
                attrs = ['pos', 'vel', 'id', 'mass', 'age', 'gmet', 'pot', 'gz']
            else:
                stardatafile = None  # snapshot already contains birth data
                attrs = ['pos', 'vel', 'id', 'mass', 'age', 'gmet', 'pot', 'gz', 'bpos', 'bvel']
        else:
            stardatafile = None
            attrs = ['pos', 'vel', 'id', 'mass', 'age', 'gmet', 'pot', 'gz']
        
        s = gadget_readsnap(snap, snappath=dd + '/output/', hdf5=True, loadonlytype=[4], loadonly=attrs)
        sf = load_subfind(snap, dir=dd + '/output/', hdf5=True, loadonly=['fpos', 'frc2', 'svel', 'flty', 'fnsh', 'slty'])
        s.calc_sf_indizes(sf)
        s.select_halo(sf, 3., use_principal_axis=True, use_cold_gas_spin=False, do_rotation=True)
        
        if birthdatafile and atbirth:
            attrs.append('bpos')
            attrs.append('bvel')
        g = parse_particledata(s, sf, attrs, radialcut=0.1 * sf.data['frc2'][0], stardatafile=stardatafile)
        g.prep_data()
        
        sdata = g.sgdata['sdata']
        eps2 = sdata['eps2']
        smass = sdata['mass']
        star_age = sdata['age']
        iron = sdata['Fe']
        
        if atbirth:
            star_pos = sdata['bpos']
        else:
            star_pos = sdata['pos']
        
        if slicedir == 'z':
            arr1 = star_pos[:, 0]
            arr2 = np.sqrt((star_pos[:, 1:] ** 2).sum(axis=1))
        
        elif slicedir == 'R':
            arr1 = np.sqrt((star_pos[:, 1:] ** 2).sum(axis=1))
            arr2 = star_pos[:, 0]
        
        pnum = 0
        dfap = []
        nbin = 15
        
        for j in range(nrows):
            for i in range(ncols):
                ax = figure.fig.axes[pnum]
                
                ii, = np.where((arr1 >= lolim[i] * 1e-3) & (arr1 < uplim[i] * 1e-3) & (star_age >= loage[j]) & (star_age < upage[j]) & (
                            abs(star_pos[:, 0]) < 0.005) & (iron > -1.5))
                
                ax.hist2d(abs(arr2[ii]) * 1e3, iron[ii], bins=(80, 80), range=(bra, ylim), weights=smass[ii], normed=False, rasterized=True,
                          norm=LogNorm(), cmap=cm.get_cmap('viridis'))
                
                try:
                    popt, xbin, femean = g.get_gradient(arr2[ii] * 1e3, arrwgt1=smass[ii], arrwgt2=iron[ii], range1=bra, bins=nbin)
                    ax.plot(xbin, popt[0] + popt[1] * xbin, linestyle='-', color='b', lw=0.8, label='$\\rm{\Delta = %.2f}$' % popt[1])
                    ax.plot(xbin, femean, linestyle='--', color='r', lw=0.8)
                    dfe[d * ncols * nrows + j * ncols + i] = popt[1]
                    dfap.append(popt[1])
                except:
                    dfap.append(nan)
                
                # recycling
                kk, = np.where(nrec <= 1)
                prid = star_prid[kk]
                indy, = np.where((np.in1d(sdata['id'][ii], prid) == True))
                ax.plot(arr2[ii][indy][::100] * 1e3, iron[ii][indy][::100], 'gray', marker='.', lw=0., markersize=0.4)
                popt, xbin, femean = g.get_gradient(abs(arr2[ii][indy]) * 1e3, arrwgt1=smass[ii][indy], arrwgt2=iron[ii][indy], range1=bra, bins=nbin)
                ax.plot(xbin, popt[0] + popt[1] * xbin, linestyle='-', color='gray', lw=0.8, label='$\\rm{\Delta = %.2f}$' % popt[1])
                
                kk, = np.where(nrec > 1)
                prid = star_prid[kk]
                indy, = np.where((np.in1d(sdata['id'][ii], prid) == True))
                ax.plot(arr2[ii][indy][::100] * 1e3, iron[ii][indy][::100], 'k', marker='.', lw=0., markersize=0.4)
                popt, xbin, femean = g.get_gradient(abs(arr2[ii][indy]) * 1e3, arrwgt1=smass[ii][indy], arrwgt2=iron[ii][indy], range1=bra, bins=nbin)
                ax.plot(xbin, popt[0] + popt[1] * xbin, linestyle='-', color='k', lw=0.8, label='$\\rm{\Delta = %.2f}$' % popt[1])
                
                ###
                if pnum == 0:
                    title = "$\\rm{Au \,%s}$" % runs[d].split('_')[1]
                else:
                    title = ''
                figure.set_panel_title(panel=pnum, title=title, position='top left')
                ax.text(0.1, 0.1, "$\\rm{%.1f < %s < %.1f}$ \n$\\rm{%.1f < Age < %.1f}$" % (lolim[i], slicedir, uplim[i], loage[j], upage[j]),
                        color='k', transform=ax.transAxes, fontsize=6)
                
                pnum += 1
                
                ax.legend(loc='upper right', frameon=False, prop={'size': 7})
        
        figure.reset_axis_limits()
        
        colors = ['b', 'r', 'k', 'g', 'm', 'c']
        
        for i in range(ncols):
            ax3 = figure3.fig.axes[i]
            ax3.plot(meanage, dfap[i::ncols], linestyle='-', color=colors[i], label='$\\rm{%s=%.0f \\,[kpc]}$' % (slicedir, meanlim[i]))
            ax3.legend(loc='upper left', frameon=False, prop={'size': 7})
        
        if slicedir == 'R':
            figure3.fig.savefig("%s/%s/MAP_verticalmetalgrad_%s.%s" % (outpath, runs[d], runs[d], suffix), dpi=300)
        elif slicedir == 'z':
            figure3.fig.savefig("%s/%s/MAP_radialmetalgrad_lin%s.%s" % (outpath, runs[d], runs[d], suffix), dpi=300)
        
        if pptype == 4:
            sty = 'stars'
        elif pptype == 0:
            sty = 'gas'
        
        if disc_stars:
            fstr += '_disc'
        if atbirth:
            fstr += '_birth'
        
        if slicedir == 'z':
            figname1 = '%s%s/radialmetaldist_%s_hist%s%s_new.%s' % (outpath, runs[d], sty, fstr, runs[d], suffix)
        elif slicedir == 'R':
            figname1 = '%s%s/verticalmetaldist_%s_hist%s%s.%s' % (outpath, runs[d], sty, fstr, runs[d], suffix)
        print("figname1=", figname1)
        
        figure.fig.savefig(figname1)
    
    if fig2:
        
        dfe2 = np.zeros(nrows * ncols)
        disp = np.zeros(nrows * ncols)
        
        for j in range(nrows):
            for i in range(ncols):
                dfe2[j * ncols + i] = np.median(dfe[j * ncols + i::nrows * ncols])
                disp[j * ncols + i] = np.std(dfe[j * ncols + i::nrows * ncols])
        
        for i in range(ncols):
            ax2 = figure2.fig.axes[i]
            ax2.plot(meanage, dfe2[i::ncols], linestyle='-', color=colors[i], label='$\\rm{%s=%.0f \\,[kpc]}$' % (slicedir, meanlim[i]))
            ax2.fill_between(meanage, dfe2[i::ncols] + disp[i::ncols], dfe2[i::ncols] - disp[i::ncols], color=colors[i], alpha=0.3)
            ax2.legend(loc='upper left', frameon=False, prop={'size': 7})
        
        if slicedir == 'R':
            figure2.fig.savefig("%s/plotall/MAP_verticalmetalgrad.%s" % (outpath, suffix), dpi=300)
        elif slicedir == 'z':
            figure2.fig.savefig("%s/plotall/MAP_radialmetalgrad_new.%s" % (outpath, suffix), dpi=300)