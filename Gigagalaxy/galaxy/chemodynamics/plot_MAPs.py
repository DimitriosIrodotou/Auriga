from const import *
from loadmodules import *
from pylab import *
from util import *

gstring = ['one', 'two', 'three', 'four', 'five', 'six']


def plot_MAPs(runs, dirs, outpath, outputlistfile, zlist, suffix, nrows, ncols, alphacut=0.1, ecut=False, agecut=False):
    panels = len(runs)
    
    xlim = [-2.0, 1.0]
    ylim = [0.0, 0.6]
    
    for isnap in range(len(zlist)):
        
        figure = multipanel_layout.multipanel_layout(nrows=nrows, ncols=ncols, npanels=panels, left=0.1, right=0.88, bottom=0.1, top=0.88)
        figure.set_figure_layout()
        figure.set_axis_locators(xminloc=1., xmajloc=5., ymajloc=1., yminloc=0.1)
        figure.set_axis_labels(xlabel="$\\rm{R [kpc]}$", ylabel="$\\rm{\Sigma [M_{\odot} pc^{-2}]}$")
        figure.set_fontsize()
        figure.set_axis_limits_and_aspect(xlim=[0.0, 24.0], ylim=[1e-1, 2e4], logyaxis=True)
        
        figure2 = multipanel_layout.multipanel_layout(nrows=nrows, ncols=ncols, npanels=panels, left=0.1, right=0.88, bottom=0.1, top=0.88)
        figure2.set_figure_layout()
        figure2.set_axis_locators(xminloc=1., xmajloc=5., ymajloc=2., yminloc=1.)
        figure2.set_axis_labels(xlabel="$\\rm{R [kpc]}$", ylabel="$\\rm{\overline{z} [kpc]}$")
        figure2.set_fontsize()
        figure2.set_axis_limits_and_aspect(xlim=[0.0, 24.0], ylim=[0., 9.])
        
        figure3 = multipanel_layout.multipanel_layout(nrows=nrows, ncols=ncols, npanels=panels, left=0.1, right=0.88, bottom=0.1, top=0.88)
        figure3.set_figure_layout()
        figure3.set_axis_locators(xminloc=1., xmajloc=5., ymajloc=2., yminloc=1.)
        figure3.set_axis_labels(xlabel="$\\rm{R [kpc]}$", ylabel="$\\rm{Age [Gyr]}$")
        figure3.set_fontsize()
        figure3.set_axis_limits_and_aspect(xlim=[0.0, 24.0], ylim=[0., 12.5])
        
        cbliml = [-1., 0.5]
        cblimp = [-1.25, 0.5]
        parameters = np.linspace(cblimp[0], cblimp[1], 20)
        cmapl = plt.get_cmap('Blues')  # mpl.cm.jet
        s_m_lowa = figure.get_scalar_mappable_for_colorbar(parameters, cmapl)
        
        cblimh = [-1.25, 0.25]
        cblimp = [-1.5, 0.25]
        parameters = np.linspace(cblimp[0], cblimp[1], 20)
        cmaph = plt.get_cmap('Reds')  # mpl.cm.jet
        s_m_hia = figure.get_scalar_mappable_for_colorbar(parameters, cmaph)
        
        # for isnap in range(len(zlist)):
        for d in range(len(runs)):
            dd = dirs[d] + runs[d]
            
            ax = figure.fig.axes[d]
            ax2 = figure2.fig.axes[d]
            ax3 = figure3.fig.axes[d]
            
            snap = np.int_(select_snapshot_number.select_snapshot_number(outputlistfile[d], [zlist[isnap]]))
            
            print("Doing dir %s snap %d." % (dd, snap))
            
            attrs = ['pos', 'vel', 'mass', 'age', 'gmet', 'pot']
            s = gadget_readsnap(snap, snappath=dd + '/output/', hdf5=True, loadonlytype=[4], loadonly=attrs)
            sf = load_subfind(snap, dir=dd + '/output/', hdf5=True, loadonly=['fpos', 'frc2', 'svel', 'flty', 'fnsh', 'slty', 'fmc2'])
            s.calc_sf_indizes(sf)
            s.select_halo(sf, 3., use_principal_axis=True, use_cold_gas_spin=False, do_rotation=True)
            g = parse_particledata(s, sf, attrs, radialcut=0.1 * sf.data['frc2'][0])
            g.prep_data()
            
            sdata = g.sgdata['sdata']
            eps2 = sdata['eps2']
            smass = sdata['mass']
            age_s = sdata['age']
            iron = sdata['Fe']
            alphaelem = sdata['alpha']
            alphaelem -= iron
            pos = sdata['pos']
            radius = np.sqrt((pos[:, 1:] ** 2).sum(axis=1))
            z = pos[:, 0]
            
            galrad = 0.1 * sf.data['frc2'][0]
            
            if agecut:
                jj, = np.where((age_s < agecut))
                smass = smass[jj]
                iron = iron[jj]
                alphaelem = alphaelem[jj]
                age_s = age_s[jj]
                radius = radius[jj]
                z = z[jj]
            
            nbin = 7
            xbin = np.linspace(xlim[0], xlim[1], nbin)
            ybin = np.linspace(ylim[0], ylim[1], nbin)
            
            xcen = np.zeros(nbin - 1)
            ycen = np.zeros(nbin - 1)
            xcen[:] = 0.5 * (xbin[1:] + xbin[:-1])
            ycen[:] = 0.5 * (ybin[1:] + ybin[:-1])
            
            dlistf = []
            dlista = []
            indlist = np.zeros((len(iron), ((nbin - 1) * (nbin - 1))))
            count = 0
            for j in range(nbin - 1):
                for i in range(nbin - 1):
                    ii, = np.where((iron > xbin[i]) & (iron < xbin[i + 1]) & (alphaelem > ybin[j]) & (alphaelem < ybin[j + 1]))
                    if len(ii) > 1e3:
                        strtmp = gstring[i] + gstring[j]
                        dlistf.append(xcen[i])
                        dlista.append(ycen[j])
                        indlist[:len(ii), count] = ii[:]
                        count += 1
            
            nhbin = 20
            sa = np.zeros(nhbin)
            sden = np.zeros(nhbin)
            
            for k in range(len(dlistf)):
                jj, = np.where((indlist[:, k] != 0.))
                if (dlista[k] < alphacut):
                    ii = np.zeros(len(jj))
                    ii[:] = indlist[jj, k]
                    ii = np.array(ii, dtype=int)
                    
                    # get number in each radial bin
                    n, edges = np.histogram(radius[ii], bins=nhbin, range=(0.0, galrad))
                    nn, = np.where((n > 100))  # plot only these
                    
                    mtot, edges = np.histogram(radius[ii], weights=smass[ii], bins=nhbin, range=(0.0, galrad))
                    
                    zmean, edges = np.histogram(radius[ii], weights=smass[ii] * z[ii], bins=nhbin, range=(0.0, galrad))
                    zmean /= mtot
                    binind = np.digitize(radius[ii], edges) - 1
                    hz, edges = np.histogram(radius[ii], weights=smass[ii] * (z[ii] - zmean[binind]) ** 2, bins=nhbin, range=(0.0, galrad))
                    hz /= mtot
                    hz = np.sqrt(hz)
                    
                    sage, edges = np.histogram(radius[ii], weights=smass[ii] * age_s[ii], bins=nhbin, range=(0.0, galrad))
                    sage /= mtot
                    
                    sa[:] = np.pi * (edges[1:] * edges[1:] - edges[:-1] * edges[:-1])
                    sden[:] = mtot[:] / sa[:]
                    x = np.zeros(nhbin)
                    x[:] = 0.5 * (edges[1:] + edges[:-1])
                    
                    ax.semilogy(x[nn] * 1e3, sden[nn] * 1e-2, linewidth=0.7, color=s_m_lowa.to_rgba(dlistf[k]))  # ,
                    # label=r'$\rm{Fe,\alpha=(%.2f,%.2f)}$'%(dlistf[k],dlista[k]) )
                    ax2.plot(x[nn] * 1e3, hz[nn] * 1e3, linewidth=0.7,
                             color=s_m_lowa.to_rgba(dlistf[k]))  # , label=r'$\rm{Fe,\alpha=(%.2f,%.2f)}$'%(dlistf[k],dlista[k]) )
                    ax3.plot(x[nn] * 1e3, sage[nn], linewidth=0.7,
                             color=s_m_lowa.to_rgba(dlistf[k]))  # , label=r'$\rm{Fe,  # \alpha=(%.2f,%.2f)}$'%(dlistf[k],dlista[k]) )
            
            # ax.set_color_cycle([plt.cm.hot(i) for i in np.linspace(0, 1, 5)])
            # ax2.set_color_cycle([plt.cm.hot(i) for i in np.linspace(0, 1, 5)])
            # ax3.set_color_cycle([plt.cm.hot(i) for i in np.linspace(0, 1, 5)])
            
            for k in range(len(dlistf)):
                jj, = np.where((indlist[:, k] != 0.))
                if (dlista[k] > alphacut):
                    ii = np.zeros(len(jj))
                    ii[:] = indlist[jj, k]
                    ii = np.array(ii, dtype=int)
                    
                    # get number in each radial bin
                    n, edges = np.histogram(radius[ii], bins=nhbin, range=(0.0, galrad))
                    nn, = np.where((n > 100))  # plot only these
                    
                    mtot, edges = np.histogram(radius[ii], weights=smass[ii], bins=nhbin, range=(0.0, galrad))
                    
                    zmean, edges = np.histogram(radius[ii], weights=smass[ii] * z[ii], bins=nhbin, range=(0.0, galrad))
                    zmean /= mtot
                    binind = np.digitize(radius[ii], edges) - 1
                    hz, edges = np.histogram(radius[ii], weights=smass[ii] * (z[ii] - zmean[binind]) ** 2, bins=nhbin, range=(0.0, galrad))
                    hz /= mtot
                    hz = np.sqrt(hz)
                    
                    sage, edges = np.histogram(radius[ii], weights=smass[ii] * age_s[ii], bins=nhbin, range=(0.0, galrad))
                    sage /= mtot
                    
                    sa[:] = np.pi * (edges[1:] * edges[1:] - edges[:-1] * edges[:-1])
                    sden[:] = mtot[:] / sa[:]
                    
                    x = np.zeros(nhbin)
                    x[:] = 0.5 * (edges[1:] + edges[:-1])
                    ax.semilogy(x[nn] * 1e3, sden[nn] * 1e-2, linewidth=0.7,
                                color=s_m_hia.to_rgba(dlistf[k]))  # , label=r'$\rm{Fe,\alpha=(%.2f,%.2f)}$'%(dlistf[k],dlista[k]) )
                    ax2.plot(x[nn] * 1e3, hz[nn] * 1e3, linewidth=0.7,
                             color=s_m_hia.to_rgba(dlistf[k]))  # , label=r'$\rm{Fe,\alpha=(%.2f,%.2f)}$'%(dlistf[k],dlista[k]) )
                    ax3.plot(x[nn] * 1e3, sage[nn], linewidth=0.7,
                             color=s_m_hia.to_rgba(dlistf[k]))  # , label=r'$\rm{Fe,\alpha=(%.2f,%.2f)}$'%(dlistf[k],dlista[k]) )
            
            mtot, edges = np.histogram(radius, weights=smass, bins=nhbin, range=(0.0, galrad))
            
            sden[:] = mtot[:] / sa[:]
            ax.semilogy(x * 1e3, sden * 1e-2, linewidth=0.7, color='k', linestyle='--', label='')
            
            zmean, edges = np.histogram(radius, weights=smass * z, bins=nhbin, range=(0.0, galrad))
            zmean /= mtot
            binind = np.digitize(radius, edges) - 1
            hz, edges = np.histogram(radius, weights=smass * (z - zmean[binind]) ** 2, bins=nhbin, range=(0.0, galrad))
            hz /= mtot
            hz = np.sqrt(hz)
            ax2.plot(x * 1e3, hz * 1e3, linewidth=0.7, color='k', linestyle='--', label='')
            
            sage, edges = np.histogram(radius, weights=smass * age_s, bins=nhbin, range=(0.0, galrad))
            sage /= mtot
            
            ax3.plot(x * 1e3, sage, linewidth=0.7, color='k', linestyle='--', label='')
            
            # ax.xaxis.set_ticks([0,5,10,15,20,25])
            # ax.yaxis.set_ticks([10.,100.,1000,10000,100000,1000000])
            ax.legend(loc='upper right', frameon=False, prop={'size': 5}, ncol=1)
            # ax2.legend( loc='upper right', frameon=False, prop={'size':5}, ncol=1 )
            
            ax.axvline(galrad * 1e3, color='gray', linestyle='--')
            ax2.axvline(galrad * 1e3, color='gray', linestyle='--')
            ax3.axvline(galrad * 1e3, color='gray', linestyle='--')
            
            figure.set_panel_title(panel=d, title="$\\rm{%s\,{%s}}$" % ('Au', runs[d].split('_')[1]), position='top left')
            figure2.set_panel_title(panel=d, title="$\\rm{%s\,{%s}}$" % ('Au', runs[d].split('_')[1]), position='top left')
            figure3.set_panel_title(panel=d, title="$\\rm{%s\,{%s}}$" % ('Au', runs[d].split('_')[1]), position='top right')
        
        figure.set_colorbar(cbliml, r'$\rm{ [Fe/H] \, [dex] \, (\alpha \, poor \, sequence) }$', [-1., -0.75, -0.5, -0.25, 0., 0.25, 0.5], cmap=cmapl,
                            fontsize=7, labelsize=7, orientation='vertical')
        figure.set_colorbar(cblimh, r'$\rm{ [Fe/H] \, [dex] \, (\alpha \, rich \, sequence)}$', [-1.25, -1., -0.75, -0.5, -0.25, 0., 0.25],
                            cmap=cmaph, fontsize=7, labelsize=7, orientation='horizontal')
        figure2.set_colorbar(cbliml, r'$\rm{ [Fe/H] \, [dex] \, (\alpha \, poor \, sequence)}$', [-1., -0.75, -0.5, -0.25, 0., 0.25, 0.5], cmap=cmapl,
                             fontsize=7, labelsize=7, orientation='vertical')
        figure2.set_colorbar(cblimh, r'$\rm{ [Fe/H] \, [dex] \, (\alpha \, rich \, sequence)}$', [-1.25, -1., -0.75, -0.5, -0.25, 0., 0.25],
                             cmap=cmaph, fontsize=7, labelsize=7, orientation='horizontal')
        figure3.set_colorbar(cbliml, r'$\rm{ [Fe/H] \, [dex] \, (\alpha \, poor \, sequence)}$', [-1., -0.75, -0.5, -0.25, 0., 0.25, 0.5], cmap=cmapl,
                             fontsize=7, labelsize=7, orientation='vertical')
        figure3.set_colorbar(cblimh, r'$\rm{ [Fe/H] \, [dex] \, (\alpha \, rich \, sequence)}$', [-1.25, -1., -0.75, -0.5, -0.25, 0., 0.25],
                             cmap=cmaph, fontsize=7, labelsize=7, orientation='horizontal')
        
        figure.reset_axis_limits()
        figure2.reset_axis_limits()
        figure3.reset_axis_limits()
        
        name = ''
        if agecut:
            name += '_agecut'
        figure.fig.savefig('%s/MAP_sdenprof%03d%s_new.%s' % (outpath, snap, name, suffix))
        figure2.fig.savefig('%s/MAP_height%03d%s_new.%s' % (outpath, snap, name, suffix))
        figure3.fig.savefig('%s/MAP_age%03d%s_new.%s' % (outpath, snap, name, suffix))