from const import *
from loadmodules import *
from parse_particledata import parse_particledata
from pylab import *
from util import *
from util import plot_helper

colors = ['g', 'b', 'r', 'k', 'c', 'y', 'm', 'purple']
markers = ['o', '^', 'd', 's']


def plot_stellarvstotalmass(runs, dirs, outpath, nrows, ncols, outputlistfile, zlist, suffix, subhalo=0, restest=False, colorbymass=False, dat=None):
    panels = nrows * ncols
    figure = multipanel_layout.multipanel_layout(nrows=nrows, ncols=ncols, npanels=panels)
    figure.set_fontsize(8)
    figure.set_figure_layout()
    figure.set_fontsize(8)
    figure.set_axis_labels(xlabel="$\\rm{M_{tot}\\,[M_\\odot]}$", ylabel="$\\rm{M_{stars}\\,[M_\\odot]}$")
    figure.set_axis_limits_and_aspect(xlim=[2.e10, 3.e12], ylim=[5.e8, 1.e12], logaxis=True)
    
    lencol = len(colors)
    lenmrk = len(markers)
    
    nsnaps = len(zlist)
    stellarmass = zeros(nsnaps)
    totalmass = zeros(nsnaps)
    
    masses = arange(1., 300.)
    cosmic_baryon_frac = 0.048 / 0.307
    
    loglog(1.0e10 * masses, 1.0e10 * masses * cosmic_baryon_frac, ':', color='gray')
    fill_between(1.0e10 * masses, 1.0e10 * masses * cosmic_baryon_frac, [1e12] * len(masses), color='gray', alpha=0.2, edgecolor='none')
    
    if dat:
        # initialise plot helper class
        ph = plot_helper.plot_helper()
    
    if dat == 'Guo':
        guo_high = ph.guo_abundance_matching(masses) * 10 ** (+0.2)
        guo_low = ph.guo_abundance_matching(masses) * 10 ** (-0.2)
        fill_between(1.0e10 * masses, 1.0e10 * guo_low, 1.0e10 * guo_high, color='g', hatch='///', hatchcolor='g', alpha=0.2)
        loglog(1.0e10 * masses, 1.0e10 * ph.guo_abundance_matching(masses), linestyle='-', color='g')
        labels = ["$\\rm{M_{200}\\,\\,\\,\\Omega_b / \\Omega_m}$", "Guo+ 10"]
        l1 = legend(labels, loc='upper left', fontsize=9, frameon=False)
    
    for d in range(len(runs)):
        dd = dirs[d] + runs[d]
        
        print("outputlistfile[d]=", outputlistfile[d])
        snaps = select_snapshot_number.select_snapshot_number(outputlistfile[d], zlist)
        if isinstance(snaps, int):
            snaps = [snaps]
        
        color = colors[d]
        marker = markers[d]
        for isnap in range(nsnaps):
            ax = figure.axes[isnap]
            
            # add cosmic baryon fraction curve
            ax.loglog(1.0e10 * masses, 1.0e10 * masses * cosmic_baryon_frac, ':', color='gray')
            ax.fill_between(1.0e10 * masses, 1.0e10 * masses * cosmic_baryon_frac, [1e12] * len(masses), color='gray', alpha=0.2, edgecolor='none')
            
            # add Guo abundance matching curve
            if dat == 'Guo' and zlist[d] == 0. and d == 0:
                guo_high = guo_abundance_matching(masses) * 10 ** (+0.2)
                guo_low = guo_abundance_matching(masses) * 10 ** (-0.2)
                ax.fill_between(1.0e10 * masses, 1.0e10 * guo_low, 1.0e10 * guo_high, color='g', hatch='///', hatchcolor='g', alpha=0.2)
                ax.loglog(1.0e10 * masses, 1.0e10 * guo_abundance_matching(masses), linestyle='-', color='g')
                labels = ["$\\rm{M_{200}\\,\\,\\,\\Omega_b / \\Omega_m}$", "Guo+ 10"]
                l1 = ax.legend(labels, loc='upper left', fontsize=9, frameon=False)
            
            attrs = ['pos', 'vel', 'mass', 'age']
            
            s = gadget_readsnap(snaps[isnap], snappath=dd + '/output/', hdf5=True, loadonlytype=[0, 1, 4], loadonly=attrs)
            sf = load_subfind(snaps[isnap], dir=dd + '/output/', hdf5=True, loadonly=['fpos', 'frc2'])
            s.center = sf.data['fpos'][subhalo, :]
            g = parse_particledata(s, sf, attrs, radialcut=sf.data['frc2'][subhalo])
            g.prep_data()
            sdata = g.sgdata['sdata']
            gdata = g.sgdata['gdata']
            
            # add Moster abundance matching curve
            if dat == 'Moster' and d == 0:
                ax.loglog(1.0e10 * masses, 1.0e10 * ph.moster_abundance_matching(masses, g.s.time), linestyle='-', linewidth=1., color='royalblue')
                moster_up = ph.moster_abundance_matching(masses, g.s.time) * 10 ** (+0.2)
                moster_lo = ph.moster_abundance_matching(masses, g.s.time) * 10 ** (-0.2)
                ax.fill_between(1.0e10 * masses, 1.0e10 * moster_lo, 1.0e10 * moster_up, facecolor='royalblue', alpha=0.2, edgecolor='none')
                if isnap == 0:
                    labels = ["$\\rm{M_{200}\\,\\,\\,\\Omega_b / \\Omega_m}$", "Moster+ 13"]
                    l1 = ax.legend(labels, loc='upper left', fontsize=6, frameon=False)
            
            totalmass[isnap] = s.data['mass'][(s.r() < g.sf.data['frc2'][subhalo])].sum()
            stellarmass[isnap] = sdata['mass'].sum()
            
            if isnap == nsnaps - 1 and (restest or not colorbymass):
                label = "$\\rm{%s\, %s}$" % ('Au', runs[d].split('_')[1])
            else:
                label = ''
            
            if restest:
                for i, level in enumerate(restest):
                    lst = 'level%01d' % level
                    if lst in dirs[d]:
                        marker = markers[i]  # label += " $\\rm{lvl %01d}$"%level
                
                color = colors[d % lencol]
            else:
                marker = 'o'
            
            if colorbymass:
                pmin = 0.1;
                pmax = 10.
                params = np.linspace(pmin, pmax, 20)
                cmap = plt.get_cmap('magma')
                s_m = figure.get_scalar_mappable_for_colorbar(params, cmap)
                color = s_m.to_rgba(sdata['mass'].sum())
            
            print("1.0e10 * totalmass[isnap], 1.0e10 * stellarmass[isnap]=", 1.0e10 * totalmass[isnap], 1.0e10 * stellarmass[isnap])
            ax.loglog(1.0e10 * totalmass[isnap], 1.0e10 * stellarmass[isnap], marker=marker, mfc='none', mec=color, ms=5., mew=1., linestyle='None',
                      label=label)
            
            ax.text(0.7, 0.1, "$\\rm{z=%01d}$" % zlist[isnap], transform=ax.transAxes, fontsize=10)
    
    if colorbymass:
        pticks = np.linspace(log10(pmin * 1e10), log10(pmax * 1e10), 5)
        pticks = [round(elem, 1) for elem in pticks]
        figure.set_colorbar([log10(pmin * 1e10), log10(pmax * 1e10)], '$\\rm{ log_{10}(M_{*} \, [M_{\odot}])}$', pticks, cmap=cmap)
    
    ax.legend(loc='upper left', frameon=False, prop={'size': 5}, numpoints=1, ncol=1)
    
    if restest:
        figure.fig.savefig("%s/stellarvstotalmass%03d_restest.%s" % (outpath, snaps[-1], suffix), dpi=300)
    else:
        figure.fig.savefig("%s/stellarvstotalmass%03d.%s" % (outpath, snaps[-1], suffix), dpi=300)