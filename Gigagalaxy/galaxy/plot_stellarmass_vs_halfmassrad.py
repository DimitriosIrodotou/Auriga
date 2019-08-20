from const import *
from loadmodules import *
from pylab import *
from util import *

marker = ['o', '^', 'd', 's']
colors = ['b', 'gray', 'r', 'k', 'g', 'c', 'y', 'm', 'purple']
markers = ['o', '^', 'd', 's']
band = {'Umag': 0, 'Bmag': 1, 'Vmag': 2, 'Kmag': 3, 'gmag': 4, 'rmag': 5, 'imag': 6, 'zmag': 7}
Msunabs = [5.61, 5.48, 4.83, 3.30, 5.12, 4.68, 4.57, 4.60]


def plot_stellarmass_vs_halfmassrad(runs, dirs, outpath, nrows, ncols, outputlistfile, zlist, bandi, suffix, restest=False, colorbymass=False):
    if bandi not in band:
        raise ValueError('%s not in band. Must be one of %s' % (bandi, band))
    
    panels = nrows * ncols
    figure = multipanel_layout.multipanel_layout(nrows=nrows, ncols=ncols, npanels=panels)
    figure.set_fontsize(8)
    figure.set_figure_layout()
    figure.set_fontsize(8)
    figure.set_axis_labels(xlabel="$\\rm{M_{stars}\\,[M_\\odot]}$", ylabel="$\\rm{R_{half}\\,[kpc]}$")
    figure.set_axis_limits_and_aspect(xlim=[1.e8, 5.e11], ylim=[0.9, 20.], logaxis=True)
    
    p = plot_helper.plot_helper()
    
    nsnaps = len(zlist)
    
    for isnap, snap in enumerate(snaps):
        ax = figure.axes[isnap]
        masses = np.linspace(1e9, 1e12, 10)
        
        # if (zlist[isnap] < 0.25):
        #       reff = vanderwellate( masses, 10**(0.86), 0.25 )
        #       sigma = 0.16
        if (zlist[isnap] > 0.5) & (zlist[isnap] < 1.1):
            reff = p.vanderwellate(masses, 10 ** (0.78), 0.22)
            sigma = 0.16
            ax.loglog(masses, reff, dashes=(2, 2), color='hotpink', label=r'$\rm{van \, der \, Wel+ 14 \, (z \sim 1.25)}$')
        if (zlist[isnap] > 1.5) & (zlist[isnap] < 2.5):
            reff = p.vanderwellate(masses, 10 ** (0.55), 0.22)
            sigma = 0.19
            ax.loglog(masses, reff, dashes=(2, 2), color='hotpink', label=r'$\rm{van \, der \, Wel+ 14 \, (z \sim 1.75)}$')
        if (zlist[isnap] > 2.5) & (zlist[isnap] < 3.5):
            reff = p.vanderwellate(masses, 10 ** (0.51), 0.18)
            sigma = 0.19
            ax.loglog(masses, reff, dashes=(2, 2), color='hotpink', label=r'$\rm{van \, der \, Wel+ 14 \, (z \sim 2.75)}$')
        
        if zlist[isnap] < 0.1:
            rshen = p.shen(masses)
            sigma = 0.2
            
            rhalf_Sab_Scd, rhalf_S0_Sa = p.lange_Sab_Scd(masses)
            
            ax.fill_between(masses, 10 ** (log10(rhalf_Sab_Scd) + sigma), 10 ** (log10(rhalf_Sab_Scd) - sigma), edgecolor='royalblue', color='none',
                            alpha=0.5)
            ax.loglog(masses, rhalf_Sab_Scd, linestyle='-', color='royalblue', label=r'$\rm{Lange+ 2016}$')
            
            # ax.fill_between( masses, 10**(log10(rshen)+sigma), 10**(log10(rshen)-sigma), edgecolor='red', color='none',   # alpha=0.5 )  #
            # ax.loglog( masses, rshen, linestyle='-', color='red', label=r'$\rm{Shen+ 2003}$' )
        
        ax.legend(loc='upper left', fontsize=10, frameon=False, numpoints=1)
        
        for d in range(len(runs)):
            dd = dirs[d] + runs[d]
            
            snaps = np.int_(select_snapshot_number.select_snapshot_number(outputlistfile[d], zlist))
            if isinstance(snaps, int):
                snaps = [snaps]
            print("Doing dir %s." % (dd))
            
            attrs = ['pos', 'mass', 'age', 'id', 'gsph']
            s = gadget_readsnap(snaps[isnap], snappath=dd + '/output/', hdf5=True, loadonlytype=[4], loadonly=attrs, verbose=False)
            sf = load_subfind(snaps[isnap], dir=dd + '/output/', hdf5=True, loadonly=['fpos', 'frc2', 'fnsh', 'slty', 'spos'])
            s.center = sf.data['fpos'][0, :]
            s.centerat(s.center)
            g = parse_particledata(s, sf, attrs, radialcut=0.1 * sf.data['frc2'][0])
            g.prep_data()
            sdata = g.sgdata['sdata']
            
            luminosity = 10 ** (-0.4 * (sdata[bandi] - Msunabs[band[bandi]]))
            
            mass = sdata['mass']
            radius = np.sqrt((sdata['pos'][:, :] ** 2).sum(axis=1))
            rsort = radius.argsort()
            radius = radius[rsort]
            mass = mass[rsort]
            masscum = np.cumsum(mass)
            masscum /= masscum[-1]
            luminosity = luminosity[rsort]
            lumcum = np.cumsum(luminosity)
            lumcum /= lumcum[-1]
            
            indy, = np.where((lumcum > 0.5))
            rhalf = radius[indy[0]]
            stellarmass = mass.sum()
            
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
                color = 'k'
                marker = 'o'
            
            if colorbymass:
                pmin = 0.1;
                pmax = 10.
                params = np.linspace(pmin, pmax, 20)
                cmap = plt.get_cmap('magma')
                s_m = figure.get_scalar_mappable_for_colorbar(params, cmap)
                color = s_m.to_rgba(mass.sum())
            
            ax.loglog(1.0e10 * stellarmass, rhalf * 1e3, marker=marker, mfc='none', mec=color, ms=5., mew=1., linestyle='--', label='')
        
        ax.text(0.8, 0.1, "$\\rm{z=%01d}$" % zlist[isnap], transform=ax.transAxes, fontsize=12)
    
    if colorbymass:
        pticks = np.linspace(log10(pmin * 1e10), log10(pmax * 1e10), 5)
        pticks = [round(elem, 1) for elem in pticks]
        figure.set_colorbar([log10(pmin * 1e10), log10(pmax * 1e10)], '$\\rm{ log_{10}(M_{*} \, [M_{\odot}])}$', pticks, cmap=cmap)
    
    if restest:
        figure.fig.savefig("%s/stellarmass_rhalf%03d_restest.%s" % (outpath, snap, suffix), dpi=300)
    else:
        figure.fig.savefig("%s/stellarmass_rhalf%03d.%s" % (outpath, snap, suffix), dpi=300)