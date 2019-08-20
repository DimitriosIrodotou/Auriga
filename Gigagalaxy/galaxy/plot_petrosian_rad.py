from const import *
from loadmodules import *
from pylab import *
from util import *

colors = ['g', 'b', 'r', 'k', 'c', 'y', 'm', 'purple']
band = {'Umag': 0, 'Bmag': 1, 'Vmag': 2, 'Kmag': 3, 'gmag': 4, 'rmag': 5, 'imag': 6, 'zmag': 7}
Msunabs = [5.61, 5.48, 4.83, 3.30, 5.12, 4.68, 4.57, 4.60]


def plot_petrosian_rad(runs, dirs, outpath, nrows, ncols, outputlistfile, zlist, iband, suffix, restest=False, colorbymass=False):
    panels = nrows * ncols
    
    figure = multipanel_layout.multipanel_layout(nrows=nrows, ncols=ncols, npanels=panels)
    figure.set_fontsize(8)
    figure.set_figure_layout()
    figure.set_fontsize(8)
    figure.set_axis_labels(xlabel="$\\rm{M_{r}}$", ylabel="$\\rm{R_{petrosian}\\,[kpc]}$")
    figure.set_axis_limits_and_aspect(xlim=[-23., -21.], ylim=[1., 20.], logaxis=True)
    figure.set_axis_locators(xminloc=1., xmajloc=1., yminloc=1., ymajloc=10.)
    
    p = plot_helper.plot_helper()
    
    nsnaps = len(zlist)
    
    # f = open('petrosian_rad.txt', 'w')
    # header = "%12s%10s%10s%10s%10s\n" % ('#run', 'r_petr', 'r_50', 'M_r', 'r_max' )
    # f.write(header)
    
    for isnap in range(nsnaps):
        ax = figure.axes[isnap]
        if zlist[isnap] < 0.1:
            mag = arange(-24., -20., 0.5)
            ax.fill_between(mag, p.shen_lower_env(mag), p.shen_upper_env(mag), color='lightgray', edgecolor='None')
            ax.semilogy(mag, p.shen_best_fit(mag), 'k-', label="$\\rm{Shen+ 03}$")
        for d in range(len(runs)):
            dd = dirs[d] + runs[d]
            
            snaps = np.int_(select_snapshot_number.select_snapshot_number(outputlistfile[d], zlist))
            if isinstance(snaps, int):
                snaps = [snaps]
            
            attrs = ['pos', 'vel', 'mass', 'age', 'gsph']
            
            s = gadget_readsnap(snaps[isnap], snappath=dd + '/output/', hdf5=True, loadonlytype=[4], loadonly=attrs)
            sf = load_subfind(snaps[isnap], dir=dd + '/output/', hdf5=True, loadonly=['fpos', 'frc2', 'flty', 'fnsh', 'slty', 'svel'])
            s.calc_sf_indizes(sf)
            s.select_halo(sf, 3., use_principal_axis=True, use_cold_gas_spin=False, do_rotation=True)
            
            g = parse_particledata(s, sf, attrs, radialcut=0.2 * sf.data['frc2'][0])
            g.prep_data()
            sdata = g.sgdata['sdata']
            
            mass = sdata['mass'].astype('float64')
            pos = sdata['pos'].astype('float64')
            
            # after the rotation the galaxy's spin is aligned to the z axis
            r = np.sqrt(pos[:, 1] * pos[:, 1] + pos[:, 2] * pos[:, 2])
            
            luminosity = 10 ** (-0.4 * (sdata[iband] - Msunabs[band[iband]]))
            
            high = len(r) + 1
            low = 0
            mid = (high + low) / 2
            
            while (high - low) > 1:
                r_tmp = r[mid]
                k, = np.where(r < r_tmp)
                j, = np.where((r < 1.25 * r_tmp) & (r > 0.8 * r_tmp))
                avg_lum = luminosity[k].sum() / (np.pi * r_tmp * r_tmp)
                avg_annulus_lum = luminosity[j].sum() / (np.pi * r_tmp * r_tmp * (1.25 * 1.25 - 0.8 * 0.8))
                ratio = avg_annulus_lum / avg_lum
                if ratio < 0.2:
                    high = mid
                    mid = (high + low) / 2
                    if (high - low) == 1:
                        petrosian_rad = r[mid]
                        break
                elif ratio > 0.2:
                    low = mid
                    mid = (high + low) / 2
                    if (high - low) == 1:
                        petrosian_rad = r[mid]
                        break
                else:
                    petrosian_rad = r[i]
                    break
            
            k, = np.where(r < 2.0 * petrosian_rad)
            luminosity = luminosity[k]
            r = r[k]
            
            # order stars by radius
            isr = np.argsort(r)
            luminosity = luminosity[isr]
            r = r[isr]
            
            totlum = luminosity.sum()
            
            idx = 0
            mm = 0.0
            while mm <= 0.5 * totlum:
                mm += luminosity[idx]
                idx += 1
            
            r50 = r[idx]
            totlum = -2.5 * log10(totlum) + Msunabs[band[iband]]
            
            if isnap == nsnaps - 1 and (restest or not colorbymass):
                label = "$\\rm{%s\, %s}$" % ('Au', runs[d].split('_')[1])
            else:
                label = ''
            
            if restest:
                for i, level in enumerate(restest):
                    lst = 'level%01d' % level
                    if lst in dirs[d]:
                        label += '_%01d' % level
                
                color = colors[d % len(colors)]
            else:
                color = 'k'
            
            if colorbymass:
                pmin = 0.1;
                pmax = 10.
                params = np.linspace(pmin, pmax, 20)
                cmap = plt.get_cmap('magma')
                s_m = figure.get_scalar_mappable_for_colorbar(params, cmap)
                color = s_m.to_rgba(mass.sum())
            
            ax.semilogy(totlum, 1.0e3 * r50, 'o', ms=8., mec=color, mfc='none', mew=1., label=label)
            
            if d == 0:
                ax.text(0.7, 0.1, "$\\rm{z=%2.1f}$" % (zlist[isnap]), size=8, transform=ax.transAxes)
            
            ax.legend(loc='lower left', frameon=False, prop={'size': 7}, numpoints=1)
    
    if colorbymass:
        pticks = np.linspace(log10(pmin * 1e10), log10(pmax * 1e10), 3)
        pticks = [round(elem, 1) for elem in pticks]
        figure.set_colorbar([log10(pmin * 1e10), log10(pmax * 1e10)], '$\\rm{ log_{10}(M_{*} \, [M_{\odot}])}$', pticks, cmap=cmap)
    
    if restest:
        figure.fig.savefig("%s/petrosianradius_band%s_%03d_restest.%s" % (outpath, iband, snaps[isnap], suffix), dpi=300)
    else:
        figure.fig.savefig("%s/petrosianradius_band%s_%03d.%s" % (outpath, iband, snaps[isnap], suffix), dpi=300)