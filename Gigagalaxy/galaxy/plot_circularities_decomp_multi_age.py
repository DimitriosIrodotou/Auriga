from const import *
from loadmodules import *
from pylab import *
from util import *
from util import multipanel_layout

colors = ['g', 'b', 'r', 'k', 'c', 'y', 'm', 'purple']


def plot_circularities_decomp_multi_age(runs, dirs, outpath, outputlistfile, zlist, suffix, nrows, ncols, fs, subhalo=0, restest=False,
                                        normalize_bins=True, lzcirc=False, accretedfiledir=None):
    if restest:
        panels = len(runs) / len(restest)
    else:
        panels = nrows * ncols
    
    figure = multipanel_layout.multipanel_layout(nrows=nrows, ncols=ncols, npanels=panels)
    figure.set_fontsize(3)
    figure.set_figure_layout()
    figure.set_axis_limits_and_aspect(xlim=[-1.8, 1.8], ylim=[0.0, 3.4])
    if normalize_bins:
        figure.set_axis_locators(xminloc=0.25, xmajloc=1.0, yminloc=0.1, ymajloc=0.5)
    else:
        figure.set_axis_locators(xminloc=0.25, xmajloc=1.0, yminloc=0.01, ymajloc=0.05)
    figure.set_fontsize(10)
    figure.set_axis_labels(xlabel="$\\rm{\epsilon}$", ylabel="$\\rm{f(\epsilon)}$")
    
    fileout = outpath + '/stellar_age.txt'
    f = open(fileout, 'w')
    header = "%12s%12s%12s\n" % ("Run", "Age disk", "Age bulge")
    f.write(header)
    
    for d in range(len(runs)):
        dd = dirs[d] + runs[d]
        
        if not restest:
            ax = figure.axes[d]
        else:
            ax = figure.axes[int(d / len(restest))]
        
        snap = np.int_(select_snapshot_number.select_snapshot_number(outputlistfile[d], zlist))
        
        print("Doing dir %s, snap %d" % (dd, snap))
        if accretedfiledir:
            accretedfile = accretedfiledir + runs[d] + '/%sstarID_accreted_all.dat' % runs[d]
        else:
            accretedfile = None
        
        attrs = ['pos', 'vel', 'mass', 'pot', 'age', 'id']
        s = gadget_readsnap(snap, snappath=dd + '/output/', hdf5=True, loadonlytype=[0, 1, 4], loadonly=attrs)
        sf = load_subfind(snap, dir=dd + '/output/', hdf5=True, loadonly=['fpos', 'frc2', 'svel', 'flty', 'fnsh', 'slty'])
        s.calc_sf_indizes(sf)
        s.select_halo(sf, 3., use_principal_axis=True, use_cold_gas_spin=False, do_rotation=True)
        
        g = parse_particledata(s, sf, attrs, radialcut=0.1 * sf.data['frc2'][subhalo], accretedfile=accretedfile, docircularities=True)
        g.prep_data(lzcirc=lzcirc)
        
        if accretedfile:
            asind = g.select_accreted_stars(accreted=True)
        
        sdata = g.sgdata['sdata']
        eps2 = sdata['eps2']
        smass = sdata['mass']
        star_age = sdata['age']
        
        # compute total new (it is used for bulge disk decomposition and plotted afterwards)
        ydatatot, edges = histogram(eps2, bins=100, weights=smass / smass.sum(), range=[-1.7, 1.7])
        if accretedfile:
            ydatatot_acc, edges = histogram(eps2[asind], bins=100, weights=smass[asind] / smass.sum(), range=[-1.7, 1.7])
        
        xdatatot = 0.5 * (edges[1:] + edges[:-1])
        
        # such that \int f(\epsilon) d\epsilon = 1
        if normalize_bins:
            binwidth = edges[1:] - edges[:-1]
            ydatatot[:] /= binwidth[:]
            if accretedfile:
                ydatatot_acc[:] /= binwidth[:]
        
        if not restest:
            xdata = np.zeros(len(xdatatot))
            ydata = np.zeros(len(ydatatot))
            ydatad = np.zeros(len(ydatatot))
            
            kk, = np.where(xdatatot <= 0.0)
            pivot = np.max(kk)
            
            # bulge
            ydata[:] = ydatatot[:]
            xdata[:] = xdatatot[:]
            # mirror part with negative epsilon (if that is too big keep the actual value)
            if len(ydata) % 2 == 0:
                for i in range(0, np.int_(len(ydata)) // 2):
                    if ydata[pivot + i + 1] > ydata[pivot - i]:
                        ydata[pivot + i + 1] = ydata[pivot - i]
            else:
                for i in range(1, np.int_(len(ydata)) // 2):
                    if ydata[pivot + i] > ydata[pivot - i]:
                        ydata[pivot + i] = ydata[pivot - i]
            
            ax.fill_between(xdata, ydata, hatch='\\\\\\\\\\\\\\\\\\', color='none', edgecolor='tomato', lw=1., label=r'$\rm{spheroid}$')
            
            # disk
            ydatad[:] = ydatatot[:]
            xdata[:] = xdatatot[:]
            # subtract the bulge part
            ydatad[:] -= ydata[:]
            ax.fill_between(xdata, ydatad, hatch='////////////', color='none', edgecolor='royalblue', lw=1., label=r'$\rm{disc}$')
            
            color = 'k'
            label = r'$\rm{total}$'
        else:
            for i, level in enumerate(restest):
                lst = 'level%01d' % level
                if lst in dirs[d]:
                    label = " $\\rm{Au %s lvl %01d}$" % (runs[d].split('_')[1], level)
                else:
                    label = "$\\rm{Au %s}$" % (runs[d].split('_')[1])
            color = colors[d % len(colors)]
        
        # plot total new
        ax.plot(xdatatot, ydatatot, color, label=label)
        if accretedfile:
            ax.plot(xdatatot, ydatatot_acc, color, dashes=(2, 2), label=r'$\rm{acc\,(%4.2f)}$' % (float(size(np.array(asind))) / float(len(smass))))
        
        # total old
        if 'eps' in sdata:
            ydata, edges = histogram(sdata['eps'], bins=100, weights=smass / smass.sum(), range=[-1.7, 1.7])
            xdata = 0.5 * (edges[1:] + edges[:-1])
            
            # such that \int f(\epsilon) d\epsilon = 1
            if normalize_bins:
                binwidth = edges[1:] - edges[:-1]
                ydata[:] /= binwidth[:]
            
            ax.plot(xdata, ydata, 'gray', lw=0.7, label="$\\rm{f(\\epsilon_{v})}$")
        
        jj, = np.where((eps2 > 0.7) & (eps2 < 1.7))
        kk, = np.where((eps2 > -1.7) & (eps2 <= 0.7))
        ll, = np.where((eps2 > -1.7) & (eps2 < 1.7))
        alt_frac = smass[jj].sum() / smass[ll].sum()
        
        if not restest:
            age_disk = (star_age[jj] * smass[jj]).sum() / smass[jj].sum()
            age_bulge = (star_age[kk] * smass[kk]).sum() / smass[kk].sum()
            header = "%12s%12.2f%12.2f\n" % (runs[d], age_disk, age_bulge)
            f.write(header)
        
        if not restest:
            figure.set_panel_title(panel=d, title="$\\rm{%s\,{%s}}$" % ("Au", runs[d].split('_')[1]), position='bottom left', color=color,
                                   fontsize=10)
        
        ax.legend(loc='upper left', frameon=False, prop={'size': 7})
        if not restest:
            ax.text(0.06, 0.35, r"$\rm{D/T = %4.2f [%4.2f]}$" % (alt_frac, ydatad.sum() / ydatatot.sum()), size=7, transform=ax.transAxes)
    
    f.close()
    if restest:
        figure.fig.savefig('%s/epsilon_multi_age%03d%s_restest.%s' % (outpath, snap, fs, suffix))
    else:
        figure.fig.savefig('%s/epsilon_multi_age%03d%s_new.%s' % (outpath, snap, fs, suffix))