import bisect

from const import *
from gadget import *
from gadget_subfind import *
from matplotlib.ticker import FormatStrFormatter
from pylab import *
from util import multipanel_layout

colors = ['g', 'b', 'r', 'k', 'c', 'y', 'm', 'purple']
toinch = 0.393700787


def plot_metal_fractions(runs, dirs, outpath, snap, suffix, nrows, ncols, fgr=True):
    panels = len(runs)
    
    figure = multipanel_layout.multipanel_layout(nrows=nrows, ncols=ncols, npanels=panels)
    figure.set_figure_layout()
    figure.set_axis_limits_and_aspect(ylim=[0.0, 1.0], xlim=[0.001, 2.5], logaxis=False, logxaxis=True)
    figure.set_axis_locators(xminloc=0.1, yminloc=0.05)
    figure.set_axis_labels(xlabel="$\\rm{r/r_{vir}}$", ylabel="$\\rm{f_{Z}\\,(<\\,r/r_{vir})}$")
    figure.set_fontsize()
    
    if fgr == True:
        for d in range(len(runs)):
            dd = dirs[d] + runs[d]
            ax = figure.fig.axes[d]
            
            print
            "Doing dir %s. snapshot %d" % (dd, snap)
            
            s = gadget_readsnap(snap, snappath=dd + '/output/', hdf5=True, loadonlytype=[0, 4], loadonly=['pos', 'mass', 'gz', 'sfr'])
            sf = load_subfind(snap, dir=dd + '/output/', hdf5=True, loadonly=['fpos', 'frc2'])
            
            s.center = sf.data['fpos'][0, :]
            galrad = 0.1 * sf.data['frc2'][0]
            metalrad = 0.150  # sf.data['frc2'][0]
            halorad = sf.data['frc2'][0]
            maxrad = 2.5 * sf.data['frc2'][0]
            
            print
            'Select gas cells'
            istars, = np.where((s.r() < maxrad) & (s.type == 4))
            igas, = np.where((s.r() < maxrad) & (s.type == 0))
            igassf, = np.where(s.data['sfr'][igas] > 0.0)
            igasnsf, = np.where(s.data['sfr'][igas] <= 0.0)
            
            ngas = len(igas)
            nstars = len(istars)
            ngassf = len(igassf)
            ngasnsf = len(igasnsf)
            ntot = ngas + nstars
            
            radii = np.zeros(ntot)
            metalmasses = np.zeros(ntot)
            metalmassesstar = np.zeros(ntot)
            metalmassessfgas = np.zeros(ntot)
            metalmassesnsfgas = np.zeros(ntot)
            
            first_star = s.nparticlesall[:4].sum()
            
            radii[0:ngas] = s.r()[igas]
            radii[ngas:] = s.r()[istars]
            metalmasses[0:ngas] = s.data['mass'][igas].astype('float64') * s.data['gz'][igas].astype('float64')
            metalmasses[ngas:] = s.data['mass'][istars].astype('float64') * s.data['gz'][istars - first_star].astype('float64')
            metalmassesstar[ngas:] = metalmasses[ngas:]
            
            metalmassessfgas[igassf] = metalmasses[igassf]
            metalmassesnsfgas[igasnsf] = metalmasses[igasnsf]
            
            print
            'Compute profiles'
            index = np.argsort(radii)
            radii = radii[index]
            metalmassesstar = np.cumsum(metalmassesstar[index])
            metalmassessfgas = np.cumsum(metalmassessfgas[index])
            metalmassesnsfgas = np.cumsum(metalmassesnsfgas[index])
            metalmasses = np.cumsum(metalmasses[index])
            
            i = bisect.bisect_right(radii, galrad)
            j = bisect.bisect_right(radii, metalrad)
            
            xlim = ax.get_xlim()
            
            ax.semilogx(radii / halorad, metalmassesstar / metalmasses[ntot - 1], 'g-')
            ax.semilogx(radii / halorad, (metalmassesstar + metalmassessfgas) / metalmasses[ntot - 1], 'b-')
            ax.semilogx(radii / halorad, (metalmassesstar + metalmassessfgas + metalmassesnsfgas) / metalmasses[ntot - 1], 'r-')
            ax.hlines(metalmasses[i] / metalmasses[ntot - 1], xlim[0], xlim[1], colors='gray', linestyles='dashed')
            ax.hlines(metalmasses[j] / metalmasses[ntot - 1], xlim[0], xlim[1], colors='gray', linestyles='dashed')
            
            ax.text(galrad / halorad - 0.02, metalmasses[i] / metalmasses[ntot - 1] + 0.02,
                    "$\\rm{%.2f\\,\\,\,M_{Z}}$" % (metalmasses[i] / metalmasses[ntot - 1]), size=5, transform=ax.transData, ha='right')
            ax.text(metalrad / halorad - 0.02, metalmasses[j] / metalmasses[ntot - 1] + 0.02,
                    "$\\rm{%.2f\\,\\,\,M_{Z}}$" % (metalmasses[j] / metalmasses[ntot - 1]), size=5, transform=ax.transData, ha='right')
            ax.text(0.06, 0.97, "$\\rm{M_{Z}}\\,=\\,%.2fx10^{10}\\,M_{\\odot}$" % metalmasses[ntot - 1], color='black', size=5,
                    horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
            
            print
            xlim, metalmasses[i]
            
            format = FormatStrFormatter("%g")
            ax.xaxis.set_major_formatter(format)
            
            figure.set_panel_title(panel=d, title="$\\rm{%s-{%s}}$" % (runs[d].split('_')[0], runs[d].split('_')[1]), position='bottom right')
            
            figure.reset_axis_limits()
        
        figure.fig.savefig("%s/metalfractions_multi%03d.%s" % (outpath, snap, suffix), dpi=300)