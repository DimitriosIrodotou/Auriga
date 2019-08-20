from const import *
from loadmodules import *
from pylab import *
from util import *

colors = ['r', 'm', 'c', 'b']


def plot_birthradiusdist(runs, dirs, outpath, outputlistfile, redshift, zcut=0.5, rcut=[[3., 7., 11., 15.], [5., 9., 13., 17.]], accretedfiledir=None,
                         birthdatafile=None, alphaelement=None, disc_stars=False):
    rbinlo = rcut[0]
    rbinhi = rcut[1]
    
    nrows = 1
    ncols = len(rbinlo)
    panels = nrows * ncols
    
    xran = [0., 20.]
    yran = [0., 0.29]
    
    for d in range(len(runs)):
        
        figure = multipanel_layout.multipanel_layout(nrows=nrows, ncols=ncols, npanels=panels, bottom=0.2, top=0.98)
        figure.set_figure_layout()
        figure.set_axis_limits_and_aspect(xlim=xran, ylim=yran, logaxis=False)
        figure.set_axis_locators(xminloc=1., ymajloc=0.1, yminloc=0.05)
        figure.set_axis_labels(xlabel="$\\rm{R\,[kpc]}$", ylabel=r"$\rm{dN}$")
        figure.set_fontsize(8)
        
        cblimp = [1., 11.]
        parameters = np.linspace(cblimp[0], cblimp[1], 20)
        cmapl = plt.get_cmap('viridis')  # mpl.cm.jet
        s_m = figure.get_scalar_mappable_for_colorbar(parameters, cmapl)
        
        dd = dirs[d] + runs[d]
        wpath = outpath + runs[d]
        
        snap = np.int_(select_snapshot_number.select_snapshot_number(outputlistfile[d], redshift))
        
        print("doing %s, snap=%03d" % (wpath, snap))
        
        if birthdatafile:  # read birth data from post-processed file
            stardatafile = outpath + runs[d] + birthdatafile
            rotmatfile = None
            galcenfile = None
            attrs = ['pos', 'vel', 'id', 'mass', 'age', 'gmet', 'pot', 'gz']
        else:
            stardatafile = None  # snapshot already contains birth data
            rotmatfile = dd + '/output/rotmatlist_%s.txt' % (runs[d])
            galcenfile = dd + '/output/galcen_%s.txt' % (runs[d])
            attrs = ['pos', 'vel', 'id', 'mass', 'age', 'gmet', 'pot', 'gz', 'bpos', 'bvel']
        
        s = gadget_readsnap(snap, snappath=dd + '/output/', hdf5=True, loadonlytype=[4], loadonly=attrs)
        sf = load_subfind(snap, dir=dd + '/output/', hdf5=True, loadonly=['fpos', 'frc2', 'svel', 'flty', 'fnsh', 'slty'])
        s.calc_sf_indizes(sf)
        s.select_halo(sf, 3., use_principal_axis=True, use_cold_gas_spin=False, do_rotation=True)
        
        if accretedfiledir:
            accretedfile = accretedfiledir + runs[d] + '/%sstarID_accreted_all.dat' % runs[d]
        else:
            accretedfile = None
        if birthdatafile:
            attrs.append('bpos')
            attrs.append('bvel')
        g = parse_particledata(s, sf, attrs, rotmatfile=rotmatfile, galcenfile=galcenfile, accretedfile=accretedfile, stardatafile=stardatafile,
                               radialcut=sf.data['frc2'][0])
        g.prep_data()
        
        sdata = g.sgdata['sdata']
        
        bradius = np.sqrt((sdata['bpos'][:, 1:] ** 2).sum(axis=1))
        # bradius = sdata['bradg']
        radius = np.sqrt((sdata['pos'][:, 1:] ** 2).sum(axis=1))
        bheight = sdata['bpos'][:, 0]
        height = sdata['pos'][:, 0]
        eps2 = sdata['eps2']
        star_age = sdata['age']
        vrad = (sdata['pos'][:, 1] * sdata['vel'][:, 1] + sdata['pos'][:, 2] * sdata['vel'][:, 2]) / radius
        
        ind = np.in1d(sdata['id'], sdata['bid'], assume_unique=True)
        
        binx = 80
        biny = 80
        # conv = (binx / (feran[1]-feran[0])) * (biny / (alran[1] - alran[0]))
        vmin = 0.005
        vmax = 0.5
        if accretedfile:
            vmin /= 10.
            vmax /= 10.
        
        agelo = np.array([0., 2., 4., 6., 8., 10., 0.])
        agehi = np.array([2., 4., 6., 8., 10., 12., 12.])
        
        print("agelo,agehi=", agelo, agehi)
        pnum = 0
        for j in range(len(rbinlo)):
            ax = figure.axes[pnum]
            pnum += 1
            
            for i in range(len(agelo)):
                
                age = 0.5 * (agelo[i] + agehi[i])
                
                if disc_stars:
                    jj, = np.where((radius * 1e3 > rbinlo[j]) & (radius * 1e3 < rbinhi[j]) & (abs(height * 1e3) < zcut) & (eps2 > 0.7) & (
                                star_age < agehi[i]) & (star_age > agelo[i]))
                else:
                    jj, = np.where((radius * 1e3 > rbinlo[j]) & (radius * 1e3 < rbinhi[j]) & (abs(height * 1e3) < zcut) & (star_age < agehi[i]) & (
                                star_age > agelo[i]))
                
                n, edges = np.histogram(bradius[jj] * 1e3, bins=20, range=[0., 20.])
                
                n = np.float_(n)
                n /= float(n.sum())
                
                xbin = 0.5 * (edges[:-1] + edges[1:])
                # radial Vdisp
                sigr = np.sqrt(np.mean(vrad[jj] ** 2))
                
                if i == 0:
                    ax.axvspan(rbinlo[j], rbinhi[j], alpha=0.2, facecolor='gray', edgecolor='none')
                if i == len(agelo) - 1:
                    ax.plot(xbin, n, '--', color='k')
                else:
                    ax.plot(xbin, n, '-', color=s_m.to_rgba(age))
            
            if pnum == 0:
                title = "$\\rm{Au \,%s ,\, t=%2.1f}$" % (runs[d].split('_')[1], time)
                figure.set_panel_title(panel=pnum, title=title, position='bottom left')
            
            ax.text(0.5, 0.85, "$\\rm{%.0f < R < %.0f}$" % (rbinlo[j], rbinhi[j]), color='gray', transform=ax.transAxes, fontsize=6)
        
        figure.reset_axis_limits()
        
        figure.set_colorbar([1., 11.], r'$\rm{ age \, [Gyr] }$', [1., 3., 5., 7., 9., 11.], cmap=cmapl, fontsize=7, labelsize=7,
                            orientation='vertical')
        
        name = ''
        
        if disc_stars:
            name += '_disc'
        if accretedfiledir:
            name += '_accreted'
        
        figname1 = '%s/bradiusdist_radslice%s%03d.pdf' % (wpath, name, snap)
        figure.fig.savefig(figname1)