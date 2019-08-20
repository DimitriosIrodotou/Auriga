from pylab import *
from loadmodules import *
from const import *
from util import *
from scipy import interpolate

Gcosmo = 43.0071 * 1e-7


def plot_ToomresQ(runs, dirs, outpath, outputlistfile, zlist, suffix, nrows, ncols, gas=True):
    panels = len(runs)
    figure = multipanel_layout.multipanel_layout(nrows=nrows, ncols=ncols, npanels=panels)
    figure.set_figure_layout()
    figure.set_axis_limits_and_aspect(xlim=[0.0, 14.], ylim=[0.0, 8.0], logaxis=False)
    figure.set_axis_locators(xminloc=1.0, xmajloc=5.0, yminloc=0.5, ymajloc=1.0)
    figure.set_axis_labels(xlabel="$\\rm{R [kpc]}$", ylabel="$\\rm{Q}$")
    figure.set_fontsize(8.)
    
    colors = ['r', 'b', 'c', 'g', 'k']
    lss = ['-', '--', '.-', ':']
    
    for d in range(len(runs)):
        dd = dirs[d] + runs[d]
        ax = figure.fig.axes[d]
        snaps = np.int_(select_snapshot_number.select_snapshot_number(outputlistfile[d], zlist))
        if isinstance(snaps, int):
            snaps = [snaps]
        
        wpath = outpath + runs[d] + '/'
        if not os.path.exists(wpath):
            os.makedirs(wpath)
        
        for isnap, snap in enumerate(snaps):
            print("Doing dir %s snap %d." % (dd, snap))
            attrs = ['pos', 'vel', 'mass', 'age', 'pot', 'sfr', 'u']
            s = gadget_readsnap(snap, snappath=dd + '/output/', hdf5=True, loadonlytype=[0, 1, 4, 5], loadonly=attrs)
            sf = load_subfind(snap, dir=dd + '/output/', hdf5=True, loadonly=['fpos', 'frc2', 'svel', 'flty', 'fnsh', 'slty'])
            s.calc_sf_indizes(sf)
            s.select_halo(sf, 3., use_principal_axis=True, use_cold_gas_spin=False, do_rotation=True)
            
            g = parse_particledata(s, sf, attrs, radialcut=0.1 * sf.data['frc2'][0])
            g.prep_data()
            
            sdata = g.sgdata['sdata']
            gdata = g.sgdata['gdata']
            
            gamma = 5. / 3.
            
            iall, = np.where((s.r() < 0.1 * sf.data['frc2'][0]))
            nall = size(iall)
            
            mass = s.data['mass'][iall]
            radius = np.sqrt((s.pos[iall, 1:] ** 2).sum(axis=1))
            
            rr = pylab.sqrt((s.pos[iall, :] ** 2).sum(axis=1))
            msort = rr.argsort()
            mass_all = mass
            msum = pylab.zeros(nall)
            msum[msort[:]] = np.cumsum(mass_all[msort[:]])
            galrad = 0.1 * sf.data['frc2'][0]
            
            ## Get circular vel curve ##
            nshells = 60
            rxy = np.linspace(0.001, galrad, nshells)
            crad = pylab.zeros(nshells)
            vc = pylab.zeros(nshells)
            print("rxy,rr,msum=", rxy, rr, msum)
            for i in range(nshells):
                ii, = np.where((rr < rxy[i]))
                mtot = np.sum(mass_all[ii])
                vc[i] = np.sqrt(Gcosmo * mtot * 1e10 / (rxy[i] * 1e3))
            
            zcut = 0.005
            
            ii, = np.where((abs(sdata['pos'][:, 0]) < zcut) & (sdata['eps2'] > 0.7))
            
            smass = sdata['mass'][ii]
            star_age = sdata['age'][ii]
            star_radius = np.sqrt((sdata['pos'][ii, 1:] ** 2).sum(axis=1))
            star_vrad = (sdata['pos'][ii, 1] * sdata['vel'][ii, 1] + sdata['pos'][ii, 2] * sdata['vel'][ii, 2]) / star_radius
            star_vphi = (sdata['pos'][ii, 1] * sdata['vel'][ii, 0] - sdata['pos'][ii, 2] * sdata['vel'][ii, 1]) / star_radius
            ii, = np.where((abs(gdata['pos'][:, 0]) < zcut))
            
            if gas:
                gmass = gdata['mass'][ii]
                gas_radius = np.sqrt((gdata['pos'][ii, 1:] ** 2).sum(axis=1))
                gas_vrad = (gdata['pos'][ii, 1] * gdata['vel'][ii, 1] + gdata['pos'][ii, 2] * gdata['vel'][ii, 2]) / gas_radius
                gas_vphi = (gdata['pos'][ii, 1] * gdata['vel'][ii, 0] - gdata['pos'][ii, 2] * gdata['vel'][ii, 1]) / gas_radius
                gas_cs = np.sqrt(gamma * (gamma - 1.) * gdata['u'][ii])
            
            ns, edges = np.histogram(star_radius, nshells, range=(0.001, galrad))
            mshell, edges = np.histogram(star_radius, nshells, weights=smass, range=(0.001, galrad))
            x = np.zeros(nshells)
            x[:] = 0.5 * (edges[1:] + edges[:-1])
            sden = np.zeros(nshells)
            sden[:] = 1e10 * mshell[:] / (np.pi * ((edges[1:] * 1e3) ** 2 - (edges[:-1] * 1e3) ** 2))
            
            ns, edges = np.histogram(gas_radius, nshells, range=(0.001, galrad))
            mshell, edges = np.histogram(gas_radius, nshells, weights=gmass, range=(0.001, galrad))
            x = np.zeros(nshells)
            x[:] = 0.5 * (edges[1:] + edges[:-1])
            gsden = np.zeros(nshells)
            gsden[:] = 1e10 * mshell[:] / (np.pi * ((edges[1:] * 1e3) ** 2 - (edges[:-1] * 1e3) ** 2))
            
            omega = 1e-3 * vc / rxy
            omega2 = omega ** 2
            gomega = omega
            gomega2 = gomega ** 2
            
            tk = interpolate.splrep(array(rxy) * 1e3, array(omega2), k=3, s=0)
            dom = interpolate.splev(array(rxy) * 1e3, tk, der=1)
            
            kappa = np.sqrt(rxy * 1e3 * dom + 4. * omega2)
            
            ni, ibins = np.histogram(star_radius, bins=nshells, range=(0.001, galrad))
            sni, ibins = np.histogram(star_radius, bins=nshells, weights=star_vrad, range=(0.001, galrad))
            sni2, ibins = np.histogram(star_radius, bins=nshells, weights=star_vrad * star_vrad, range=(0.001, galrad))
            smeani = sni / ni
            yi = np.sqrt(sni2 / ni - smeani * smeani)
            xi = np.zeros(nshells)
            xi[:] = 0.5 * (ibins[1:] + ibins[:-1])
            
            Q = (kappa[:] * yi[:]) / (3.36 * Gcosmo * sden[:])
            Qtot = Q
            
            if gas:
                gkappa = np.sqrt(rxy * 1e3 * dom + 4. * gomega2)
                
                ni, ibins = np.histogram(gas_radius, bins=nshells, range=(0.001, galrad))
                sni, ibins = np.histogram(gas_radius, bins=nshells, weights=gas_cs, range=(0.001, galrad))
                gyi = sni / ni
                gxi = np.zeros(len(ibins) - 1)
                gxi[:] = 0.5 * (ibins[1:] + ibins[:-1])
            
            if gas:
                w = 2. * yi * gyi / (yi ** 2 + gyi ** 2)
                
                Qg = (gkappa[:] * gyi[:]) / (np.pi * Gcosmo * gsden[:])
                Qtot = np.zeros(len(Q))
                
                for j in range(len(Q)):
                    if Q[j] > Qg[j]:
                        Qtot[j] = 1. / ((w[j] / Q[j]) + (1. / Qg[j]))
                    if Qg[j] > Q[j]:
                        Qtot[j] = 1. / ((1. / Q[j]) + (w[j] / Qg[j]))
            
            print("Q,Qg,Qtot=", Q, Qg, Qtot)
            
            xnew = np.linspace(0., max(x), 100)
            Q1 = [1.] * 100
            
            time = s.cosmology_get_lookback_time_from_a(s.time, is_flat=True)
            
            ax.plot(x * 1e3, Qtot, linestyle=lss[isnap % len(lss)], color=colors[isnap % len(colors)], lw=0.7, label='Time = %.1f' % time)
            ax.plot(xnew[:-1] * 1e3, Q1[:-1], linestyle=':', color='k', lw=0.5)
            # ax.plot(x*1e3,Q,linestyle='--',color=clist[isnap],lw=0.5)
            ax.xaxis.set_ticks([0, 5, 10, 15, 20, 25])
            ax.yaxis.set_ticks([0, 2, 4, 6, 8])
        
        figure.set_panel_title(panel=d, title="$\\rm{%s%s}$" % ("Au", runs[d].split('_')[1]), position='bottom right')
        if d == 0:
            ax.legend(loc='upper right', frameon=False, prop={'size': 3})
    
    figure.reset_axis_limits()
    figure.fig.savefig('%s/ToomreQ.%s' % (outpath, suffix))