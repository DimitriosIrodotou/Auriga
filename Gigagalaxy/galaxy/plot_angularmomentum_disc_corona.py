from const import *
from loadmodules import *
from pylab import *
from util import *

colors = ['k', 'cornflowerblue', 'seagreen', 'r', 'hotpink']


def plot_angularmomentum_disc_corona(runs, dirs, outpath, outputlistfile, zlist, suffix, nrows, ncols, types=[0], mergertree=True):
    panels = nrows * ncols
    
    figure = multipanel_layout.multipanel_layout(nrows=nrows, ncols=ncols, npanels=panels, aspect_fac=0.7)
    figure.set_figure_layout()
    figure.set_axis_limits_and_aspect(xlim=[0.0, 52.0], ylim=[-1., 3.8])
    # figure.set_axis_locators( xminloc=1., xmajloc=5., yminloc=0.5, ymajloc=1. )
    figure.set_fontsize(8)
    figure.set_axis_labels(xlabel="$\\rm{R [kpc]}$", ylabel="$\\rm{ log_{10} l_z \,[km\,s^{-1}]}$")
    
    lstyle = ['-', '--', ':']
    
    for d in range(len(runs)):
        dd = dirs[d] + runs[d]
        ax = figure.axes[d]
        # ax2 = figure2.axes[d]
        
        if runs[d] == 'halo_L7':
            zlist = [0., 0.25, 0.5, 2.]
        else:
            zlist = [0., 0.25, 0.5, 1.]
        snaps = np.int_(select_snapshot_number.select_snapshot_number(outputlistfile[d], zlist))
        if isinstance(snaps, int):
            snaps = [snaps]
        
        # get disc scale length from file
        filename = outpath + runs[d] + '/radprof/' + 'fit_table_%s_L.txt' % runs[d]
        pdat = np.loadtxt(filename, comments='#', skiprows=1, dtype={
            'names':   ("Run", "Time", "virial_mass", "virial_radius", "stellar_mass", "disc_mass", "R_d", "bulge_mass", "r_eff", "n", "D/T"),
            'formats': ('S1', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4')})
        
        if mergertree:
            snap0 = np.int_(select_snapshot_number.select_snapshot_number(outputlistfile[d], [0.]))
            treepath = '%s/mergertrees/Au-%s/trees_sf1_%03d' % (dirs[d], runs[d].split('_')[1], snap0)
            t = load_tree(0, 0, base=treepath)
            snap_numbers_main, redshifts_main, subfind_indices_main, first_progs_indices_main, ff_tree_indices_main, fof_indices_main, \
            prog_mass_main, next_prog_indices = t.return_first_next_mass_progenitors(
                0)
        
        ldir = np.zeros((len(snaps), 3))
        gasangle = np.zeros(len(snaps))
        tumbleangle = np.zeros(len(snaps))
        times = np.zeros(len(snaps))
        
        for isnap in range(len(snaps)):
            snap = snaps[isnap]
            # if runs[d] == 'halo_L7' and snap == 78:
            #       snap = 65
            print("Doing dir %s snap %d." % (dd, snap))
            attrs = ['pos', 'vel', 'mass', 'age', 'id', 'sfr', 'age', 'gmet', 'nh', 'ne', 'u', 'gz']
            
            s = gadget_readsnap(snap, snappath=dd + '/output/', hdf5=True, loadonlytype=[0, 4], loadonly=attrs, forcesingleprec=True)
            sf = load_subfind(snap, dir=dd + '/output/', hdf5=True, loadonly=['fpos', 'frc2', 'svel', 'flty', 'fnsh', 'slty', 'spos', 'ffsh'])
            s.calc_sf_indizes(sf)
            
            if mergertree:
                fof = fof_indices_main[snap0 - snap]
                sub = subfind_indices_main[snap0 - snap]
                shind = sub - sf.data['fnsh'][:fof].sum()
                s.select_halo(sf, 3., use_principal_axis=True, use_cold_gas_spin=False, do_rotation=True, haloid=fof, subhalo=shind)
            else:
                s.select_halo(sf, 3., use_principal_axis=True, use_cold_gas_spin=False, do_rotation=True)
            galrad = 0.5 * sf.data['frc2'][0]
            g = parse_particledata(s, sf, attrs, radialcut=galrad)
            g.prep_data()
            
            dstr = g.datatypes[0]
            data = g.sgdata[dstr]
            mass = data['mass']
            rr = np.sqrt((data['pos'] ** 2).sum(axis=1))
            rad = np.sqrt((data['pos'][:, 1:] ** 2).sum(axis=1))
            vel = data['vel']
            pos = data['pos']
            sfr = data['sfr']
            ne = data['ne']
            XH = data['gmet'][:, 0]
            metallicity = data['gz']
            yhelium = (1 - XH - metallicity) / (4. * XH);
            mu = (1 + 4 * yhelium) / (1 + yhelium + ne)
            temp = data['u'] * 1e10 * mu * PROTONMASS / BOLTZMANN
            print("min max temp=", temp.min(), temp.max())
            
            # set up data for stellar disc
            sdata = g.sgdata['sdata']
            smass = sdata['mass']
            svel = sdata['vel']
            spos = sdata['pos']
            srr = np.sqrt((spos ** 2).sum(axis=1))
            srad = np.sqrt((spos[:, 1:] ** 2).sum(axis=1))
            
            times[isnap] = s.cosmology_get_lookback_time_from_a(s.time.astype('float64'))
            
            rd = pdat['R_d'][(pdat['Time'] == np.around(times[isnap], decimals=3))]
            rb = pdat['r_eff'][(pdat['Time'] == np.around(times[isnap], decimals=3))]
            
            idisc, = np.where((abs(spos[:, 0]) < 0.005) & (srr < galrad))
            ihalo, = np.where((sfr <= 0.) & (temp < 1e6) & (temp > 1e4))
            
            jdisc = np.squeeze(np.cross(spos[idisc], svel[idisc]))
            jhalo = np.squeeze(np.cross(pos[ihalo], vel[ihalo]))
            rdisc = srad[idisc]
            rhalo = rad[ihalo]
            
            jtotdisc = jdisc.sum(axis=0)
            jdirdisc = jtotdisc / (sqrt((jtotdisc ** 2).sum()))
            
            jtothalo = jhalo.sum(axis=0)
            jdirhalo = jtothalo / (sqrt((jtothalo ** 2).sum()))
            
            jzdisc = jdisc[:, 0]
            jzhalo = jhalo[:, 0]
            
            vphi = jzdisc / rdisc
            
            nshells = 100
            
            n, edges = np.histogram(rhalo, bins=nshells, range=(0., galrad))
            jzhbin, edges = np.histogram(rhalo, bins=nshells, range=(0., galrad), weights=jzhalo)
            jzhbin /= n
            
            n, edges = np.histogram(rdisc, bins=nshells, range=(0., galrad))
            vdbin, edges = np.histogram(rdisc, bins=nshells, range=(0., galrad), weights=vphi)
            vdbin /= n
            
            rbin = 0.5 * (edges[1:] + edges[:-1])
            
            # if runs[d] == 'halo_L7' and snap == 60:
            #        rd = np.array([1.612])
            
            if rd:
                dind = find_nearest(rbin * 1e3, rd)
                rs = rd
            else:
                #       dind = find_nearest( rbin*1e3, rb )
                #       rs = rb
                rs = 0.
                dind = None
            if dind:
                vd = vdbin[dind]
            else:
                vd = 0.
            
            lzdisc = 2. * vd * rs
            print("lzdisc=", lzdisc)
            
            if d == 2 or d == 0:
                label = '$\\rm{z=%.2f}$' % zlist[isnap]
            else:
                label = ''
            
            ax.plot(rbin * 1e3, jzhbin, linestyle='-', lw=0.7, color=colors[isnap], label=label)  # , color=s_m.to_rgba(zlist[isnap]) )
            
            # use this approx only for small negative values!!
            if lzdisc <= 0.:
                loglz = 0.
            else:
                loglz = np.log10(lzdisc)
            ax.plot(rbin * 1e3, [loglz] * nshells, linestyle='--', color=colors[isnap])  # , color=s_m.to_rgba(zlist[isnap]) )
        
        if d == 0:
            ax.legend(loc='lower right', frameon=False, prop={'size': 5}, ncol=1)
        else:
            ax.legend(loc='upper left', frameon=False, prop={'size': 5}, ncol=2)
        
        figure.set_panel_title(panel=d, title="$\\rm{%s\,{%s}}$" % ("Au", runs[d].split('_')[1]), position='bottom left')
    
    figure.fig.savefig('%s/lz_disc-corona_L8L7.%s' % (outpath, suffix))


def find_nearest(array, value):
    if len(value) == 1:
        idx = (np.abs(array - value)).argmin()
    else:
        idx = np.zeros(len(value))
        for i in range(len(value)):
            idx[i] = (np.abs(array - value[i])).argmin()
    
    return idx