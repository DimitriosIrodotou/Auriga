from const import *
from loadmodules import *
from parallel_decorators import vectorize_parallel
from pylab import *
from util import *

colors = ['g', 'b', 'r', 'k', 'c', 'y', 'm', 'purple', 'gray', 'hotpink']
markers = ['o', '^', 'd', 's', 'v', '>']


def plot_disctumbling(runs, dirs, outpath, outputlistfile, zlast, suffix, nrows, ncols, types=[0, 4], mergertree=True):
    lencol = len(colors)
    
    panels = len(runs)
    
    figure = multipanel_layout.multipanel_layout(nrows=nrows, ncols=ncols, npanels=panels)
    figure.set_figure_layout()
    figure.set_axis_limits_and_aspect(xlim=[13.5, 0.], ylim=[-1.2, 1.2])
    figure.set_axis_locators(xminloc=2., xmajloc=2., yminloc=0.2, ymajloc=0.4)
    figure.set_fontsize(6)
    figure.set_axis_labels(xlabel="$\\rm{t_{lookback} [Gyr]}$", ylabel="$\\rm{\cos (angle)}$")
    
    lstyle = ['-', '--', ':']
    
    for d in range(len(runs)):
        dd = dirs[d] + runs[d]
        ax = figure.axes[d]
        # ax2 = figure.axes[1]
        
        snaphz = np.int_(select_snapshot_number.select_snapshot_number(outputlistfile[d], zlast))
        snap0 = np.int_(select_snapshot_number.select_snapshot_number(outputlistfile[d], [0.]))
        # snaps = list(range(60,array(snaps).max()+1))
        snaps = list(range(snaphz, snap0 + 1))
        snaps = snaps[::-1]
        # snaps = snaps[::4]
        
        if mergertree:
            treepath = '%s/mergertrees/Au-%s/trees_sf1_%03d' % (dirs[d], runs[d].split('_')[1], snap0)
            print("treepath=", treepath)
            t = load_tree(0, 0, base=treepath)
            snap_numbers_main, redshifts_main, subfind_indices_main, first_progs_indices_main, ff_tree_indices_main, fof_indices_main, \
            prog_mass_main, next_prog_indices = t.return_first_next_mass_progenitors(
                0)
            
            print("fof_indices_main,subfind_indices_main=", fof_indices_main, subfind_indices_main)
        
        # check if we need to trace ID msot bound particles
        idmb_flag = 0
        
        nin = snap0 - snaphz
        flag_indx, = np.where((fof_indices_main[:nin] > 0) & (subfind_indices_main[:nin] > 0))
        if size(flag_indx) >= 1:
            print("we have found %d snaps out of %d that have FoF and SubFind > 0" % (size(flag_indx), nin))  # idmb_flag = 1
        
        if runs[d] == 'halo_L6':
            idmb_flag = 1
        
        if runs[d] == 'halo_L10':
            snap0 = snaps[7]
            snaps = snaps[7:]
        
        if idmb_flag:
            types = [0, 1, 4]
        else:
            types = [0, 4]
        
        ldir = np.zeros((len(snaps), 3))
        
        filename = outpath + '/%s/disctumbling_%s.txt' % (runs[d], runs[d])
        f2 = open(filename, 'w')
        header = "%18s%18s%18s\n" % ("Time", "Tumbleangle", "Gasdischaloangle")
        f2.write(header)
        
        attrs = ['pos', 'vel', 'mass', 'age', 'id', 'sfr', 'gmet', 'nh', 'ne', 'u', 'gz']
        s = gadget_readsnap(snap0, snappath=dd + '/output/', hdf5=True, loadonlytype=types, loadonly=attrs, forcesingleprec=True)
        sf = load_subfind(snap0, dir=dd + '/output/', hdf5=True, loadonly=['fpos', 'frc2', 'svel', 'flty', 'fnsh', 'slty', 'spos', 'ffsh'])
        s.calc_sf_indizes(sf)
        s.centerat(sf.data['fpos'][0])
        # s.select_halo( sf, 3., use_principal_axis=True, use_cold_gas_spin=False, do_rotation=False )
        
        galrad = 0.5 * sf.data['frc2'][0]
        g = parse_particledata(s, sf, attrs, radialcut=galrad)
        g.prep_data()
        
        sdata = g.sgdata[g.datatypes[4]]
        age = sdata['age']
        spos = sdata['pos']
        svel = sdata['vel']
        smass = sdata['mass']
        rrstars = np.sqrt((sdata['pos'] ** 2).sum(axis=1))
        
        istars, = np.where((rrstars < 0.01) & (rrstars > 0.) & (age > 0.))
        lang = np.cross(spos[istars], (svel[istars] * smass[istars, None]))
        ltot = lang.sum(axis=0)
        rdir = ltot / sqrt((ltot ** 2).sum())
        
        arr = []
        
        if idmb_flag:
            idmb = list(s.get_most_bound_dm_particles())
        else:
            idmb = None
        
        arr.append(get_angles(snaps, dd, attrs, fof_indices_main, subfind_indices_main, snap0, rdir, types, idmb=idmb))
        
        arr = np.array(arr).ravel()
        times = arr[::3]
        gasangle = arr[1::3]
        tumbleangle = arr[2::3]
        
        color = colors[d % lencol]
        # marker = markers[int(d/lencol)]
        if d == 0:
            label1 = "$\\rm{tumble}$"
            label2 = "$\\rm{disc-corona}$"
        else:
            label1 = ''
            label2 = ''
        
        ax.plot(times, tumbleangle, linestyle='-', lw=0.7, color=color, label=label1)
        ax.plot(times, gasangle, dashes=(2, 2), lw=0.7, color=color, label=label2)
        
        np.savetxt(f2, np.column_stack((times, tumbleangle, gasangle)))
        f2.close()
        
        figure.set_panel_title(panel=d, title="$\\rm %s\,%s$" % ("Au", runs[d].split('_')[1]), position='bottom right', color='k', fontsize=8)
        
        ax.legend(loc='lower left', frameon=False, prop={'size': 6}, ncol=1)
    
    print("outpath=", outpath)
    figure.fig.savefig('%s/disc_tumbling.%s' % (outpath, suffix))


@vectorize_parallel(method='processes', num_procs=20)
def get_angles(snap, dd, attrs, fof_indices_main, subfind_indices_main, snap0, rot, types, idmb=None):
    s = gadget_readsnap(snap, snappath=dd + '/output/', hdf5=True, loadonlytype=types, loadonly=attrs, forcesingleprec=True)
    sf = load_subfind(snap, dir=dd + '/output/', hdf5=True, loadonly=['fpos', 'frc2', 'svel', 'flty', 'fnsh', 'slty', 'spos', 'ffsh'])
    s.calc_sf_indizes(sf)
    
    if idmb:
        centre = s.pos[np.where(np.in1d(s.id, [idmb[0]]) == True), :].ravel()
        s.centerat(centre)
        # s.select_halo( sf, 3., centre=list(centre), use_principal_axis=True, use_cold_gas_spin=False, do_rotation=False )
        s.calc_sf_indizes(sf)
        fof = fof_indices_main[snap0 - snap]
        sub = subfind_indices_main[snap0 - snap]
        shind = sub - sf.data['fnsh'][:fof].sum()
    else:
        fof = fof_indices_main[snap0 - snap]
        sub = subfind_indices_main[snap0 - snap]
        shind = sub - sf.data['fnsh'][:fof].sum()
        s.select_halo(sf, 3., use_principal_axis=True, use_cold_gas_spin=False, do_rotation=False, haloid=fof, subhalo=shind)
    
    galrad = 0.5 * sf.data['frc2'][0]
    print("attrs=", attrs)
    g = parse_particledata(s, sf, attrs, radialcut=galrad)
    g.prep_data()
    
    sdata = g.sgdata[g.datatypes[4]]
    age = sdata['age']
    spos = sdata['pos']
    svel = sdata['vel']
    smass = sdata['mass']
    rrstars = np.sqrt((sdata['pos'] ** 2).sum(axis=1))
    
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
    
    # istars, = np.where( (rrstars < 0.1 * sf.data['frc2'][0]) & (rrstars > 0.) & (age > 0.) )
    istars, = np.where((rrstars < 0.01) & (rrstars > 0.) & (age > 0.))
    lang = np.cross(spos[istars], (svel[istars] * smass[istars, None]))
    ltot = lang.sum(axis=0)
    ldir = ltot / sqrt((ltot ** 2).sum())
    tumbleangle = np.dot(ldir, rot)
    
    times = s.cosmology_get_lookback_time_from_a(s.time.astype('float64'))
    
    idisc, = np.where((sfr > 0.) & (rr < 0.1 * sf.data['frc2'][0]))
    # ihalo, = np.where( (sfr <= 0.) )
    ihalo, = np.where((sfr <= 0.) & (temp < 1e5) & (temp > 1e4) & (rr < sf.data['frc2'][0]))
    
    jdisc = np.squeeze(np.cross(pos[idisc], vel[idisc]))
    jhalo = np.squeeze(np.cross(pos[ihalo], vel[ihalo]))
    rdisc = rad[idisc]
    rhalo = rad[ihalo]
    
    jtotdisc = jdisc.sum(axis=0)
    jdirdisc = jtotdisc / (sqrt((jtotdisc ** 2).sum()))
    
    jtothalo = jhalo.sum(axis=0)
    jdirhalo = jtothalo / (sqrt((jtothalo ** 2).sum()))
    
    # gasangle = np.dot( jdirdisc, jdirhalo )
    gasangle = np.dot(ldir, jdirhalo)
    
    print("times,gasangle,tumbleangle=", times, gasangle, tumbleangle)
    arr = np.array([times, gasangle, tumbleangle])
    
    return arr