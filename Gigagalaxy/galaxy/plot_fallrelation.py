import matplotlib.colors as colors
from const import *
from loadmodules import *
from parallel_decorators import vectorize_parallel
# from util import multipanel_layout
from util import *

colors = ['g', 'b', 'k', 'c', 'y', 'r', 'm', 'purple', 'gray', 'hotpink']
lines = ['-', ':']
marker = ['o', '^', 'd', 's']
toinch = 0.393700787
Gcosmo = 43.0071


# colors = ['g', 'hotpink']

def plot_fallrelation(runs, dirs, outpath, snap, outputlistfile):
    nruns = len(runs)
    
    figure = multipanel_layout.multipanel_layout(nrows=1, ncols=1, npanels=1, aspect_fac=0.7)
    figure.set_fontsize(8)
    figure.set_figure_layout()
    figure.set_fontsize(8)
    figure.set_axis_locators(xminloc=0.5, xmajloc=0.5, yminloc=0.5, ymajloc=0.5)
    figure.set_axis_labels(xlabel="$\\rm{log_{10} \\, M_{*} \\,[M_{\odot}]}$", ylabel="$\\rm{log_{10} \\, l_z}\\,[kpc \\, km \\,s^{-1}]}$")
    xran = [9., 11.1]
    figure.set_axis_limits_and_aspect(xlim=xran, ylim=[2., 3.4])
    
    fallslope = 0.67
    fallnormdisc = 3.17
    fallnormbulge = 2.25
    xs = np.linspace(xran[0], xran[1], 10)
    ydisc = (xs - 10.5) * 0.67 + fallnormdisc
    ybulge = (xs - 10.5) * 0.67 + fallnormbulge
    
    parameters = np.linspace(0., 0.8, 9)
    cmapl = plt.get_cmap('viridis')
    s_m = figure.get_scalar_mappable_for_colorbar(parameters, cmapl)
    
    for d in range(nruns):
        
        nsk = 4
        snaphz = np.int_(select_snapshot_number.select_snapshot_number(outputlistfile[d], [3.]))
        snap0 = np.int_(select_snapshot_number.select_snapshot_number(outputlistfile[d], [0.]))
        snaps = list(range(snaphz, snap0 + 1))
        snaps = snaps[::-1]
        
        if runs[d] == 'halo_L10':
            # snap0 =snaps[7]
            snaps = snaps[7:]
        if runs[d] == 'halo_L5':
            snaps = snaps[5:]
        # if runs[d] == 'halo_L4':
        #       snaps = snaps[3:]
        
        snaps = snaps[::nsk]
        
        dd = dirs[d] + runs[d]
        ax = figure.axes[0]
        print("Doing dir %s snap %d." % (dd, snap))
        
        treepath = '%s/mergertrees/Au-%s/trees_sf1_%03d' % (dirs[d], runs[d].split('_')[1], snap0)
        print("treepath=", treepath)
        t = load_tree(0, 0, base=treepath)
        snap_numbers_main, redshifts_main, subfind_indices_main, first_progs_indices_main, ff_tree_indices_main, fof_indices_main, prog_mass_main, \
        next_prog_indices = t.return_first_next_mass_progenitors(
            0)
        
        # check if we need to trace ID msot bound particles
        idmb_flag = 0
        nin = snap0 - snaphz
        flag_indx, = np.where((fof_indices_main[:nin] > 0) & (subfind_indices_main[:nin] > 0))
        if size(flag_indx) >= 1:
            print("we have found %d snaps out of %d that have FoF and SubFind > 0" % (size(flag_indx), nin))  # idmb_flag = 1
        
        if runs[d] == 'halo_L6':
            idmb_flag = 1
        
        if idmb_flag:
            attrs = ['pos', 'vel', 'mass', 'age', 'pot', 'id', 'sfr']
            s = gadget_readsnap(snap0, snappath=dd + '/output/', hdf5=True, loadonlytype=[1, 4], loadonly=attrs)
            sf = load_subfind(snap0, dir=dd + '/output/', hdf5=True, loadonly=['fpos', 'frc2', 'spos', 'ffsh', 'fnsh'])
            s.centerat(sf.data['fpos'][0])
            idmb = s.get_most_bound_dm_particles()
            arr = []
            arr.append(get_fallrelation_idmb(snaps, dd, attrs, idmb, snap0))
        
        else:
            attrs = ['pos', 'vel', 'mass', 'age', 'pot', 'sfr']
            arr = []
            arr.append(get_fallrelation(snaps, dd, attrs, fof_indices_main, subfind_indices_main, snap0))
        
        arr = np.array(arr).ravel()
        
        mtot = arr[::2]
        jstar = arr[1::2]
        
        print("np.log10(mtot*1e10), np.log10(jstar*1e3)=", np.log10(mtot * 1e10), np.log10(jstar * 1e3))
        ax.plot(xs, ydisc, dashes=(2, 2), color='b')
        ax.plot(xs, ybulge, dashes=(2, 2), color='r')
        # ax.plot( np.log10(mtot*1e10), np.log10(jstar*1e3), marker=markers[runs[d]], mfc='k', mec=colors[runs[d]], ms=5., mew=1. )
        # ax.plot( np.log10(mtot*1e10), np.log10(jstar*1e3), marker='o', mfc=s_m.to_rgba(btrat), ms=5., mew=0., alpha=0.8 )
        
        ax.plot(np.log10(mtot[0] * 1e10), np.log10(jstar[0] * 1e3), marker='o', mfc=colors[d], ms=5., mew=0., alpha=0.8, lw=0.)
        
        if runs[d] == 'halo_L8' or runs[d] == 'halo_L7':  # or runs[d] == 'halo_L10' or runs[d] == 'halo_L2' or runs[d] == 'halo_L4':
            ax.plot(np.log10(mtot * 1e10), np.log10(jstar * 1e3), color=colors[d], lw=1.)
            imark = np.array([snap0 - 95, snap0 - 78]) / nsk
            ax.plot(np.log10(mtot[np.int_(imark)] * 1e10), np.log10(jstar[np.int_(imark)] * 1e3), marker='s', mfc=colors[d], ms=4, mew=0., alpha=0.8,
                    lw=0.)
    
    # figure.set_colorbar([0.,0.8],r'$\rm{ B/T }$',[0, 0.2, 0.4, 0.6, 0.8], cmap=cmapl, fontsize=7, labelsize=7, orientation='horizontal')
    
    figure.fig.savefig('%s/plotall/fallrelation_new.pdf' % (outpath))


@vectorize_parallel(method='processes', num_procs=20)
def get_fallrelation(snap, dd, attrs, fof_indices_main, subfind_indices_main, snap0):
    s = gadget_readsnap(snap, snappath=dd + '/output/', hdf5=True, loadonly=attrs, loadonlytype=[0, 4])
    sf = load_subfind(snap, dir=dd + '/output/', hdf5=True, loadonly=['fpos', 'frc2', 'fmc2', 'svel', 'flty', 'fnsh', 'slty', 'spos', 'ffsh'])
    s.calc_sf_indizes(sf)
    
    fof = fof_indices_main[snap0 - snap]
    sub = subfind_indices_main[snap0 - snap]
    shind = sub - sf.data['fnsh'][:fof].sum()
    s.select_halo(sf, 3., use_principal_axis=True, use_cold_gas_spin=False, do_rotation=True, haloid=fof, subhalo=shind)
    
    accretedfile = None
    g = parse_particledata(s, sf, attrs, radialcut=0.2 * sf.data['frc2'][fof], accretedfile=accretedfile)
    # g = parse_particledata( s, sf, attrs, radialcut=0.03, accretedfile=accretedfile )
    g.prep_data()
    
    sdata = g.sgdata['sdata']
    
    mass = sdata['mass']
    pos = sdata['pos']
    vel = sdata['vel']
    eps = sdata['eps2']
    
    gdata = g.sgdata['gdata']
    gmass = gdata['mass']
    gpos = gdata['pos']
    gvel = gdata['vel']
    sfr = gdata['sfr']
    
    btrat = 2. * (np.sum(mass[np.where(eps < 0.)])) / mass.sum()
    
    j = mass[:, None] * pylab.cross(pos[:, :], vel[:, :])
    
    isfr, = np.where(sfr > 0.)
    mtot = mass.sum()  # + gmass[isfr].sum()
    gj = gmass[:, None] * pylab.cross(gpos[:, :], gvel[:, :])
    gj = gj[isfr]
    
    jz = j[:, 0]
    gjz = gj[:, 0]
    # jz /= mtot[None]
    jstar = np.sum(jz)  # + np.sum( gjz )
    jstar /= mtot
    
    arr = np.array([mtot, jstar])
    return arr


@vectorize_parallel(method='processes', num_procs=20)
def get_fallrelation_idmb(snap, dd, attrs, idmb, snap0):
    s = gadget_readsnap(snap, snappath=dd + '/output/', hdf5=True, loadonly=attrs, loadonlytype=[0, 1, 4])
    sf = load_subfind(snap, dir=dd + '/output/', hdf5=True, loadonly=['fpos', 'frc2', 'fmc2', 'svel', 'flty', 'fnsh', 'slty'])
    centre = s.pos[np.where(np.in1d(s.id, [idmb[0]]) == True), :].ravel()
    s.calc_sf_indizes(sf)
    s.select_halo(sf, 3., centre=list(centre), use_principal_axis=True, use_cold_gas_spin=False, do_rotation=True)
    
    accretedfile = None
    g = parse_particledata(s, sf, attrs, radialcut=0.03, accretedfile=accretedfile)
    g.prep_data()
    
    sdata = g.sgdata['sdata']
    
    mass = sdata['mass']
    pos = sdata['pos']
    vel = sdata['vel']
    eps = sdata['eps2']
    
    gdata = g.sgdata['gdata']
    gmass = gdata['mass']
    gpos = gdata['pos']
    gvel = gdata['vel']
    sfr = gdata['sfr']
    
    mtot = mass.sum()
    btrat = 2. * (np.sum(mass[np.where(eps < 0.)])) / mass.sum()
    
    isfr, = np.where(sfr > 0.)
    mtot = mass.sum()  # + gmass[isfr].sum()
    
    gj = gmass[:, None] * pylab.cross(gpos[:, :], gvel[:, :])
    gj = gj[isfr]
    j = mass[:, None] * pylab.cross(pos[:, :], vel[:, :])
    
    jz = j[:, 0]
    gjz = gj[:, 0]
    jstar = np.sum(jz)  # + np.sum( gjz )
    jstar /= mtot
    
    # jz = j[:,0]
    # jz /= mtot[None]
    # jstar = np.sum( jz )
    
    arr = np.array([mtot, jstar])
    return arr