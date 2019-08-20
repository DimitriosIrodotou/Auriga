import matplotlib

matplotlib.use('Agg')
from loadmodules import *
import matplotlib.pyplot as plt

toinch = 0.393700787


def plot_xyz_tracers(dirs, runs, snaplist, name, outpath, suffix='', boxsize=0.05, loadonlytype=[], rcut=0.005, numthreads=1, targetgasmass=False,
                     accreted=False, fhcen=False):
    for d in range(len(runs)):
        
        path = '%s/%s%s/' % (dirs[d], runs[d], suffix)
        
        wpath = '%s/%s/projections/' % (outpath, runs[d])
        if not os.path.exists(wpath):
            os.makedirs(wpath)
        wpath = str(wpath)
        
        res = 512
        pylab.figure(figsize=(22 * toinch, 22 * toinch), dpi=300)
        
        if not loadonlytype:
            loadonlytype = [i for i in range(6)]
        print("loadonlytype=", loadonlytype)
        
        loadonlytype = [0, 4, 6]
        sidmallnow = []
        
        nb = 0
        nbt = 0
        
        # snaplist should be passed in in reverse
        snap = snaplist[0]
        
        s = gadget_readsnap(snap, snappath=path + '/output/', loadonlytype=loadonlytype,
                            loadonly=['pos', 'vel', 'mass', 'age', 'gsph', 'id', 'trid', 'prid', 'hrgm'], hdf5=True, forcesingleprec=True)
        sf = load_subfind(snap, dir=path + '/output/', hdf5=True, loadonly=['fpos', 'slty', 'frc2', 'svel', 'sidm', 'smty', 'spos', 'fnsh', 'flty'],
                          forcesingleprec=True)
        s.calc_sf_indizes(sf)
        subhalostarmass = sf.data['smty'][0:sf.data['fnsh'][0], 4]
        
        s.select_halo(sf, [], 3., use_principal_axis=True, use_cold_gas_spin=False, do_rotation=True)
        
        time = s.cosmology_get_lookback_time_from_a(s.time.astype('float64'))
        
        ids = s.data['id'].astype('int64')
        idg = s.data['id'][s.type == 0].astype('int64')
        prid = s.data['prid'].astype('int64')
        trid = s.data['trid'].astype('int64')
        ptype = s.data['type'].astype('int64')
        
        # age = s.data['age']
        ii, = np.where((s.data['age'] >= 0.))
        star_age = -np.ones(len(s.data['age']))
        star_age[ii] = s.cosmology_get_lookback_time_from_a(s.data['age'][ii].astype('float64'), is_flat=True)
        s.data['age'] = np.zeros(s.npartall)
        st = s.nparticlesall[:4].sum()
        en = st + s.nparticlesall[4]
        s.data['age'][st:en] = star_age
        
        fname = '%s/%s/tracers/nrecycle_tracers.txt' % (outpath, runs[d])
        fin = open(fname, 'r')
        data = np.loadtxt(fin)
        star_rad = data[:, 0]
        star_age = data[:, 1]
        star_met = data[:, 2]
        
        if len(data[0, :]) == 4:
            nrecycle = data[:, 3]
        else:
            star_prid = data[:, 3]
            nrecycle = data[:, 4]
        
        # make selection
        ii, = np.where((nrecycle >= 4))
        # ii, = np.where( (nrecycle > -1) )
        starprid = star_prid[ii]
        indy = np.in1d(prid, starprid)
        startrid = trid[indy]
        
        ii, = np.where((nrecycle == 0))
        # ii, = np.where( (nrecycle > -1) )
        starprid = star_prid[ii]
        indy = np.in1d(prid, starprid)
        startrid2 = trid[indy]
        
        # now look for them in previous snapshots
        for isnap, snap in enumerate(snaplist[::4]):
            
            print("doing snap ", snap)
            fig, ((ax, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True)
            # plt.setp(ax.flat, aspect=1., adjustable='box-forced')
            s = gadget_readsnap(snap, snappath=path + '/output/', loadonlytype=loadonlytype,
                                loadonly=['pos', 'vel', 'mass', 'age', 'gsph', 'id', 'trid', 'prid', 'hrgm'], hdf5=True, forcesingleprec=True)
            
            sf = load_subfind(snap, dir=path + '/output/', hdf5=True,
                              loadonly=['fpos', 'slty', 'frc2', 'svel', 'sidm', 'smty', 'spos', 'fnsh', 'flty'], forcesingleprec=True)
            s.calc_sf_indizes(sf)
            subhalostarmass = sf.data['smty'][0:sf.data['fnsh'][0], 4]
            
            s.select_halo(sf, [], 3., use_principal_axis=True, use_cold_gas_spin=False, do_rotation=True)
            
            time = s.cosmology_get_lookback_time_from_a(s.time.astype('float64'))
            
            ids = s.data['id'].astype('int64')
            idg = s.data['id'][s.type == 0].astype('int64')
            prid = s.data['prid'].astype('int64')
            trid = s.data['trid'].astype('int64')
            ptype = s.data['type'].astype('int64')
            
            mass_variable = s.data['mass'].astype('float64')
            
            # stars
            mass = s.data['mass'].astype('float64')
            temp_pos = s.pos[:, :].astype('float64')
            pos = np.zeros((size(ptype), 3))
            pos[:, 0] = temp_pos[:, 1]
            pos[:, 1] = temp_pos[:, 2]
            pos[:, 2] = temp_pos[:, 0]
            temp_vel = s.vel[:, :].astype('float64')
            vel = np.zeros((size(ptype), 3))
            vel[:, 0] = temp_vel[:, 1]
            vel[:, 1] = temp_vel[:, 2]
            vel[:, 2] = temp_vel[:, 0]
            
            # total stellar mass
            # get tracer parent ID
            tind = np.in1d(trid, startrid)
            TracerID = s.trid[tind]
            ParentID = s.prid[tind]
            
            i, = np.where((s.mass < 2. * targetgasmass / s.hubbleparam) & (s.type == 4))
            spos = pos[i, :]
            smass = s.mass[i].astype('float64')
            ids = s.id[i].astype('int64')
            psind = np.in1d(ids, ParentID)
            tspos = spos[psind, :]
            
            # gas
            mass_variable = s.data['mass'][s.type == 0].astype('float64')
            i, = np.where((s.hrgm > 0.9 * mass_variable))
            idg = s.id[i].astype('int64')
            gind = np.in1d(idg, ParentID)
            
            gpos = pos[i, :]
            gvel = vel[i, :]
            gmass = s.mass[i].astype('float64')  # mass[i]
            
            tgpos = gpos[gind, :]
            tgmass = gmass[gind]
            tgvel = gvel[gind, :]
            
            # second selection
            tind = np.in1d(trid, startrid2)
            TracerID = s.trid[tind]
            ParentID = s.prid[tind]
            
            psind = np.in1d(ids, ParentID)
            tspos2 = spos[psind, :]
            
            gind = np.in1d(idg, ParentID)
            tgpos2 = gpos[gind, :]
            tgvel2 = gvel[gind, :]
            tgmass2 = gmass[gind]
            
            if isnap > 0:
                lzg = np.mean(np.cross(tgpos, tgvel)[2])
                lzg2 = np.mean(np.cross(tgpos2, tgvel2)[2])
                print("** lz (nrec>4 vs. nrec=0)=", lzg, lzg2)
            
            grad = np.sqrt((tgpos ** 2).sum(axis=1))
            grad2 = np.sqrt((tgpos2 ** 2).sum(axis=1))
            srad = np.sqrt((tspos ** 2).sum(axis=1))
            srad2 = np.sqrt((tspos2 ** 2).sum(axis=1))
            
            # plot it
            ax.plot(tspos[:, 1], tspos[:, 2], 'r.', markersize=0.3)
            ax2.plot(tspos[:, 1], tspos[:, 0], 'r.', markersize=0.3)
            ax3.plot(tgpos[:, 1], tgpos[:, 2], 'r.', markersize=0.3)
            ax4.plot(tgpos[:, 1], tgpos[:, 0], 'r.', markersize=0.3)
            
            ax.plot(tspos2[:, 1], tspos2[:, 2], 'k.', markersize=0.3)
            ax2.plot(tspos2[:, 1], tspos2[:, 0], 'k.', markersize=0.3)
            ax3.plot(tgpos2[:, 1], tgpos2[:, 2], 'k.', markersize=0.3)
            ax4.plot(tgpos2[:, 1], tgpos2[:, 0], 'k.', markersize=0.3)
            
            ax.text(0.1, 0.9, "$\\rm{stars}$", color='k', transform=ax.transAxes, fontsize=15)
            ax3.text(0.1, 0.9, "$\\rm{gas}$", color='k', transform=ax3.transAxes, fontsize=15)
            
            # axlim = 0.2
            axlim = 0.5 * boxsize
            ax.set_xlim([-axlim, axlim])
            ax.set_ylim([-axlim, axlim])
            ax2.set_xlim([-axlim, axlim])
            ax2.set_ylim([-axlim, axlim])
            ax3.set_xlim([-axlim, axlim])
            ax3.set_ylim([-axlim, axlim])
            ax4.set_xlim([-axlim, axlim])
            ax4.set_ylim([-axlim, axlim])
            del s
            del sf
            
            fig.savefig('%s/xyz_trace%s%s_%03d.png' % (wpath, runs[d], name, snap))