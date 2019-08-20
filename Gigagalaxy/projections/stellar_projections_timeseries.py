import copy
import os

import matplotlib.pyplot as plt
import numpy as np
from loadmodules import *
from util import select_snapshot_number

toinch = 0.393700787


def project_stars(dir, run, rot, snaplist, outpath, outputlistfile, suffix='', boxsize=0.05, loadonlytype=[], subhaloid=None, numthreads=1,
                  outputdir='output', mergertree=True):
    path = '%s/%s%s/' % (dir, run, suffix)
    outpath += ('/%s/projections/stars' % run)
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    
    if not loadonlytype:
        loadonlytype = [i for i in range(6)]
    
    sidmallnow = []
    loadptype = [4]
    
    if mergertree:
        snap0 = np.int_(select_snapshot_number.select_snapshot_number(outputlistfile, [0.]))
        treepath = '%s/mergertrees/Au-%s/trees_sf1_%03d' % (dir, run.split('_')[1], snap0)
        t = load_tree(0, 0, base=treepath)
        snap_numbers_main, redshifts_main, subfind_indices_main, first_progs_indices_main, ff_tree_indices_main, fof_indices_main, prog_mass_main, \
        next_prog_indices = t.return_first_next_mass_progenitors(
            0)
    
    for snap in snaplist:
        res = 512
        pylab.figure(figsize=(10 * toinch, 15 * toinch), dpi=300)
        ax1 = axes([0., 0.33, 1., 0.66])
        ax2 = axes([0., 0., 1., 0.33])
        # pylab.figure( figsize=(8*toinch,16*toinch), dpi=300 )
        # ax1 = axes( [0., 0.5, 1., 0.5] )
        # ax2 = axes( [0., 0., 1., 0.5] )
        
        if outputdir != 'output_red':
            print("Doing full snapshot")
            s = gadget_readsnap(snap, snappath=path + outputdir, loadonlytype=loadptype, loadonly=['pos', 'vel', 'mass', 'age', 'gsph', 'id'],
                                hdf5=True, forcesingleprec=True)
            sf = load_subfind(snap, dir=path + '/output/', hdf5=True, verbose=False,
                              loadonly=['fpos', 'frc2', 'svel', 'flty', 'fnsh', 'slty', 'spos', 'ffsh'])
            s.calc_sf_indizes(sf, verbose=False)
            age_select = 3.
            
            if subhaloid:
                s.select_halo(sf, 3., use_principal_axis=True, use_cold_gas_spin=False, do_rotation=True, haloid=0, subhalo=subhaloid)
            elif mergertree:
                fof = fof_indices_main[snap0 - snap]
                sub = subfind_indices_main[snap0 - snap]
                print("fof,sub=", fof, sub)
                shind = sub - sf.data['fnsh'][:fof].sum()
                s.select_halo(sf, 3., use_principal_axis=True, use_cold_gas_spin=False, do_rotation=True, haloid=fof, subhalo=shind)
            else:
                s.select_halo(sf, age_select, remove_bulk_vel=False, use_principal_axis=True, euler_rotation=False, rotate_disk=True,
                              do_rotation=True)
            
            time = s.cosmology_get_lookback_time_from_a(s.time, is_flat=True)
            if time < 0.0001:
                time = 0.0
            print("Time =", time)
            
            dist = np.max(np.abs(s.pos - s.center[None, :]), axis=1)
            istars, = np.where((s.type == 4) & (dist < 1.5 * boxsize))  # & (s.data['age'] > 0.) )
            
            # remove wind particles from stars ...
            first_star = 0
            for ptype in loadptype:
                if ptype < 4:
                    first_star += s.nparticlesall[ptype]
            
            istars -= np.int_(first_star)
            j, = np.where(s.data['age'][istars] > 0.)
            jstars = istars[j] + np.int_(first_star)
            
            temp_pos = s.pos[jstars, :].astype('float64')
            temp_vel = s.vel[jstars, :].astype('float64')
            mass = s.data['mass'][jstars].astype('float64')
            rho = np.ones(size(jstars))
        
        else:
            print("Doing reduced snapshot")
            s = gadget_readsnap(snap, snappath=path + '/%s/snapdir_%03d/' % (outputdir, snap), snapbase='snapshot_reduced_', hdf5=True,
                                loadonlytype=[4], loadonly=['pos', 'vel', 'mass', 'age', 'gsph'])
            dist = np.sqrt((s.pos ** 2).sum(axis=1))
            jstars, = np.where((s.type == 4) & (dist < 1.5 * boxsize) & (s.data['age'] > 0.))
            
            temp_pos = s.pos[jstars, :].astype('float64')
            temp_vel = s.vel[jstars, :].astype('float64')
            mass = s.data['mass'][jstars].astype('float64')
            rho = np.ones(size(jstars))
            time = 0.
        
        pos = np.zeros((size(jstars), 3))
        # 1 0 2 for euler rotations...
        # 0 1 2 for normal...
        pos[:, 0] = temp_pos[:, 1]  # 0:1  # y-axis
        pos[:, 1] = temp_pos[:, 2]  # 1:2  # x-axis
        pos[:, 2] = temp_pos[:, 0]  # 2:0
        
        vel = np.zeros((size(jstars), 3))
        vel[:, 0] = temp_vel[:, 1]  # 0:1  # y-axis
        vel[:, 1] = temp_vel[:, 2]  # 1:2  # x-axis
        vel[:, 2] = temp_vel[:, 0]  # 2:0
        
        srad = np.zeros(len(jstars))
        srad[:] = np.sqrt((pos[:, :] ** 2).sum(axis=1))
        
        nn, = np.where((srad < 0.01))
        
        trotate = False
        if trotate:
            # theta = 70. * (2.*np.pi / 360.)
            alpha = 30. * (2. * np.pi / 360.)
            beta = 30. * (2. * np.pi / 360.)
            rotz = [[cos(alpha), -sin(alpha), 0.], [sin(alpha), cos(alpha), 0.], [0., 0., 1.]]
            
            ##roty = [ [1., 0., 0.], [0., cos(beta), -sin(beta)], [0., sin(beta), cos(beta)] ]
            
            roty = [[cos(beta), 0., sin(beta)], [0., 1., 0.], [-sin(beta), 0., cos(beta)]]
            
            pos = np.dot(pos, rotz)
            pos = np.dot(pos, roty)
            vel = np.dot(vel, rotz)
            vel = np.dot(vel, roty)
        
        srad = np.zeros(len(jstars))
        srad[:] = np.sqrt((pos[:, :] ** 2).sum(axis=1))
        
        print("type pos, mass=", type(pos), type(mass))
        tree = makeTree(pos)
        hsml = tree.calcHsmlMulti(pos, pos, mass, 48, numthreads=numthreads)
        hsml = np.minimum(hsml, 4. * boxsize / res)
        hsml = np.maximum(hsml, 1.001 * boxsize / res * 0.5)
        
        datarange = np.array([[4003.36, 800672.], [199.370, 132913.], [133.698, 200548.]])
        # datarange = np.array( [ [4003.36,400672.], [199.370,42913.], [133.698,30548.] ] )
        
        fac = (512. / res) ** 2 * (0.5 * boxsize / 0.025) ** 2
        datarange *= fac
        
        data = np.zeros((res, res, 3))
        
        for k in range(3):
            iband = [3, 1, 0][k]
            band = 10 ** (-2.0 * s.data['gsph'][jstars, iband].astype('float64') / 5.0)
            grid = calcGrid.calcGrid(pos, hsml, band, rho, rho, res, res, res, boxsize, boxsize, boxsize, s.center[0], s.center[1], s.center[2], 1, 1,
                                     numthreads=numthreads)
            drange = datarange[k]
            grid = np.minimum(np.maximum(grid, drange[0]), drange[1])
            
            loggrid = np.log10(grid)
            
            logdrange = np.log10(drange)
            
            data[:, :, k] = (loggrid - logdrange[0]) / (logdrange[1] - logdrange[0])
            A = np.zeros((res, res))
            A[:, :] = data[:, :, k]
            data[:, :, k] = np.fliplr(A[:, :])  # .T
        
        gr, xedges, yedges = np.histogram2d(pos[0], pos[1], range=[[-boxsize, boxsize], [-boxsize, boxsize]], bins=[50, 50])
        
        # plt.subplot(2,1,1, frameon=False)
        ax1.imshow(data, interpolation='nearest')
        # ax = plt.gca()
        axis('image')
        ax1.set_xticks([])
        ax1.set_yticks([])
        # plt.title("z = %1.2f, t = %1.2f Gyr" % (s.redshift, time))
        
        """
        pix_per_kpc = float(res) / boxsize
        barlen = 0.01 * pix_per_kpc
        asb = AnchoredSizeBar(ax1.transData,
                              barlen,
                              r"$\rm{10\, kpc}$",
                              fontproperties=fm.FontProperties(size=15, family='monospace'),
                              loc=3,
                              pad=0.15, borderpad=0.5, sep=5, color='w',
                              frameon=False)
        ax1.add_artist(asb)

        ax1.scatter( [res/2 - (0.008/boxsize)*res], [res/2], marker='x', s=20, lw=0.8, edgecolor='w', facecolors='w' )
        ax1.scatter( [res/2 + (0.008/boxsize)*res], [res/2], marker='+', s=20, lw=0.8, edgecolor='w', facecolors='w' )
        ax1.scatter( [res/2], [res/2 - (0.008/boxsize)*res], marker='+', s=20, lw=0.8, edgecolor='w', facecolors='w' )
        ax1.scatter( [res/2], [res/2 + (0.008/boxsize)*res], marker='+', s=20, lw=0.8, edgecolor='w', facecolors='w' )
        """
        
        # ax1.arrow(res/2, res/2, ldir[0]*100, ldir[1]*100, color='w', ec='w')
        # ax1.arrow(res/2, res/2, lon[0]*100, lon[1]*100, color='r', ec='r')
        # text( 0.1, 0.9, "z = %1.2f, t = %1.2f Gyr" % (s.redshift, time), color='w', transform=ax.transAxes )
        # text( 0.9, 0.1, "$\\rm{Au \,%s}$" % (run.split('_')[1]), color='w', transform=ax1.transAxes, fontsize=22 )
        
        temp_pos2 = copy.copy(pos)
        pos[:, 0] = temp_pos2[:, 2]  # 2,0,1
        pos[:, 1] = temp_pos2[:, 1]
        pos[:, 2] = temp_pos2[:, 0]
        
        data = np.zeros((np.int_(res / 2), res, 3))
        # data = np.zeros( (res,res,3) )
        for k in range(3):
            iband = [3, 1, 0][k]
            band = 10 ** (-2.0 * s.data['gsph'][jstars, iband].astype('float64') / 5.0)
            grid = calcGrid.calcGrid(pos, hsml, band, rho, rho, np.int_(res / 2), res, res, boxsize / 2, boxsize, boxsize, s.center[0], s.center[1],
                                     s.center[2], 1, 1, numthreads=numthreads)
            # grid = calcGrid.calcGrid( pos, hsml, band, rho, rho, res, res, res, boxsize, boxsize, boxsize, s.center[0], s.center[1],
            # s.center[2], 1, 1, numthreads=numthreads )
            drange = datarange[k]
            grid = np.minimum(np.maximum(grid, drange[0]), drange[1])
            loggrid = np.log10(grid)
            print("logrid=", loggrid, np.min(loggrid), np.max(loggrid))
            logdrange = np.log10(drange)
            print("logdrange=", logdrange)
            data[:, :, k] = (loggrid - logdrange[0]) / (logdrange[1] - logdrange[0])
            print("data min max=", data[:, :, k], data[:, :, k].min(), data[:, :, k].max())
            
            A = np.zeros((np.int_(res / 2), res))
            # A = np.zeros((res, res))
            A[:, :] = data[:, :, k]
            data[:, :, k] = np.fliplr(A[:, :])  # .T
        
        # plt.subplot(2,1,2, frameon=False)
        ax2.imshow(data, interpolation='nearest')
        # ax = plt.gca()
        axis('image')
        ax2.set_xticks([])
        ax2.set_yticks([])
        
        # plt.tight_layout()
        
        print("saving file in:", outpath)
        
        del s
        if outputdir != 'output_red':
            del sf
        if res <= 256:
            savefig('%s/stars_%s_snap%04d_%06d_lres.png' % (outpath, run, snap, int(boxsize * 1e3)), dpi=300, transparent=True, bbox_inches='tight',
                    pad_inches=0)
        else:
            savefig('%s/stars_%s_snap%04d_%06d.png' % (outpath, run, snap, int(boxsize * 1e3)), dpi=300, transparent=True, bbox_inches='tight',
                    pad_inches=0)
        del ax1
        del ax2
        
        plt.clf()