import struct

from const import *
from loadmodules import *
from pylab import *
from util import *
from util import satellite_utilities as sat

toinch = 0.393700787
Gcosmo = 43.0071


def plot_mainsubhalo_properties(runs, dirs, outpath, outputlistfile, firstsnap, lastsnap, suffix, nrows, ncols):
    # snaplist = np.arange(41,128)
    # snaplist = snaplist[::-1]
    # nsnap = len(snaplist)
    
    HubbleParam = 0.6777
    
    panels = nrows * ncols
    
    for d in range(len(runs)):
        
        ofile = outputlistfile[d]  # + '/expfac_snapshotlist_%s.txt'%runs[d]
        
        snap = np.int_(select_snapshot_number.select_snapshot_number(ofile, lastsnap))
        snapf = np.int_(select_snapshot_number.select_snapshot_number(ofile, firstsnap))
        snaplist = np.arange(snapf, snap + 1)[::-1]
        nsnap = len(snaplist)
        
        figure = multipanel_layout.multipanel_layout(nrows=nrows, ncols=ncols, npanels=panels, twinyaxis=True)
        figure.set_fontsize(4)
        figure.set_figure_layout()
        figure.set_axis_locators(xminloc=1., xmajloc=2., yminloc=1., ymajloc=2.)
        figure.set_fontsize(10)
        figure.set_axis_labels(xlabel="$\\rm{t_{lookback}\\, [Gyr]}$", ylabel="$\\rm{d\\,[kpc]}$",
                               y2label="$\\rm{Z_{sub}/Z_{main}}$\n $\\rm{M/M_{peak}}$")
        figure.set_axis_limits_and_aspect(xlim=[12.5, 0.], ylim=[1e0, 5e2], logyaxis=True)
        
        figure2 = multipanel_layout.multipanel_layout(nrows=nrows, ncols=ncols, npanels=panels, twinyaxis=False)
        figure2.set_fontsize(4)
        figure2.set_figure_layout()
        figure2.set_axis_locators(xminloc=1., xmajloc=2., yminloc=0.4, ymajloc=0.4)
        figure2.set_fontsize(10)
        figure2.set_axis_labels(xlabel="$\\rm{t_{lookback}\\, [Gyr]}$", ylabel=r"$\rm{cos(\alpha)}$")
        figure2.set_axis_limits_and_aspect(xlim=[12.5, 0.], ylim=[-1.1, 1.1], logyaxis=False)
        
        filename = outpath + '/%s/mergers_flybys_%s.txt' % (runs[d], runs[d])
        f1 = open(filename, 'w')
        header = "%4s%6s%12s%12s%12s%12s%12s%12s%12s%12s%12s%12s%12s%12s%12s%12s\n" % (
        "type", "id", "time", "peaktotmass", "peaksmass", "peakgmass", "peaksrat", "peakgrat", "totmass", "smassrat", "gmassrat", "dist", "peakgmet",
        "peakgmetrat", "gmetrat", "pkmassid")
        f1.write(header)
        
        filename = '%s/lists/%sstarID_accreted_all_newmtree.dat' % (dirs[d], runs[d])
        print("filename=", filename)
        fin = open(filename, "rb")
        nacc = struct.unpack('i', fin.read(4))[0]
        idacc = numpy.array(struct.unpack('%sd' % nacc, fin.read(nacc * 8)), dtype=int64)
        subacc = numpy.array(struct.unpack('%sd' % nacc, fin.read(nacc * 8)), dtype=int64)
        snapacc = numpy.array(struct.unpack('%sd' % nacc, fin.read(nacc * 8)), dtype=int64)
        aflag = numpy.array(struct.unpack('%sd' % nacc, fin.read(nacc * 8)))
        fofflag = numpy.array(struct.unpack('%sd' % nacc, fin.read(nacc * 8)))
        rootid = numpy.array(struct.unpack('%sd' % nacc, fin.read(nacc * 8)), dtype=int64)
        pkmassid = numpy.array(struct.unpack('%sd' % nacc, fin.read(nacc * 8)), dtype=int64)
        nins = struct.unpack('i', fin.read(4))[0]
        idins = numpy.array(struct.unpack('%sd' % nins, fin.read(nins * 8)), dtype=int64)
        fin.close()
        
        first_prog = np.unique(pkmassid)
        stars_in_subhalo = np.zeros(len(first_prog))
        for i, pid in enumerate(first_prog):
            stars_in_subhalo[i] = sum(pkmassid == pid)  # stars_in_subhalo[i] = sum( rootid==pid )
        
        nsort = np.argsort(stars_in_subhalo)[::-1]
        stars_in_subhalo = stars_in_subhalo[nsort]
        first_prog = first_prog[nsort]
        
        print("first_prog=", first_prog)
        
        np_indices = {}
        np_snaps = {}
        np_reds = {}
        nsub = panels
        print('outputlistfile[d]=', outputlistfile[d], snap)
        treebase = 'trees_sf1_%03d' % snap
        treepath = '%s/mergertrees/Au-%s/%s' % (dirs[d], runs[d].split('_')[1], treebase)
        print('treepath=', treepath)
        t = load_tree(0, 0, base=treepath)
        
        snap_numbers, redshifts, subfind_indices, tree_indices, ff_tree_indices, fof_indices = t.return_subhalo_first_progenitors(0)
        
        subhalo0_pos = t.data['spos'][tree_indices] / (1. + redshifts[:, None]) / HubbleParam
        subhalo0_vel = t.data['svel'][tree_indices] * np.sqrt((1. / (1. + redshifts[:, None])))
        subhalo0_snum = t.data['snum'][tree_indices]
        subhalo0_mass = t.data['smty'][tree_indices][:].sum(axis=1) / HubbleParam
        subhalo0_smass = t.data['smty'][tree_indices][:, 4] / HubbleParam
        subhalo0_gmass = t.data['smty'][tree_indices][:, 0] / HubbleParam
        subhalo0_gmet = t.data['sgmt'][tree_indices]
        
        for i in range(0, nsub):
            indlist = []
            index = first_prog[i]
            
            while t.data['desc'][index] != -1:
                indlist.append(index)
                index = t.data['desc'][index]
            
            indlist = np.array(indlist)
            indlist = indlist[~np.in1d(indlist, tree_indices)]
            
            if len(indlist):
                index = indlist[-1]
            
            ## if main prog of main halo, go to next one
            if index == 0:
                nsub += 1
                continue
            
            np_indices[i] = np.array([], dtype=int)
            np_snaps[i] = np.array([], dtype=int)
            np_reds[i] = np.array([], dtype=int)
            
            while index != -1:
                np_indices[i] = np.append(np_indices[i], index)
                np_snaps[i] = np.append(np_snaps[i], t.data['snum'][index])
                np_reds[i] = np.append(np_reds[i], t.data['reds'][index])
                index = t.data['fpin'][index]
            
            ind_bef_merge = np.in1d(np_indices[i], tree_indices)
        
        dd = dirs[d] + runs[d]
        
        pkid = []
        l = 0
        
        # lzvec
        age_select = 3.
        if runs[d] == 'halo_L4':
            loadonlytype = [1, 4]
        else:
            loadonlytype = [4]
        
        ldir = np.zeros((nsnap, 3))
        time = np.zeros(nsnap)
        snap0 = snaplist[0]
        
        for isnap, snap in enumerate(snaplist):
            
            print("Doing dir %s snap %d." % (dd, snap))
            
            s = gadget_readsnap(snap, snappath=dd + '/output/', hdf5=True, loadonly=['pos', 'vel', 'mass', 'age', 'id', 'sfr'],
                                loadonlytype=loadonlytype)
            sf = load_subfind(snap, dir=dd + '/output/', hdf5=True,
                              loadonly=['fpos', 'frc2', 'svel', 'flty', 'fnsh', 'slty', 'spos', 'smas', 'smty', 'ffsh'])
            s.calc_sf_indizes(sf)
            
            if 1 in loadonlytype:
                if isnap == 0:
                    # s.calc_sf_indizes( sf,dosubhalos=False, halolist=[0] )
                    xdir, ydir, zdir = s.select_halo(sf, age_select, use_principal_axis=True, use_cold_gas_spin=False, do_rotation=False)
                    idmb = s.get_most_bound_dm_particles()
                else:
                    centre = s.pos[np.where(np.in1d(s.id, [idmb[0]]) == True), :].ravel()
                    center = centre
                    centre = list(centre)
                    subdint = 99.
                    for i in range(len(sf.data['spos'])):
                        subd = np.sum((center - sf.data['spos'][i, :]) ** 2)
                        if subd < subdint:
                            subdint = subd
                    
                    xdir, ydir, zdir = s.select_halo(sf, age_select, centre=centre, use_principal_axis=True, use_cold_gas_spin=False,
                                                     do_rotation=False)
            
            else:
                centre = subhalo0_pos[isnap]
                
                center = centre
                centre = list(centre)
                xdir, ydir, zdir = s.select_halo(sf, age_select, centre=centre, use_principal_axis=True, use_cold_gas_spin=False, do_rotation=False)
            
            na = s.nparticlesall
            
            time[isnap] = s.cosmology_get_lookback_time_from_a(s.time.astype('float64'), is_flat=True)
            
            galrad = 0.1 * sf.data['frc2'][0]
            
            rot = np.array([xdir, ydir, zdir])
            
            st = na[:4].sum();
            en = st + na[4]
            age = np.zeros(s.npartall)
            age[st:en] = s.data['age']
            
            iall, = np.where((s.r() < galrad) & (s.r() > 0.))
            istars, = np.where((s.r() < galrad) & (s.r() > 0.) & (s.type == 4) & (age > 0.))
            
            pos = s.pos[iall, :].astype('float64')
            vel = s.vel[iall, :].astype('float64')
            mass = s.data['mass'][iall].astype('float64')
            
            lang = np.cross(s.pos[istars, :], (s.vel[istars, :] * s.data['mass'][istars][:, None]))
            ltot = lang.sum(axis=0)
            ldir[isnap] = ltot / sqrt((ltot ** 2).sum())
        
        for j in np_indices:
            
            cosangle = []
            dist = []
            msub = []
            smsub = []
            gmsub = []
            smassrat = []
            gmassrat = []
            gmetsub = []
            massmain = []
            gmetmain = []
            
            nfail = 0
            for i, treei in enumerate(np_indices[j]):
                
                sub0i = where(np.array(subhalo0_snum) == np.array(np_snaps[j][i]))
                
                try:
                    sub0pos = subhalo0_pos[sub0i][0]
                    sub0vel = subhalo0_vel[sub0i][0]
                    sub0mass = subhalo0_mass[sub0i][0]
                    sub0smass = subhalo0_smass[sub0i][0]
                    sub0gmass = subhalo0_gmass[sub0i][0]
                    sub0gmet = subhalo0_gmet[sub0i][0]
                except:
                    pass
                
                msub.append(np.log10(t.data['smty'][treei][:].sum() * 1e10 / HubbleParam))
                gmsub.append(np.log10(t.data['smty'][treei][0] * 1e10 / HubbleParam))
                
                smsub.append(np.log10(t.data['smty'][treei][4] * 1e10 / HubbleParam))
                
                gmetsub.append(t.data['sgmt'][treei])
                
                # zdist = (1e3 / HubbleParam) * (sub0pos[0] - t.data['spos'][treei][0])
                # ydist = (1e3 / HubbleParam) * (sub0pos[1] - t.data['spos'][treei][1])
                # xdist = (1e3 / HubbleParam) * (sub0pos[2] - t.data['spos'][treei][2])
                
                zdist = 1e3 * (sub0pos[0] - t.data['spos'][treei][0] / (1. + t.data['reds'][treei]) / HubbleParam)
                ydist = 1e3 * (sub0pos[1] - t.data['spos'][treei][1] / (1. + t.data['reds'][treei]) / HubbleParam)
                xdist = 1e3 * (sub0pos[2] - t.data['spos'][treei][2] / (1. + t.data['reds'][treei]) / HubbleParam)
                
                pos = np.array([zdist, ydist, xdist])
                
                smassrat.append((t.data['smty'][treei][4]) / sub0smass)
                gmassrat.append((t.data['smty'][treei][0]) / sub0gmass)
                massmain.append(sub0mass)
                
                gmetmain.append(sub0gmet)
                
                dist.append((xdist ** 2. + ydist ** 2. + zdist ** 2.) ** 0.5)
                
                zvel = (sub0vel[0] - t.data['svel'][treei][0] * np.sqrt((1. / (1. + t.data['reds'][treei]))))
                yvel = (sub0vel[1] - t.data['svel'][treei][1] * np.sqrt((1. / (1. + t.data['reds'][treei]))))
                xvel = (sub0vel[2] - t.data['svel'][treei][2] * np.sqrt((1. / (1. + t.data['reds'][treei]))))
                
                vel = np.array([zvel, yvel, xvel]) / HubbleParam
                
                reds = np_reds[j]
                a = 1. / (1. + reds)
                times = sat.return_lookbacktime_from_a(a)
                
                lsat = np.cross(pos, vel)
                
                ltot = np.sqrt((lsat ** 2).sum())
                lsdir = lsat / ltot
                
                ii = where(snaplist == np_snaps[j][i])
                
                try:
                    cosangle.append(np.dot(ldir[ii], lsdir[:])[0])
                except:
                    cosangle.append(0.)
            
            distance = np.array(dist)
            alpha = np.array(cosangle)
            submass = np.array(msub)
            subsmass = np.array(smsub)
            subgmass = np.array(gmsub)
            smassrat = np.array(smassrat)
            gmassrat = np.array(gmassrat)
            gmetsub = np.array(gmetsub)
            times = np.array(times)
            gmetmain = np.array(gmetmain)
            massmain = np.array(massmain)
            
            if (distance[0] < 180. and np_indices[j][0] not in pkid):
                perind = np.zeros(len(times))
                for k in range(1, len(times) - (nfail + 1)):
                    if (distance[k] < distance[k - 1]) & (distance[k] < distance[k + 1]) & (distance[k] < 1e3):
                        perind[k] = 1
                        # fly-by
                        line = "%01d%10d%12.2f%12.3f%12.3f%12.3f%12.3f%12.3f%12.3f%12.3f%12.3f%12.3f%12.3f%12.3f%12.3f%10d\n" % (
                        0, np_indices[j][0], times[k], submass.max(), subsmass.max(), subgmass[k], smassrat[subsmass == subsmass.max()],
                        gmassrat[subsmass == subsmass.max()], submass[k], smassrat[k], gmassrat[k], distance[k], gmetsub[subsmass == subsmass.max()],
                        gmetsub[subsmass == subsmass.max()] / gmetmain[subsmass == subsmass.max()], gmetsub[k] / gmetmain[k], first_prog[j])
                        f1.write(line)
                
                # "time", "peaksmass", "peakgmass", "peaksrat", "peakgrat", "smassrat", "gmassrat", "dist", "peakgmet", "peakgmetrat", "gmetrat"
                if times[0] > 1e-5:  # merger?
                    print('np_indices[j][0]=', np_indices[j][0])
                    print('times[0]=', times[0])
                    print('submass=', submass)
                    line = "%01d%10d%12.2f%12.3f%12.3f%12.3f%12.3f%12.3f%12.3f%12.3f%12.3f%12.3f%12.3f%12.3f%12.3f%10d\n" % (
                    1, np_indices[j][0], times[0], submass.max(), subsmass.max(), subgmass.max(), smassrat[subsmass == subsmass.max()],
                    gmassrat[subsmass == subsmass.max()], submass[0], smassrat[0], gmassrat[0], distance[0], gmetsub[subsmass == subsmass.max()],
                    gmetsub[subsmass == subsmass.max()] / gmetmain[subsmass == subsmass.max()], gmetsub[0] / gmetmain[0], first_prog[j])
                    f1.write(line)
                
                ax3 = figure2.axes[l]
                ax3.plot(times, alpha, lw=1., color='k')
                
                ar2 = np.around(np.linspace(-0.2, 1.2, 8), 1)
                ax = figure.axes[l]
                ax2 = figure.twinyaxes[l]
                ax2.set_ylim([ar2[0], ar2[-1]])
                ax2.set_yticks(abs(ar2[1:]))
                ax2.set_yticklabels(abs(ar2[1:]))
                
                ax2.plot(times, subsmass / subsmass.max(), lw=1., color='b')
                ax2.plot(times, subgmass / subgmass.max(), lw=1., color='k')
                ax.semilogy(times, distance, lw=1., color='g')
                ax2.plot(times[gmetsub > 0.], gmetsub[gmetsub > 0.] / gmetmain[gmetsub > 0.], lw=1., color='r', linestyle='--')
                
                ax.text(0.05, 0.15, r'$\rm{M_{*}}=%.1f$' % subsmass.max(), color='b', transform=ax.transAxes, fontsize=5)
                ax.text(0.35, 0.15, r'$\rm{M_{rat}}=%.2f$' % smassrat[subsmass == subsmass.max()], color='b', transform=ax.transAxes, fontsize=5)
                ax.text(0.7, 0.15, r'$\rm{M_{g}}=%.1f$' % gmassrat[subsmass == subsmass.max()], color='k', transform=ax.transAxes, fontsize=5)
                ax.text(0.05, 0.05, r'$\rm{t_{m}}=%.1f$' % times[0], color='g', transform=ax.transAxes, fontsize=5)
                ax.text(0.35, 0.05, r'$\rm{f_{Z}}=%.1f$' % (gmetsub[subsmass == subsmass.max()] / gmetmain[subsmass == subsmass.max()]), color='r',
                        transform=ax.transAxes, fontsize=5)
                
                ax.text(0.7, 0.9, r'$\rm{id=%02d}$' % first_prog[j], color='k', transform=ax.transAxes, fontsize=5)
                
                l += 1
                pkid.append(np_indices[j][0])
        
        # bounds = [9., 9.5, 10., 10.5, 11.]
        # norm = mpl.colors.Normalize(vmin=9.,vmax=11.)
        # cb = mpl.colorbar.ColorbarBase( cax, cmap=cmap, norm=norm, orientation='vertical', ticks=bounds )
        # cb.set_label('$\\rm{log_{10} [M/M_{\odot}]}$', fontsize=25)
        # cax.xaxis.set_label_position('top')
        # cax.xaxis.set_ticks_position('top')
        # cax.xaxis.set_tick_params(labelsize=25)
        
        # text( 0.8, 0.1, "$\\rm{Au \,%s}$" % (runs[d].split('_')[1]), color='k', transform=ax.transAxes, fontsize=25 )
        
        figure.fig.savefig('%s/mainsubhalo_dist-vs-time_%s.%s' % (outpath + runs[d], runs[d], suffix), dpi=300)
        figure2.fig.savefig('%s/mainsubhalo_alpha-vs-time_%s.%s' % (outpath + runs[d], runs[d], suffix), dpi=300)
        f1.close()