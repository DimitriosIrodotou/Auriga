import matplotlib

matplotlib.use('Agg')
from loadmodules import *

toinch = 0.393700787


def trace_allstellar_tracers(dirs, runs, snaplist, name, outpath, outputlistfile, suffix='', boxsize=0.05, loadonlytype=[], rcut=0.005, numthreads=1,
                             targetgasmass=False, accreted=False, fhcen=False, agerange=[0., 13.]):
    for d in range(len(runs)):
        run = runs[d]
        dir = dirs[d]
        
        path = '%s/%s%s/' % (dir, run, suffix)
        
        agelo = agerange[0]
        ageup = agerange[1]
        wpath = '%s/%s/tracers/' % (outpath, run)
        if not os.path.exists(wpath):
            os.makedirs(wpath)
        wpath = str(wpath)
        
        if runs[d] == 'halo_L6':  # or runs[d] == 'halo_L5':
            loadonlytype = [0, 1, 4, 6]
        else:
            loadonlytype = [0, 4, 6]
        sidmallnow = []
        
        nb = 0
        nbt = 0
        
        fexp = '/u/rgrand/expfaclists/' + 'expfac_snapshotlist_%s.txt' % run
        expansion_fact = np.loadtxt(fexp)
        
        snap0 = snaplist[0]
        print('snap0=', snap0)
        
        # if 1 not in loadonlytype:
        treebase = 'trees_sf1_%03d' % snap0
        treepath = '%s/mergertrees/Au-%s/%s' % (dirs[d], runs[d].split('_')[1], treebase)
        print("treepath=", treepath)
        t = load_tree(0, 0, base=treepath)
        snap_numbers_main, redshifts_main, subfind_indices_main, first_progs_indices_main, ff_tree_indices_main, fof_indices_main, prog_mass_main, \
        next_prog_indices = t.return_first_next_mass_progenitors(
            0)
        print("fof_indices_main,subfind_indices_main=", fof_indices_main, subfind_indices_main)
        
        snapfirst = snap0
        snaplist = np.arange(array(snaplist).min(), snapfirst + 1)
        snaplist = snaplist[::-1]
        
        s = gadget_readsnap(snapfirst, snappath=path + '/output/', loadonlytype=loadonlytype,
                            loadonly=['pos', 'vel', 'mass', 'age', 'gsph', 'id', 'trid', 'prid', 'hrgm'], hdf5=True, forcesingleprec=True)
        sf = load_subfind(snapfirst, dir=path + '/output/', hdf5=True,
                          loadonly=['fpos', 'slty', 'frc2', 'svel', 'sidm', 'smty', 'spos', 'fnsh', 'flty'], forcesingleprec=True)
        s.calc_sf_indizes(sf)
        subhalostarmass = sf.data['smty'][0:sf.data['fnsh'][0], 4]
        
        s.select_halo(sf, 3., use_principal_axis=True, use_cold_gas_spin=False, do_rotation=True)
        if 1 in loadonlytype:
            idmb = s.get_most_bound_dm_particles()
        
        time = s.cosmology_get_lookback_time_from_a(s.time.astype('float64'))
        
        prid = s.data['prid'].astype('int64')
        trid = s.data['trid'].astype('int64')
        ptype = s.data['type'].astype('int64')
        
        # age = s.data['age']
        ii, = np.where((s.data['age'] >= 0.))
        star_age = -np.ones(len(s.data['age']))
        star_age[ii] = s.cosmology_get_lookback_time_from_a(s.data['age'][ii].astype('float64'), is_flat=True)
        
        st = s.nparticlesall[:4].sum()
        en = st + s.nparticlesall[4]
        afactor = np.zeros(s.npartall)
        afactor[st:en] = s.data['age']
        s.data['age'] = np.zeros(s.npartall)
        s.data['age'][st:en] = star_age
        
        # if isnap == 0:
        i, = np.where(
            (s.mass < 2. * targetgasmass / s.hubbleparam) & (s.type == 4) & (s.r() < 0.03) & (s.data['age'] > agelo) & (s.data['age'] < ageup))
        s.data['age'][st:en] = afactor[st:en]
        
        rad3d = np.sqrt((s.pos[i, :] ** 2).sum(axis=1))
        ids = s.data['id'][i].astype('int64')
        tage = s.data['age'][i]
        afactor = afactor[i]
        indy, = np.where(np.in1d(prid, ids) == True)
        pridt = prid[indy]
        tridt = trid[indy]
        bornflag = [1] * size(pridt)
        
        """
        # get list of tracers in gas
        i, = np.where( (s.mass < 2.*targetgasmass / s.hubbleparam) & (s.type==0) & (s.r() < sf.data['frc2'][0]) )
        idg = s.data['id'][i].astype('int64')
        indy, = np.where(np.in1d( prid, idg )==True)
        pridt = np.hstack( (pridt,prid[indy]) )
        tridt = np.hstack( (tridt,trid[indy]) )
        bornflag = np.hstack ( (bornflag, [-1]*size(indy)) )
        """
        print("Looking for %d tracers" % size(pridt))
        print("looking for %d tracers in total" % (size(tridt)))
        
        tsort = np.argsort(tridt)
        tridt = tridt[tsort]
        pridt = pridt[tsort]
        
        # now look for them in previous snapshots
        for isnap, snap in enumerate(snaplist):
            print('doing snap=', snap)
            if 1 in loadonlytype:
                fname = '%s/tracers_stars_snap%d_dmcen.dat' % (wpath, snap)
            else:
                fname = '%s/tracers_stars_snap%d.dat' % (wpath, snap)
            fout = open(fname, 'wb')
            
            s = gadget_readsnap(snap, snappath=path + '/output/', loadonlytype=loadonlytype,
                                loadonly=['pos', 'vel', 'mass', 'age', 'gsph', 'id', 'trid', 'prid', 'hrgm', 'sfr', 'gz'], hdf5=True,
                                forcesingleprec=True)
            sf = load_subfind(snap, dir=path + '/output/', hdf5=True,
                              loadonly=['fpos', 'slty', 'frc2', 'svel', 'sidm', 'smty', 'spos', 'fnsh', 'flty', 'slty', 'ffsh'], forcesingleprec=True)
            
            if 1 in loadonlytype:
                # idmbcentres = s.pos[np.where(np.in1d(s.id,idmb)==True),:]
                # idmbcentres = np.reshape(idmbcentres.ravel(), (len(idmb),3))
                # print('idmbcentres=',idmbcentres)
                # centre = idmbcentres[35,:] # 35 for halo_L6
                centre = s.pos[np.where(np.in1d(s.id, [idmb[0]]) == True), :].ravel()
                center = centre
                centre = list(centre)
                s.calc_sf_indizes(sf)
                s.select_halo(sf, centre=centre, use_principal_axis=True, use_cold_gas_spin=False, do_rotation=True)
                
                idm = s.data['id'][s.type == 1].astype('int64')
                last_dm = 0
                fof0 = 0
                sub0 = 0
                
                for ll in range(len(sf.data['fnsh'])):
                    offsetdm = sf.data['flty'][:ll, 1].sum()
                    first_dm = offsetdm
                    nsub_thisfof = sf.data['fnsh'][ll]
                    for l in range(nsub_thisfof):
                        subind = l + sf.data['fnsh'][:ll].sum()
                        last_dm = first_dm + sf.data['slty'][subind, 1]
                        idtmp = idm[np.int_(first_dm):np.int_(last_dm)]
                        index, = np.where(np.in1d(idtmp, idmb[0]) == True)
                        first_dm = last_dm
                        
                        if np.size(index):
                            print('index=', index)
                            fof0 = ll
                            sub0 = subind  # l
                            break
                    if np.size(index):
                        break
                
                print('fof0 and sub0 selected from MostBoundDM: ', fof0, sub0)
            
            else:
                # try:
                fof0 = fof_indices_main[snap0 - snap]
                sub0 = subfind_indices_main[snap0 - snap]
                # if sub0 > sf.data['fnsh'][fof0]:
                #        sub0 = 0
                #        fof0 = 0
                #        print("<<Warning: we have just set fof and sub number manually to zero. Take care.")
                # except:
                #        print('<<< Warning: setting fof and sub number manually to snap0 values.')
                #        fof0 = fof_indices_main[0]
                #        sub0 = subfind_indices_main[0]
                
                print('fof0,sub0=', fof0, sub0)
                # shind = sf.data['fnsh'][:fof0].sum() + sub0
                center = sf.data['spos'][sub0, :]
                shind = sub0 - sf.data['fnsh'][:fof0].sum()
                s.calc_sf_indizes(sf)
                s.select_halo(sf, 3., use_principal_axis=True, use_cold_gas_spin=False, do_rotation=True, haloid=fof0, subhalo=shind)
            
            loadedpart = np.zeros(7, dtype=np.int64)
            for type in loadonlytype:
                loadedpart[type] = s.nparticlesall[type]
            
            time = s.cosmology_get_lookback_time_from_a(s.time.astype('float64'))
            rvir = sf.data['frc2'][0]
            
            istar, = np.where((s.type == 4))
            igas, = np.where((s.type == 0))
            
            ids = s.data['id'][istar].astype('int64')
            idg = s.data['id'][igas].astype('int64')
            idbh = s.data['id'][s.type == 5].astype('int64')
            prid = s.data['prid'].astype('int64')
            trid = s.data['trid'].astype('int64')
            ptype = s.data['type'].astype('int64')
            
            # set up age array
            age = np.zeros(s.npartall)
            st = s.nparticlesall[:4].sum()
            en = st + s.nparticlesall[4]
            age[st:en] = s.data['age']
            
            # set up metallicity array
            metallicity = np.zeros(s.npartall)
            metallicity[:s.nparticlesall[0]] = s.data['gz'][:s.nparticlesall[0]]
            metallicity[s.nparticlesall[:4].sum():s.nparticlesall[:4].sum() + s.nparticlesall[4]] = s.data['gz'][
                                                                                                    s.nparticlesall[0]:s.nparticlesall[0] +
                                                                                                                       s.nparticlesall[4]]
            
            mass_variable = s.data['mass'].astype('float64')
            mass = s.data['mass'].astype('float64')
            
            pos = s.pos
            vel = s.vel
            
            # total stellar mass
            # get tracer parent ID
            trind, = np.where(np.in1d(trid, tridt) == True)
            TracerID = trid[trind]
            ParentID = prid[trind]
            
            psort = np.argsort(ParentID)
            TracerID = TracerID[psort]
            ParentID = ParentID[psort]
            
            i, = np.where((s.type == 4))
            
            # Find which of our tracers are in stars
            pindex, = np.where(np.in1d(ParentID, ids) == True)
            trid_star = TracerID[pindex]
            
            unq, unq_ind, unq_idx, unq_cnt = np.unique(ParentID[pindex], return_index=True, return_inverse=True, return_counts=True)
            print("unq, unq_ind, unq_idx, unq_cnt=", unq, unq_ind, unq_idx, unq_cnt)
            
            pindy, = np.where(np.in1d(ids, unq) == True)
            pindy = pindy[np.argsort(ids[pindy])]
            psind = np.repeat(pindy, unq_cnt)
            
            spos = pos[i, :]
            svel = vel[i, :]
            jz = np.cross(spos, svel)[:, 0]
            srad = np.sqrt((spos ** 2).sum(axis=1))  # / rvir
            smass = s.mass[i].astype('float64')
            sage = age[i]
            smetallicity = metallicity[i]
            
            tspos = spos[psind, :]
            sage = sage[psind]
            prid_star = ids[psind]
            srad = srad[psind]
            sheight = spos[psind, 0]
            smetallicity = smetallicity[psind]
            jz = jz[psind]
            htype_star = -9. * np.ones(len(trid_star))
            
            # gas
            i, = np.where((s.type == 0))
            idg = s.data['id'][i].astype('int64')
            
            pindex, = np.where(np.in1d(ParentID, idg) == True)
            trid_gas = TracerID[pindex]
            unq, unq_ind, unq_idx, unq_cnt = np.unique(ParentID[pindex], return_index=True, return_inverse=True, return_counts=True)
            
            sfr = s.data['sfr'][i]
            gpos = pos[i, :]
            gvel = vel[i, :]
            gjz = np.cross(gpos, gvel)[:, 0]
            grad = np.sqrt((gpos ** 2).sum(axis=1))  # / rvir
            gmass = s.mass[i].astype('float64')  # mass[i]
            gmetallicity = metallicity[i]
            
            gindy, = np.where(np.in1d(idg, unq) == True)
            gindy = gindy[np.argsort(idg[gindy])]
            if size(gindy):
                gind = np.repeat(gindy, unq_cnt)
                tgpos = gpos[gind, :]
                gjz = gjz[gind]
                tgmass = gmass[gind]
                sfr = sfr[gind]
                prid_gas = idg[gind]
                grad = grad[gind]
                gheight = gpos[gind, 0]
                gmetallicity = gmetallicity[gind]
                
                # halo type
                htype_gas = np.zeros(len(trid_gas))  # 0 for unbound
                
                # mainsub_index = sub0 + sf.data['fnsh'][:fof0].sum()
                # mainsub_first = sf.data['flty'][:fof0,0].sum() + sf.data['slty'][:mainsub_index,0].sum()
                # mainsub_last = mainsub_first + sf.data['slty'][mainsub_index,0]
                # print('first and last index of main subhalo=',mainsub_first,mainsub_last)
                print('fof0,sub0=', fof0, sub0)
                last_gas = 0
                for ll in range(len(sf.data['fnsh'])):
                    offset = sf.data['flty'][:ll, 0].sum()
                    first_gas = offset
                    nsub_thisfof = sf.data['fnsh'][ll]
                    for l in range(nsub_thisfof):
                        subind = l + sf.data['fnsh'][:ll].sum()
                        last_gas = first_gas + sf.data['slty'][subind, 0]
                        idtmp = s.data['id'][first_gas:last_gas]
                        if ll == fof0 and subind == sub0:
                            index, = np.where(np.in1d(prid_gas, idtmp) == True)
                            index0 = index
                            # print('index (fofsub0)=',index,len(index))
                            if np.size(index):
                                htype_gas[index] = -1  # in the main halo
                        else:
                            index, = np.where(np.in1d(prid_gas, idtmp) == True)
                            # print('index(sub)=',index,len(index))
                            if np.size(index):
                                htype_gas[index] = 1  # in a sub - halo
                        
                        first_gas = last_gas
                print('indices of main halo elements=', htype_gas[index0], len(index0))
                
                # mainsub_ids = s.data['id'][mainsub_first:mainsub_last]  # hind_gas, = np.where(np.in1d(prid_gas,  # mainsub_ids)==True)  #
                # htype_gas[hind_gas] = -1 # in the main halo
            
            else:
                trid_gas = []
                prid_gas = []
                grad = []
                gjz = []
                gmetallicity = []
                gheight = []
                htype_gas = []
            
            # make tracer type (star, wind particle, SF gas...) array
            # do stars first
            stype = -np.ones(size(trid_star))
            stype[sage >= 0.] = 4  # star particle
            stype[sage < 0] = 3  # wind particle
            
            # do gas cells
            if size(gindy):
                gtype = -np.ones(size(trid_gas))
                gtype[sfr > 0.] = 2
                gtype[sfr <= 0.] = 1
                gage = -np.ones(size(trid_gas))
            else:
                gtype = []
                gage = []
            
            # stack everything
            htype = np.hstack((htype_star, htype_gas)).astype('int32')
            ttype = np.hstack((stype, gtype)).astype('int32')
            trid_all = np.hstack((trid_star, trid_gas)).astype('int64')
            # trid_all = TracerID
            prid_all = np.hstack((prid_star, prid_gas)).astype('int64')
            rad_all = np.hstack((srad, grad))
            
            height_all = np.hstack((sheight, gheight))
            lz_all = np.hstack((jz, gjz))
            metallicity_all = np.hstack((smetallicity, gmetallicity))
            
            age_all = np.hstack((sage, gage))
            
            # sort everything by tracer ID
            trsort = np.argsort(trid_all)
            
            trid_all = trid_all[trsort]
            prid_all = prid_all[trsort]
            
            htype = htype[trsort]
            ttype = ttype[trsort]
            rad_all = rad_all[trsort]
            height_all = height_all[trsort]
            lz_all = lz_all[trsort]
            metallicity_all = metallicity_all[trsort]
            age_all = age_all[trsort]
            print('rad<30 kpc=', np.size(np.where(rad_all < 0.03)))
            if isnap == 0:
                ttype_old = ttype
            
            k, = np.where((ttype != ttype_old))
            ctype = np.zeros(len(trsort), dtype=int)
            merge_ctype = np.core.defchararray.add(ttype_old.astype('str'), ttype.astype('str')).astype('int')
            ctype[k] = np.int_(merge_ctype[k])
            ttype_old = ttype
            
            print('htype=', htype, htype[htype != -9])
            
            print("rad first ten tracers=", rad_all[np.where(np.in1d(trid_all, tridt[:10]) == True)])
            print("prid first ten tracers=", prid_all[np.where(np.in1d(trid_all, tridt[:10]) == True)])
            print("num stars greater than 30 kpc=", size(np.where(rad_all[np.where(np.in1d(trid_all, tridt) == True)] > 0.03)),
                  rad_all[np.where(rad_all[np.where(np.in1d(trid_all, tridt) == True)] > 0.03)])
            print("type first ten tracers=", ttype[np.where(np.in1d(trid_all, tridt[:10]) == True)])
            print("tracer ID first ten tracers=", trid_all[np.where(np.in1d(trid_all, tridt[:10]) == True)])
            
            ntracers = len(trid_all)
            
            fout.write(struct.pack('d', time))
            fout.write(struct.pack('d', rvir))
            fout.write(struct.pack('i', ntracers))
            fout.write(struct.pack('%sd' % ntracers, *trid_all))
            fout.write(struct.pack('%si' % ntracers, *htype))
            fout.write(struct.pack('%si' % ntracers, *ttype))
            fout.write(struct.pack('%si' % ntracers, *ctype))
            # fout.write(struct.pack('%si' % ntracers, *idsub_all))
            fout.write(struct.pack('%sd' % ntracers, *prid_all))
            fout.write(struct.pack('%sf' % ntracers, *rad_all))
            fout.write(struct.pack('%sf' % ntracers, *height_all))
            fout.write(struct.pack('%sf' % ntracers, *lz_all))
            fout.write(struct.pack('%sf' % ntracers, *age_all))
            fout.write(struct.pack('%sf' % ntracers, *metallicity_all))
            # fout.write(struct.pack('%si' % ntracers, *bflag_all))
            fout.close()
            
            print("#####")
            print("time=", time)
            print("number of types 1,2,3,4=", size(np.where(ttype == 1)), size(np.where(ttype == 2)), size(np.where(ttype == 3)),
                  size(np.where(ttype == 4)))