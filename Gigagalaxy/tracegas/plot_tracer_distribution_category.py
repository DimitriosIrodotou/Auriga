import matplotlib

matplotlib.use('Agg')
from loadmodules import *
import matplotlib.pyplot as plt
from parallel_decorators import vectorize_parallel

toinch = 0.393700787


def plot_tracer_distribution_category(dirs, runs, snaplist, name, outpath, outputlistfile, suffix='', targetgasmass=False):
    for d in range(len(runs)):
        run = runs[d]
        dir = dirs[d]
        
        path = '%s/%s%s/' % (dir, run, suffix)
        
        wpath = '%s/%s/tracers/' % (outpath, run)
        if not os.path.exists(wpath):
            os.makedirs(wpath)
        wpath = str(wpath)
        
        fname = '%s/nrecycle_tracers_sf.txt' % (wpath)
        fout = open(fname, 'w')
        
        # a = np.array([0., 3., 4., 2., 5.])
        # b = a
        # c = a
        # np.savetxt( fout, np.column_stack((a,b,c)) )
        
        loadonlytype = [0, 4, 6]
        sidmallnow = []
        
        nb = 0
        nbt = 0
        
        # ctype_evo = np.zeros((size(tridt),(len(snaplist))))
        # prid_evo = np.zeros((size(tridt),(len(snaplist))))
        
        lbtime = np.zeros(len(snaplist) - 1)
        tlist = np.arange(len(snaplist) - 1)
        print("len lbtime,tlist=", len(lbtime), len(tlist))
        # now look for them in previous snapshots
        for isnap, snap in enumerate(snaplist[:-1]):
            if runs[d] == 'halo_L5' or runs[d] == 'halo_L6':
                fname = '%s/tracers_stars_snap%d_dmcen.dat' % (wpath, snap)
            else:
                fname = '%s/tracers_stars_snap%d.dat' % (wpath, snap)
            
            print("Reading %s" % fname)
            fin = open(fname, 'rb')
            time = struct.unpack('d', fin.read(8))[0]
            rvir = struct.unpack('d', fin.read(8))[0]
            ntracers = struct.unpack('i', fin.read(4))[0]
            trid_all = numpy.array(struct.unpack('%sd' % ntracers, fin.read(ntracers * 8)), dtype=int64)
            ttype = numpy.array(struct.unpack('%si' % ntracers, fin.read(ntracers * 4)), dtype=int32)
            ctype = numpy.array(struct.unpack('%si' % ntracers, fin.read(ntracers * 4)), dtype=int32)
            # idsub_all = numpy.array(struct.unpack('%si' % ntracers, fin.read(ntracers*4)), dtype=int32)
            prid_all = numpy.array(struct.unpack('%sd' % ntracers, fin.read(ntracers * 8)), dtype=int64)
            rad_all = numpy.array(struct.unpack('%sf' % ntracers, fin.read(ntracers * 4)), dtype=float64)
            height_all = numpy.array(struct.unpack('%sf' % ntracers, fin.read(ntracers * 4)), dtype=float64)
            lz_all = numpy.array(struct.unpack('%sf' % ntracers, fin.read(ntracers * 4)), dtype=float64)
            age_all = numpy.array(struct.unpack('%sf' % ntracers, fin.read(ntracers * 4)), dtype=float64)
            metallicity_all = numpy.array(struct.unpack('%sf' % ntracers, fin.read(ntracers * 4)), dtype=float64)
            # bflag = numpy.array(struct.unpack('%si' % ntracers, fin.read(ntracers*4)), dtype=int32)
            fin.close()
            
            print("ntracers=", ntracers)
            
            if isnap == 0:
                ctype_evo = np.zeros((ntracers, (len(snaplist))))
                prid_evo = np.zeros((ntracers, (len(snaplist))))
                ttype_evo = np.zeros((ntracers, (len(snaplist))))
            
            ctype_evo[:, isnap] = ctype
            prid_evo[:, isnap] = prid_all
            ttype_evo[:, isnap] = ttype
            lbtime[isnap] = time
            
            if snap == 127:
                age = age_all
                rad = rad_all * rvir
                prid_now = prid_all
                met = metallicity_all
                lz = lz_all
                print("prid_now,age=", prid_now, age)
        
        # needs to be made consistent from here onwards
        # f = open(outputlistfile[d], 'r')
        # expansion_fact = np.loadtxt(f, usecols=0)
        # fexp = path + '/expfac_snapshotlist_%s.txt' % run
        fexp = '/u/rgrand/expfaclists/' + 'expfac_snapshotlist_%s.txt' % run
        expansion_fact = np.loadtxt(fexp)
        expansion_fact = expansion_fact[::-1]
        print("expansion_fact=", expansion_fact)
        
        psort = np.argsort(prid_now)
        prid_now = prid_now[psort]
        prid_evo = prid_evo[psort, :]
        ctype_evo = ctype_evo[psort, :]
        ttype_evo = ttype_evo[psort, :]
        age = age[psort]
        rad = rad[psort]
        met = met[psort]
        lz = lz[psort]
        
        istars, = np.where((age > 0.) & (rad < 0.03))
        print("star age=", age[istars])
        
        star_age = age[istars]
        star_rad = rad[istars]
        star_met = met[istars]
        star_prid = prid_now[istars]
        star_lz = lz[istars]
        
        prid_unq, unq_idx = np.unique(star_prid, return_index=True)
        npart = size(prid_unq)
        
        print("max prid index=", len(prid_now), unq_idx.max())
        print("star_age<0.=", np.where(star_age < 0.), star_age)
        print("star_age[unq_idx]<0.=", np.where(star_age[unq_idx] < 0.), star_age[unq_idx])
        print("star_age[unq_idx] min max=", star_age[unq_idx].min(), star_age[unq_idx].max())
        
        print("age,expansion_fact=", age, expansion_fact)
        nind = np.digitize(age, bins=expansion_fact)
        print("age where not -1=", age[age != -1.])
        nind = np.minimum(nind, [len(snaplist) - 1] * len(nind))
        nind -= 1
        nind = np.maximum(nind, [0] * len(nind))
        print("nind=", nind[age != -1.])
        
        recycled = np.zeros(npart)
        recycled2 = np.zeros(npart)
        
        print("len unq_idx=", len(unq_idx), len(age))
        
        star_rad = star_rad[unq_idx]
        star_met = star_met[unq_idx]
        star_prid = star_prid[unq_idx]
        star_lz = star_lz[unq_idx]
        
        # path = '/hits/universe/GigaGalaxy/level4_MHD/halo_6/'
        s = gadget_readsnap(127, snappath=path + '/output/', loadonlytype=[4], loadonly=['pos', 'vel', 'mass'], hdf5=True, forcesingleprec=True)
        page = s.cosmology_get_lookback_time_from_a(star_age[unq_idx].astype('float64'), is_flat=True)
        print("page=", page)
        
        print("%d particles selected" % len(prid_unq))
        
        recycled = calculate_nrecycle(prid_unq, prid_now, prid_evo, nind, ctype_evo)
        recycled = np.array(recycled)
        for i in range(-1, 6):
            print("recycled %d times: %d" % (i, np.size(np.where(recycled == i))))
        
        recycled[recycled == -1] = 0
        
        print("len star rad, age, met, recycled=", len(star_rad), len(page), len(star_met), len(recycled))
        
        np.savetxt(fout, np.column_stack((star_lz, page, star_met, star_prid, recycled)))
        fout.close()
        
        for j in range(10):
            print("number of elements recycled %d times by SNII= %d" % (j, size(np.where(recycled == j))))
            # print "number of elements recycled %d times by SNIa and AGB stars= %d"%(j,size(np.where( recycled2==j )))
            
            indy, = np.where((recycled == j))  # & (rad > 0.0015))# & (rad < 0.001) )
            # sage = age[indy]
            sage = page[indy]
            
            print("mean age=", sage.mean(), np.median(sage))
        
        ageup = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]
        agelo = [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]
        
        # ageup = [2., 4., 6., 8.]
        # agelo = [0., 2., 4., 6.]
        
        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 4))
        ax = axes.flatten()
        for i in range(len(ageup)):
            jj, = np.where((page < ageup[i]) & (page > agelo[i]))
            ax[i].hist(recycled[jj], bins=10, range=[0, 10], label='%1.0f' % agelo[i])
            
            ax[i].legend(frameon=False, prop={'size': 5})
        plt.savefig('./recycled_test_sf.pdf')


@vectorize_parallel(method='processes', num_procs=24)
def calculate_nrecycle(pid, prid_now, prid_evo, nind, ctype_evo):
    recycled = -1
    
    index_forage = np.where(np.in1d(prid_now, pid) == True)
    ageindex = nind[index_forage]
    # print "ageindex=",ageindex
    trid_indx_for_part = np.ravel(np.where(np.in1d(prid_evo[:, ageindex[0]], pid) == True))
    # print "trid_indx_for_part=",trid_indx_for_part
    for i in range(len(trid_indx_for_part)):
        cindex = trid_indx_for_part[i]
        # print "cindex=",cindex
        # nrecsn2 = np.size( np.where( (ctype_evo[cindex,ageindex[0]:] == 23) | (ctype_evo[cindex,ageindex[0]:] == 13) | (ctype_evo[cindex,
        # ageindex[0]:] == 43 ) ) )
        # print "nrecsn2=",nrecsn2
        nrecsn2 = np.size(
            np.where((ctype_evo[cindex, ageindex[0]:] == 21) | (ctype_evo[cindex, ageindex[0]:] == 23) | (ctype_evo[cindex, ageindex[0]:] == 24)))
        recycled += nrecsn2
    # print "recycled tot=",recycled
    
    return recycled


def find_nearest(array, value):
    if len(value) == 1:
        idx = (np.abs(array - value)).argmin()
    else:
        idx = np.zeros(len(value))
        for i in range(len(value)):
            idx[i] = (np.abs(array - value[i])).argmin()
    
    return idx