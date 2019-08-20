from loadmodules import *
from util import get_halo_centres
from util import select_snapshot_number

toinch = 0.393700787


def get_rotation_vector(dir, halonum, snap, suffix='', loadonlytype=[]):
    # path = '%s/halo_%d.newSNIa/output/'% (dir, halonum)
    print("dir =", dir)
    path = '%s/halo_%d%s/' % (dir, halonum, suffix)
    print("path =", path)
    if not loadonlytype:
        loadonlytype = [i for i in range(6)]
    
    s = gadget_readsnap(snap, snappath=path + '/output/', loadonlytype=loadonlytype, loadonly=['pos', 'vel', 'mass'], hdf5=True, forcesingleprec=True)
    sf = load_subfind(snap, dir=path + '/output/', hdf5=True, verbose=False, loadonly=['fpos', 'frc2', 'svel', 'flty', 'fnsh', 'slty'],
                      forcesingleprec=True)
    s.calc_sf_indizes(sf)
    rot = s.select_halo(sf, use_principal_axis=True)
    
    del s
    del sf
    
    return rot


def project_stars(dir, halonum, rot, name, inputfile1, inputfile2, suffix='', boxsize=0.05, loadonlytype=[], numthreads=1, accreted=False,
                  wpath='../plots/projections/'):
    path = '%s/halo_%d%s/' % (dir, halonum, suffix)
    
    # if not os.path.exists(wpath):
    #       os.makedirs(wpath)
    
    res = 512
    pylab.figure(figsize=(22 * toinch, 22 * toinch), dpi=300)
    
    if not loadonlytype:
        loadonlytype = [i for i in range(6)]
    
    alist = [i for i in range(29)]
    snaplist = select_snapshot_number.match_expansion_factor_files(inputfile1, inputfile2, alist)
    
    if accreted == True:
        fnamein = '/home/grandrt/analysis/level4/halo_%d/sfr/halo_%dstarID_accreted.txt' % (halonum, halonum)
        insitu = np.loadtxt(fnamein, delimiter=None, dtype=int)
        idins = np.array(insitu.astype('int64'))
    
    print("snaplist last=", snaplist[-1])
    if halonum in [1, 11]:
        sidmallnow = get_halo_centres.get_bound_DMparticles(snaplist[-1], path, [0, 1, 4])
        loadptype = [1, 4]
    else:
        sidmallnow = []
        loadptype = [4]
    
    # for snap in range(5, 23):
    for snap in range(5, 29):
        
        s = gadget_readsnap(snaplist[snap], snappath=path + '/output/', loadonlytype=loadptype, loadonly=['pos', 'vel', 'mass', 'age', 'gsph', 'id'],
                            hdf5=True, forcesingleprec=True)
        print(s.time, s.redshift)
        sf = load_subfind(snaplist[snap], dir=path + '/output/', hdf5=True, loadonly=['fpos', 'frc2', 'svel', 'flty', 'fnsh', 'slty', 'spos'],
                          forcesingleprec=True)
        age_select = 3.
        s.select_halo(sf, list(sidmallnow), age_select, remove_bulk_vel=False, use_principal_axis=True, rotate_disk=False)
        # if halonum in [1, 11]:
        # jj, center = s.force_halo_centering( sf, sidmallnow )
        # s.center = sf.data['spos'][jj,:]
        
        s.rotateto(rot[0], dir2=rot[1], dir3=rot[2])
        
        time = s.cosmology_get_lookback_time_from_a(s.time.astype('float64'))
        
        dist = np.max(np.abs(s.pos - s.center[None, :]), axis=1)
        istars, = np.where((s.type == 4) & (dist < 0.5 * boxsize))
        
        # remove wind particles from stars ...
        first_star = 0
        for ptype in loadptype:
            if ptype < 4:
                first_star += s.nparticlesall[ptype]
        
        istars -= np.int_(first_star)
        # star_age = np.zeros(size(istars))
        # star_age[:] = s.cosmology_get_lookback_time_from_a( s.data['age'][istars], is_flat=True )
        j, = np.where((s.data['age'][istars] > 0.))
        # j, = np.where( (star_age > 0.) & ( (star_age - time) < 1. ) )
        jstars = istars[j] + np.int_(first_star)
        
        if accreted == True:
            ids = s.id[jstars].astype('int64')
            idlist_true = list(set(ids).intersection(idins))
            arrayf = np.arange(len(jstars))
            idict = dict(zip(ids, arrayf))
            cc = 0
            sind = []
            while cc < len(idlist_true):
                nnf = idict[idlist_true[cc]]
                sind.append(nnf)
                cc += 1
            asind = np.array(sind)
            
            temp_pos = s.pos[jstars, :].astype('float64')
            spos = np.zeros((size(jstars), 3))
            spos[:, 0] = temp_pos[:, 1]
            spos[:, 1] = temp_pos[:, 2]
            spos[:, 2] = temp_pos[:, 0]
            pos = np.zeros((size(asind), 3))
            pos[:, 0] = spos[asind, 0]
            pos[:, 1] = spos[asind, 1]
            pos[:, 2] = spos[asind, 2]
            
            smass = s.data['mass'][jstars].astype('float64')
            mass = smass[asind]
            j = asind
        else:
            
            temp_pos = s.pos[jstars, :].astype('float64')
            pos = np.zeros((size(jstars), 3))
            pos[:, 0] = temp_pos[:, 1]
            pos[:, 1] = temp_pos[:, 2]
            pos[:, 2] = temp_pos[:, 0]
            
            mass = s.data['mass'][jstars].astype('float64')
        
        tree = makeTree(pos)
        hsml = tree.calcHsmlMulti(pos, pos, mass, 48, numthreads=numthreads)
        hsml = np.minimum(hsml, 4. * boxsize / res)
        hsml = np.maximum(hsml, 1.001 * boxsize / res * 0.5)
        if accreted == True:
            rho = np.ones(size(j))
        else:
            rho = np.ones(size(jstars))
        
        datarange = np.array([[4003.36, 800672.], [199.370, 132913.], [133.698, 200548.]])
        # fac = (512./res)**2 * (0.5 * boxsize / (0.025 * s.hubbleparam))**2
        fac = (512. / res) ** 2 * (0.5 * boxsize / 0.025) ** 2
        datarange *= fac
        
        data = np.zeros((res, res, 3))
        for k in range(3):
            iband = [3, 1, 0][k]
            band = 10 ** (-2.0 * s.data['gsph'][j, iband].astype('float64') / 5.0)
            grid = calcGrid.calcGrid(pos, hsml, band, rho, rho, res, res, res, boxsize, boxsize, boxsize, s.center[0], s.center[1], s.center[2], 1, 1,
                                     numthreads=numthreads)
            
            drange = datarange[k]
            grid = np.minimum(np.maximum(grid, drange[0]), drange[1])
            
            loggrid = np.log10(grid)
            
            logdrange = np.log10(drange)
            print(loggrid.min(), loggrid.max(), logdrange)
            
            data[:, :, k] = (loggrid - logdrange[0]) / (logdrange[1] - logdrange[0])
            print(data[:, :, k].min(), data[:, :, k].max(), data[:, :, k].sum() / res ** 2)
        
        ix = (snap - 5) % 6
        iy = (snap - 5) / 6
        
        x = ix * (1. / 6.) + 1. / 6. * 0.05 * 0.5
        y = iy * (1. / 4.) + 1. / 6. * 0.05 * 0.5 + 1. / 12.
        
        ax = axes([x, y, 1 / 6. * 0.95, 1. / 6. * 0.95], frameon=False)
        imshow(data, interpolation='nearest')
        axis('image')
        ax.set_xticks([])
        ax.set_yticks([])
        text(0.6, 0.05, "z = %1.2f" % (s.redshift), color='w', transform=ax.transAxes)
        
        if accreted == True:
            temp_pos = s.pos[jstars, :].astype('float64')
            spos = np.zeros((size(jstars), 3))
            spos[:, 0] = temp_pos[:, 0]
            spos[:, 1] = temp_pos[:, 2]
            spos[:, 2] = temp_pos[:, 1]
            pos = np.zeros((size(asind), 3))
            pos[:, 0] = spos[asind, 0]
            pos[:, 1] = spos[asind, 1]
            pos[:, 2] = spos[asind, 2]
            
            smass = s.data['mass'][jstars].astype('float64')
            mass = smass[asind]
            j = asind
        else:
            temp_pos = s.pos[jstars, :].astype('float64')
            pos = np.zeros((size(jstars), 3))
            pos[:, 0] = temp_pos[:, 0]
            pos[:, 1] = temp_pos[:, 2]
            pos[:, 2] = temp_pos[:, 1]
        
        data = np.zeros((res / 2, res, 3))
        for k in range(3):
            iband = [3, 1, 0][k]
            band = 10 ** (-2.0 * s.data['gsph'][j, iband].astype('float64') / 5.0)
            grid = calcGrid.calcGrid(pos, hsml, band, rho, rho, res / 2, res, res, boxsize / 2, boxsize, boxsize, s.center[0], s.center[1],
                                     s.center[2], 1, 1, numthreads=numthreads)
            
            drange = datarange[k]
            grid = np.minimum(np.maximum(grid, drange[0]), drange[1])
            loggrid = np.log10(grid)
            logdrange = np.log10(drange)
            data[:, :, k] = (loggrid - logdrange[0]) / (logdrange[1] - logdrange[0])
        
        x = ix * (1. / 6.) + 1. / 6. * 0.05 * 0.5
        y = iy * (1. / 4.) + 1. / 6. * 0.05 * 0.5
        
        ax = axes([x, y, 1 / 6. * 0.95, 1. / 12. * 0.95], frameon=False)
        imshow(data, interpolation='nearest')
        axis('image')
        ax.set_xticks([])
        ax.set_yticks([])
        
        del s
        del sf
    
    save_name = '%s/halo_%d%s' % (wpath, halonum, name)
    if accreted:
        save_name += '_acc'
    save_name += '.png'
    print('saving', save_name)
    savefig(save_name)