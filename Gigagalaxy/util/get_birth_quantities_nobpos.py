from const import *
from gadget import *
from gadget_subfind import *
from gglibs import discy
from pylab import *
from util import select_snapshot_number
from util import select_stars_for_radial_migration

toinch = 0.393700787
Gcosmo = 43.0071

ZSUN = 0.0127

element = {'H': 0, 'He': 1, 'C': 2, 'N': 3, 'O': 4, 'Ne': 5, 'Mg': 6, 'Si': 7, 'Fe': 8}

# from Asplund et al. (2009) Table 5
SUNABUNDANCES = {'H': 12.0, 'He': 10.98, 'C': 8.47, 'N': 7.87, 'O': 8.73, 'Ne': 7.97, 'Mg': 7.64, 'Si': 7.55, 'Fe': 7.54}

LUKPC = 1000.0


def get_birthradius(runs, dirs, nrows, ncols, outputlistfile, outpath, accreted=False, ecut=False):
    for d in range(len(runs)):
        wpath = outpath + runs[d] + '/rm4/'
        if not os.path.exists(wpath):
            os.makedirs(wpath)
        
        print("wpath=", wpath)
        ### Open file to write grid data ###
        if ecut == False:
            filename = wpath + '/birth_current.dat'
        elif ecut == True:
            filename = wpath + '/birth_current-e0p7.dat'
        f1 = open(filename, 'w')
        filename2 = wpath + '%sstarID_age.txt' % runs[d]
        f2 = open(filename2, 'w')
        idpath = outpath + runs[d] + '/sfr/'
        if accreted == True:
            fnamein = '%s/%sstarID_insitu.txt' % (idpath, runs[d])
            data = np.loadtxt(fnamein, delimiter=None, dtype=int)
            insitu = data[:, 0]
            idins = np.array(insitu.astype('int64'))
        
        print("idins=", idins)
        dd = dirs[d] + runs[d]
        print("Doing run:", dd)
        
        if runs[d] == 'halo_19':
            age_select = 1.0
        else:
            age_select = 3.0
        
        dr_sig = np.zeros(3)
        # snap = 63
        snap = 127
        sname = str(snap)
        s = gadget_readsnap(snap, snappath=dd + '/output/', hdf5=True, loadonly=['pos', 'age', 'id', 'mass', 'vel', 'gmet', 'pot'])
        sf = load_subfind(snap, dir=dd + '/output/', hdf5=True, loadonly=['fpos', 'frc2', 'svel', 'flty', 'fnsh', 'slty'])
        s.calc_sf_indizes(sf)
        center = None
        s.select_halo(sf, center, age_select, use_principal_axis=True, use_cold_gas_spin=False, do_rotation=True)
        galrad = 0.15 * sf.data['frc2'][0]
        print("galrad=", galrad)
        age = np.zeros(s.npartall)
        na = s.nparticlesall
        st = na[:4].sum();
        en = st + na[4]
        age[st:en] = s.data['age']
        ipall, = np.where((s.r() > 0.) & (s.r() < galrad))
        istars, = np.where((s.r() > 0.) & (s.r() < galrad) & (s.type == 4) & (age > 0.))
        nall = size(ipall)
        ntoday = size(istars)
        ### Calculate some star quantities ###
        starage = pylab.zeros(ntoday)
        starage[:] = s.cosmology_get_lookback_time_from_a(age[istars], is_flat=True)
        metal = pylab.zeros([s.npartall, 2])
        metal[st:en, 0] = s.data['gmet'][na[0]:, element['Fe']]
        metal[st:en, 1] = s.data['gmet'][na[0]:, element['H']]
        smetal = pylab.zeros([ntoday, 2])
        smetal[:, :] = metal[istars, :]
        feabund = np.log10(smetal[:, 0] / smetal[:, 1] / 56.) - (SUNABUNDANCES['Fe'] - SUNABUNDANCES['H'])
        
        vz = pylab.zeros(ntoday)
        vz[:] = s.vel[istars, 0]
        z = pylab.zeros(ntoday)
        z[:] = s.pos[istars, 0]
        
        ### Sort out Lzc curve... ###
        nshells = 100
        rr = pylab.sqrt(s.pos[ipall, 0] ** 2 + s.pos[ipall, 1] ** 2 + s.pos[ipall, 2] ** 2)
        rr = rr.ravel()
        mass = s.data['mass'][ipall].astype('float64')
        rbin = np.linspace(0.0, 1.5 * galrad, nshells)
        radius = pylab.zeros(nshells)
        vc = pylab.zeros(nshells)
        lc = pylab.zeros(nshells)
        for i in range(1, nshells):
            ii, = np.where((rr < rbin[i]))
            mtot = np.sum(mass[ii])
            radius[i] = rbin[i]
            vc[i] = pylab.sqrt(G * mtot * 1e10 * msol / (radius[i] * 1e6 * parsec)) / 1e5
            lc[i] = vc[i] * radius[i] * LUKPC
        lc = np.array(lc)
        ### Match specific Lz of particle to Lzc to infer guiding centre ###
        lz_spec = pylab.zeros(ntoday)
        lz_spec = pylab.cross(s.pos[istars, :], s.vel[istars, :])
        lz_spec = abs(lz_spec[:, 0])
        indy = find_nearest(lc, lz_spec * 1e3).astype('int64')
        tradg = pylab.zeros(ntoday)
        tradg[:] = radius[indy]
        ### Calculate present day radius too ###
        trad = pylab.zeros(ntoday)
        trad[:] = pylab.sqrt(s.pos[istars, 1] ** 2 + s.pos[istars, 2] ** 2)
        ## and orbital circularity ##
        nn = istars - st
        pos = s.pos[ipall, :].astype('float64')
        vel = s.vel[ipall, :].astype('float64')
        pot = s.data['pot'][ipall].astype('float64')
        rxy = pylab.sqrt((s.pos[ipall, 1:] ** 2).sum(axis=1))
        
        eps2 = np.zeros(ntoday)
        spec_energy = np.zeros(ntoday)
        
        j = pylab.cross(pos[nn, :], vel[nn, :])
        jz = j[:, 0]
        spec_energy[:] = 0.5 * (vel[nn, :] ** 2).sum(axis=1) + pot[nn]
        eps2[:] = jz
        iensort = np.argsort(spec_energy)
        eps2 = eps2[iensort]  # eps2orig=[5,4,3,2,1], iensort=[5,4,3,2,1]
        eps2 = discy.disc(eps2)
        ilist = list(range(0, ntoday))
        ilist = np.array(ilist)  # ilist=1,2,3,4,5
        ilist = ilist[iensort]  # ilist=5,4,3,2,1
        resort = np.argsort(ilist)  # resort=5,4,3,2,1
        eps2 = eps2[resort]  # eps2=5,4,3,2,1
        
        ### Match age of each star to snapshot it was born in ###
        ages = age[istars]
        redshifts = 1. / ages - 1.
        snapshot_list = select_snapshot_number.select_snapshot_number(outputlistfile, list(redshifts), verbose=False)
        snapshot_list = np.array(snapshot_list)
        snapshot_list += 1
        
        ### Sort everything by snap number ###
        asort = snapshot_list.argsort()
        ids = s.data['id'][istars].astype('int64')
        ids = ids[asort]
        snapshot_list = snapshot_list[asort]
        ll, = np.where((snapshot_list == (snap + 1)))
        if size(ll) > 0:
            snapshot_list[ll] -= 1
        print("snapshot_list,max=", snapshot_list, snapshot_list.max())
        trad = trad[asort]
        tradg = tradg[asort]
        feabund = feabund[asort]
        vz = vz[asort]
        z = z[asort]
        eps2 = eps2[asort]
        starage = starage[asort]
        
        print("Final number of stars =", len(ids))
        snaplist_unique = np.unique(snapshot_list)
        print("List of unique snapshots =", snaplist_unique)
        
        ### Now go through each snapshot for which we have a newborn star ###
        brad = []
        ff = 0
        for isnap, snap in enumerate(snaplist_unique):
            
            print('isnap =', isnap)
            print("Doing snap %s" % snap)
            # if snap != 63:
            ipt = np.where((snapshot_list == snap))  # Take only particles born at this time
            idlist_temp = ids[ipt]
            
            s = gadget_readsnap(snap, snappath=dd + '/output/', hdf5=True, loadonlytype=[4], loadonly=['pos', 'age', 'id', 'vel', 'mass'])
            sf = load_subfind(snap, dir=dd + '/output/', hdf5=True, loadonly=['fpos', 'frc2', 'svel', 'flty', 'fnsh', 'slty'])
            s.calc_sf_indizes(sf)
            na = s.nparticlesall  # All particles
            center = None
            try:
                s.select_halo(sf, center, age_select, use_principal_axis=True, use_cold_gas_spin=False,
                              do_rotation=True)  # Get halo from subfind and read data
            except:
                cc = 0
                while cc < len(idlist_temp):
                    dumrad = -1
                    brad.append(dumrad)
                    cc += 1
                    ff += 1
                continue
            
            st = na[:4].sum();
            en = st + na[4]  # Get start and end point
            print('total number of particles before cut =', en - st)
            age = np.zeros(s.npartall)
            age[st:en] = s.data['age']
            print('size of age array=', size(age[st:en]))
            
            jstars, = np.where((s.r() > 0) & (s.type == 4) & (age > 0.))
            nallstars = size(jstars)
            nstars = size(idlist_temp)
            
            idk = s.id[jstars].astype('int64')
            if len(idlist_temp) != len(list(set(ids).intersection(idlist_temp))):
                raise ValueError("not all selected particles are present in current snapshot")
            idict = dict(zip(idk, np.ravel(jstars)))
            
            cc = 0
            while cc < nstars:
                try:
                    nn = idict[idlist_temp[cc]]
                    dumrad = pylab.sqrt(s.pos[nn, 1] ** 2 + s.pos[nn, 2] ** 2)
                    brad.append(dumrad)
                    cc += 1
                except:
                    dumrad = -1.
                    brad.append(dumrad)
                    cc += 1
                    ff += 1
        
        print("Star particles missing:", ff)
        print("out of total:", ntoday)
        brad = np.array(brad)
        jj, = np.where((brad > 0.))
        if ecut == False:
            ii, = np.where((brad > 0.))
        elif ecut == True:
            ii, = np.where((eps2[jj] > 0.7))
        ids = ids[ii]  ## new
        brad = brad[ii]
        trad = trad[ii]
        tradg = tradg[ii]
        starage = starage[ii]
        feabund = feabund[ii]
        vz = vz[ii]
        z = z[ii]
        eps2 = eps2[ii]
        nfinal = size(ii)
        ### Remove ex-situ stars ###
        if accreted == True:
            arrayf = np.arange(len(ii))
            asind = select_stars_for_radial_migration.select_insitu_stars(ids, idins, arrayf)
            
            brad = brad[asind]
            trad = trad[asind]
            tradg = tradg[asind]
            starage = starage[asind]
            feabund = feabund[asind]
            vz = vz[asind]
            z = z[asind]
            eps2 = eps2[asind]
            nfinal = len(asind)
        
        np.savetxt(f1, np.column_stack((brad, trad, tradg, starage, feabund, z, vz, eps2)))
        f1.close()


def find_nearest(array, value):
    if len(value) == 1:
        idx = (np.abs(array - value)).argmin()
    else:
        idx = np.zeros(len(value))
        for i in range(len(value)):
            idx[i] = (np.abs(array - value[i])).argmin()
    
    return idx