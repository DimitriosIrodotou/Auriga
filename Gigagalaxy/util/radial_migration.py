from const import *
from gadget import *
from gadget_subfind import *
from pylab import *
from util import select_stars_for_radial_migration

toinch = 0.393700787
Gcosmo = 43.0071
LUKPC = 1000.0
ZSUN = 0.0127
element = {'H': 0, 'He': 1, 'C': 2, 'N': 3, 'O': 4, 'Ne': 5, 'Mg': 6, 'Si': 7, 'Fe': 8}

# from Asplund et al. (2009) Table 5
SUNABUNDANCES = {'H': 12.0, 'He': 10.98, 'C': 8.47, 'N': 7.87, 'O': 8.73, 'Ne': 7.97, 'Mg': 7.64, 'Si': 7.55, 'Fe': 7.54}


def radial_migration(snaplist, dd, fpath, ecut, idpath, runs, accreted=False):
    print("Beginning Radial migration routine...")
    print("snaplist=", snaplist)
    
    if accreted == True:
        sname = str(snaplist[1])
        fnameacc = '%s/%sstarID_accreted.txt' % (idpath, runs[0])
        fnamein = '%s/%sstarID_insitu.txt' % (idpath, runs[0])
        insitu = np.loadtxt(fnamein, delimiter=None, dtype=int)
        idins = np.array(insitu.astype('int64'))
    else:
        idins = 0
    
    time = pylab.zeros(len(snaplist))
    ### End of first loop ###
    for isnap, snap in enumerate(snaplist):
        
        print("snap=", snap)
        
        s = gadget_readsnap(snap, snappath=dd + '/output/', hdf5=True, loadonly=['pos', 'age', 'id', 'mass', 'vel', 'gmet', 'pot'])
        sf = load_subfind(snap, dir=dd + '/output/', hdf5=True)
        s.calc_sf_indizes(sf)
        na = s.nparticlesall  # All particles
        s.select_halo(sf, use_principal_axis=True, use_cold_gas_spin=False, do_rotation=True, age_select=True)  # Get halo from subfind and read data
        galrad = 0.1 * sf.data['frc2'][0]  # Get some max radius for cut
        galrad = 0.04
        print('galrad:', galrad)
        ###
        st = na[:4].sum();
        en = st + na[4]  # Get start and end point
        age = np.zeros(s.npartall)
        age[st:en] = s.data['age']
        
        time[isnap] = s.cosmology_get_lookback_time_from_a(s.time.astype('float64'))
        ### FOr calculating radial mass profile ##
        ipall, = np.where((s.r() > 0) & (s.r() < galrad))
        nev = size(ipall)
        istars, = np.where((s.r() > 0) & (s.type == 4) & (age > 0.) & (s.r() < galrad))
        nstars = size(istars)
        
        rr = pylab.sqrt(s.pos[ipall, 0] ** 2 + s.pos[ipall, 1] ** 2 + s.pos[ipall, 2] ** 2)
        rr = rr.ravel()
        rr2d = pylab.sqrt(s.pos[ipall, 1] ** 2 + s.pos[ipall, 2] ** 2)
        rr2d = rr2d.ravel()
        # mass = s.data['mass'].astype('float64')
        mass_all = s.data['mass'][ipall].astype('float64')
        nshells = 100
        rbin = np.linspace(0.0, galrad, nshells)
        radius = pylab.zeros(nshells)
        vc = pylab.zeros(nshells)
        lc = pylab.zeros(nshells)
        for i in range(1, nshells):
            ii, = np.where((rr < rbin[i]))
            mtot = np.sum(mass_all[ii])
            # print("mtot=", mtot)
            radius[i] = rbin[i]
            vc[i] = pylab.sqrt(Gcosmo * mtot / (radius[i]))
            lc[i] = vc[i] * radius[i] * LUKPC
            radius[i] *= LUKPC
        lc = np.array(lc)
        
        # print("vc=",vc)
        
        ### Get interior mass for each particle ###
        rr = pylab.sqrt((s.pos[ipall, :] ** 2).sum(axis=1))
        msort = rr.argsort()
        # mass_all = mass[ipall]
        msum = pylab.zeros(nev)
        msum[msort[:]] = np.cumsum(mass_all[msort[:]])
        
        idall = s.id[ipall].astype('int64')
        
        age = age[ipall]
        pos = np.reshape(s.pos[ipall, :].astype('float64'), (nev, 3))
        vel = np.reshape(s.vel[ipall, :].astype('float64'), (nev, 3))
        mass = s.data['mass'][ipall].astype('float64')
        ptype = s.data['type'][ipall]
        pot = s.data['pot'][ipall].astype('float64')
        metal = np.zeros([s.npartall, 2])
        metal[st:en, 0] = s.data['gmet'][na[0]:, element['Fe']]
        metal[st:en, 1] = s.data['gmet'][na[0]:, element['H']]
        metal = metal[ipall, :]
        smetal = pylab.zeros([nstars, 2])
        nn, = np.where((ptype[:] == 4) & (age[:] > 0.))
        
        smetal[:, :] = metal[nn, :]
        feabund = log10(smetal[:, 0] / smetal[:, 1] / 56.) - (SUNABUNDANCES['Fe'] - SUNABUNDANCES['H'])
        
        lang = pylab.zeros(nstars)
        lz = pylab.zeros(nstars)
        eng = pylab.zeros(nstars)
        etot = pylab.zeros(nstars)
        epot = pylab.zeros(nstars)
        vrad = pylab.zeros(nstars)
        vz = pylab.zeros(nstars)
        z = pylab.zeros(nstars)
        eran = pylab.zeros(nstars)
        rad = pylab.zeros(nstars)
        speceng = pylab.zeros(nstars)
        m = pylab.zeros(nstars)
        
        if isnap == 0:
            epsilon = pylab.zeros(nstars)
            jcmax = pylab.zeros(nstars)
            rad1 = pylab.zeros(nstars)
            r1g = pylab.zeros(nstars)
            r_g1 = pylab.zeros(nstars)
        elif isnap == 1:
            rad2 = pylab.zeros(nstars)
            r2g = pylab.zeros(nstars)
            r_g2 = pylab.zeros(nstars)
        
        nn, = np.where((ptype[:] == 4) & (age[:] > 0.))
        rad = pylab.sqrt(pos[nn, 1] ** 2 + pos[nn, 2] ** 2)
        z = pos[nn, 0]
        vz = vel[nn, 0]
        lang = pylab.cross(pos[nn, :], vel[nn, :])
        lang_spec = abs(lang[:, 0])
        lz = lang_spec * mass[nn]
        indy = find_nearest(lc, lang_spec * LUKPC).astype('int64')
        speceng = 0.5 * (vel[nn, :] ** 2).sum(axis=1) + pot[nn]
        m = mass[nn]
        if isnap == 0:
            epsilon = lang[:, 0]
            rad1 = rad
            r_g1[:] = radius[indy]
            idk0 = idall[nn]
        if isnap == 1:
            rad2 = rad
            r_g2[:] = radius[indy]
            idk1 = idall[nn]
        
        if isnap == 0:
            iensort = np.argsort(speceng)
            epsilon = epsilon[iensort]
            for i in range(nstars):
                indy = i
                nn0 = indy - 50
                nn1 = indy + 50
                if nn0 < 0:
                    nn1 += -nn0
                    nn0 = 0
                if nn1 >= nstars:
                    nn0 -= (nn1 - (nstars - 1))
                    nn1 = nstars - 1
                
                jcmax[i] = np.max(epsilon[nn0:nn1])
            
            epsilon[:] /= jcmax[:]
            
            z1 = z[iensort]
            vz1 = vz[iensort]
            m1 = m[iensort]
            r_g1 = r_g1[iensort]
            rad1 = rad1[iensort]
            idk0 = idk0[iensort]
            feabund = feabund[iensort]
            ## For normal ecut
            jj, = np.where((epsilon > ecut))
            ## For prograde bulge stars
            # jj, = np.where((epsilon < ecut) & (epsilon > 0.))
            z1 = z1[jj]
            vz1 = vz1[jj]
            m1 = m1[jj]
            r_g1 = r_g1[jj]
            rad1 = rad1[jj]
            idk0 = idk0[jj]
            feabund = feabund[jj]
        
        if isnap == 1:
            z2 = z
            vz2 = vz
            m2 = m
    
    nstars0 = size(idk0)
    nstars1 = size(idk1)
    
    asind0, asind1 = select_stars_for_radial_migration.select_stars_given_ids(idk0, idk1, nstars0, nstars1, idins, accreted=accreted)
    
    rg1 = r_g1[asind0] / LUKPC
    rg2 = r_g2[asind1] / LUKPC
    r1 = rad1[asind0]
    r2 = rad2[asind1]
    z1 = z1[asind0]
    z2 = z2[asind1]
    vz1 = vz1[asind0]
    vz2 = vz2[asind1]
    m1 = m1[asind0]
    m2 = m2[asind1]
    idk0 = idk0[asind0]
    idk1 = idk1[asind1]
    feabund = feabund[asind0]
    
    return (rg1, rg2, r1, r2, z1, z2, vz1, vz2, m1, m2, time, feabund)


def find_nearest(array, value):
    if len(value) == 1:
        idx = (np.abs(array - value)).argmin()
    else:
        idx = np.zeros(len(value))
        for i in range(len(value)):
            idx[i] = (np.abs(array - value[i])).argmin()
    
    return idx