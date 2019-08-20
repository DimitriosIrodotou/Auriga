from const import *
from gadget import *
from gadget_subfind import *
from pylab import *
# import calcGrid
from util import select_snapshot_number

Gcosmo = 43.0071

outputlistfile = '/home/marinafo/scratch/Aquarius/AgeOfAquariusOutputs/ExpansionList_128'
inputfile1 = '/home/marinafo/scratch/Aquarius/AgeOfAquariusOutputs/ExpansionList_128'
inputfile2 = '/home/marinafo/scratch/Aquarius/AgeOfAquariusOutputs/ExpansionList_128'


def select_stars_given_ids(idk0, idk1, nstars0, nstars1):
    idlist_true = list(set(idk1).intersection(idk0))  # Take only IDs present in both lists
    idlist_true_array = np.array(idlist_true)
    nstars = size(idlist_true)
    print("Size of IDlisttrue=", nstars)
    
    array0 = np.arange(nstars0)
    array1 = np.arange(nstars1)
    
    idict0 = dict(zip(idk0, array0))
    idict1 = dict(zip(idk1, array1))
    
    cc = 0
    sind0 = []
    sind1 = []
    # Find position in data list of each ID
    while cc < nstars:
        nn0 = idict0[idlist_true[cc]]
        nn1 = idict1[idlist_true[cc]]
        sind0.append(nn0)
        sind1.append(nn1)
        cc += 1
    asind0 = np.array(sind0)
    asind1 = np.array(sind1)
    
    return asind0, asind1


def select_stars(dd, snaplist, thick_disc=False):
    print("dd=", dd)
    print("snaplist=", snaplist)
    time = pylab.zeros(len(snaplist))
    for isnap, snap in enumerate(snaplist):
        
        print('dd=', dd)
        print('isnap,snap=', isnap, snap)
        
        s = gadget_readsnap(snap, snappath=dd + '/', hdf5=True)
        sf = load_subfind(snap, dir=dd + '/', hdf5=True)
        s.calc_sf_indizes(sf)
        na = s.nparticlesall  # All particles
        time[isnap] = s.cosmology_get_lookback_time_from_a(s.time.astype('float64'))
        print("Sel.Lookback Time (Gyr) = ", time[isnap])
        
        galrad = 0.2 * sf.data['frc2'][0]  # Get some max radius for cut
        print('galrad:', galrad)
        s.select_halo(sf, use_principal_axis=True, use_cold_gas_spin=False, do_rotation=True)  # Get halo from subfind and read data
        # na is 1d array that stores the numbers particles of subhaloes
        st = na[:4].sum();
        en = st + na[4]  # Get start and end point
        print('total number of particles before cut =', en - st)
        age = np.zeros(s.npartall)
        age[st:en] = s.data['age']
        ## Mod(27/04/15)
        if thick_disc == True:
            abundances[st:en, :] = s.data['gmet'][na[0]:, :]
            k, = np.where(abundances[:, element['Fe']] <= 0.0)
            abundances[k, element['Fe']] = 1.0e-40
            k, = np.where(abundances[:, element['O']] <= 0.0)
            abundances[k, element['O']] = 1.0e-40
            iron = log10(abundances[:, element['Fe']] / abundances[:, element['H']] / 56.0)
            alphaelem = log10(abundances[:, element['O']] / abundances[:, element['H']] / 16.)
        
        # Apply radial cut to particles, and get array element number where condition is satisfied
        if isnap == 0:
            
            iall, = np.where((s.r() > 0.0) & (s.r() < galrad))
            istars, = np.where((s.r() > 0.0) & (s.r() < galrad) & (s.type == 4) & (age > 0.))
            nstars = size(istars)  # number of stars after cut
            nall = size(iall)  # number of particles after cut
            
            idk = s.id[istars].astype('int64')
        
        elif isnap == 1:
            
            iall1 = np.where((s.r() > 0.))
            if thick_disc == True:
                istars1 = np.where(
                    (s.r() > 0.) & (s.type == 4) & (age > 0.) & (alphaelem > 0.275))  ### Geta single final idlist here, then do anoter loop
            else:
                istars1 = np.where((s.r() > 0.) & (s.type == 4) & (age > 0.))  ### Geta single final idlist here, then do anoter loop
            nall1 = size(iall1)
            nstars1 = size(istars1)
            idlist1 = pylab.zeros(nstars1)
            idk1 = s.id[istars1].astype('int64')
            idict = dict(zip(idk1, np.ravel(istars1)))
            idd1 = list(idk1)
    
    if len(snaplist) > 1:
        idlist_true = list(set(idk1).intersection(idk))  # Take only IDs present in both lists
        idlist_true_array = np.array(idlist_true)
        nstars = size(idlist_true)
        ids = s.id[istars1].astype('int64')
        idict = dict(zip(ids, np.ravel(istars1)))
    else:
        idlist_true = list(idk)
        idlist_true_array = np.array(idlist_true)
        nstars = size(idlist_true)
        ids = s.id[istars].astype('int64')
        idict = dict(zip(ids, np.ravel(istars)))
    cc = 0
    sind = []
    # Find position in data list of each ID
    while cc < nstars:
        nn = idict[idlist_true[cc]]
        sind.append(nn)
        cc += 1
    asind = np.array(sind)
    age = age[asind]
    # Make lists based on stellar age
    ages = pylab.zeros(nstars)
    age_gyr = pylab.zeros(nstars)
    red_shift = pylab.zeros(nstars)
    # Get lookback time of each particle
    k = 0
    for i in range(nstars):
        ages[k] = age[i]
        red_shift[k] = 1. / ages[i] - 1.
        age_gyr[k] = s.cosmology_get_lookback_time_from_a(age[i])
        k += 1
    
    asort = age_gyr.argsort()
    idlist_true_array = idlist_true_array[asort]
    ages = ages[asort]  # scale factor
    age_gyr = age_gyr[asort]  # lookback time in Gyr
    red_shift = red_shift[asort]
    # Get snapshot corresponding to particle age of each particle
    snaplist_index = select_snapshot_number.select_snapshot_number(outputlistfile, red_shift, exp_fact=False, verbose=False)
    
    slist = select_snapshot_number.match_expansion_factor_files(inputfile1, inputfile2, snaplist_index)
    
    snapshot_list = np.array(slist)
    print("length of true array=", len(idlist_true_array))
    print("length of snapshotlist=", len(snapshot_list))
    
    return idlist_true_array, snapshot_list, age_gyr


# def select_thickdisc_stars(dd, snaplist):

# return select_stars(dd, snaplist, thick_disc=True)

def select_disc_stars(dd, snap):
    print("Selecting disc stars for run", dd, "and step ", snap)
    s = gadget_readsnap(snap, snappath=dd + '/', hdf5=True)
    sf = load_subfind(snap, dir=dd + '/', hdf5=True)
    s.calc_sf_indizes(sf)
    na = s.nparticlesall  # All particles
    
    galrad = 0.2 * sf.data['frc2'][0]  # Get some max radius for cut
    s.select_halo(sf, use_principal_axis=True, use_cold_gas_spin=False, do_rotation=True)
    mass = s.data['mass'].astype('float64')
    st = na[:4].sum();
    en = st + na[4]  # Get start and end point
    age = np.zeros(s.npartall)
    age[st:en] = s.data['age']
    iall, = np.where((s.r() < galrad) & (s.r() > 0.))
    istars, = np.where((s.r() < galrad) & (s.r() > 0.) & (s.type == 4) & (age > 0.))
    
    nstars = size(istars)
    nall = size(iall)
    rr = pylab.sqrt((s.pos[iall, :] ** 2).sum(axis=1))
    msort = rr.argsort()
    mass_all = mass[iall]
    
    msum = pylab.zeros(nall)
    msum[msort[0]] = mass_all[msort[0]]
    for nn in range(1, nall):
        msum[msort[nn]] = msum[msort[nn - 1]] + mass_all[msort[nn]]
    pos = s.pos[iall, :].astype('float64')
    vel = s.vel[iall, :].astype('float64')
    mass = s.data['mass'][iall].astype('float64')
    ptype = s.data['type'][iall]
    pot = s.data['pot'][iall].astype('float64')
    radius = pylab.sqrt((s.pos[iall, 1:] ** 2).sum(axis=1))
    idp = s.id[iall].astype('int64')
    ids = s.id[istars].astype('int64')
    age = age[iall]
    
    eps = pylab.zeros(nstars)
    eps2 = pylab.zeros(nstars)
    smass = pylab.zeros(nstars)
    jcmax = pylab.zeros(nstars)
    spec_energy = pylab.zeros(nstars)
    # ids = pylab.zeros(nstars)
    
    print("Nall =", nall)
    cc = 0
    for nn in range(nall):
        if ptype[nn] == 4 and age[nn] > 0.:
            energy = 0.5 * (vel[nn, :] ** 2).sum() + pot[nn]
            j = pylab.cross(pos[nn, :], vel[nn, :])
            jc = radius[nn] * pylab.sqrt(Gcosmo * msum[nn] / radius[nn])
            jz = j[0]
            
            eps[cc] = jz / jc
            eps2[cc] = jz
            spec_energy[cc] = energy
            smass[cc] = mass[nn]
            # ids[cc] = idp[nn]
            
            cc += 1
    
    print('cc=', cc)
    print("ids=", ids)
    # sort particle by specific energy
    iensort = np.argsort(spec_energy)
    eps = eps[iensort]
    eps2 = eps2[iensort]
    spec_energy = spec_energy[iensort]
    smass = smass[iensort]
    ids = ids[iensort]
    print("ids=", ids)
    
    for nn in range(nstars):
        nn0 = nn - 50
        nn1 = nn + 50
        
        if nn0 < 0:
            nn1 += -nn0
            nn0 = 0
        if nn1 >= nstars:
            nn0 -= (nn1 - (nstars - 1))
            nn1 = nstars - 1
        
        jcmax[nn] = np.max(eps2[nn0:nn1])
    
    smass /= smass.sum()
    eps2[:] /= jcmax[:]
    
    print("eps2=", eps2)
    
    jj, = np.where((eps2 > 0.7))
    idlist_circ = ids[jj]
    idlist_circ = np.array(idlist_circ)
    print("idlist_circ=", idlist_circ)
    
    return idlist_circ


def select_accreted_insitu_stars(ids, fnameacc, fnamein):
    fout = fnameacc
    fid = open(fout, 'r')
    
    idsacc = []
    for line in fid:
        line = line.strip()
        col = line.split()
        idsacc.append(col[0])
    
    fid.close()
    idsacc = np.array(idsacc)
    
    idlist_acc = list(set(idsacc).intersection(ids))
    ##
    
    fout = fnamein
    fid = open(fout, 'r')
    
    idsin = []
    for line in fid:
        line = line.strip()
        col = line.split()
        idsin.append(int64(col[0]))
    idsin = np.array(idsin)
    
    fid.close()
    
    idlist_in = list(set(idsin).intersection(ids))
    
    return idlist_acc, idlist_in