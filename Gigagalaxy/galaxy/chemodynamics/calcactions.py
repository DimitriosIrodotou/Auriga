import struct

from actions import *
from const import *
from galpy import *
from loadmodules import *
from pylab import *
from util import *

colors = ['r', 'm', 'c', 'b']


def calcactions(runs, dirs, outpath, outputlistfile, zlist, suffix, nrows, ncols, birthdatafile=None, potfiles=None, weight=False, accreted=False,
                disc_stars=False, atbirth=False, birthonly=False, alpha=False, apogeesf=False, gaiasf=False):
    panels = len(runs)
    
    potlist = ["bulge_hernquist", "disk_exp", "dmhalo_NFW", "gasdisk_exp"]
    
    for d in range(len(runs)):
        
        snaps = np.int_(select_snapshot_number.select_snapshot_number(outputlistfile[d], zlist))
        if isinstance(snaps, int):
            snaps = [snaps]
        
        print("snaps=", snaps)
        
        dd = dirs[d] + runs[d]
        wpath = outpath + runs[d]
        if not os.path.exists(wpath):
            os.makedirs(wpath)
        
        apath = outpath + runs[d] + '/actions/'
        if not os.path.exists(apath):
            os.makedirs(apath)
        
        fstr = '_new'
        fa = apath + '/actions_%s%s.dat' % (runs[d], fstr)
        
        filebase_potargs = outpath + runs[d] + '/potentialdecomp/'
        
        if atbirth:
            if birthdatafile:  # read birth data from post-processed file
                stardatafile = outpath + runs[d] + birthdatafile
                attrs = ['pos', 'vel', 'id', 'mass', 'age', 'gmet', 'pot', 'gz']
            else:
                stardatafile = None  # snapshot already contains birth data
                attrs = ['pos', 'vel', 'id', 'mass', 'age', 'gmet', 'pot', 'gz', 'bpos', 'bvel']
        else:
            stardatafile = None
            attrs = ['pos', 'vel', 'id', 'mass', 'age', 'gmet', 'gz']
        
        s = gadget_readsnap(snaps[-1], snappath=dd + '/output/', hdf5=True, loadonlytype=[0, 1, 4, 5], loadonly=attrs)
        sf = load_subfind(snaps[-1], dir=dd + '/output/', hdf5=True, loadonly=['fpos', 'frc2', 'svel', 'flty', 'fnsh', 'slty', 'fmc2'])
        s.calc_sf_indizes(sf)
        s.select_halo(sf, 3., use_principal_axis=True, use_cold_gas_spin=False, do_rotation=True)
        if birthdatafile and atbirth:
            attrs.append('bpos')
            attrs.append('bvel')
        
        rotmatfile = dd + '/output/rotmatlist_%s.txt' % runs[d]
        galcenfile = dd + '/output/galcen_%s.txt' % runs[d]
        g = parse_particledata(s, sf, attrs, rotmatfile, galcenfile, radialcut=sf.data['frc2'][0], stardatafile=stardatafile)
        g.prep_data()
        
        nstars = np.int_(g.numpart[4])
        print("number of stars=", nstars)
        
        a = actions(potlist, dobject=g)
        
        sdata = g.sgdata['sdata']
        
        # Read parameters from files
        print("poftiles=", potfiles)
        paramset = {}
        for fname in potfiles:
            f = wpath + '/densityprof/' + fname
            parameters = np.loadtxt(f, delimiter=None, skiprows=1)
            print("parameters=", parameters)
            if fname == 'fitstars_hR_z0.txt' or fname == 'fitbaryon_hR_z0.txt':
                paramset["bulge_hernquist"] = parameters[:, 1:3]  # .ravel()
                paramset["disk_exp"] = parameters[:, 3:6]  # .ravel()
                timelist = parameters[:, 0]
            if fname == 'fitgas_hR_z0.txt':
                paramset["gasdisk_exp"] = parameters[:, 1:4]  # .ravel()
            if fname == 'fitDM_density.txt':
                paramset["dmhalo_NFW"] = parameters[:, 1:4]  # .ravel()
        
        print("paramset=", paramset)
        print("timelist=", timelist)
        
        print("paramset t0=", paramset["disk_exp"][0])
        
        # sys.exit()
        
        if atbirth:
            ageindex = np.digitize(sdata['age'], timelist)
            ageinduni = np.unique(ageindex)
            
            jRb = pylab.zeros(nstars)
            jzb = pylab.zeros(nstars)
            lzb = pylab.zeros(nstars)
            
            # loop to calculate birth actions using potential of nearest snapshot
            for j, t in enumerate(ageinduni):
                ind, = np.where(ageindex == t)
                
                a.normalise_components(paramset, j)
                
                R = np.sqrt((sdata['bpos'][ind, 1:] ** 2).sum(axis=1))
                vR = (sdata['bvel'][ind, 1] * sdata['bpos'][ind, 1] + sdata['bvel'][ind, 2] * sdata['bpos'][ind, 2]) / R
                vT = - (sdata['bvel'][ind, 1] * sdata['bpos'][ind, 2] - sdata['bvel'][ind, 2] * sdata['bpos'][ind, 1]) / R
                
                phase_space_dat = zip(R, vR, vT, sdata['bpos'][ind, 0], sdata['bvel'][ind, 0])
                
                jRb[ind], jzb[ind], lzb[ind] = a.calculate_actions(phase_space_dat, paramset, j)
                print("jRb[ind], jzb[ind], lzb[ind]=", jRb[ind], jzb[ind], lzb[ind])
        
        index = 0
        a.normalise_components(paramset, index)
        
        # present day actions for all stars
        R = np.sqrt((sdata['pos'][:, 1:] ** 2).sum(axis=1))
        vR = (sdata['vel'][:, 1] * sdata['pos'][:, 1] + sdata['vel'][:, 2] * sdata['pos'][:, 2]) / R
        vT = - (sdata['vel'][:, 1] * sdata['pos'][:, 2] - sdata['vel'][:, 2] * sdata['pos'][:, 1]) / R
        z = sdata['pos'][:, 0]
        vz = sdata['vel'][:, 0]
        
        phase_space_dat = zip(R, vR, vT, z, vz)
        jR, jz, lz = a.calculate_actions(phase_space_dat, paramset, index)
        
        print("jR,jz,lz=", jR, jz, lz)
        
        f1 = open(fa, 'wb')
        f1.write(struct.pack('i', nstars))
        f1.write(struct.pack('%sd' % nstars, *sdata['id']))
        f1.write(struct.pack('%sd' % nstars, *jR))
        f1.write(struct.pack('%sd' % nstars, *jz))
        f1.write(struct.pack('%sd' % nstars, *lz))
        if atbirth:
            f1.write(struct.pack('%sd' % nstars, *jRb))
            f1.write(struct.pack('%sd' % nstars, *jzb))
            f1.write(struct.pack('%sd' % nstars, *lzb))
        f1.close()


def find_nearest(array, value):
    if len(value) == 1:
        idx = (np.abs(array - value)).argmin()
    else:
        idx = np.zeros(len(value))
        for i in range(len(value)):
            idx[i] = (np.abs(array - value[i])).argmin()
    
    return idx