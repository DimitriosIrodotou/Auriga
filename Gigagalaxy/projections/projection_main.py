import pylab
from loadmodules import *
from projections.column_density_projections import *
from projections.column_temperature_projections import *
from projections.temperature_metallicity_projections import *

toinch = 0.393700787


def rotate_value(sf, value, matrix):
    rotmat = np.array(matrix.transpose(), dtype=sf.data[value].dtype)
    sf.data[value] = np.dot(sf.data[value], rotmat)
    return


def plot_gas_projections(prefix, run, snap, outpath, suffix, func, name, daxes, numthreads=1):
    path = prefix + run + '/output/'
    outpath += run
    s = gadget_readsnap(snap, snappath=path, hdf5=True, loadonly=['pos', 'vel', 'mass', 'u', 'rho', 'vol', 'ne', 'nh', 'sfr', 'gmet', 'gz'],
                        forcesingleprec=True)
    sf = load_subfind(snap, dir=path, hdf5=True, loadonly=['fpos', 'frc2', 'flty', 'fnsh', 'spos', 'svel', 'slty', 'smty'], forcesingleprec=True)
    s.calc_sf_indizes(sf, dosubhalos=False, halolist=[0])
    rotmatrix = s.select_halo(sf, use_principal_axis=True, do_rotation=True)
    
    # center the satellites
    sf.data['spos'][:, 0] -= sf.data['fpos'][0, 0]
    sf.data['spos'][:, 1] -= sf.data['fpos'][0, 1]
    sf.data['spos'][:, 2] -= sf.data['fpos'][0, 2]
    # rotate subhalo positions
    rotate_value(sf, 'spos', rotmatrix)
    
    pxsize = 11.45
    pysize = 5.7 * len(daxes)  # + 5.7 * ((len(daxes) - 1) / len(daxes))
    
    psize = 1.8
    offsetx = 0.1
    offsety = 0.8  # 0.38
    offset = 0.36
    
    fig = pylab.figure(figsize=(np.array([pxsize, pysize]) * toinch), dpi=300)
    
    res = 256
    # res = 128
    boxsize = 100.  # 1.0e3 # 1.0Mpc
    fact = 0.5  # projection lenght will be 2.0 * fact * boxsize
    
    for j in range(len(daxes)):
        for iplot in range(3):
            ix = iplot % 4
            iy = j % len(daxes)
            print("ix, iplot=", ix, iplot)
            x = ix * (2. * psize + offsetx) / pxsize + offsetx / pysize
            
            y = iy * (2. * psize + offsety) / pysize + offsety / pysize + j * 0.15
            ax1 = axes([x, y, 2. * psize / pxsize, 2. * psize / pysize], frameon=True)
            
            y = (2. * psize + 3. * offset) / pysize + 0.15 * psize / pysize
            cax = axes([x, y, 2. * psize / pxsize, psize / pysize / 15.], frameon=False)
            
            if func[iplot] == plot_hydrogen:
                func[iplot](ax1, cax, s, sf, boxsize, fact, res, dextoshow=8, daxes=daxes[j], numthreads=numthreads)
            else:
                func[iplot](ax1, cax, s, sf, boxsize, fact, res, dextoshow=6, daxes=daxes[j], numthreads=numthreads)
            
            for label in cax.xaxis.get_ticklabels():
                label.set_fontsize(6)
            
            for label in ax1.xaxis.get_ticklabels():
                label.set_fontsize(6)
            
            # ax1.xaxis.set_major_formatter( NullFormatter() )
            ax1.yaxis.set_major_formatter(NullFormatter())
            
            minorLocator = MultipleLocator(50.0)
            ax1.xaxis.set_minor_locator(minorLocator)
            ax1.yaxis.set_minor_locator(minorLocator)
            ax1.set_xlabel("$\\rm{x\\,[kpc]}$", size=6)
    
    fig.savefig('%s/projections/%s_%s.%s' % (outpath, name, run, suffix), transparent=True, dpi=300)