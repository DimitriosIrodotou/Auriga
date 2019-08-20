import calcGrid
import matplotlib
import numpy
import pysph
from gadget import gadget_readsnap
from gadget_subfind import load_subfind
from matplotlib import pyplot

bands = ['U', 'B', 'V', 'K', 'g', 'r', 'i', 'z']
band_array = {'U': 0, 'B': 1, 'V': 2, 'K': 3, 'g': 4, 'r': 5, 'i': 6, 'z': 7}
Msunabs = [5.61, 5.48, 4.83, 3.30, 5.12, 4.68, 4.57, 4.60]

# gsph is GFM_StellarPhotometrics, from the AREPO documentation:
""" Create stellar_photometrics.hdf5 file using BC03 models, as used for Illustris and IllustrisTNG runs.
Bands: UBVK (Buser U,B3,V,IR K filter + Palomar200 IR detectors + atmosphere.57) in Vega, griz (sdss) in AB
Requires: http://www.bruzual.org/bc03/Original_version_2003/bc03.models.padova_1994_chabrier_imf.tar.gz
Produces: 87f665fe5cdac109b229973a2b48f848  stellar_photometrics.hdf5
Original: f4bcd628b35036f346b4e47f4997d55e  stellar_photometrics.hdf5
  (all datasets between the two satisfy np.allclose(rtol=1e-8,atol=8e-4))
"""


def get_stellar_projection(s, mask, idir, res=1024, boxsize=0.05, center=None, type="light", maxHsml=True, numthreads=8, verbose=False):
    # Tree really wants double. Let's give it double :-)
    pos_orig = s.pos[mask].astype("f8")
    mass = s.mass[mask].astype("f8")
    pos = numpy.zeros((numpy.size(mass), 3))
    
    # Chances are that data is going to be gsph, but gsph is only available for
    # star particles. This could break if other particletypes are also read in
    # So we deploy a little trick :-)
    
    na = s.nparticlesall;
    st = na[:4].sum();
    en = st + na[4]
    gsph = numpy.zeros((s.npartall, 8))
    gsph[st:en] = s.data["gsph"].astype("f8")
    data = gsph[mask, :]
    
    if not center: center = s.center
    
    if idir == 0:  # XY plane ?
        pos[:, 0] = pos_orig[:, 1]
        pos[:, 1] = pos_orig[:, 2]
        pos[:, 2] = pos_orig[:, 0]
        
        xres = res
        yres = res
        
        boxx = boxsize
        boxy = boxsize
    elif idir == 1:  # XZ Plane?
        pos[:, 0] = pos_orig[:, 0]
        pos[:, 1] = pos_orig[:, 2]
        pos[:, 2] = pos_orig[:, 1]
        
        xres = res // 2
        yres = res
        
        boxx = boxsize / 2.
        boxy = boxsize
    elif idir == 2:
        pos[:, 0] = pos_orig[:, 1]
        pos[:, 1] = pos_orig[:, 0]
        pos[:, 2] = pos_orig[:, 2]
        
        xres = res
        yres = res // 2
        
        boxx = boxsize
        boxy = boxsize / 2.
    
    tree = pysph.makeTree(pos)
    hsml = tree.calcHsmlMulti(pos, pos, mass, 48, numthreads=8)
    if maxHsml:
        hsml = numpy.minimum(hsml, 4. * boxsize / res)
    hsml = numpy.maximum(hsml, 1.001 * boxsize / res * 0.5)
    rho = numpy.ones(numpy.size(mass))
    
    # What is this magic?
    datarange = numpy.array([[4003.36, 800672.], [199.370, 132913.], [133.698, 200548.]])
    fac = (512. / res) ** 2 * (0.5 * boxsize / 0.025) ** 2
    datarange *= fac
    
    boxz = max(boxx, boxy)
    
    if type == 'light':
        proj = numpy.zeros((xres, yres, 3))
        for k in range(3):
            iband = [3, 1, 0][k]  # K B U
            band = 10 ** (-2.0 * data[:, iband] / 5.0)
            
            grid = calcGrid.calcGrid(pos, hsml, band, rho, rho, xres, yres, 256, boxx, boxy, boxz, center[0], center[1], center[2], 1, 1,
                                     numthreads=numthreads)
            
            if verbose:
                print("Computing band: {0} (k={1})".format(bands[k], k))
                print("  grid.sum() = {0}\n  band.sum() = {1}".format(grid.sum(), band.sum()))
            
            drange = datarange[k]
            if verbose:
                print("  grid.max() = {0}\n  drange[0] = {1}\n  drange[1] = {2}".format(grid.max(), drange[0], drange[1]))
            
            grid = numpy.minimum(numpy.maximum(grid, drange[0]), drange[1])
            if verbose:
                print("  grid.min() = {0}\n  grid.max() = {1}".format(grid.min(), grid.max()))
            
            loggrid = numpy.log10(grid)
            logdrange = numpy.log10(drange)
            if verbose:
                print("  loggrid.min() = {0}\n  loggrid.max() = {1}\n  logdrange = {2}".format(loggrid.min(), loggrid.max(), logdrange))
            
            proj[:, :, k] = (loggrid - logdrange[0]) / (logdrange[1] - logdrange[0])
            if verbose:
                print("  proj[:,:,k].min() = {0}\n  proj[:,:,k].max() = {1}".format(proj[:, :, k].min(), proj[:, :, k].max()))
                print("proj[:,:,k].sum()/res**2 = {0}\n".format(proj[:, :, k].sum() / res ** 2))
    
    elif type == 'mass':
        proj = calcGrid.calcGrid(pos, hsml, data, rho, rho, xres, yres, 256, boxx, boxy, boxz, 0., 0., 0., 1, 1, numthreads=numthreads)
    else:
        print("Type %s not found." % type)
        return None
    
    return proj


if __name__ == "__main__":
    pyplot.switch_backend("agg")
    level = 4
    halo_number = 24
    snapid = 127
    
    basedir = "/hits/universe/GigaGalaxy/level{0}_MHD/".format(level)
    halodir = basedir + "halo_{0}/".format(halo_number)
    snappath = halodir + "output/"
    
    print("basedir : {0}".format(basedir))
    print("halodir : {0}".format(halodir))
    print("snappath: {0}".format(snappath))
    
    for snapid in range(77, 128, 1):
        print("\n\nProducing Stellar Projections Plot for snapid={0}".format(snapid))
        # if snapid != 127: continue
        sf = load_subfind(snapid, dir=snappath)
        s = gadget_readsnap(snapid, snappath=snappath, lazy_load=True, subfind=sf, loadonlytype=[4])
        s.subfind = sf
        s.haloname = halo_number
        
        # print("redshift: {0}".format(s.redshift))
        # print("time    : {0}".format(s.time))
        # print("center  : {0}".format(s.center))
        # print("#stars  : {0}".format(len(stars[0])))
        
        # Trying to get stellar projections
        s.calc_sf_indizes(s.subfind)
        s.select_halo(s.subfind, rotate_disk=True, do_rotation=True, use_principal_axis=True)
        
        res = 1024
        boxsize = 0.05
        istars, = numpy.where((s.r() < 2. * boxsize) & (s.data['age'] > 0.))
        
        fig, (ax, ax_r) = pyplot.subplots(2, 2)
        gs1 = matplotlib.gridspec.GridSpec(3, 3)
        gs1.update(hspace=0)
        ax1 = pyplot.subplot(gs1[:-1, :])
        ax2 = pyplot.subplot(gs1[-1, :])
        
        face_on = get_projection(s.pos[istars, :].astype('f8'), s.mass[istars].astype('f8'), s.data['gsph'][istars, :].astype('f8'), 0, res, boxsize,
                                 'light')
        ax1.imshow(face_on, interpolation='nearest')
        ax1.set_xticks([], []);
        ax1.set_yticks([], [])
        
        edge_on = get_projection(s.pos[istars, :].astype('f8'), s.mass[istars].astype('f8'), s.data['gsph'][istars, :].astype('f8'), 1, res, boxsize,
                                 'light')
        ax2.imshow(edge_on, interpolation='nearest')
        ax2.set_xticks([], []);
        ax2.set_yticks([], [])
        
        ax1.text(0.05, 0.92, "Au{0}-{1}".format(s.haloname, level), color='w', fontsize=12, transform=ax1.transAxes)
        ax1.text(0.95, 0.92, "z = {0:.2f}".format(s.redshift), color='w', ha="right", fontsize=12, transform=ax1.transAxes)
        
        pyplot.tight_layout()
        pyplot.savefig("out/Auriga_level{0}_halo{1}_snapid_{2}.png".format(level, halo_number, snapid), dpi=600)
        pyplot.close()