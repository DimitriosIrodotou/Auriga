import calcGrid
import matplotlib as mpl
from const import *
from gadget import *
from gadget_subfind import *
from pylab import *
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import kde

LUKPC = 1000.0

inputfile1 = '/home/marinafo/scratch/Aquarius/AgeOfAquariusOutputs/ExpansionList_short'
inputfile2 = '/home/marinafo/scratch/Aquarius/AgeOfAquariusOutputs/ExpansionList_128'


def bin_quantity(nr, xqu, weight):
    n, bins = np.histogram(xqu, bins=nr)
    wn, wbins = np.histogram(xqu, bins=nr, weights=weight)
    
    qumean = wn / n
    x = np.zeros(len(bins) - 1)
    for cc in range(len(bins) - 1):
        x[cc] = (bins[cc] + bins[cc + 1]) / 2.
    
    return x, wn, bins, qumean


def cumulative_bin(nr, val, weight):
    wn, bins = np.histogram(val, bins=nr, weights=weight)
    cval = pylab.zeros(nr)
    for i in range(1, nr):
        cval[i] = cval[i - 1] + wn[i]
    return bins[1:], cval


def plot_sigmaz_rm(nr, xval, binval):
    ni, ibins = np.histogram(xval, bins=nr)  # For 1st step
    sni, ibins = np.histogram(xval, bins=nr, weights=binval)
    sni2, ibins = np.histogram(xval, bins=nr, weights=binval * binval)
    
    smeani = sni / ni  # 1st step
    sstdi = np.sqrt(abs(sni2 / ni) - abs(smeani * smeani))
    xi = np.zeros(len(ibins) - 1)
    yi = np.zeros(len(ibins) - 1)
    for cc in range(len(ibins) - 1):
        xi[cc] = (ibins[cc] + ibins[cc + 1]) / 2.0
        yi[cc] = sstdi[cc]
    
    return xi, yi


def plot_sigmaz_rm_massweighted(nr, xval, binval, massval):
    ni, ibins = np.histogram(xval, bins=nr)
    sni, ibins = np.histogram(xval, bins=nr, weights=binval)
    sni2, ibins = np.histogram(xval, bins=nr, weights=binval * binval)
    mi, ibins = np.histogram(xval, bins=nr, weights=massval)
    
    # print("mi=", mi)
    # print("sni=", sni)
    # sni /= mi
    # sni2 /= (mi*mi)
    smeani = sni / mi
    # print("smeani=", smeani)
    sstdi = np.sqrt(abs(sni2 / mi) - abs(smeani * smeani))
    xi = np.zeros(len(ibins) - 1)
    yi = np.zeros(len(ibins) - 1)
    for cc in range(len(ibins) - 1):
        xi[cc] = (ibins[cc] + ibins[cc + 1]) / 2.0
        yi[cc] = sstdi[cc]
    
    return xi, yi


def plot_rgini_deltar_rmp(ax3, nr, rg1, rg2):
    #
    kdata = np.array(zip(rg1, (rg2 - rg1)))
    xk, yk = kdata.T
    nbins = 50
    k = kde.gaussian_kde(kdata.T)
    xl, yl = np.mgrid[0.0:0.025:nbins * 1j, -0.0125:0.0125:nbins * 1j]
    #    xl, yl = np.mgrid[0.0:25.:nbins*1j, -12.5:12.5:nbins*1j]
    zl = np.log10(k(np.vstack([xl.flatten(), yl.flatten()])))
    zmin = 0.0
    zmax = 4.0
    cmap = mpl.cm.jet
    ax3.pcolormesh(xl * LUKPC, yl * LUKPC, zl.reshape(xl.shape), vmin=zmin, vmax=zmax)
    return ax3


def plot_rgini_deltar_rm(nbin, x, y, wgt, hrange):
    hist, xedges, yedges = np.histogram2d(x, y, bins=(nbin, nbin), weights=wgt, range=hrange)
    
    dx = (hrange[0][0] - hrange[0][1]) / nbin
    dxinv = 1. / dx
    h_g = np.log10(gaussian_filter(hist, 2.0) * dxinv ** 2)
    # h_g = gaussian_filter(hist,0.5)*dxinv**2
    
    return h_g


def plot_2dhist(axx, nr, xx, yy, zz, hrange, vval):
    n, xedges, yedges = np.histogram2d(xx, yy, bins=nr, range=hrange)
    sz, xedges, yedges = np.histogram2d(xx, yy, bins=nr, range=hrange, weights=zz)
    sz2, xedges, yedges = np.histogram2d(xx, yy, bins=nr, range=hrange, weights=zz * zz)
    kk, ll = np.where((n < 200.))  # ~40 for level5, 200 for level4
    
    xedges.shape, yedges.shape
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    mean = sz / n
    std = np.sqrt(sz2 / n - mean * mean)
    std[kk, ll] = nan
    
    cmap = mpl.cm.jet
    
    im = axx.imshow(np.flipud(std.T), extent=extent, aspect='auto', interpolation='nearest', vmin=vval[0], vmax=vval[1], cmap=cmap)
    # im=axx.imshow(np.fliplr(std.T),extent=extent,aspect='auto',
    #     interpolation='nearest', vmin=vval[0],vmax=vval[1],cmap=cmap)
    
    return im


def plot_scaleheight_rm(nr, r1, r2, z1, z2, m1, m2):
    ni, ibins = np.histogram(r1, bins=nr)  # For 1st step
    sni, ibins = np.histogram(r1, bins=nr, weights=m1 * m1)
    sni2, ibins = np.histogram(r1, bins=nr, weights=m1 * m1 * z1 * z1)
    
    nf, bins = np.histogram(r2, bins=nr)  # For 2nd step
    snf, bins = np.histogram(r2, bins=nr, weights=m2 * m2)
    snf2, bins = np.histogram(r2, bins=nr, weights=m2 * m2 * z2 * z2)
    
    rmmeanzi = np.sqrt(sni2 / sni)
    xi = np.zeros(len(ibins) - 1)
    yi = np.zeros(len(ibins) - 1)
    for cc in range(len(ibins) - 1):
        xi[cc] = (ibins[cc] + ibins[cc + 1]) / 2.0
        yi[cc] = rmmeanzi[cc]
    
    rmmeanz = np.sqrt(snf2 / snf)
    x = np.zeros(len(bins) - 1)
    y = np.zeros(len(bins) - 1)
    for cc in range(len(bins) - 1):
        x[cc] = (bins[cc] + bins[cc + 1]) / 2.0
        y[cc] = rmmeanz[cc]
    
    return x, y, xi, yi


def rotcurve(dd, snap, nshells, rmax):
    #    snap = select_snapshot_number.match_expansion_factor_files(inputfile1, inputfile2, snapshot)
    print("Doing dir %s. snapshot %d" % (dd, snap))
    
    s = gadget_readsnap(snap, snappath=dd + '/', hdf5=True)
    sf = load_subfind(snap, dir=dd + '/', hdf5=True)
    
    s.center = sf.data['fpos'][0, :]
    rad = sf.data['frc2'][0]
    
    dr = rmax / float(nshells)
    
    na = s.nparticlesall
    end = na.copy()
    for i in range(1, len(end)):
        end[i] += end[i - 1]
    
    start = zeros(len(na), dtype='int32')
    for i in range(1, len(start)):
        start[i] = end[i - 1]
    
    mass = pylab.zeros((nshells, 6))
    vel = pylab.zeros((nshells, 6))
    for i in range(6):
        rp = calcGrid.calcRadialProfile(s.pos[start[i]:end[i], :].astype('float64'), s.data['mass'][start[i]:end[i]].astype('float64'), 0, nshells,
                dr, s.center[0], s.center[1], s.center[2])
        
        radius = rp[1, :]
        mass[:, i] = rp[0, :]
        
        for j in range(1, nshells):
            mass[j, i] += mass[j - 1, i]
        
        vel[:, i] = pylab.sqrt(G * mass[:, i] * 1e10 * msol / (radius * 1e6 * parsec)) / 1e5
    
    rp = calcGrid.calcRadialProfile(s.pos.astype('float64'), s.data['mass'].astype('float64'), 0, nshells, dr, s.center[0], s.center[1], s.center[2])
    
    radius = rp[1, :]
    mtot = rp[0, :]
    
    for j in range(1, nshells):
        mtot[j] += mtot[j - 1]
    
    vtot = pylab.sqrt(G * mtot * 1e10 * msol / (radius * 1e6 * parsec)) / 1e5
    lc = vtot * radius * 1e3
    
    return (radius * 1e3, vtot, lc)