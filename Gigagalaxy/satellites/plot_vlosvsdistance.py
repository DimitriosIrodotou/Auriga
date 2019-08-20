import numpy as np
import read_McConnachie_table as read_data
from pylab import *

colors = ['g', 'b', 'r', 'k', 'c', 'y', 'm', 'purple']
toinch = 0.393700787
base = '/output/fof_subhalo_tab_'

GRAVITY = 43.0071


def escape_velocity(mass, radius):
    vesc = 2.0 * GRAVITY * mass / radius
    return np.sqrt(vesc)


def plot_vlosvsdistance(outpath, suffix):
    fac = 1.0
    fig = figure(figsize=np.array([16.5 * fac, 16.3]) * toinch * 0.5, dpi=300)
    ax = axes([0.17 / fac, 0.13, 0.8 / fac, 0.8])
    
    data = read_data.read_data_catalogue(filename='./data/McConnachie2012_extended.dat')
    data.read_table()
    
    radii = arange(0.1, 600.)
    vesc = escape_velocity(mass=100., radius=radii * 1.0e-3)
    
    i, = np.where((data.MWdistance < 9000) & (data.MWvelocity < 9000))
    ax.plot(radii, vesc, ':', ms=3.0, color='gray')
    ax.plot(radii, -vesc, ':', ms=3.0, color='gray')
    ax.axhline(y=0, xmin=0, xmax=1, color='gray', ls=':')
    ax.plot(data.MWdistance[i], sqrt(3.0) * data.MWvelocity[i], 'o', ms=5.0, mec='None', mfc='b')
    
    # legend( loc='upper left', fontsize=6, frameon=False, numpoints=1 )
    
    xlim(0.0, 600.)
    ylim(-500.0, 500.)
    
    minorLocator = MultipleLocator(20.0)
    ax.xaxis.set_minor_locator(minorLocator)
    minorLocator = MultipleLocator(50.0)
    ax.yaxis.set_minor_locator(minorLocator)
    
    xlabel("$\\rm{D\\,[kpc]}$")
    ylabel("$\\rm{\\sqrt{3}\,v_{los}\\,[km\\,s^{-1}]}$")
    
    savefig("%s/vlosvsdistance.%s" % (outpath, suffix), dpi=300)


outpath = '../plots'
suffix = 'pdf'

plot_vlosvsdistance(outpath, suffix)