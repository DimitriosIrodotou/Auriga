import matplotlib

matplotlib.use('Agg')
from loadmodules import *

# standard modules
# chemodynamic modules

MHD = True
level = 4
fast = False
ht = False
L = False

rtype = 'MHD'  # DM, MHD, HD

if level == 5:
    dir5 = '/hits/universe/GigaGalaxy/level5/'
    outpath = '/home/grandrt/analysis/level5_MHD/'
    runs = ['halo6_MHD', 'halo16_MHD', 'halo24_MHD']
    runs = ['halo_6', 'halo_6_metal1cell', 'halo_6_TNGyields', 'halo_6_metal1cell_TNGyields']
    outputlistfile = '/home/grandrt/explists/ExpansionList_128'
    outputlistgyr = '/home/grandrt/explists/timegyrlist_128'
    
    dirs = [dir5] * len(runs)
    nrows, ncols = 2, 2

if level == 4:
    # runs = ['halo_6']
    
    ##### Discy halos (21) #####
    # runs = ['halo_2', 'halo_3', 'halo_4', 'halo_5', 'halo_6', 'halo_7', 'halo_8', 'halo_9','halo_12', 'halo_14', 'halo_15', 'halo_16', 'halo_17',
    # 'halo_18', 'halo_19', 'halo_20', 'halo_21', 'halo_23', 'halo_24', 'halo_25', 'halo_27']
    
    # nrows, ncols = 6, 4
    
    ##### Non-discy haloes #####
    # runs = ['halo_4','halo_7','halo_8', 'halo_10', 'halo_13','halo_14', 'halo_20', 'halo_22','halo_26','halo_28','halo_29','halo_30']
    
    #####  For all halos (30) #####
    # runs = ['halo_1','halo_2', 'halo_3','halo_4','halo_5', 'halo_6','halo_7','halo_8', 'halo_9', 'halo_10','halo_11','halo_12', 'halo_13',
    # 'halo_14', 'halo_15', 'halo_16', 'halo_17', 'halo_18', 'halo_19','halo_20', 'halo_21','halo_22', 'halo_23', 'halo_24', 'halo_25','halo_26',
    # 'halo_27','halo_28','halo_29','halo_30']
    
    # runs = ['halo_L1','halo_L2', 'halo_L3','halo_L4','halo_L5', 'halo_L6','halo_L7','halo_L8', 'halo_L9', 'halo_L10']
    
    ##### Size ordering #####
    # scatter plots
    # runs = ['halo_1','halo_4', 'halo_5', 'halo_6','halo_7', 'halo_9','halo_12', 'halo_13','halo_14', 'halo_15', 'halo_17', 'halo_18', 'halo_19',
    # 'halo_21', 'halo_23', 'halo_24', 'halo_27','halo_30','halo_2','halo_3','halo_8', 'halo_16','halo_20', 'halo_25', 'halo_10','halo_11',
    # 'halo_22','halo_26','halo_28','halo_29']
    # multipanel plots
    # runs = ['halo_2','halo_3','halo_8', 'halo_16','halo_20', 'halo_25', 'halo_10','halo_11','halo_22','halo_26','halo_28','halo_29','halo_1',
    # 'halo_4', 'halo_5', 'halo_6','halo_7', 'halo_9','halo_12', 'halo_13','halo_14', 'halo_15', 'halo_17', 'halo_18', 'halo_19', 'halo_21',
    # 'halo_23', 'halo_24', 'halo_27','halo_30']
    
    nruns = len(runs)
    
    if rtype is 'MHD':
        if not L:
            dir5 = '/hits/universe/GigaGalaxy/level4_MHD/'
        else:
            dir5 = '/hits/universe/GigaGalaxy/level4_MHD_new/'
        
        outputlistfile = ['/home/grandrt/explists/ExpansionList_128']
        outputlistgyr = ['/home/grandrt/explists/timegyrlist_128']
        outpath = '/home/grandrt/analysis/level4_MHD/'
    elif rtype is 'HD':
        dir5 = '/hits/universe/GigaGalaxy/level4/'
        outputlistfile = ['/home/grandrt/explists/ExpansionList_hrt']
        outputlistgyr = ['/home/grandrt/explists/timegyrlist_128']  # need to make hrt!
        outpath = '/home/grandrt/analysis/level4/'
    elif rtype is 'DM':
        dir5 = '/hits/universe/GigaGalaxy/level4_DM/'
        outputlistfile = ['/home/grandrt/explists/ExpansionList_128']
        outputlistgyr = ['/home/grandrt/explists/timegyrlist_128']  # need to make hrt!
        outpath = '/home/grandrt/analysis/level4_DM/'
    else:
        raise ValueError('inappropriate rtype value given. Stopping.')
    
    dirs = [dir5] * nruns
    # outpath2 *= nruns
    outputlistfile *= nruns
    nrows, ncols = 1, nruns

elif level == 3:
    dir5 = '/hits/universe/GigaGalaxy/level3_MHD/'
    outputlistfile = '/home/grandrt/explists/ExpansionList_64'
    outputlistgyr = '/home/grandrt/explists/timegyrlist_64'
    outpath = '/home/grandrt/analysis/level3_MHD/'
    runs = ['halo_6', 'halo_16', 'halo_21', 'halo_23', 'halo_24', 'halo_27']
    dirs = [dir5] * len(runs)
    nrows, ncols = 1, len(runs)
    nrows, ncols = 2, 3
    lastsnap = 63

Aq = False
if Aq:
    dir5 = '/hits/universe/marinafo/tap/Aquarius/'
    outputlistfile = '/home/grandrt/explists/ExpansionList_64'
    outputlistgyr = '/home/grandrt/explists/timegyrlist_64'
    outpath = '/home/grandrt/analysis/Aq4/'

## High time res run
if ht:
    runs = ['halo_6_1024snaps']
    outputlistfile = ['/home/grandrt/explists/ExpansionList_1024'] * len(runs)

dirs = [dir5] * len(runs)

#####  For resolution study #####
res = False
# res = [5,4]
if res:
    runs = ['halo_6', 'halo_6']  # ,'halo6', 'halo16_MHD','halo_16','halo16', 'halo24_MHD','halo_24','halo24']
    nruns = int(len(runs) / len(res))
    dirs = []
    outpath2 = []
    outputlistfile = []
    for i in res:
        dirstr = '/hits/universe/GigaGalaxy/level%01d_MHD/' % i
        dirs.append(dirstr)
        outpathstr = '/home/grandrt/analysis/level%01d_MHD/' % i
        outpath2.append(outpathstr)
        if i == 5:
            outputlistfile.append('/home/grandrt/explists/ExpansionList_64')
        elif i == 4:
            outputlistfile.append('/home/grandrt/explists/ExpansionList_128')
        elif i == 3:
            outputlistfile.append('/home/grandrt/explists/ExpansionList_64')
    
    dirs *= nruns
    outpath2 *= nruns
    outputlistfile *= 2
    nrows, ncols = 1, nruns

suffix = 'pdf'
zlist = [3., 2., 1., 0.]
zlist = [2., 1., 0.8, 0.6, 0.4, 0.2, 0.1, 0.]
# nrows, ncols = 1, len(zlist)

print("runs=", runs)

################ main galaxy properties ########################################

# plot_stellarvstotalmass.plot_stellarvstotalmass( runs, dirs, outpath+'/plotall/', nrows, ncols, outputlistfile, zlist, suffix, restest=res,
# colorbymass=True, dat='Moster' )

# plot_stellarmass_vs_age.plot_stellarmass_vs_age( runs, dirs, outpath, nrows, ncols, outputlistfile, zlist, suffix, restest=res,
# colorbymass=False, dat=True )

# plot_stellarmass_vs_metallicity.plot_stellarmass_vs_metallicity( runs, dirs, outpath, nrows, ncols, outputlistfile, zlist, suffix, restest=res,
# colorbymass=False, dat=True )

# plot_tullyfisher_mag.plot_tullyfisher_mag( runs, dirs, outpath, nrows, ncols, outputlistfile, zlist, suffix, restest=res, colorbymass=False,
# plot_max_val=False, dat=['pizagno','verheijen','courteau','dutton','tiley'] )

# plot_alphavsiron_stars.plot_alphavsiron_stars( runs, dirs, outpath, outputlistfile, suffix, nrows, ncols, disc_stars=False )

# plot_sfr_vs_stellarmass.plot_sfr_vs_stellarmass( runs, dirs, outpath, nrows, ncols, outputlistfile, zlist, suffix, time_lim=0.5, restest=res,
# colorbymass=False, dat=True )

# plot_rotcurve_multi.plot_rotcurve_multi( runs, dirs, outpath, nrows, ncols, outputlistfile, zlist, suffix, restest=res, dat=None )

# plot_circularities_decomp_multi_age.plot_circularities_decomp_multi_age( runs, dirs, outpath+'/plotall/', outputlistfile, [0.], suffix, nrows,
# ncols, '', restest=res, normalize_bins=True, lzcirc=False, accretedfiledir=outpath )

# plot_decompsfr_multi_z0.plot_decompsfr_multi( runs, dirs, outpath+'/plotall/', '/plots_grand/', outputlistfile, suffix, nrows, ncols, restest=res )

# plot_stellarmass_vs_halfmassrad.plot_stellarmass_vs_halfmassrad( runs, dirs, outpath+'/plotall/', nrows, ncols, outputlistfile, zlist, 'rmag',
# suffix, restest=res )

# plot_staroxygen_profiles.plot_staroxygen_profiles( runs, dirs, outpath, outputlistfile, [0.], suffix, nrows, ncols, nshells=35, reltosun=False )

# plot_petrosian_rad.plot_petrosian_rad( runs, dirs, outpath+'/plotall/', 1, len(zlist), outputlistfile, zlist, 'rmag', suffix, restest=res,
# colorbymass=True )

# plot_stellarsurfdensity.plot_stellar_surfden(runs, dirs, outpath+'/plotall/', outputlistfile, zlist, suffix, nrows, ncols, iband=None, restest=res)

# plot_angularmomentum_profile.plot_angularmomentum_profile( runs, dirs, outpath+'/plotall/', outputlistfile, zlist, suffix, nrows, ncols )

# plot_vertical_density.plot_vertical_density(runs, dirs, outpath, outputlistfile, [0.], suffix, nrows, ncols, disc_stars=False)

################ chemodynamic plots ########################################

# plot_sigma_figure.plot_sigma_figure( runs, dirs, outpath+'/plotall/', outputlistfile, zlist, suffix, nrows, ncols, agecut=[[0., 1., 5., 0.],[1.,
# 5., 13., 13.]], ecut=0.7, restest=res)

# plot_birth_dist.plot_birth_dist( runs, dirs, outpath, lastsnap, alpha=0, disc_stars=False )

# plot_AVR.plot_AVR( runs, dirs, outpath, nrows, ncols, outputlistfile, suffix, ddir='z', rbinfac=[1., 2., 3., 4.],
# birthdatafile='/birth_current-pdat_IDs.dat', restest=res, accreted=False)

# get_birth_quantities_nobpos.get_birth_quantities_nobpos( runs, dirs, outputlistfile, outpath )

# get_accreted_insitu_particles.get_accreted_insitu_particles( outputlistfile, runs, dirs, outpath, lastsnap, src=True, idwrite=True,
# raw_age=False, mergertree=True )

# plot_stellarsurfdensity.plot_stellar_surfden(runs, dirs, outpath+'/plotall/', outputlistfile, zlist, suffix, nrows, ncols, iband=None, restest=res)

# plot_staroxygen_profiles.plot_staroxygen_profiles( runs, dirs, outpath, outputlistfile, [0.], suffix, nrows, ncols, nshells=35, reltosun=False )

# plot_alpha_vs_iron_Rz.plot_alpha_vs_iron_Rz( runs, dirs, outpath, outputlistfile, [0.], zcut=[[1., 0.5, 0.],[2., 1., 0.5]], rcut=[[3., 7., 11.,
# 15.],[5., 9., 13., 17.]], accretedfiledir=outpath, birthdatafile='/birth_current-pdat_IDs.dat', atbirth=False, disc_stars=False)

# plot_metal_gradient_star.plot_metal_gradient_star( runs, dirs, outpath, outputlistfile, [0.], suffix, nrows, ncols, slicedir='z', pptype=4,
# birthdatafile='/birth_current-pdat_IDs.dat', atbirth=False, disc_stars=False )

# plot_angmom_fluxtensor_map.plot_angmom_fluxtensor( runs, dirs, outpath, outputlistfile, [0.], suffix, nrows, ncols )

# plot_ToomresQ.plot_ToomresQ( runs, dirs, outpath, outputlistfile, zlist, suffix, nrows, ncols, gas=True )

# plot_vertical_density_evol.plot_vertical_density_evol(runs, dirs, outpath, outputlistfile, zlist, suffix, nrows, ncols)

# plot_MAPs.plot_MAPs( runs, dirs, outpath+'/plotall/', outputlistfile, zlist, suffix, nrows, ncols, ecut=False, agecut=False )

# plot_angularmomentum_profile.plot_angularmomentum_profile( runs, dirs, outpath+'/plotall/', outputlistfile, zlist, suffix, nrows, ncols )