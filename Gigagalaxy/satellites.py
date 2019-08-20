from satellites import plot_luminosity_distribution
from satellites import plot_mass_distribution
from satellites import plot_stellar_mass_distribution
from satellites import plot_vcirc_distribution

# outputlistfile = '/home/marinafo/scratch/Aquarius/AgeOfAquariusOutputs/ExpansionList_64'

runs = ['CDMlv2', 'ETHOSlv2']
dirs = ['/n/home00/fmarinacci/SIDM/CDM/', '/n/home00/fmarinacci/SIDM/ETHOS-4/']

nruns = len(runs)

outpath = '../plots'
suffix = 'pdf'

# redshift = [2.0, 1.0, 0.5, 0.0]
# snaps = select_snapshot_number.select_snapshot_number( outputlistfile, redshift )

# print 'Selected snapshot numbers ', snaps
# print 'Corresponding to expansion factor ', redshift #redshift contains the expansion factor after the function call
# print

# redshift = [0.0]
# snap = select_snapshot_number.select_snapshot_number( outputlistfile, redshift )

# print 'Selected snapshot number' , snap
# print 'Corresponding to expansion factor ', redshift #redshift contains the expansion factor after the function call
# print

snap = 104

############################################################################################################################
# plot_halflightradii.plot_halflightradii( runs, dirs, outpath, snap, suffix, bandlist=['V'] )

# plot_masstolightratios.plot_masstolightratios( runs, dirs, outpath, snap, suffix )

# plot_dynamicalmasses.plot_dynamicalmasses( runs, dirs, outpath, snap, suffix )

# plot_metallicities.plot_metallicities( runs, dirs, outpath, snap, suffix )

# plot_surfbrightness.plot_surfacebrightness( runs, dirs, outpath, snap, suffix )

# plot_halflightvshalfmass.plot_halflightvshalfmass( runs, dirs, outpath, snap, suffix, bandlist=['V'] )

# plot_sigmastars.plot_sigmastars( runs, dirs, outpath, snap, suffix )

# plot_HIfracvsradius.plot_HIfracvsradius( runs, dirs, outpath, snap, suffix )

########################################3 POSSIBLE ADDITIONAL PLOTS ########################################################
############################################################################################################################

# plot_mgasvsradius.plot_mgasvsradius( runs, dirs, outpath, snap, suffix )

# plot_mstarvsradius.plot_mstarvsradius( runs, dirs, outpath, snap, suffix )

# plot_sfrvsradius.plot_sfrvsradius( runs, dirs, outpath, snap, suffix )

# plot_mstarvsvcirc.plot_mstarvsvcirc( runs, dirs, outpath, snap, suffix )

# plot_mstarvsphotometry.plot_all_photometric_bands( runs, dirs, outpath, snap, suffix )

# plot_vcircvsphotometry.plot_all_photometric_bands( runs, dirs, outpath, snap, suffix )

# plot_mstarvsmhalo.plot_mstarvsmhalo( runs, dirs, outpath, snap, suffix )

# plot_mstarvssize.plot_mstarvssize( runs, dirs, outpath, snap, suffix )

# plot_mhalovssize.plot_mhalovssize( runs, dirs, outpath, snap, suffix )

# plot_gasfracvsradius.plot_gasfracvsradius( runs, dirs, outpath, snap, suffix )

plot_luminosity_distribution.cumulative_luminosity_function(runs, dirs, outpath, snap, suffix, rcut=300., mhd=False, newfig=False)

# plot_msfrgasvsradius.plot_msfrgasvsradius( runs, dirs, outpath, snap, suffix )

# plot_oxygenvsiron.plot_oxygenvsiron( runs, dirs, outpath, snap, suffix )

# plot_ironvsphotometry.plot_ironvsphotometry( runs, dirs, outpath, snap, suffix )

# plot_fracsfrgasvsradius.plot_fracsfrgasvsradius( runs, dirs, outpath, snap, suffix )

# plot_HImassvsradius.plot_HImassvsradius( runs, dirs, outpath, snap, suffix )

# plot_aitoffprojection.plot_aitoffprojection( runs, dirs, outpath, snap, suffix )

# plot_sfr_multi.plot_sfr_multi( runs, dirs, outpath, snap, suffix )

# plot_sfr_multi.plot_sfr_multi_cumulative( runs, dirs, outpath, snap, suffix )

# plot_3D_positions.plot_3D_positions( runs, dirs, outpath, snap, suffix )

############################################################################################################################

# plot_starvstotal_mass.plot_stellarvstotalmass( runs, dirs, outpath, snap, suffix )

plot_vcirc_distribution.plot_number_vs_vcirc_cumulative(runs, dirs, outpath, snap, suffix, rcut=300., logscale=True)

# plot_vcirc_distribution.plot_number_vs_vcirc_differential( runs, dirs, outpath, snap, suffix, logscale=True )

# plot_mass_distribution.plot_number_vs_mass_differential( runs, dirs, outpath, snap, suffix, logscale=True )

plot_mass_distribution.plot_number_vs_mass_cumulative(runs, dirs, outpath, snap, suffix, rcut=300., logscale=True)

# plot_luminosity_distribution.differential_luminosity_function( runs, dirs, outpath, snap, suffix )

# plot_stellar_mass_distribution.plot_number_vs_stellar_mass_differential( runs, dirs, outpath, snap, suffix, logscale=True )

plot_stellar_mass_distribution.plot_number_vs_stellar_mass_cumulative(runs, dirs, outpath, snap, suffix, rcut=300., logscale=True)

############################################################################################################################