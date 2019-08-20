from CGM import plot_densprofiles_werk
from CGM import plot_luminosity_profiles_diff
from util import select_snapshot_number

dir1 = '/hits/universe/marinafo/GigaGalaxy/'
dir2 = '/hits/universe/pakmorrr/GigaGalaxy/level5/'
# dir3 = '/cosma5/data/Gigagalaxy/marinafo/campbell/'
dir3 = '/hits/universe/marinafo/GigaGalaxy/level_4/'
dir4 = '/hits/universe/marinafo/GigaGalaxy/level_3/'

# runs = ['halo_10', 'halo_11', 'halo_12', 'halo_13','halo_14', 'halo_15', 'halo_16', 'halo_17', 'halo_18', 'halo_19', 'halo_20', 'halo_16.newSNIa']
# dirs = [ dir1, dir1, dir1, dir1, dir1, dir1, dir2, dir2, dir2, dir2, dir2, dir1]

# runs = ['halo_30', 'halo_31', 'halo_32', 'halo_33','halo_34', 'halo_35', 'halo_36', 'halo_37', 'halo_38', 'halo_39', 'halo_40']
# dirs = [ dir1, dir1, dir1, dir1, dir1, dir1, dir2, dir2, dir2, dir2, dir2]

# runs = ['halo_1', 'halo_2', 'halo_3', 'halo_4','halo_5', 'halo_6', 'halo_7', 'halo_8', 'halo_9', 'halo_10', 'halo_11']
# dirs = [ dir3, dir3, dir3, dir3, dir3, dir3, dir3, dir3, dir3, dir3, dir3]

# runs = ['halo_21', 'halo_22', 'halo_23', 'halo_24','halo_25', 'halo_26', 'halo_27', 'halo_28', 'halo_29', 'halo_30', 'halo_31']
# dirs = [ dir3, dir3, dir3, dir3, dir3, dir3, dir3, dir3, dir3, dir3, dir3]

runs = ['halo_28']
dirs = [dir4]

# runs = ['halo_9', 'halo_9.hiergrav']
# dirs = [dir4, dir3]

nruns = len(runs)

# set here the number of rows and columns to which arrange the panels in the figures
nrows = 1
ncols = 1

outpath = '../plots/CGM'
suffix = 'pdf'
# suffix = 'eps'

# outputlistfile = '/home/marinafo/scratch/Aquarius/AgeOfAquariusOutputs/ExpansionList_256'
outputlistfile = '/home/marinafo/scratch/Aquarius/AgeOfAquariusOutputs/ExpansionList_64'

z = [4.0]
firstsnap = select_snapshot_number.select_snapshot_number(outputlistfile, z)

z = [0.0]
lastsnap = select_snapshot_number.select_snapshot_number(outputlistfile, z)

z = [0.0]
snap = select_snapshot_number.select_snapshot_number(outputlistfile, z)

redshift = [2.0, 1.0, 0.5, 0.0]
snaps = select_snapshot_number.select_snapshot_number(outputlistfile, redshift)

print('Selected snapshot numbers ', snaps)

snap = 42
lastsnap = snap

################ GCM plots ######################################################

plot_densprofiles_werk.plot_densprofiles_werk(runs, dirs, outpath, snap, suffix, nrows, ncols)

# plot_massprofiles.plot_massprofiles( runs, dirs, outpath, snap, suffix, nrows, ncols )

plot_luminosity_profiles_diff.plot_luminosity_profiles_diff_decomp(runs, dirs, outpath, snap, suffix, nrows, ncols, nshells=100)

# plot_phasediagram.plot_phasediagram( runs, dirs, outpath, snap, suffix, nrows, ncols )

# plot_gasmetalposition_referee.plot_gasmetalposition_histogram( runs, dirs, outpath, snap, suffix, nrows, ncols )

# plot_metal_fractions.plot_metal_fractions( runs, dirs, outpath, snap, suffix, nrows, ncols, fgr=True )