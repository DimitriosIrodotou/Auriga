from blackhole import plot_bh_massdifference
from util import select_snapshot_number

dir1 = '/hits/universe/marinafo/GigaGalaxy/level_3/'
dir2 = '/hits/universe/pakmorrr/GigaGalaxy/level5/'
# dir3 = '/cosma5/data/Gigagalaxy/marinafo/campbell/'
dir3 = '/hits/universe/marinafo/GigaGalaxy/level_4/'

outpath = '../plots/blackhole'
suffix = 'pdf'

outputlistfile = '/home/marinafo/scratch/Aquarius/AgeOfAquariusOutputs/ExpansionList_short'
outputlistfile = '/home/marinafo/scratch/Aquarius/AgeOfAquariusOutputs/ExpansionList_64'

runs = ['halo_28']
dirs = [dir1]

z = [10.0]
firstsnap = select_snapshot_number.select_snapshot_number(outputlistfile, z)
z = [0.0]
lastsnap = 45

# plot_bh_hsml.plot_bh_hsml( runs, dirs, outpath, firstsnap, lastsnap, suffix )
# plot_bh_rho.plot_bh_rho( runs, dirs, outpath, firstsnap, lastsnap, suffix )
# plot_bh_press.plot_bh_press( runs, dirs, outpath, firstsnap, lastsnap, suffix )
# plot_bh_utherm.plot_bh_utherm( runs, dirs, outpath, firstsnap, lastsnap, suffix )
# plot_bh_massgrowth.plot_bh_massgrowth( runs, dirs, outpath, firstsnap, lastsnap, suffix )
# plot_bh_mass.plot_bh_mass( runs, dirs, outpath, firstsnap, lastsnap, suffix )
plot_bh_massdifference.plot_bh_massdifference(runs, dirs, outpath, firstsnap, lastsnap,
                                              suffix)  # plot_bh_centerdistance.plot_bh_centerdistance(  # runs, dirs, outpath, firstsnap, lastsnap, suffix )