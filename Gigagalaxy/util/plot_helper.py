# from scipy.ndimage.filters import gaussian_filter
# from scipy.interpolate import interp1d
# from scipy.ndimage import map_coordinates
from astropy.io import fits
from const import *
from pylab import *

LUKPC = 1000.0

inputfile1 = '/home/marinafo/scratch/Aquarius/AgeOfAquariusOutputs/ExpansionList_short'
inputfile2 = '/home/marinafo/scratch/Aquarius/AgeOfAquariusOutputs/ExpansionList_128'

# see http://www.astro.umd.edu/~ssm/ASTR620/mags.html
Msunabs = [5.61, 5.48, 4.83, 3.30, 5.12, 4.68, 4.57, 4.60]
photo_band = ['U', 'B', 'V', 'K', 'g', 'r', 'i', 'z']


class plot_helper:
    def __init__(self):
        print("Initializing plot_helper")
    
    def find_nearest(self, array, value):
        
        if len(value) == 1:
            idx = (np.abs(array - value)).argmin()
        else:
            idx = np.zeros(len(value))
            for i in range(len(value)):
                idx[i] = (np.abs(array - value[i])).argmin()
        
        return idx
    
    # profile fits
    def hernquist_bulge(self, x, po, a):
        return po / ((x / a) * ((1. + x / a) ** 3))
    
    def NFW(self, r, po, a):
        return po / ((r / a) * ((1. + r / a) ** 2))
    
    def Miyamoto_Nagai(self, x, z, mn, a, b):
        return (b * b * mn / 4 * np.pi) * ((a * x * x + (a + 3 * sqrt(z * z + b * b)) * (a + sqrt(z * z + b * b)) ** 2) / (
                    ((z * z + b * b) ** 1.5) * (x * x + (a + sqrt(z * z + b * b)) ** 2) ** 2.5))
    
    def dble_exponential_disc(self, x, z, mn, a, b):
        return (mn / (4. * np.pi * a * a)) * np.exp(-x / a) * np.exp(-z / (2. * b))
    
    def exponential_disc(self, x, mn, a):
        return (mn / (4. * np.pi * a * a)) * np.exp(-x / a)
    
    def total_baryons(self, p, x, z, y):
        return self.hernquist_bulge(sqrt(x * x + z * z), p[0], p[1]) + self.Miyamoto_Nagai(x, z, p[2], p[3], p[4]) - y
    
    @staticmethod
    def fit_profile(profile, x, y, guess, sigma, bounds):
        (popt, pcov) = curve_fit(profile, x, y, guess, sigma=sigma, bounds=bounds)
        return (popt, pcov)
    
    # profiles
    @staticmethod
    def gaussian_kernel(y, x, sigma):
        kern = (exp(- 0.5 * x * x / (sigma * sigma))) / (sigma * sqrt(2. * np.pi))  # / (2. * np.pi )
        return (kern * y).sum() / kern.sum()
    
    @staticmethod
    def sech2(x):
        return (2. / (np.exp(x) - np.exp(-x))) ** 2
    
    @staticmethod
    def double_exp(r, n, p1, r0, p2):
        y = r.copy()
        y[np.where(r < r0)] = n + r[np.where(r < r0)] * p1
        y[np.where(r >= r0)] = n + r[np.where(r >= r0)] * p2 + r0 * (p1 - p2)
        return y
    
    # @staticmethod
    def sersic_prof1(self, x, ba, b, n):
        return ba * exp(-(x / b) ** n)  # warning!! b is not Reff here!!!
    
    # @staticmethod
    def exp_prof(self, x, da, h):
        return da * exp(-x / h)
    
    def total_profile2(self, x, da, h, ba, b, n, da2, h2):
        y = self.exp_prof(x, da, h) + self.sersic_prof1(x, ba, b, n) + self.exp_prof(x, da2, h2)
        return (y)
    
    # @staticmethod
    def total_profile(self, x, da, h, ba, b, n):
        y = self.exp_prof(x, da, h) + self.sersic_prof1(x, ba, b, n)
        return (y)
    
    def sersic_prof1_log(self, x, ba, b, n):
        return ba - (x / b) ** n
    
    def exp_prof_log(self, x, da, h):
        return da - x / h
    
    def total_profile_log(self, x, da, h, ba, b, n):
        y = self.exp_prof_log(x, da, h) + self.sersic_prof1_log(x, ba, b, n)
        return (y)
    
    def total_profilede(self, x, da, h, da2, h2):
        y = self.exp_prof(x, da, h) + self.exp_prof(x, da2, h2)
        return (y)
    
    # from MacArthur+03 appendix A
    @staticmethod
    def sersic_b_param(n):
        if n <= 0.36:
            b = 0.01945 + n * (- 0.8902 + n * (10.95 + n * (- 19.67 + n * 13.43)))
        else:
            x = 1.0 / n
            b = -1.0 / 3.0 + 2. * n + x * (4.0 / 405. + x * (46. / 25515. + x * (131. / 1148175 - x * 2194697. / 30690717750.)))
        return b
    
    @staticmethod
    def exp_prof_break(x, ab, hi, ho):
        return ab * np.exp(-8. / hi) * np.exp(-(x - 8.) / ho)
    
    # relations
    # table 7 form Rudolph at al (2006)
    @staticmethod
    def bestfit_rudolph_FIR(radius):
        abundance = -0.041 * (radius - 8.5) + 8.42
        return abundance
    
    @staticmethod
    def bestfit_rudolph_OPT(radius):
        abundance = -0.06 * (radius - 8.5) + 8.67
        return abundance
    
    # from Bovy et al. 2014 (Fig. 18)
    @staticmethod
    def bestfit_bovy(radius):
        abundance = -0.09 * (radius - 8.) + 0.03
        return abundance
    
    @staticmethod
    def karim11(mass, c, beta):
        return c * ((mass / 1e11) ** (beta)) * mass * 1e-9
    
    @staticmethod
    def daddi07(mass):
        return 200. * (mass ** 0.9)
    
    @staticmethod
    def elbaz07(mass, redshift):
        if redshift == 0:
            return 8.7 * (mass ** 0.77)
        if redshift == 1:
            return 7.2 * (mass ** 0.9)
    
    @staticmethod
    def elbaz11(mass):
        return 0.12 + (mass - 5e8) / 4e9
    
    @staticmethod
    def magdis10(mass):
        return 350. * ((mass) ** 0.91)
    
    def get_pizagno_data(self):
        tablename = "./data/pizagno.txt"
        
        rmag_p = np.genfromtxt(tablename, comments='#', usecols=3)
        gmag_p = np.genfromtxt(tablename, comments='#', usecols=2)
        vcirc_p = np.genfromtxt(tablename, comments='#', usecols=9)
        color_p = np.genfromtxt(tablename, comments='#', usecols=5)
        mass_p = self.pizagno_convert_color_to_mass(color_p, rmag_p)
        
        return mass_p, vcirc_p
    
    def get_verheijen_data(self):
        tablename = "./data/verheijen.txt"
        
        Bmag_v = np.genfromtxt(tablename, comments='#', usecols=1)
        Rmag_v = np.genfromtxt(tablename, comments='#', usecols=2)
        vcirc_v = np.genfromtxt(tablename, comments='#', usecols=7)
        color_v = Bmag_v - Rmag_v
        mass_v = self.verheijen_convert_color_to_mass(color_v, Bmag_v)
        
        return mass_v, vcirc_v
    
    def get_courteau_data(self):
        tablename = "./data/courteau.txt"
        
        loglum_c = np.genfromtxt(tablename, comments='#', usecols=6)
        vcirc_c = np.genfromtxt(tablename, comments='#', usecols=8)
        mass_c = self.courteau_convert_luminosity_to_mass(loglum_c)
        
        return mass_c, vcirc_c
    
    def get_dutton_bestfit(self, masses):
        return log10(self.obs_tullyfisher_fit(masses))
    
    # The observational fit is taken from equation 24 of Dutton et al. 2011
    def obs_tullyfisher_fit(self, masses):
        vel = 10 ** (2.179 + 0.259 * np.log10(masses / 10 ** 0.3))
        return vel
    
    # see Bell et al. 2003. Table in the appendix
    @staticmethod
    def pizagno_convert_color_to_mass(color, magnitude, band=5):
        mass_to_light = 10 ** (-0.306 + 1.097 * color)
        luminosity = 10 ** (2.0 * (Msunabs[band] - magnitude) / 5.0)
        stellar_mass = log10(mass_to_light * luminosity * 1.0e-10)  # in 10^10 M_sun
        # this converts in the MPA stellar masses eq. 20 of Dutton+11. There is an offset
        # of -0.1 dex that takes into account the Chabrier IMF see sect. 3.5
        stellar_mass = (stellar_mass - 0.230) / 0.922
        return 10 ** stellar_mass
    
    @staticmethod
    def verheijen_convert_color_to_mass(color, magnitude, band=1):
        mass_to_light = 10 ** (-0.976 + 1.111 * color)
        luminosity = 10 ** (2.0 * (Msunabs[band] - magnitude) / 5.0)
        stellar_mass = log10(mass_to_light * luminosity * 1.0e-10)  # in 10^10 M_sun
        # this converts in the MPA stellar masses eq. 20 of Dutton+11. There is an offset
        # of -0.1 dex that takes into account the Chabrier IMF see sect. 3.5
        stellar_mass = (stellar_mass - 0.230) / 0.922
        return 10 ** stellar_mass
    
    # from Dutton et al. (2007) eq. 28
    @staticmethod
    def courteau_convert_luminosity_to_mass(loglum):
        mass_to_light = 10 ** (0.172 + 0.144 * (loglum - 10.3))
        luminosity = 10 ** (loglum)
        stellar_mass = log10(mass_to_light * luminosity * 1.0e-10)  # in 10^10 M_sun
        # this converts in the MPA stellar masses eq. 20 of Dutton+11. There is an offset
        # of -0.1 dex that takes into account the Chabrier IMF see sect. 3.5
        stellar_mass = (stellar_mass - 0.230) / 0.922
        return 10 ** stellar_mass
    
    @staticmethod
    def tiley_bestfit(logm):
        logv = (logm - 1.61) / 4.04
        logvhi = (logm - 1.43) / 3.95
        logvlo = (logm - 1.79) / 4.13
        
        return logv, logvlo, logvhi
    
    @staticmethod
    def guo_abundance_matching(mass):
        # equation 3 of Guo et al. 2010. Mass MUST be given in 10^10 M_sun
        c = 0.129
        M_zero = 10 ** 1.4
        alpha = -0.926
        beta = 0.261
        gamma = -2.44
        
        val = c * mass * ((mass / M_zero) ** alpha + (mass / M_zero) ** beta) ** gamma
        return val
    
    @staticmethod
    def moster_abundance_matching(mass, afac):
        # equation 2 of Moster et al. 2013. Mass units in 10^10 Msun.
        m0 = 11.59
        m1 = 1.195
        n0 = 0.0351
        n1 = -0.0247
        beta0 = 1.376
        beta1 = -0.826
        gamma0 = 0.608
        gamma1 = 0.329
        
        m = 10. ** (m0 + (1 - afac) * m1) * 1e-10
        n = n0 + (1 - afac) * n1
        beta = beta0 + (1 - afac) * beta1
        gamma = gamma0 + (1 - afac) * gamma1
        
        val = 2. * n / ((mass / m) ** (-beta) + (mass / m) ** gamma)
        
        return val * mass
    
    @staticmethod
    def vanderwellate(mstar, a, alpha):
        # Reff vs. stellar mass fits to late type CANDELS galaxies (Table 1., van er Wel 2014)
        return a * (mstar / 5e10) ** alpha
    
    @staticmethod
    def vanderwelearly(mstar, redshift):
        # Reff vs. stellar mass fits to late type CANDELS galaxies (Table 1., van er Wel 2014)
        reff = (10 ** 0.6) * (mstar / 5e10) ** 0.75
        
        return reff
    
    @staticmethod
    def shen(mstar):
        
        alpha = 0.14
        beta = 0.39
        gamma = 0.1
        m0 = 3.98e10
        
        rhalf = gamma * (mstar ** alpha) * ((1 + mstar / m0) ** (beta - alpha))
        
        return rhalf
    
    @staticmethod
    def lange_Sab_Scd(mstar):
        # See Table 1 of Lange et al. 2016
        # For Sab - Scd galaxy types
        a = 5.285
        b = 0.333
        # Lange+ 2015
        # a = 3.971
        # b = 0.204
        
        rhalf_Sab_Scd = a * ((mstar * 1e-10) ** b)
        
        # For S0 - Sa galaxy types
        a = 2.574
        b = 0.326
        
        rhalf_S0_Sa = a * ((mstar * 1e-10) ** b)
        
        return rhalf_Sab_Scd, rhalf_S0_Sa
    
    def shen_best_fit(self, mag, alpha=0.21, beta=0.53, gamma=-1.31, mzero=-20.52):
        R = gamma + (beta - alpha) * log10(1.0 + 10. ** (-0.4 * (mag - mzero))) - 0.4 * alpha * mag
        return 10. ** R
    
    def shen_sigma_fit(self, mag, sigma1=0.48, sigma2=0.25, mzero=-20.52):
        sigmaR = sigma2 + (sigma1 - sigma2) / (1.0 + 10. ** (-0.8 * (mag - mzero)))
        return sigmaR
    
    def shen_upper_env(self, mag):
        R = np.log(self.shen_best_fit(mag)) + self.shen_sigma_fit(mag)
        return np.exp(R)
    
    def shen_lower_env(self, mag):
        R = np.log(self.shen_best_fit(mag)) - self.shen_sigma_fit(mag)
        return np.exp(R)
    
    def get_fits_value(self, fitsfile=None, fitsfield=None, fitsflag=None, index=None):
        """
        fitsfile: the name of the fits file to be read (str)
        fitsfield: the name of the field in the fits file to be read (str)
        fitsflag: list of 2 entries: a name (str) and a value (float)
        """
        
        if fitsfile is not None:
            infofits = fits.open(fitsfile)
        else:
            raise ValueError('no fits file specified')
        
        if fitsfield is not None:
            tmp = infofits[1].data.field(fitsfield)
            if fitsflag:
                warning = infofits[1].data.field(fitsflag[0])
                mask = warning == 0
                tmp = tmp[mask]
                mask = tmp < fitsflag[1]
                if index:
                    value = tmp[mask, index]
                else:
                    value = tmp[mask]
            elif index >= 0:
                value = tmp[:, index]
            else:
                value = tmp
        
        infofits.close()
        
        return value