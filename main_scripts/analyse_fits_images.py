import re
import os
import PIL
import time
import glob
import plot_tools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style

from PIL import Image
from astropy.io import fits
from matplotlib import gridspec
from scipy.optimize import curve_fit
from matplotlib.patches import Circle
from photutils.isophote import Ellipse
from photutils.isophote import EllipseGeometry
from scipy.ndimage.filters import gaussian_filter

style.use("classic")
plt.rcParams.update({'font.family':'serif'})
date = time.strftime("%d_%m_%y_%H%M")


def convert_to_fit(name):
    """
    Add Gaussian blur, convert to gray scale and save an image in 256x256 along with its fits version.
    :param name: name of the file.
    :return:
    """
    # Load a png image and convert to 256x256 #
    image_png = Image.open(name + '.png').convert("L")
    image_png = image_png.resize((256, 256), PIL.Image.NEAREST)

    # Create and save a fits version of the rescaled image #
    hdu = fits.PrimaryHDU(image_png)
    # hdu.scale('int16', 'old')  # Convert to 16 bit image.
    hdu.writeto(name + '_ctf.fits', overwrite=True)

    # Increase the intensity of all pixels by the same factor #
    image_fits = fits.getdata(name + '_ctf.fits', ext=0)
    # image_fits *= 20

    # Generate the figure and set its parameters #
    figure, axis = plt.subplots(1, frameon=False)
    plt.axis('off')
    axis.set_aspect('equal')

    # Add Gaussian blur #
    FWHM = 2
    sigma = FWHM / np.sqrt(8 * np.log(2))
    image_fits = gaussian_filter(image_fits, sigma=sigma)
    array = np.asarray(image_fits)

    # Generate the figure and set its parameters #
    figure, axis = plt.subplots(1, frameon=False)
    plt.axis('off')
    axis.set_aspect('equal')

    # Create and save the image #
    plt.imsave(name + '_ctf.png', array, cmap='gray_r')
    plt.close()

    # Create and save a fits version of the image #
    hdu = fits.PrimaryHDU(array)
    hdu.writeto(name + '_ctf.fits', overwrite=True)
    return None


def fit_isophotal_ellipses(name, ellipticity):
    """
    Use the photutils package to fit isophotal ellipses on fits images.
    :param name: name of the file.
    :param ellipticity: initial ellipticity.
    :return: None
    """
    # Load the fits image and calculate the centre #
    image_fits = fits.getdata(name + '_ctf.fits', ext=0)
    centre = get_image_centre(name)

    # Provide the elliptical isophote fitter with an initial ellipse (geometry) and fit multiple isophotes to the
    # image array #
    geometry = EllipseGeometry(x0=centre[0], y0=centre[1], fix_center=True, sma=centre[0] / 10, eps=ellipticity,
                               pa=1e-2)
    ellipse = Ellipse(image_fits, geometry)
    isolist = ellipse.fit_image(minsma=1, maxsma=centre[0], step=0.25)
    print(isolist.to_table())  # Print the isophote values as a table sorted by the semi-major axis length.
    print('average ellipticity:', np.mean(isolist.eps))

    # Fit a Sersic plus exponential profile #
    # popt, pcov = curve_fit(fit_total_profile, isolist.sma, isolist.intens,
    #     p0=[isolist.intens[0], 2, isolist.intens[0], 2, 4])  # p0 = [I_0d, R_d, I_0b, b, n]
    # I_0d, R_d, I_0b, b, n = popt[0], popt[1], popt[2], popt[3], popt[4]
    # R_eff = b * sersic_b_n(n) ** n
    # print('I_0d:', I_0d, 'h:', R_d, 'I_0b:', I_0b, 'n:', n, 'R_eff:', R_eff)

    # Fit an exponential profile #
    popt, pcov = curve_fit(fit_exponential_profile, isolist.sma, isolist.intens, p0=[isolist.intens[0], 1])
    I_0d, R_d = popt[0], popt[1]
    print('I_0d:', I_0d, 'h:', R_d)
    return None


def combine_images(name, ellipticity):
    """
    Load and show the fits images and combine with ctf and isophote sample.
    :param name: name of the file.
    :param ellipticity: initial ellipticity.
    :return: None
    """
    # Load the fits image and show it #
    centre = get_image_centre(name)
    image_fits = fits.getdata(name + '_ctf.fits', ext=0)
    os.chdir(Imfit_path + name)  # Change to each halo's Imfit directory
    model = fits.getdata(name + '_model.fits', ext=0)
    residual = fits.getdata(name + '_residual.fits', ext=0)

    # Provide the elliptical isophote fitter with an initial ellipse (geometry) and fit multiple isophotes to the
    # image array #
    geometry = EllipseGeometry(x0=centre[0], y0=centre[1], fix_center=True, sma=centre[0] / 10, eps=ellipticity,
                               pa=1e-2)
    ellipse = Ellipse(image_fits, geometry)
    isolist = ellipse.fit_image(minsma=1, maxsma=centre[0], step=0.25)

    # Plot the original data with some of the isophotes, the elliptical model image, and the residual image #
    # Generate the figure and set its parameters #
    figure = plt.figure(figsize=(20, 7.5))
    gs = gridspec.GridSpec(2, 4, hspace=0.3, wspace=0.4)
    axis00, axis01, axis02, axis03 = plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[0, 2]), plt.subplot(
        gs[0, 3])
    axis10, axis11, axis12, axis13 = plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1]), plt.subplot(gs[1, 2]), plt.subplot(
        gs[1, 3])

    axes = [axis00, axis01, axis02, axis03]
    images = [image_fits, model, image_fits/model, residual]
    for axis, image in zip(axes, images):
        plot_tools.set_axis(axis, xlim=[0, 256], ylim=[0, 256], xlabel=r'$\mathrm{x/pix}$', ylabel=r'$\mathrm{y/pix}$',
                            which='major', aspect=None)
        axis.grid(True, color='gray', linestyle='-')
        axis.imshow(image, origin='upper', cmap='gray_r', interpolation='bicubic')

    for axis in [axis00, axis01, axis02, axis03]:
        labels = ['0', '', '100', '', '200', '']
        axis.set_xticklabels(labels)
        axis.set_yticklabels(labels)

    figure.text(0.01, 0.92, r'$\mathrm{%s}$' % str(name), color='k', fontsize=16, transform=axis00.transAxes)
    figure.text(0.01, 0.92, r'$\mathrm{Model}$', color='k', fontsize=16, transform=axis01.transAxes)
    figure.text(0.01, 0.92, r'$\mathrm{Input/Model}$', color='k', fontsize=16, transform=axis02.transAxes)
    figure.text(0.01, 0.92, r'$\mathrm{Residual}$', color='k', fontsize=16, transform=axis03.transAxes)

    # Provide the elliptical isophote fitter with an initial ellipse (geometry) and fit multiple isophotes to the
    # image array #
    geometry = EllipseGeometry(x0=centre[0], y0=centre[1], fix_center=True, sma=centre[0] / 10, eps=ellipticity,
                               pa=1e-2)
    ellipse = Ellipse(model, geometry)
    isolist_model = ellipse.fit_image(minsma=1, maxsma=centre[0], step=0.25)

    axes = [axis10, axis11, axis12, axis13]
    y_labels = [r'$\mathrm{Ellipticity}$', r'$\mathrm{PA/\deg}$', r'$\mathrm{Pixels\;inside\;each\;ellipse}$',
                r'$\mathrm{Mean\,intensity}$']
    y_values = [isolist.eps, isolist.pa / np.pi * 180., isolist.tflux_e, isolist.intens]
    y_lims = [(-0.1, 1.1), (-20, 220), (0.8 * min(isolist.tflux_e), 1.2 * max(isolist.tflux_e)),
              (0.8 * min(isolist.intens), 1.2 * max(isolist.intens))]
    y_errors = [isolist.ellip_err, isolist.pa_err / np.pi * 180., np.zeros(len(isolist.x0_err)), isolist.int_err]
    for axis, y_label, y_value, y_error, y_lim in zip(axes, y_labels, y_values, y_errors, y_lims):
        axis.errorbar(isolist.sma, y_value, yerr=y_error, fmt='o', color='k', markersize=10)
        plot_tools.set_axis(axis, xlim=[1e0, centre[0]], ylim=y_lim, xscale='log', xlabel=r'$\mathrm{sma/pix}$',
                            ylabel=y_label, which='major', aspect=None)
    axis12.set_yscale('log')
    axis12.set_ylim(1e3, 2e7)

    y_values = [isolist_model.eps, isolist_model.pa / np.pi * 180., isolist_model.tflux_e, isolist_model.intens]
    y_errors = [isolist_model.ellip_err, isolist_model.pa_err / np.pi * 180., np.zeros(len(isolist_model.x0_err)),
                isolist_model.int_err]
    for axis, y_value, y_error in zip(axes, y_values, y_errors):
        axis.errorbar(isolist_model.sma, y_value, yerr=y_error, fmt='o', color='tab:red', markersize=10,
                      label=r'$\mathrm{Model}$')
    axis10.legend(loc='upper right', fontsize=16, frameon=False, numpoints=1)

    # # Fit a Sersic plus exponential profile #
    # popt, pcov = curve_fit(fit_total_profile, isolist.sma, isolist.intens,
    #                        p0=[isolist.intens[0], 2, isolist.intens[0], 2, 4])  # p0 = [I_0d, R_d, I_0b, b, n]
    # I_0d, R_d, I_0b, b, n = popt[0], popt[1], popt[2], popt[3], popt[4]
    # axis13.plot(isolist.sma, fit_total_profile(isolist.sma, popt[0], popt[1], popt[2], popt[3], popt[4]),
    # color='purple')
    # R_eff = b * sersic_b_n(n) ** n
    # axis13.axvline(x=R_d, color='red')
    # axis13.axvline(x=R_eff, color='blue')
    #
    # # Fit an exponential profile #
    # popt, pcov = curve_fit(fit_exponential_profile, isolist.sma, isolist.intens, p0=[isolist.intens[0], 1])
    # axis13.plot(isolist.sma, fit_exponential_profile(isolist.sma, popt[0], popt[1]), color='black')
    # I_0d, R_d = popt[0], popt[1]
    # axis13.axvline(x=R_d, color='black')

    plt.savefig(name + '_model_' + str(date) + '.eps', bbox_inches='tight')  # Save the figure.
    return None


def fit_exponential_profile(x, I_0, R_d):
    """
    Calculate an exponential profile.
    :param x: x data.
    :param I_0: Central intensity.
    :param R_d: Scale length.
    :return: I_0 * np.exp(-x / R_d)
    """
    return I_0 * np.exp(-x / R_d)


def fit_sersic_profile(r, I_0b, b, n):
    """
    Calculate a Sersic profile.
    :param r: radius.
    :param I_0b: Spheroid central intensity.
    :param b: Sersic b parameter
    :param n: Sersic index
    :return: I_0b * np.exp(-(r / b) ** (1 / n))
    """
    return I_0b * np.exp(-(r / b) ** (1 / n))  # b = R_eff / b_n ^ n


def fit_total_profile(r, I_0d, R_d, I_0b, b, n):
    """
    Calculate a total (Sersic + exponential) profile.
    :param r: radius.
    :param I_0d: Disc central intensity.
    :param R_d: Disc scale length.
    :param I_0b: Spheroid central intensity.
    :param b: Sersic b parameter.
    :param n: Sersic index.
    :return: exponential_profile(r, I_0d, R_d) + sersic_profile(r, I_0b, b, n)
    """
    y = fit_exponential_profile(r, I_0d, R_d) + fit_sersic_profile(r, I_0b, b, n)
    return y


def sersic_b_n(n):
    """
    Calculate the Sersic b parameter.
    :param n: Sersic index.
    :return: b_n
    """
    if n <= 0.36:
        b_n = 0.01945 + n * (- 0.8902 + n * (10.95 + n * (- 19.67 + n * 13.43)))
    else:
        x = 1.0 / n
        b_n = -1.0 / 3.0 + 2. * n + x * (
            4.0 / 405. + x * (46. / 25515. + x * (131. / 1148175 - x * 2194697. / 30690717750.)))
    return b_n


def get_image_centre(name):
    """
    Get the centre of the image in pixels.
    :param name: name of the file.
    :return: centre
    """
    # Load the png image and calculate the centre #
    im = Image.open(name + '_ctf.png')

    (X, Y) = im.size
    centre = X / 2, Y / 2
    return centre


def plot_fits_image(name):
    """
    Load and show a fits image.
    :param name: name of the file.
    :return: None
    """
    # Load the fits image and show it #
    image_fits = fits.getdata(name + '.fits', ext=0)

    # Generate the figure and set its parameters #
    figure, axis = plt.subplots(1, figsize=(10, 10), frameon=False)
    plt.axis('off')
    axis.set_aspect('equal')

    # Create and save a gray scaled version of the image #
    plt.imsave(name + str(date) + '.png', image_fits, cmap='gray')
    plt.close()
    return None


def plot_fit_data(h=0.0, R_eff=0.0):
    """
    Over-plot the scale length and effective radius on the fits image.
    :return: None
    """
    # Load the fits image and show it #
    image_fits = fits.getdata(name + '.fits', ext=0)

    # Generate the figure and set its parameters #
    figure, axis = plt.subplots(1, figsize=(10, 10), frameon=False)
    plt.axis('off')
    axis.set_aspect('equal')

    axis.imshow(image_fits, cmap='gray')
    centre = get_image_centre(name)
    # Plot the scale length
    for radius in [h, R_eff]:
        circle = Circle((centre[0], centre[1]), radius, color='tab:red', fill=False)
        axis.add_patch(circle)

    # Create and save a gray scaled version of the image #
    plt.savefig(name + '_2_' + str(date) + '.eps', cmap='gray', bbox_inches='tight')
    plt.close()
    return None


# Define the local paths to the images #
Imfit_path = '/Users/Bam/PycharmProjects/Auriga/Imfit/Auriga/'
plots_path = '/Users/Bam/PycharmProjects/Auriga/plots/projections/Imfit/'

# Get the names and sort them #
names = glob.glob(plots_path + 'Au-')
names = [re.split('/Imfit|/', name)[-1] for name in names]
names.sort()

# Loop over all Auriga rbm images, convert them to the appropriate format and fit isophotal ellipses #
for name in names:
    # Prepare the image and fit isophotal ellipses #
    ellipticity = 0.55  # Set to 0.5 the first time you fit a galaxy and then to the minimum.
    os.chdir(plots_path + name)  # Change to each halo's plots directory
    # convert_to_fit(name)
    # fit_isophotal_ellipses(name, ellipticity)

    # Use Imfit to analyse the image #  # --bootstrap 15
    os.chdir(Imfit_path + name)  # Change to each halo's Imfit directory.
    # os.system('../../makeimage -o %s_psf.fits %s_config_Gaussian_psf.dat' % (name, name))  # Create the PSF image
    os.system('../../imfit -c %s_config.dat --nm --cashstat '
              '../../../plots/projections/Imfit/%s/%s_ctf.fits '
              '--save-model=%s_model.fits '
              '--save-residual=%s_residual.fits --save-params=%s_bestfit_%s.dat' % (
                  name, name, name, name, name, name, date))
    os.system('../../makeimage %s_bestfit_%s.dat --nosave --print-fluxes --estimation-size 256' % (
        name, date))  # Print the flux ratios.

    # Plot the image, model and residual #
    # plot_fits_image(name + '_model')
    os.chdir(plots_path + name)  # Change to each halo's plots directory
    combine_images(name, ellipticity)

# ellipticity
# {'Au-06':0.29, 'Au-6NoR':0.3, 'Au-06NoRNoQ':0.56}
# {'Au-17':0.57, 'Au-17NoR':0.44, 'Au-17NoRNoQ':0.51}
# {'Au-18':0.42, 'Au-18NoR':0.55, 'Au-18NoRNoQ':0.52}
