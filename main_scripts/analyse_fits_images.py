import re
import os
import PIL
import time
import glob
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from astropy.io import fits
from matplotlib import gridspec
from scipy.optimize import curve_fit
from matplotlib.patches import Circle
from photutils.isophote import Ellipse
from photutils.isophote import EllipseGeometry
from scipy.ndimage.filters import gaussian_filter
from photutils.isophote import build_ellipse_model

date = time.strftime("%d_%m_%y_%H%M")


def convert_for_fit(name, min_intensity):
    """
    Add Gaussian noise, Gaussian blur, convert to gray scale and save an image in 512x512 along with its fits version.
    :param min_intensity: the minimum intensity used to add noise to the image.
    :param name: name of the file.
    :return:
    """
    # Load a png image and convert to 512x512 #
    image_png = Image.open(name + '.png').convert("L")
    image_png = image_png.resize((512, 512), PIL.Image.NEAREST)
    
    # Create and save a fits version of the gray scaled image #
    hdu = fits.PrimaryHDU(image_png)
    hdu.writeto(name + '_ctf.fits', overwrite=True)
    
    # Add flat background noise, with specified local variance at each point #
    image_fits = fits.getdata(name + '_ctf.fits', ext=0)
    image_fits = image_fits + min_intensity
    
    # Add Gaussian blur #
    FWHM = 3
    sigma = FWHM / np.sqrt(8 * np.log(2))
    image_fits = gaussian_filter(image_fits, sigma=sigma)
    array = np.asarray(image_fits)
    
    # Generate the figure and set its parameters #
    figure, axis = plt.subplots(1, figsize=(10, 10), frameon=False)
    plt.axis('off')
    axis.set_aspect('equal')
    
    # Create and save a gray scaled version of the image #
    plt.imsave(name + '_ctf.png', array, cmap='gray')
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
    
    # Provide the elliptical isophote fitter with an initial ellipse (geometry) and fit multiple isophotes to the image array #
    geometry = EllipseGeometry(x0=centre[0], y0=centre[1], sma=centre[0] / 10, eps=ellipticity, pa=1e-2)
    ellipse = Ellipse(image_fits, geometry)
    isolist = ellipse.fit_image(minsma=1, maxsma=centre[0], step=0.3)
    print(isolist.to_table())  # Print the isophote values as a table sorted by the semi-major axis length.
    
    # Plot the ellipticity, position angle, and the center x and y position as a function of the semi-major axis length.
    # Generate the figure and set its parameters #
    plt.figure(figsize=(10, 7.5))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.35)
    axis00 = plt.subplot(gs[0, 0])
    axis01 = plt.subplot(gs[0, 1])
    axis02 = plt.subplot(gs[0, 2])
    axis10 = plt.subplot(gs[1, 0])
    axis11 = plt.subplot(gs[1, 1])
    axis12 = plt.subplot(gs[1, 2])
    
    axes = [axis00, axis01, axis02, axis10, axis11, axis12]
    y_labels = [r'$\mathrm{Ellipticity}$', r'$\mathrm{PA\;[\deg]}$', r'$\mathrm{Pixels\;inside\;each\;ellipse}$', r'$\mathrm{x_{0}\;[pix]}$',
                r'$\mathrm{y_{0}\;[pix]}$', r'$\mathrm{Mean\,intensity}$']
    y_values = [isolist.eps, isolist.pa / np.pi * 180., isolist.tflux_e, isolist.x0, isolist.y0, isolist.intens]
    y_lims = [(0, 1), (-20, 200), (0.9 * min(isolist.tflux_e), 1.1 * max(isolist.tflux_e)), (250, 260), (250, 260),
              (0.9 * min(isolist.intens), 1.1 * max(isolist.intens))]
    y_errors = [isolist.ellip_err, isolist.pa_err / np.pi * 180., np.zeros(len(isolist.x0_err)), isolist.x0_err, isolist.y0_err, isolist.int_err]
    for axis, y_label, y_value, y_error, y_lim in zip(axes, y_labels, y_values, y_errors, y_lims):
        axis.set_ylim(y_lim)
        axis.set_xscale('log')
        axis.set_ylabel(y_label)
        axis.set_xlim(1e0, centre[0])
        axis.grid(True, color='gray', linestyle='-')
        axis.set_xlabel(r'$\mathrm{Semi-major\;axis\;length\;[pix]}$')
        axis.errorbar(isolist.sma, y_value, yerr=y_error, fmt='o', markersize=3)
    
    axis02.set_yscale('log')
    popt, pcov = curve_fit(fit_exponential_profile, isolist.sma, isolist.intens, p0=[isolist.intens[0], 1])
    print('I_0:', popt[0], 'h:', popt[1])
    axis12.plot(isolist.sma, fit_exponential_profile(isolist.sma, popt[0], popt[1]), 'b-')
    axis12.axvline(x=popt[1])
    plt.savefig(name + '_isolist_' + str(date) + '.png', bbox_inches='tight')  # Save the figure.
    
    # Build an elliptical model #
    # model_image = build_ellipse_model(image_fits.shape, isolist)
    # residual = image_fits - model_image
    
    # Plot the original data with some of the isophotes, the elliptical model image, and the residual image #
    # Generate the figure and set its parameters #
    plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(1, 2, hspace=0.2, wspace=0.2)
    axis00 = plt.subplot(gs[0, 0])
    axis01 = plt.subplot(gs[0, 1])
    # axis10 = plt.subplot(gs[1, 0])
    # axis11 = plt.subplot(gs[1, 1])
    
    axes = [axis00, axis01]  # , axis10, axis11]
    titles = [r'$\mathrm{Auriga-}$' + name, r'$\mathrm{Sample\;of\;isophotes}$']  # , r'$\mathrm{Ellipse\;model}$', r'$\mathrm{Residual}$']
    images = [image_fits, image_fits]  # , model_image, residual]
    for axis, title, image in zip(axes, titles, images):
        axis.grid(True, color='gray', linestyle='-')
        axis.set_title(title)
        axis.imshow(image, origin='lower', cmap='gray')
    
    smas = np.linspace(centre[0] / 10, centre[0], 10)
    for sma in smas:
        iso = isolist.get_closest(sma)
        x, y, = iso.sampled_coordinates()
        axis01.plot(x, y, color='tab:red')
    
    plt.savefig(name + '_model_' + str(date) + '.png', bbox_inches='tight')  # Save the figure.
    
    return None


def fit_exponential_profile(x, I_0, R_d):
    """
    Calculate an exponential profile.
    :param x: x data.
    :param I_0: Central intensity.
    :param R_d: Scale length.
    :return: I_0 * np.exp(-r / R_d)
    """
    return I_0 * np.exp(-x / R_d)


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
    plt.savefig(name + '_2_' + str(date) + '.png', cmap='gray', bbox_inches='tight')
    plt.close()
    return None


# Define the paths to the images #
Imfit_path = '/Users/Bam/PycharmProjects/Auriga/Imfit/Auriga/'
plots_path = '/Users/Bam/PycharmProjects/Auriga/plots/projections/Imfit/'

# Get the names and sort them #
names = glob.glob(plots_path + 'Au-06NoRNoQ')
names = [re.split('/Imfit|/', name)[-1] for name in names]
names.sort()

# Loop over all Auriga rbm images, convert them to the appropriate format and fit isophotal ellipses #
for name in names:
    os.chdir(plots_path + name)  # Change to each halo's plots directory
    # Prepare the image and fit isophotal ellipses #
    # min_intensity = 47.86  # {'Au-06':85.57, 'Au-06NoRNoQ':47.86}
    # convert_for_fit(name, min_intensity)
    # ellipticity = 0.6  # {'Au-06NoRNoQ':0.6}
    # fit_isophotal_ellipses(name, ellipticity)
    
    # Use Imfit to analyse the image #
    # --bootstrap 15
    os.chdir(Imfit_path + name)  # Change to each halo's Imfit directory
    os.system(
        '../../imfit -c %s_config.dat --nm --model-errors --cashstat ../../../plots/projections/Imfit/%s/%s_ctf.fits --save-model=%s_model.fits '
        '--save-residual=%s_resid.fits --save-params=%s_bestfit.dat' % (name, name, name, name, name, name))
    
    # Plot the image model and residual  #
    plot_fits_image(name + '_resid')
    plot_fits_image(name + '_model')
