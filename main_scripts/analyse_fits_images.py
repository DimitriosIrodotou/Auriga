import re
import PIL
import glob
import skimage
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from astropy.io import fits
from matplotlib import gridspec
from scipy.optimize import curve_fit
from matplotlib.patches import Circle
from photutils.isophote import Ellipse
from photutils import EllipticalAperture
from photutils.isophote import EllipseGeometry
from scipy.ndimage.filters import gaussian_filter
from photutils.isophote import build_ellipse_model


def convert_for_fit(run):
    """
    Add Gaussian noise, Gaussian blur, convert to gray scale and save an image in 512x512 along with its fits version.
    :param run:
    :return:
    """
    # Load a png image and convert to 512x512 #
    image_png = Image.open(run + '.png').convert("L")
    image_png = image_png.resize((512, 512), PIL.Image.NEAREST)
    
    # Create and save a fits version of the gray scaled image #
    name = re.split('rbm-|.png', run)[1]
    hdu = fits.PrimaryHDU(image_png)
    hdu.writeto(plots_path + str(name) + '_ctf.fits', overwrite=True)
    
    # Add flat background noise, with specified local variance at each point #
    image_fits = fits.getdata(plots_path + str(name) + '_ctf.fits', ext=0)
    image_fits = image_fits + 61 / 255
    # image_fits = skimage.util.random_noise(image_fits, mode='gaussian', seed=10, clip=True)
    plt.imsave(plots_path + str(name) + '_an.png', image_fits, cmap='gray')
    
    # Add Gaussian blur #
    FWHM = 3
    sigma = FWHM / np.sqrt(8 * np.log(2))
    image_fits = gaussian_filter(image_fits, sigma=sigma)
    array = np.asarray(image_fits)
    
    # Generate the figure and define its parameters #
    figure, axis = plt.subplots(1, figsize=(10, 10), frameon=False)
    plt.axis('off')
    axis.set_aspect('equal')
    
    # Create and save a gray scaled version of the image #
    plt.imsave(plots_path + str(name) + '_ctf.png', array, cmap='gray')
    plt.close()
    
    # Create and save a fits version of the image #
    hdu = fits.PrimaryHDU(array)
    hdu.writeto(plots_path + str(name) + '_ctf.fits', overwrite=True)
    return None


def plot_fits_image():
    """
    Load and show a fits image.
    :return: None
    """
    # Load the fits image and show it #
    image_fits = fits.getdata(run + '_ctf.fits', ext=0)
    
    # Generate the figure and define its parameters #
    figure, axis = plt.subplots(1, figsize=(10, 10), frameon=False)
    plt.axis('off')
    axis.set_aspect('equal')
    
    # Create and save a gray scaled version of the image #
    plt.imsave(run + '_pfi.png', image_fits, cmap='gray')
    plt.close()
    return None


def image_centre():
    """
    Get the centre of the image in pixels.
    :return: centre
    """
    # Load the png image and calculate the centre #
    im = Image.open(run + '_ctf.png')
    
    (X, Y) = im.size
    centre = X / 2, Y / 2
    return centre


def image_intensity(name):
    """
    Get the centre of the image in pixels.
    :param name: name of the file.
    :return: None
    """
    # Load the png image and calculate the centre #
    im = Image.open(plots_path + str(name) + '.png').convert("L")
    
    intensity = np.array([[pixel, value] for pixel, value in enumerate(im.histogram())])
    print(1 / intensity[0, 1])
    plt.bar(intensity[:, 0], intensity[:, 1], width=2, log=True)
    plt.show()
    return None


def plot_fit_data(h=0.0, R_eff=0.0):
    """
    Over-plot the scale length and effective radius on the fits image.
    :return: None
    """
    # Load the fits image and show it #
    image_fits = fits.getdata(run + '.fits', ext=0)
    
    # Generate the figure and define its parameters #
    figure, axis = plt.subplots(1, figsize=(10, 10), frameon=False)
    plt.axis('off')
    axis.set_aspect('equal')
    
    axis.imshow(image_fits, cmap='gray')
    centre = image_centre()
    # Plot the scale length
    for radius in [h, R_eff]:
        circle = Circle((centre[0], centre[1]), radius, color='tab:red', fill=False)
        axis.add_patch(circle)
    
    # Create and save a gray scaled version of the image #
    plt.savefig(run + '_2.png', cmap='gray', bbox_inches='tight')
    plt.close()
    return None


def fit_isophotal_ellipses():
    """
    Use the photutils package to fit isophotal ellipses on fits images.
    :return: None
    """
    # Load the fits image and calculate the centre #
    name = re.split('Auriga/|/', run)[-1]
    image_fits = fits.getdata(run + '_ctf.fits', ext=0)
    array = np.asarray(image_fits)
    centre = image_centre()
    
    # Provide the elliptical isophote fitter with an initial ellipse (geometry) and fit multiple isophotes to the image array. #
    geometry = EllipseGeometry(x0=centre[0], y0=centre[1], sma=centre[0] / 10, eps=0.5, pa=0.0)
    ellipse = Ellipse(image_fits, geometry)
    isolist = ellipse.fit_image(minsma=1, maxsma=centre[0], step=0.3)
    print(isolist.to_table())  # Print the isophote values as a table sorted by the semi-major axis length.
    
    # Plot the ellipticity, position angle, and the center x and y position as a function of the semi-major axis length.
    # Generate the figure and define its parameters #
    plt.figure(figsize=(10, 7.5))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.35)
    ax00 = plt.subplot(gs[0, 0])
    ax01 = plt.subplot(gs[0, 1])
    ax02 = plt.subplot(gs[0, 2])
    ax10 = plt.subplot(gs[1, 0])
    ax11 = plt.subplot(gs[1, 1])
    ax12 = plt.subplot(gs[1, 2])
    
    axes = [ax00, ax01, ax02, ax10, ax11, ax12]
    y_labels = [r'$\mathrm{Ellipticity}$', r'$\mathrm{PA\;[\deg]}$', r'$\mathrm{Pixels\;inside\;each\;ellipse}$', r'$\mathrm{x_{0}\;[pix]}$',
                r'$\mathrm{y_{0}\;[pix]}$', r'$\mathrm{Mean\,intensity}$']
    y_values = [isolist.eps, isolist.pa / np.pi * 180., isolist.tflux_e, isolist.x0, isolist.y0, isolist.intens]
    y_lims = [(0, 1), (-20, 200), (1e0, 1e7), (250, 260), (250, 260), (50, 260)]
    y_errors = [isolist.ellip_err, isolist.pa_err / np.pi * 180., np.zeros(len(isolist.x0_err)), isolist.x0_err, isolist.y0_err, isolist.int_err]
    for axis, y_label, y_value, y_error, y_lim in zip(axes, y_labels, y_values, y_errors, y_lims):
        axis.set_ylim(y_lim)
        axis.set_ylabel(y_label)
        axis.set_xscale('log')
        axis.set_xlim(1e0, centre[0])
        axis.set_xlabel(r'$\mathrm{Semi-major\;axis\;length\;[pix]}$')
        axis.errorbar(isolist.sma, y_value, yerr=y_error, fmt='o', markersize=3)
    
    ax02.set_yscale('log')
    ax12.set_xscale('linear')
    popt, pcov = curve_fit(exponential_profile, isolist.sma, isolist.intens, p0=[isolist.intens[0], 1])
    ax12.plot(isolist.sma, exponential_profile(isolist.sma, popt[0], popt[1]), 'b-')
    plt.savefig(run + '_fie_isolist.png', bbox_inches='tight')  # Save the figure.
    
    # Build an elliptical model #
    # model_image = build_ellipse_model(image_fits.shape, isolist)
    # residual = image_fits - model_image
    
    # Plot the original data with some of the isophotes, the elliptical model image, and the residual image #
    # Generate the figure and define its parameters #
    plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(1, 2, hspace=0.2, wspace=0.2)
    ax00 = plt.subplot(gs[0, 0])
    ax01 = plt.subplot(gs[0, 1])
    # ax10 = plt.subplot(gs[1, 0])
    # ax11 = plt.subplot(gs[1, 1])
    
    axes = [ax00, ax01]  # , ax10, ax11]
    titles = [r'$\mathrm{Auriga-}$' + str(name), r'$\mathrm{Sample\;of\;isophotes}$']  # , r'$\mathrm{Ellipse\;model}$', r'$\mathrm{Residual}$']
    images = [image_fits, image_fits]  # , model_image, residual]
    for axis, title, image in zip(axes, titles, images):
        axis.set_title(title)
        axis.imshow(image, origin='lower', cmap='gray')
    
    smas = np.linspace(centre[0] / 10, centre[0], 10)
    for sma in smas:
        iso = isolist.get_closest(sma)
        x, y, = iso.sampled_coordinates()
        ax01.plot(x, y, color='tab:red')
    
    plt.savefig(run + '_fie_model.png', bbox_inches='tight')  # Save the figure.
    
    return None


def exponential_profile(x, I_0, R_d):
    """
    Calculate an exponential profile.
    :param x: x data.
    :param I_0: Central intensity.
    :param R_d: Scale length.
    :return: I_0 * np.exp(-r / R_d)
    """
    return I_0 * np.exp(-x / R_d)


# Define the paths to the images #
name = 'rbm-06NOAGN'
output_path = '/Users/Bam/PycharmProjects/Auriga/Imfit/Auriga/'
plots_path = '/Users/Bam/PycharmProjects/Auriga/plots/projections/'
run = plots_path + re.split('rbm-|.png', name)[1]

# Loop over all Auriga rbm images, convert them to the appropriate fit format and fit isophotal ellipses #
# names = glob.glob(plots_path + 'rbm*')
# names = [re.split('projections/|.png', name)[1] for name in names]
# for name in names:
#     convert_for_fit(plots_path + name)
#
#     name = re.split('rbm-|.png', name)[1]
#     fit_isophotal_ellipses(plots_path + name)

convert_for_fit(plots_path + name)
fit_isophotal_ellipses()
