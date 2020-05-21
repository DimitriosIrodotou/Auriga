import re
import PIL
import glob
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from astropy.io import fits
from matplotlib import gridspec
from matplotlib.patches import Circle
from photutils.isophote import Ellipse
from photutils import EllipticalAperture
from photutils.isophote import EllipseGeometry
from scipy.ndimage.filters import gaussian_filter
from photutils.isophote import build_ellipse_model


def convert_to_grayscale(run):
    """
    Convert a png image to gray scale and save it along with a fits version of it.
    :param run: name of the file.
    :return: None
    """
    # Load png image and convert to a 512x512 grayscale Gaussian blurred array #
    image = Image.open(run + '.png').convert("L")
    image = image.resize((512, 512), PIL.Image.NEAREST)
    FWHM = 2
    sigma = FWHM / np.sqrt(8 * np.log(2))
    image = gaussian_filter(image, sigma=sigma)
    array = np.asarray(image)
    
    # Generate the figure and define its parameters #
    figure, axis = plt.subplots(1, figsize=(10, 10), frameon=False)
    plt.axis('off')
    axis.set_aspect('equal')
    
    # Create and save a gray scaled version of the image #
    name = re.split('rbm-|.png', run)[1]
    plt.imsave(plots_path + str(name) + '_ctg.png', array, cmap='gray')
    plt.close()
    
    # Create and save a fits version of the gray scaled image #
    hdu = fits.PrimaryHDU(array)
    hdu.writeto(plots_path + str(name) + '_ctg.fits', overwrite=True)
    return None


def plot_fits_image(run):
    """
    Load and show a fits image.
    :param run: name of the file.
    :return: None
    """
    # Load the fits image and show it #
    image_data = fits.getdata(run + '_ctg.fits', ext=0)
    
    # Generate the figure and define its parameters #
    figure, axis = plt.subplots(1, figsize=(10, 10), frameon=False)
    plt.axis('off')
    axis.set_aspect('equal')
    
    # Create and save a gray scaled version of the image #
    plt.imsave(run + '_pfi.png', image_data, cmap='gray')
    plt.close()
    return None


def image_centre(run):
    """
    Get the centre of the image in pixels.
    :param run: name of the file.
    :return: centre
    """
    # Load the png image and calculate the centre #
    im = Image.open(run + '_ctg.png')
    
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


def plot_fit_data(run, h=0.0, R_eff=0.0):
    """
    Over-plot the scale length and effective radius on the fits image.
    :param run: name of the file.
    :return: None
    """
    # Load the fits image and show it #
    image_data = fits.getdata(run + '.fits', ext=0)
    
    # Generate the figure and define its parameters #
    figure, axis = plt.subplots(1, figsize=(10, 10), frameon=False)
    plt.axis('off')
    axis.set_aspect('equal')
    
    axis.imshow(image_data, cmap='gray')
    centre = image_centre(run)
    # Plot the scale length
    for radius in [h, R_eff]:
        circle = Circle((centre[0], centre[1]), radius, color='tab:red', fill=False)
        axis.add_patch(circle)
    
    # Create and save a gray scaled version of the image #
    plt.savefig(plots_path + str(run) + '_2.png', cmap='gray', bbox_inches='tight')
    plt.close()
    return None


def fit_isophotal_ellipses(run):
    """
    Use the photutils package to fit isophotal ellipses on fits images.
    :param run: name of the file.
    :return: None
    """
    # Load the fits image and calculate the centre #
    image_data = fits.getdata(run + '_ctg.fits', ext=0)
    centre = image_centre(run)
    
    # aper = EllipticalAperture((geometry.x0, geometry.y0), geometry.sma, geometry.sma * (1 - geometry.eps), geometry.pa)
    # Generate the figure and define its parameters #
    # figure, axis = plt.subplots(1, figsize=(10, 10), frameon=False)
    # plt.axis('off')
    # axis.set_aspect('equal')
    #
    # plt.imshow(image_data, origin='lower')
    # aper.plot(color='white')
    
    # Create and save a gray scaled version of the image #
    # plt.savefig(run + '_fie.png', bbox_inches='tight')
    # image = Image.open(run + '2.png').convert("L")
    # image = image.resize((512, 512), PIL.Image.NEAREST)
    # plt.imsave(run + '2.png', image, cmap='gray')
    # plt.close()
    
    # Provide the elliptical isophote fitter with an initial ellipse (geometry) and fit multiple isophotes to the image array. #
    
    geometry = EllipseGeometry(x0=centre[0], y0=centre[1], sma=5, eps=0.4, pa=0)
    ellipse = Ellipse(image_data, geometry)
    isolist = ellipse.fit_image(maxsma=centre[0])
    print(isolist.to_table())  # Print the isophote values as a table sorted by the semi-major axis length.
    
    # Plot the ellipticity, position angle, and the center x and y position as a function of the semi-major axis length.
    figure = plt.figure(figsize=(10, 7.5))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.35)
    ax00 = plt.subplot(gs[0, 0])
    ax01 = plt.subplot(gs[0, 1])
    ax10 = plt.subplot(gs[1, 0])
    ax11 = plt.subplot(gs[1, 1])
    ax12 = plt.subplot(gs[1, 2])
    
    axes = [ax00, ax01, ax10, ax11,ax12]
    y_labels = [r'$\mathrm{Ellipticity}$', r'$PA\;[\deg]$', r'$\mathrm{x_{0}\;[pix]}$', r'$\mathrm{y_{0}\;[pix]}$', r'$\mathrm{Mean\,intensity}$']
    y_values = [isolist.eps, isolist.pa / np.pi * 180., isolist.x0, isolist.y0, isolist.intens]
    y_errors = [isolist.ellip_err, isolist.pa_err / np.pi * 180., isolist.x0_err, isolist.y0_err, isolist.int_err]
    for axis, y_label, y_value, y_error in zip(axes, y_labels, y_values, y_errors):
        axis.set_ylabel(y_label)
        axis.set_xlabel(r'$\mathrm{Semi-major\;axis\;length\;[pix]}$')
        axis.errorbar(isolist.sma, y_value, yerr=y_error, fmt='o', markersize=4)
    
    plt.savefig(run + '_fie2.png', bbox_inches='tight')
    
    # Build an elliptical model #
    # model_image = build_ellipse_model(image_data.shape, isolist)
    # residual = image_data - model_image
    #
    # # Plot the original data with some of the isophotes, the elliptical model image, and the residual image #
    # fig, (ax1, ax2, ax3) = plt.subplots(figsize=(14, 5), nrows=1, ncols=3)
    # fig.subplots_adjust(left=0.04, right=0.98, bottom=0.02, top=0.98)
    # ax1.imshow(image_data, origin='lower', cmap='gray')
    # ax1.set_title('Data')
    #
    # smas = np.linspace(10, 200, 10)
    # for sma in isolist.sma:
    #     iso = isolist.get_closest(sma)
    #     x, y, = iso.sampled_coordinates()
    #     ax1.plot(x, y, color='white')
    #
    # ax2.imshow(model_image, origin='lower', cmap='gray')
    # ax2.set_title('Ellipse Model')
    #
    # ax3.imshow(residual, origin='lower', cmap='gray')
    # ax3.set_title('Residual')
    # plt.savefig(run + '_fie3.png', bbox_inches='tight')
    #
    return None


plots_path = '/Users/Bam/PycharmProjects/Auriga/plots/projections/'
output_path = '/Users/Bam/PycharmProjects/Auriga/Imfit/Auriga/'
run = plots_path + str('06')

# names = glob.glob(plots_path + 'rbm*')
# names = [re.split('projections/|.png', name)[1] for name in names]
# print(names)
# for name in names:
#     if '.fits' in names:
#         continue
#     convert_to_grayscale(plots_path + name)

# plot_fits_image(run)
fit_isophotal_ellipses(run)  # plot_fit_data('Au-06_edge_ong', h=79.8589, R_eff=128.589)
