import re
import PIL
import cv2
import glob
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
    plt.savefig(run + '_2.png', cmap='gray', bbox_inches='tight')
    plt.close()
    return None


def fit_isophotal_ellipses(run):
    """
    Use the photutils package to fit isophotal ellipses on fits images.
    :param run: name of the file.
    :return: None
    """
    # Load the fits image and calculate the centre #
    name = re.split('Auriga/|/', run)[-1]
    image_data = fits.getdata(run + '_ctg.fits', ext=0)
    centre = image_centre(run)
    
    # Provide the elliptical isophote fitter with an initial ellipse (geometry) and fit multiple isophotes to the image array. #
    geometry = EllipseGeometry(x0=centre[0], y0=centre[1], sma=centre[0] / 10, eps=0.6, pa=0.0)
    ellipse = Ellipse(image_data, geometry)
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
    y_lims = [(0, 1), (-20, 200), (1e2, 1e7), (250, 260), (250, 260), (50, 260)]
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
    # model_image = build_ellipse_model(image_data.shape, isolist)
    # residual = image_data - model_image
    
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
    images = [image_data, image_data]  # , model_image, residual]
    for axis, title, image in zip(axes, titles, images):
        axis.set_title(title)
        axis.imshow(image, origin='lower', cmap='gray')
    
    smas = np.linspace(10, centre[0], 10)
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


def add_Gaussian_noise(run):
    img = fits.getdata(run + '_ctg.fits', ext=0) / 255
    noise = np.random.normal(loc=0, scale=1, size=img.shape)
    
    # noise overlaid over image
    noisy = np.clip((img + noise * 0.1), 0, 1)
    noisy2 = np.clip((img + noise * 0.4), 0, 1)

    # noise multiplied by image:
    # whites can go to black but blacks cannot go to white
    noisy2mul = np.clip((img * (1 + noise * 0.2)), 0, 1)
    noisy4mul = np.clip((img * (1 + noise * 0.4)), 0, 1)

    noisy2mul = np.clip((img * (1 + noise * 0.2)), 0, 1)
    noisy4mul = np.clip((img * (1 + noise * 0.4)), 0, 1)
    noise2 = (noise - noise.min()) / (noise.max() - noise.min())
    # noise multiplied by bottom and top half images,
    # whites stay white blacks black, noise is added to center
    img2 = img * 2
    n2 = np.clip(np.where(img2 <= 1, (img2 * (1 + noise * 0.2)), (1 - img2 + 1) * (1 + noise * 0.2) * -1 + 2) / 2, 0, 1)
    n4 = np.clip(np.where(img2 <= 1, (img2 * (1 + noise * 0.4)), (1 - img2 + 1) * (1 + noise * 0.4) * -1 + 2) / 2, 0, 1)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(np.vstack((np.hstack((img, noise2)), np.hstack((noisy, noisy2)), np.hstack((noisy2mul, noisy4mul)), np.hstack((n2, n4)))),
               origin='lower', cmap='gray')
    # plt.imshow(noisy, origin='lower', cmap='gray')
    plt.show()





plots_path = '/Users/Bam/PycharmProjects/Auriga/plots/projections/'
output_path = '/Users/Bam/PycharmProjects/Auriga/Imfit/Auriga/'
run = plots_path + str('06NOAGN')
#
# names = glob.glob(plots_path + 'rbm*')
# names = [re.split('projections/|.png', name)[1] for name in names]
# for name in names:
#     convert_to_grayscale(plots_path + name)
#
#     name = re.split('rbm-|.png', name)[1]
#     fit_isophotal_ellipses(plots_path + name)

# fit_isophotal_ellipses(run)
# add_Gaussian_noise(run)

# plot_fits_image(run)
# plot_fit_data('Au-06_edge_ong', h=79.8589, R_eff=128.589)

import skimage
def plotnoise(img, mode, r, c, i):
    plt.subplot(r,c,i)
    if mode is not None:
        gimg = skimage.util.random_noise(img, mode=mode)
        plt.imshow(gimg, origin='lower', cmap='gray')
    else:
        plt.imshow(img, origin='lower', cmap='gray')
    plt.title(mode)
    plt.axis("off")

plt.figure(figsize=(18,24))
r=4
c=2
img = fits.getdata(run + '_ctg.fits', ext=0) / 255
plotnoise(img, "gaussian", r,c,1)
plotnoise(img, "localvar", r,c,2)
plotnoise(img, "poisson", r,c,3)
plotnoise(img, "salt", r,c,4)
# plotnoise(img, "pepper", r,c,5)
# plotnoise(img, "s&p", r,c,6)
plotnoise(img, "speckle", r,c,4)
# plotnoise(img, None, r,c,8)
plt.savefig(run + "poisson", bbox_inches='tight')  # Save the figure.