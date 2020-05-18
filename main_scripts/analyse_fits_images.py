import re
import PIL
import glob
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from astropy.io import fits
from matplotlib.patches import Circle
from photutils.isophote import Ellipse
from photutils import EllipticalAperture
from photutils.isophote import EllipseGeometry
from scipy.ndimage.filters import gaussian_filter

plots_path = '/Users/Bam/PycharmProjects/Auriga/plots/projections/'


def convert_to_grayscale(name):
    """
    Convert a png image to gray scale and save it along with a fits version of it.
    :param name: name of the image.
    :return: None
    """
    # Load png image and convert to a 512x512 grayscale Gaussian blurred array #
    image = Image.open(plots_path + str(name) + '.png').convert("L")
    image = image.resize((512, 512), PIL.Image.NEAREST)
    FWHM = 3
    sigma = FWHM / np.sqrt(8 * np.log(2))
    image = gaussian_filter(image, sigma=sigma)
    array = np.asarray(image)
    
    # Generate the figure and define its parameters #
    figure, ax = plt.subplots(1, figsize=(10, 10), frameon=False)
    plt.axis('off')
    ax.set_aspect('equal')
    
    # Create and save a gray scaled version of the image #
    plt.imsave(plots_path + str(name) + 'g.png', array, cmap='gray')
    plt.close()
    
    # Create and save a fits version of the gray scaled image #
    hdu = fits.PrimaryHDU(array)
    hdu.writeto(plots_path + str(name) + 'g.fits', overwrite=True)
    return None


def plot_fits_image(name):
    """
    Load and show a fits image.
    :param name: name of the image.
    :return: None
    """
    # Load the fits image and show it #
    image_data = fits.getdata(str(name) + '.fits', ext=0)
    
    # Generate the figure and define its parameters #
    figure, ax = plt.subplots(1, figsize=(10, 10), frameon=False)
    plt.axis('off')
    ax.set_aspect('equal')
    
    # Create and save a gray scaled version of the image #
    plt.imsave(plots_path + str(name) + '.png', image_data, cmap='gray')
    plt.close()
    return None


def image_centre(name):
    """
    Get the centre of the image in pixels.
    :param name: name of the image.
    :return: centre
    """
    # Load the png image and calculate the centre #
    im = Image.open(plots_path + str(name) + '.png')
    
    (X, Y) = im.size
    centre = X / 2, Y / 2
    return centre


def image_intensity(name):
    """
    Get the centre of the image in pixels.
    :param name: name of the image.
    :return: centre
    """
    # Load the png image and calculate the centre #
    im = Image.open(plots_path + str(name) + '.png').convert("L")
    
    intensity = np.array([[pixel, value] for pixel, value in enumerate(im.histogram())])
    print(1 / intensity[0, 1])
    plt.bar(intensity[:, 0], intensity[:, 1], width=2, log=True)
    plt.show()
    return None


def plot_fit_data(name, h=0.0, R_eff=0.0):
    """
    Over-plot the scale length and effective radius on the fits image.
    :param name: name of the image.
    :return: None
    """
    # Load the fits image and show it #
    image_data = fits.getdata(plots_path + str(name) + '.fits', ext=0)
    
    # Generate the figure and define its parameters #
    figure, ax = plt.subplots(1, figsize=(10, 10), frameon=False)
    plt.axis('off')
    ax.set_aspect('equal')
    
    ax.imshow(image_data, cmap='gray')
    centre = image_centre(name)
    # Plot the scale length
    for radius in [h, R_eff]:
        circle = Circle((centre[0], centre[1]), radius, color='tab:red', fill=False)
        ax.add_patch(circle)
    
    # Create and save a gray scaled version of the image #
    plt.savefig(plots_path + str(name) + '2.png', cmap='gray', bbox_inches='tight')
    plt.close()
    return None


def fit_isophotal_ellipses(name):
    # Load the png image and calculate the centre #
    # image_data = Image.open(plots_path + str(name) + '.png').convert("L")
    
    # Load the fits image and show it #
    image_data = fits.getdata(plots_path + str(name) + '.fits', ext=0)
    
    # Provide the elliptical isophote fitter with an initial ellipse. #
    centre = image_centre(name)
    geometry = EllipseGeometry(x0=centre[0], y0=centre[1], sma=20, eps=0.5, pa=0)
    aper = EllipticalAperture((geometry.x0, geometry.y0), geometry.sma, geometry.sma * (1 - geometry.eps), geometry.pa)

    # Generate the figure and define its parameters #
    figure, ax = plt.subplots(1, figsize=(10, 10), frameon=False)
    plt.axis('off')
    ax.set_aspect('equal')
    
    plt.imshow(image_data, origin='lower')
    aper.plot(color='white')
    
    # Create and save a gray scaled version of the image #
    plt.savefig(plots_path + str(name) + '2.png', bbox_inches='tight')
    image = Image.open(plots_path + str(name) + '2.png').convert("L")
    image = image.resize((512, 512), PIL.Image.NEAREST)
    plt.imsave(plots_path + str(name) + '2.png', image, cmap='gray')
    plt.close()
    
    # Perform the elliptical isophote fit #
    ellipse = Ellipse(image_data, geometry)
    isolist = ellipse.fit_image()
    
    return None


# names = glob.glob(plots_path + '*edge_on.png')
# names.extend(glob.glob(plots_path + '*face_on.png'))
# names = [re.split('projections/|.png', name)[1] for name in names]
# for name in names:
# if name + '.fits' in names:
#     continue
# convert_to_grayscale(name)

# image_centre('Au-18_face_on')
# image_intensity('06N_fg')
# plot_fits_image('model_E_06N_fg')
# plot_fits_image('resid_E_06N_fg')
# plot_fit_data('Au-06_edge_ong', h=79.8589, R_eff=128.589)

fit_isophotal_ellipses('Au-06_face_ong')
