import re
import glob

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from astropy.io import fits

plots_path = '/Users/Bam/PycharmProjects/Auriga/plots/projections/'


def convert_to_grayscale_fits(name):
    """
    Convert a png image to gray scale and save it along with a fits version of it.
    :param name: name of the image.
    :return: None
    """
    # Load png image and convert to array #
    image = Image.open(plots_path + str(name) + '.png').convert("L")
    array = np.asarray(image)
    
    # Generate the figure and define its parameters #
    figure, ax = plt.subplots(1, figsize=(10, 10), frameon=False)
    plt.axis('off')
    ax.set_aspect('equal')
    
    # Create and save a gray scaled version of the image #
    ax.imshow(array, cmap='gray', interpolation='nearest', aspect='equal')
    
    plt.savefig(plots_path + str(name) + '_gs' + '.png', bbox_inches='tight', pad_inches=-0.1)
    plt.close()
    
    # Create and save a fits version of the gray scaled image #
    hdu = fits.PrimaryHDU(array)
    hdu.writeto(plots_path + str(name) + '.fits', overwrite=True)
    return None


def show_fits_image(name):
    """
    Load and show a fits image.
    :param name: name of the image.
    :return: None
    """
    # Load the fits image and show it #
    image_data = fits.getdata(plots_path + str(name) + '.fits', ext=0)
    
    plt.figure()
    plt.imshow(image_data, cmap='gray')
    plt.colorbar()
    plt.show()
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


names = glob.glob(plots_path + '*on.png')
names = [re.split('projections/|.png', name)[1] for name in names]
for name in names:
    convert_to_grayscale_fits(name)

# show_fits_image('Au-18_face_on')

# image_centre('Au-18_face_on')
