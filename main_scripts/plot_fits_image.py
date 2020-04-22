import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from astropy.io import fits


def show_fits_image(name):
    image_data = fits.getdata('examples/' + str(name) + '.fits', ext=0)
    
    plt.figure()
    # plt.xlim(50, 150)
    # plt.ylim(50, 150)
    plt.imshow(image_data, cmap='gray')
    plt.colorbar()
    plt.show()
    return None


def convert_to_grayscale(name):
    fname = str(name) + '.png'
    image = Image.open(fname).convert("L")
    arr = np.asarray(image)
    plt.imshow(arr, cmap='gray', vmin=0, vmax=255)
    plt.show()
    
    hdu = fits.PrimaryHDU(arr)
    hdu.writeto(str(name) + '.fits')
    return None


def image_centre(name):
    fname = str(name) + '.png'
    im = Image.open(fname)
    (X, Y) = im.size
    centre = X / 2, Y / 2
    return centre


# convert_to_grayscale(18)
# show_fits_image(18)
image_centre('circularities')
