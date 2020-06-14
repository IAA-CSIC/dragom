from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.time import Time

def import_fits(fits_image):
    hdul = fits.open(fits_image)
    hdr=hdul[0].header
    data = hdul[0].data
    hdul.close()
    return [hdr,data]

def write_fits(data, header, outname):
    hdu = fits.PrimaryHDU(data=data, header=header)
    nt = Time.now()
    hdu.header.set('DATE',str(nt))
    hdu.header.set('HISTORY','')
    hdu.writeto(outname+'.fits')


def show_image(data, color_bar_label='Intensity (Jy/beam)', circle=None, zoom=None):
    if circle or zoom is not None:
        fig = plt.gcf()
        ax = fig.gca()
    if circle is not None:
        x_orig=circle[0]
        y_orig=circle[1]
        r=circle[2]
        orig_pix = (x_orig,y_orig)
        plt.plot(x_orig,y_orig, 'ro')
        circle1 = plt.Circle(orig_pix, r, color='r',fill=False)
        ax.add_artist(circle1)
    if zoom is not None:
        x_min=zoom[0]
        x_max=zoom[1]
        y_min=zoom[2]
        y_max=zoom[3]
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    plt.imshow(data)
    cbar=plt.colorbar()
    cbar.set_label(color_bar_label, rotation=90)
    plt.show()

def data_subset(data,zoom):
    subset = data[zoom[0]:zoom[1],zoom[2]:zoom[3]]
    return subset
##need to fix header for correct coordinates of central point

