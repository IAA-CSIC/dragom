from astropy.io import fits
from astropy.wcs import utils
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

fits_image_filename='SVS13.fits'

hdul = fits.open(fits_image_filename)[0]
w=WCS(hdul.header)
data = hdul.data

***********************************************************
#Creamos fits con valores altos
data[data < 2e-2] =0
hduHigh = fits.PrimaryHDU(data,header=hdul.header)
hdulHigh = fits.HDUList([hduHigh])
hdulHigh.writeto('SVS13-high.fits')

#Creamos fits con valores bajos
data[data > 2e-2] =0
hduLow = fits.PrimaryHDU(data,header=hdul.header)
hdulLow = fits.HDUList([hduLow])
hdulLow.writeto('SVS13-low.fits')


*************************************************************
max_val = np.amax(data)
min_val = np.amin(data)
corte=2e-2



top = plt.cm.get_cmap('Greys', 128)
bottom = plt.cm.get_cmap('Blues', 128)
plt.cm.set_norm(bottom,colors.LogNorm(1e-3, max_val))

newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                       bottom(np.linspace(0, 1, 128))))
cmap = mpl.colors.ListedColormap(newcolors, name='GreyBlue')


# set the colormap and centre the colorbar
class MidpointNormalize(colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)
    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)
    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))



plt.clf()
plt.subplot(projection=w)
plt.imshow(data, cmap='PuOr_r',norm=MidpointNormalize(midpoint=1.5e-2))
plt.xlabel('ICRS Right Ascension')
plt.ylabel('ICRS Declination')
cbar=plt.colorbar()
cbar.set_label('Intensity (Jy/beam)', rotation=90)
plt.show()

norm = vi.ImageNormalize(data, interval=MinMaxInterval(),  stretch=vi.LogStretch())


CompositeStretch(transform_1, transform_2)


plt.clf()
plt.subplot(projection=w)
plt.imshow(data, cmap='PuOr_r',norm=norm )
plt.xlabel('ICRS Right Ascension')
plt.ylabel('ICRS Declination')
cbar=plt.colorbar()
cbar.set_label('Intensity (Jy/beam)', rotation=90)
plt.show()


plt.savefig('Midpoint1.5e-2.png', format='png',bbox_inches='tight')


plt.clf()
plt.subplot(projection=w)
plt.imshow(data, cmap='PuOr_r',norm=colors.LogNorm(1e-4,max_val))
plt.xlabel('ICRS Right Ascension')
plt.ylabel('ICRS Declination')
cbar=plt.colorbar()
cbar.set_label('Intensity (Jy/beam)', rotation=90)
plt.savefig('fit.eps', format='eps',bbox_inches='tight')
plt.savefig('LogNorm.png', format='png',bbox_inches='tight')

plt.clf()
plt.subplot(projection=w)
plt.imshow(data, cmap='PuOr_r',vmin=0, vmax=max_val)
plt.xlabel('ICRS Right Ascension')
plt.ylabel('ICRS Declination')
cbar=plt.colorbar()
cbar.set_label('Intensity (Jy/beam)', rotation=90)
plt.savefig('fit1.eps', format='eps',bbox_inches='tight')
plt.savefig('fit1.png', format='png',bbox_inches='tight')

