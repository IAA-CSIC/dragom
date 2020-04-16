import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import make_lupton_rgb


fits1='SVS13-high.fits'
fits2='SVS13-low.fits'

r_name = fits1
g_name = fits2



r = fits.open(r_name)[0].data
g = fits.open(g_name)[0].data
b = fits.open(g_name)[0].data


b=np.empty(r.shape)
b[:] = np.nan

image = make_lupton_rgb(r, g, b, filename='composite')
plt.clf()
plt.subplot()
plt.imshow(image)
plt.xlabel('ICRS Right Ascension')
plt.ylabel('ICRS Declination')
cbar=plt.colorbar()
cbar.set_label('Intensity (Jy/beam)', rotation=90)


plt.show()





