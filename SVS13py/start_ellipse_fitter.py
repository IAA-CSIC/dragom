import matplotlib.pyplot as plt

from astropy.wcs import WCS
from astropy.io import fits

from SVS13py.ellipse_fitter import EllipseFitter

import subprocess
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

result = subprocess.run(['whoami'], stdout=subprocess.PIPE)
user = result.stdout.decode().strip()

if user == 'gblazquez':
    spw17_path = '/home/gblazquez/data/spw-17-gb.610kHz.chans1536.contsub.fits'
elif user == 'guille':
    spw17_path = '/mnt/hdd/data/spw-17-gb.610kHz.chans1536.contsub.fits'
    spw17_path_lsr = \
        '/mnt/hdd/data/spw-17-gb.1221kHz.chans768.contsub.image.fits'
    lsr_path = '/mnt/hdd/data/spw-2-9-gb.contsub.lsr.fits'


hdu_spw17 = fits.open(spw17_path)[0]
hdr_spw17 = hdu_spw17.header
wcs_spw17 = WCS(hdu_spw17.header).celestial
image_data_spw17 = hdu_spw17.data[0]


hdu_spw17_lsr = fits.open(spw17_path_lsr)[0]
hdr_spw17_lsr = hdu_spw17_lsr.header
wcs_spw17_lsr = WCS(hdu_spw17_lsr.header).celestial
image_data_spw17_lsr = hdu_spw17_lsr.data[0]


hdu_lsr = fits.open(lsr_path)[0]
hdr_lsr = hdu_lsr.header
wcs_lsr = WCS(hdu_lsr.header).celestial
image_data_lsr = hdu_lsr.data[0]


box = [[556-200, 786-200], [855+200, 1164+200]]

# EF_low_res_arcs = EllipseFitter(image_data_spw17,
#                                 hdr_spw17,
#                                 db_name='little_tip',
#                                 cmap='magma',
#                                 operation=None,
#                                 photomorph='annulus',
#                                 box=box)
#
EF_lsr = EllipseFitter(image_data_lsr,
                       hdr_lsr,
#                      db_name='second_knot_rings_dubious',
                       operation=None,
                       cmap='magma',
                       photomorph='annulus',
                       box=box)

# plt.show()
plt.show(block=False)
