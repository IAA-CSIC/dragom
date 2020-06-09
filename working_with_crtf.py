import numpy as np
from astropy.wcs import WCS
from astropy.wcs import utils
from astropy import coordinates as c 


def get_coord_from_crtf(crtf_name):
    '''Reads a crtf (CASA region text format) file and gets the coordinates of the points on it.
    E.g. for the line: 'symbol [[52.26545721deg, 31.26771624deg], .] coord=ICRS, linewidth=1, linestyle=-,
    symsize=2, symthick=1, color=blue, font="DejaVu Sans", fontsize=11, fontstyle=normal, usetex=false',
    the function returns [52.26545721,31.26771624].
    The output only makes sense if the file contains only points (symbols).'''
    with open(crtf_name) as f:
        lines = f.readlines()[1:]
        coords = []
        for l in lines:
            x = l[9:20]
            y = l[24:36]
            coords.append([x,y])
    return coords


def pixels_to_list(crtf_name, fits_header, file_header=None):
    '''Gets coordinates of the points in a crtf file (crtf_name) and transforms to the corresponding pixel values
    in fits with header (fits_header). Save the pixels in a .dat file with optional header (file_header).'''
    #import CASA region with list of points end get their coordinates
    coord_vals = get_coord_from_crtf(crtf_name)
    coords = [c.SkyCoord(coord[0], coord[1], frame='icrs', unit='deg') for coord in coord_vals]
    #convert to pixels in fits_name
    pixels = [utils.skycoord_to_pixel(coord, WCS(fits_header)) for coord in coords]
    #and save to file
    if file_header is None:
        fh = 'From '+crtf_name
    else:
        fh = file_header
    np.savetxt(crtf_name[:-5]+'.dat',pixels,header=fh)
    
def save_coords(name,coords,color='red',symbol='.'):
    '''Saves a list of sky coordinates to a crtf file.'''
    f=open(name+'.crtf','w')
    f.write('#CRTFv0 CASA Region Text Format version 0\n')
    for coord in coords:
        f.write(f'symbol [[{coord.ra.deg}deg,{coord.dec.deg}deg], {symbol}] coord=ICRS, linewidth=1, linestyle=-, symsize=1, symthick=1, color={color}, font="DejaVu Sans", fontsize=11, fontstyle=normal, usetex=false\n')
    f.close()
    
def save_pixels(name,pixels,color='red',symbol='.'):
    '''Saves a list of pixels to a crtf file.'''
    f=open(name+'.crtf','w')
    f.write('#CRTFv0 CASA Region Text Format version 0\n')
    for pix in pixels:
        f.write(f'symbol [[{pix[0]}pix,{pix[1]}pix], {symbol}] coord=ICRS, linewidth=1, linestyle=-, symsize=1, symthick=1, color={color}, font="DejaVu Sans", fontsize=11, fontstyle=normal, usetex=false\n')
    f.close()
    