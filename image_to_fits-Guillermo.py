import numpy as np
import math as mt
import astropy.io.fits as pyfits
from astropy.coordinates import Angle
import argparse

parser = argparse.ArgumentParser(description='Creates a fits image of a disk with a cavity.', prefix_chars='-+')
parser.add_argument('--r_disk', type=float, help='Disk radius (au)', default=3.4)
parser.add_argument('--r_cavity', type=float, help='Cavity radius (au)',default=1.3)
parser.add_argument('--distance', type=float, help='Source distance (pc)',default=140)
parser.add_argument('--inclination', type=float, help='Angle between the visual and the disk rotation axis (deg)',default=35.0)
parser.add_argument('--PA', type=float, help='Angle of the major axis of the source, form N to E (deg)',default=140.0)
parser.add_argument('--RA', help='Right Ascention coordinates (ej: 04h31m40.0868s)',default='04h31m40.0868s')
parser.add_argument('--DEC', help='Declination coordinates (ej: 18d13m56.642s) if its negative, write: " -xxdyymzzs"',default='-62d40m46s')
parser.add_argument('--pixels', help='Number of pixel per row and per column', default=500)
parser.add_argument('--outname', help='Base name of the ouput fits',default='Model_')
parser.add_argument('--pointflux', help='Factor flux of the central point source',default='100')


def au2as(l_au, distance):
    return l_au/distance #longitud en arcsec

def make_elipse(r_disk, r_cavity, inclination, PA, distance, pixels):
    r_max = au2as(r_disk,distance) #arcsec, semi-tamaño de la imagen, estamos tomando el radio del disco 
    r_min = -1*r_max
    r_out = r_max #radio del disco 
    r_in = au2as(r_cavity,distance) #arcsec, radio de la cavidad 
    FOV_ACA = {'x':[-48,48], 'y':[-48,48]}
#    grid_x, grid_y = np.mgrid[r_min:r_max:501j, r_min:r_max:501j] #hacemos un array con las coordenadas de los pixeles de la imagen
    p = np.complex('{}j'.format(pixels))
    grid_x, grid_y = np.mgrid[FOV_ACA['x'][0]:FOV_ACA['x'][1]:p, FOV_ACA['y'][0]:FOV_ACA['y'][1]:p]
    grid_x_rot = grid_x*np.sin(PA*np.pi/180)+grid_y*np.cos(PA*np.pi/180) #nuevas coordenadas x al rotar la fuente cierto PA
    grid_y_rot = grid_x*np.cos(PA*np.pi/180)-grid_y*np.sin(PA*np.pi/180)#nuevas coordenadas y al rotar la fuente cierto PA
    intensity = np.zeros_like(grid_x)
    eq_disk = grid_x_rot**2/r_out**2 + grid_y_rot**2/(r_out*mt.cos(inclination*np.pi/180))**2 #ec de la elipse rotada cierto PA e inclinada un ang inclination (disco)
    if r_cavity != 0:
        eq_cavity = grid_x_rot**2/r_in**2 + grid_y_rot**2/(r_in*mt.cos(inclination*np.pi/180))**2 #ec de la elipse rotada cierto PA e inclinada un ang inclination (cavidad)
        intensity[(eq_cavity>=1) & (eq_disk<=1)]=1.0
    else:
        intensity[(eq_disk<=1)]=1.0   
    delta = grid_x[1,0] - grid_x[0,0] #asec
    return [intensity, delta]
    
def put_point(flux,data,pixels):
    i=int(pixels/2)
    data[0][i][i] = flux
    return data
    
def write_fits(RA, DEC, data, outname, inclination, PA):
    RA = Angle(RA)
    DEC = Angle(DEC)
    crval1 = RA.deg # RA in deg
    crval2 = DEC.deg # DEC in deg
    delta = data[1]
    I_array = data[0] # array of intensity (image) in Jy/pixel
    crpix1 = len(I_array[:,0])/2. # central pixel in RA
    crpix2 = len(I_array[0,:])/2. # central pixel in DEC
    hdu = pyfits.PrimaryHDU(I_array)
    hdu.header.set('BTYPE','Intensity')
    hdu.header.set('BUNIT','Jy/pixel ')
    hdu.header.set('EQUINOX',2.000000000000E+03)
    hdu.header.set('RADESYS','FK5     ')
    hdu.header.set('LONPOLE',1.800000000000E+02)
    hdu.header.set('LATPOLE',-2.978053866667E+01)
    hdu.header.set('CTYPE1','RA---SIN')
    hdu.header.set('CRVAL1',crval1)
    hdu.header.set('CDELT1',-delta/3600.)
    hdu.header.set('CRPIX1',crpix1)
    hdu.header.set('CUNIT1','deg     ')
    hdu.header.set('CTYPE2','DEC--SIN')
    hdu.header.set('CRVAL2',crval2)
    hdu.header.set('CDELT2',delta/3600.)
    hdu.header.set('CRPIX2',crpix2)
    hdu.header.set('CUNIT2','deg     ')
#    hdu.writeto(outname+'_i='+str(inclination)+'deg_PA='+str(PA)+'deg.fits')
    hdu.writeto(outname+'.fits')

if __name__ == "__main__": 
    args = parser.parse_args()
    data = make_elipse(args.r_disk,args.r_cavity,args.inclination, 180-args.PA, args.distance, args.pixels)
    put_point(args.pointflux,data,args.pixels)
    write_fits(args.RA,args.DEC,data,args.outname,args.inclination,args.PA)



