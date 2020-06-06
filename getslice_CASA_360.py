#from sys import argv
import numpy as np
from taskinit import *


#imput_image, origin, slice_length, angle_step, output_name = argv

# origin and slice_length in pixels
# angle_step in degrees

#origin_A = 632.353 377.38
#origin_B = 607.779 377.38
#slice_length = 160 px

def get_p2(p1, slice_length, ang): #p1 = [x0,y0], slice_length in pixels, ang en degrees
    x0, y0 = p1
    x1 = x0 - slice_length * np.sin(ang*np.pi/180)
    y1 = y0 + slice_length * np.cos(ang*np.pi/180)
    return [x1,y1]


def get_pixels(dict_slice):
    x = dict_slice['xpos']
    y = dict_slice['ypos']
    return zip(x,y)

def get_values(pixels):
    values = []
    for pix in pixels:
      val = ia.pixelvalue([pix[0],pix[1]])['value']['value']
      values.append(val)
    return values


input_image = input(">>Image? ('file_in'): ")
origin = input(">>Origin Pixel? [x0,y0]: ")
slice_length = input(">>Slice lenght? (in pixels): ")
angle_step = input(">>Angle step? (in degrees): ")
output_name = input(">>Output name? ('file_out'): ")


ia.open(input_image)

bright_u = ia.brightnessunit()
hdrtxt="Dist (pix)\tValue({})\t(pix=0.012arcsec)".format(bright_u)  #CASA does'n uses python 3 yet

ang = np.arange(0,360,angle_step)
ptos = [get_p2(origin,slice_length,a) for a in ang]
slices = [ia.getslice([origin[0],pto[0]],[origin[1],pto[1]]) for pto in ptos]

pixels = [get_pixels(sl) for sl in slices]
values = [get_values(pix) for pix in pixels]
distance = [sl['distance'] for sl in slices]

ia.close()

for a, dist, value in zip(ang,distance,values):
    out_file = output_name+'_'+str(a)+'deg.dat'
    profile = zip(dist,value)
    np.savetxt(out_file,profile,header=hdrtxt)

