import numpy as np
from taskinit import *

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

input_image = ''
origin = ''
end = ''
output_name = ''

ia.open(input_image)
info = ia.summary

bright_u = info()['unit']
pix_size = info()['incr'][1]*180/np.pi*3600 # size of a pixel in arcsec

hdrtxt='Dist (pix)\tValue({})\t(pix = {} arcsec)'.format(bright_u,pix_size)  #CASA does'n uses python 3 yet

origin_pixs = np.loadtxt(origin)
end_pixs = np.loadtxt(end)

slices = []
for orig, end in zip(origin_pixs, end_pixs):
    sl = ia.getslice([orig[0],end[0]],[orig[1],end[1]])
    slices.append(sl)

pixels = [get_pixels(sl) for sl in slices]
values = [get_values(pix) for pix in pixels]
distance = [sl['distance'] for sl in slices]

ia.close()

i=1
for dist, value in zip(distance,values):
    out_file = output_name+'_'+str(i)+'.dat'
    profile = zip(dist,value)
    np.savetxt(out_file,profile,header=hdrtxt)
    i+=1

