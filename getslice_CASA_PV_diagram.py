import numpy as np
from taskinit import *

input_image = '/home/akdiaz/Documents/Trabajo/SVS13/Images/Moms/Figuras/Momentos-final/pv-EthyGly-spw-4-11-129~167.90deg'
origin = 'pix_init_PV.txt'
end = 'pix_end_PV.txt'
output_name = 'PV_cut'

ia.open(input_image)
info = ia.summary

bright_u = info()['unit']
pix_size = info()['incr'][0] # size of a pixel in arcsec

hdrtxt='Dist (pix)\tValue({})\t(pix = {} arcsec)'.format(bright_u,pix_size)  #CASA does'n uses python 3 yet

origin_pixs = np.loadtxt(origin)
end_pixs = np.loadtxt(end)

slices = []
for orig, end in zip(origin_pixs, end_pixs):
    sl = ia.getslice(x=[orig[0],end[0]],y=[orig[1],end[1]],axes=[0,2])
    slices.append(sl)

distance = [sl['distance'] for sl in slices]
values = [sl['pixel'] for sl in slices]


ia.close()

i=1
for dist, value in zip(distance,values):
    out_file = output_name+'_'+str(i)+'.dat'
    profile = zip(dist,value)
    np.savetxt(out_file,profile,header=hdrtxt)
    i+=1

