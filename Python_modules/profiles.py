import numpy as np

def pos_in_maxs(structure_name, position_list, maxs_list):
        pos = np.where(position_list == structure_name)
        list_pos = [] 
        for i, j in zip(pos[0],pos[1]):
            list_pos.append([i, maxs_list[i][j]])
        return list_pos
    
def new_angle(ang,cut):
    new_ang = []
    for a in ang:
        if a > cut:
            new_ang.append(a-360)
        else:
            new_ang.append(a)
    return new_ang

def pixels(origin,length,ang): #origin = [x0,y0], length in pixels, ang en degrees
    x0, y0 = origin
    x1 = x0 - length * np.sin(ang*np.pi/180)
    y1 = y0 + length * np.cos(ang*np.pi/180)
    return [x1,y1]

 

def write_maxs(file_name,maxs):
    with open(file_name, 'w') as fmax:
        for prof in maxs:
            line = '\t'.join([str(i) for i in prof])+'\n' 
            fmax.write(line)
            
            
def load_maxs(file_name):
    maxs = []
    with open(file_name) as fmax:
        lines = fmax.readlines()
        for l in lines:
            nl = l.replace('\n', '').split('\t')
            m = [int(i) for i in nl] 
            maxs.append(m)
    return maxs

#needed to make the gaussian fits
def find_true_maxs(original_list, x, y):
    big_list = []
    sub_list = []
    for ix, valx in enumerate(x):
        to_append = original_list[valx][y[ix]]
   
        if ix == len(x)-1:
            sub_list.append(to_append)
            big_list.append(sub_list)
            break
        if valx == x[ix+1]:
            sub_list.append(to_append) 
        else:
            sub_list.append(to_append)
            big_list.append(sub_list)
            sub_list = []      
    return big_list


#funtion to fit (several gaussians)
def func(x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        ctr = params[i]
        amp = params[i+1]
        wid = params[i+2]
        y = y + amp * np.exp( -((x - ctr)/wid)**2)
    return y

def gaussian(x, amp, ctr, wid):
    return amp * np.exp( -((x - ctr)/wid)**2)


def spiral_log(r, ro, a):
    return 1/a * np.log(r/ro)

def spiral_arch(ang, ro, b):
    return ro + b * ang

def spiral_exp(r, a, b, c):
    return a*np.exp(r*b)+c