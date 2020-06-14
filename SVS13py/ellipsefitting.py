import numpy as np
from numpy.linalg import eig, inv

import matplotlib.pyplot as plt

def fit_ellipse(x,y):
    x = x[:,np.newaxis]
    y = y[:,np.newaxis]
    D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
    S = np.dot(D.T,D)
    C = np.zeros([6,6])
    C[0,2] = C[2,0] = 2; C[1,1] = 1
    E, V =  eig(np.dot(inv(S), C))
    n = np.argmax(np.abs(E))
    params = V[:,n]
    return params

def ellipse_center(params):
    b,c,d,f,g,a = params[1]/2, params[2], params[3]/2, params[4]/2, params[5], params[0]
    num = b*b - a*c
    x0=(c*d - b*f) / num
    y0=(a*f - b*d) / num
    return np.array([x0,y0])

def ellipse_angle_of_rotation(params):
    b,c,d,f,g,a = params[1]/2, params[2], params[3]/2, params[4]/2, params[5], params[0]
    return 0.5*np.arctan(2*b/(a-c))

def ellipse_axis_length(params):
    b,c,d,f,g,a = params[1]/2, params[2], params[3]/2, params[4]/2, params[5], params[0]
    up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    res1=np.sqrt(up/down1)
    res2=np.sqrt(up/down2)
    return np.array([res1, res2])

def ellipse_angle_of_rotation2(params):
    b,c,d,f,g,a = params[1]/2, params[2], params[3]/2, params[4]/2, params[5], params[0]
    if b == 0:
        if a > c:
            return 0
        else:
            return np.pi/2
    else:
        if a > c:
            return np.arctan(2*b/(a-c))/2
        else:
            return np.pi/2 + np.arctan(2*b/(a-c))/2

def ellipse_fitting(x, y, print_params=True):
    params = fit_ellipse(x, y)
    center = ellipse_center(params)
    phi = ellipse_angle_of_rotation(params)
    #phi = ellipse_angle_of_rotation2(params)
    axes = ellipse_axis_length(params)

    if print_params:
        print("center = ",  center)
        print("angle of rotation = ",  phi)
        print("axes = ", axes)

    return params, center, phi, axes
