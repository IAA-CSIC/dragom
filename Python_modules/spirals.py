import numpy as np

def spiral_P20(r,rc,hc,Pc,a=3/2,b=0.45):
    ''' Equation of a spiral induced by a planet at a distance rc and PA = Pc with respect to the star (Eq. 1 of Phuong+2020). Gives PA(r).

    Parameters: hc = H / r, being H the scale height of the disk at rc; a is the exponent are of the disk rotation (V_rot = r**(-a), Keplerian by default, thus a = 3/2), and b is the exponent of the radial dependence of the sound speed profile (c_sound = r**(-b), 0.45 by default).

    Angles are in radians. The spiral is wound anti-clockwise.
    '''
    r0 = r / rc
    sg = np.sign(r - rc) / hc
    B = 1 / (1 + b)
    A = 1 / (1 - a + b)
    P = Pc + sg * r0 ** (1 + b) * (B - A * r0 **(-a)) - sg * (B - A)
    return P


def spiral_M02(r,rc,hc,Pc,a=3/2,b=0.45):
    ''' Equation of a spiral induced by a planet at a distance rc and PA = Pc with respect to the star (Eq. 1 of Phuong+2020). Gives PA(r).

    Parameters: hc = H / r, being H the scale height of the disk at rc; a is the exponent are of the disk rotation (V_rot = r**(-a), Keplerian by default, thus a = 3/2), and b is the exponent of the radial dependence of the sound speed profile (c_sound = r**(-b), 0.45 by default).

    Angles are in radians. The spiral is wound anti-clockwise.
    '''
    r0 = r / rc
    sg = np.sign(r - rc) / hc
    B = 1 / (1 + b)
    A = 1 / (1 - a + b)
    P = Pc - sg * (r0 ** (1 + b) * (B - A * r0 **(-a)) - (B - A))
    return P

