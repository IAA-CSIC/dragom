from scipy import constants
from astropy import units as u
import numpy as np

K = constants.k * u.kg * u.m**2 * u.s**(-2) * u.K**(-1) #Boltzmann constant
G = constants.G * u.m**3 * u.kg**(-1) * u.s**(-2) #Gravitational constant
H = constants.h*u.J*u.s #Planck's constant
C = constants.c*100*u.cm*u.s**-1 #Speed of light in cm/s

def csound(T,m=2*1.00794*u.u,gamma=7/5):
    '''Calculates the speed of sound in a molecular gas at temperature T. By default assumes the gas is H2. m is the molecular mass and gamma the adiabatic index (7/5 for a diatomic gas and 5/3 for a monoatomic).

    It is mandatory to enter the units of the parameters: m (u,kg,g,etc) and T (K).'''
    cs = (gamma*K*T/m.to(u.kg))**(1/2) #disk sound speed
    return cs


def H_scale(r_disk, M_star, T_disk, m=2*1.00794*u.u, gamma=7/5, c_sound=None):
    ''' Calculate the disk scale height at a distance r_disk, of a disk with temperature T_disk around a star of mass M_star.

    Other parameters are: m is the molecular mass, gamma the adiabatic index, and c_sound is the speed of sound in the gas. By default assumes the gas is H2. If c_sound is None, it is calculated using this module's function csound with the input parameters.

    It is mandatory to enter the units of the parameters: M (solMass,kg,g,etc), T (K), r (au, km, m, etc) and c_sound (m/s, km/s).'''
    v = np.sqrt(G*M_star.to(u.kg)/r_disk.to(u.m)) #linear velocity at radius r
    w = v/r_disk.to(u.m) #Keplerian angular velocity
    if c_sound is None:
        c_sound = csound(T_disk, m, gamma)
    H = c_sound/w # disk scale height
    return H

def Q(M_star,M_disk,T_disk,r_disk,m=2*1.00794*u.u, gamma=7/5, c_sound=None):
    '''Calculates Toomre's parameter Q for a rotationally supported disk around a protostar (Kratter & Lodato 2016, Tobin 2016). Q < 1 means the disk is unstable (i.e. will fragment).

    Parameters are: M_star is the mass of the protostar; M_disk, T_disk and r_disk are the mass, temperature and radius of the disk, respectively; and c_sound is the speed of the sound in the gas (if it is None, it is calculated using this module's function csound with the input parameters).

    It is mandatory to enter the units of the parameters: M (solMass,kg,g,etc), T (K), r (au, km, m, etc) and c_sound (m/s).'''
    if c_sound is None:
        c_sound = csound(T_disk, m, gamma)
    HS = H_scale(r_disk, M_star, T_disk, c_sound=c_sound)
    Q = 2*M_star/M_disk * (HS/r_disk.to(u.m)).decompose()
    return Q

def MassMA01(S,T,D,k_nu,f=100):
    '''Calculates the mass of gas from the dust emission using eq (1) in Motte & André (2001).Parameters are: S
    is the dust flux density; T is the dust temperature; D is the distance to the source; k_nu is the dust opacity
    at the frequency of the observation and f is the gas-to-dust ratio (by default 100). Is mandatory to enter the
    units of the parameters S (mJy, Jy, etc), T (K), D (pc, km, m, etc), k_nu (cm^2/g).'''
    M = f*5.3e-3*(S.to(u.mJy)/(10*u.mJy))*(D.to(u.pc)/(140*u.pc))**2*(k_nu/(0.01*u.cm**2*u.g**-1))**(-1)*(T/(15*u.K))**-1
    return M*u.solMass # of dust

def MassT20(S,T,D,k_nu, wl,nu=None,f=100):
    '''Calculates the mass of gas from the dust emission using eq (1) in Tobin (2020).
    Parameters are: S is the dust flux density; T is the dust temperature; D is the distance to the source; k_nu is the dust opacity at the frequency of the observation and f is the gas-to-dust ratio (by default 100).
    It is mandatory to enter the units of the parameters S (mJy, Jy, etc), T (K), D (pc, km, m, etc), k_nu (cm^2/g).'''
    if nu is None:
        nu=C/wl.to(u.cm)
    B = 2*H*nu**3/C**2 * 1/(np.exp(H*nu/(K*T))-1)
    M = ((D.to(u.cm))**2*S/(k_nu*B)).decompose() # of dust
    return f*M.to(u.M_sun) # of gas


def EscapeVel(M,r):
    ''' Dives the escape velocity at a distance r from a star with mass M.
      It is mandatory to enter the units of the parameters M (solMass, kg, g, etc), r (au, km, cm, etc).
'''
    v = np.sqrt(2*G*M.to(u.kg)/r.to(u.m))
    return v
