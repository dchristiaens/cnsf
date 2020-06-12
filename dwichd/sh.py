'''
Python module that implements the modified SH basis as defined in Descoteaux et al. (2006).

Created on Jan 21, 2013

Copyright (c) 2012, Daan Christiaens (daan.christiaens@gmail.com)
'''

import numpy as np
from scipy.special import sph_harm


def c2s(x, y, z):
    '''
    Converts cartesian to spherical coordinates.
    '''
    r = np.hypot(x,y)
    theta = np.arctan2(r,z)
    phi = np.arctan2(y, x)
    np.hypot(r, z, out=r)
    return (theta, phi, r)


def s2c(theta, phi, r):
    '''
    Converts spherical to cartesian coordinates.
    '''
    x = r * np.cos(phi) * np.sin(theta);
    y = r * np.sin(phi) * np.sin(theta);
    z = r * np.cos(theta);
    return (x, y, z)


def n4l(L):
    '''
    Returns the number of components in the SH basis of order L.
    '''
    return (L+1)*(L+2)//2


def l2n(L):
    return n4l(L)


def l4n(R):
    '''
    Returns the order of the SH basis, given the number of coefficients.
    '''
    return int(np.sqrt(1+8*R) - 3)//2


def n2l(R):
    return l4n(R)


def modshbasis(L, theta, phi):
    '''
    Modified SH basis for spherical coordinates (theta, phi).
    '''
    assert theta.size == phi.size;
    out = np.zeros((theta.size,n4l(L)))
    rt2 = np.sqrt(2.0)
    for l in range(0,L+1,2):
        c = l*(l+1)//2
        out[:,c] = np.real(sph_harm(0,l,phi,theta))
        for m in range(1,l+1):
            sh = sph_harm(m, l, phi, theta)
            out[:,c+m] = rt2 * np.real(sh)
            out[:,c-m] = rt2 * np.imag(sh)
    return out


def modshbasiscart(L, x, y, z):
    '''
    Modified SH basis for cartesian coordinates (x, y, z).
    '''
    theta, phi, r = c2s(x, y, z)
    return modshbasis(L, theta, phi)


