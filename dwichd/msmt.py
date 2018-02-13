'''
Created on Feb 3, 2015

@author: Daan Christiaens (daan.chistiaens@gmail.com)

'''

import numpy as np
from .sh import modshbasiscart
from scipy.linalg import block_diag
from cvxopt import matrix, spmatrix, solvers
solvers.options['show_progress'] = False
solvers.options['abstol'] = 1e-3
solvers.options['reltol'] = 5e-3
solvers.options['feastol'] = 1e-3
from multiprocessing import Pool
from scipy.spatial.distance import cdist
import itertools
import sys
import time


def _n2l(n):
    return int(np.sqrt(1+8*n)-3)//2


def _l2n(l):
    return (l+1)*(l+2)//2


def zh2rh(h):
    lmax = 2*(h.shape[-1]-1)
    z2r = np.zeros((h.shape[-1], _l2n(lmax)))
    for l in range(0, lmax+1, 2):
        j1, j2 = _l2n(l-2), _l2n(l)
        z2r[l//2,j1:j2] = np.sqrt(4*np.pi/(2*l+1))
    return np.einsum('...i,ij->...j', h, z2r)


def sphconv(h, f):
    return np.einsum('ij,...j->...ij', zh2rh(h), f)


def calcsig(dwi, grad, lmax=8):
    '''
    Multi-shell DWI fit in the SH basis.
    '''
    bvals = np.unique(grad[:,3])
    B = modshbasiscart(lmax, grad[:,0], grad[:,1], grad[:,2])
    n = [_l2n(l) for l in range(0,lmax+1,2)]
    S = np.zeros(dwi.shape[:-1]+(len(bvals),n[-1]))
    for k, b in enumerate(bvals):
        bidx = grad[:,3]==b
        nn = max([a for a in n if a < np.sum(bidx)]) if b > 10.0 else 1
        pinvB = np.linalg.pinv(B[bidx,:nn])
        S[...,k,:nn] = np.dot(dwi[...,bidx], pinvB.T)
    return S


def csdsig(sig, H, dirs, lmax=None, mask=None, aux=None):
    '''
    Multi-shell multi-tissue constrained spherical deconvolution.
    '''
    # Prepare matrices
    m, n = sig.shape[-2:]
    if lmax is None:
        lmax = [2*(h.shape[1]-1) for h in H]
    nco  = [_l2n(l) for l in lmax]
    K = matrix([[matrix(np.diag([np.sqrt(4*np.pi/(2*l+1))*r[l//2] for l in range(0,lmax[j]+1,2) for k in range(2*l+1)], k=n-nco[j])[:,n-nco[j]:]) for r in h] for j, h in enumerate(H)])
    KK = K.T * K
    P = -1.0 * modshbasiscart(max(lmax), dirs[:,0], dirs[:,1], dirs[:,2])
    A = matrix(block_diag(*[P[:,:nc] for nc in nco]))
    z0 = matrix(0.0, (A.size[0], 1))
    # Solve
    F = np.zeros(sig.shape[:-2]+(len(nco),max(nco)))
    c = -K.T * matrix(sig.reshape((-1,m*n))).T
    vox = np.arange(c.size[1], dtype=int)
    if mask is not None:
        vox = vox[mask.ravel()>0.5]
    idx = np.zeros(F.shape, dtype=np.bool)
    idx.reshape((-1,len(nco),max(nco)))[vox] = np.array([[True if j < nc else False for j in range(max(nco))] for nc in nco])
    pool = _get_shared_pool()
    T = pool.map(_ParallelQP(KK, c, A, z0), vox.tolist())
    F[idx] = np.ravel(T)
    res = np.sum((sig.reshape((-1,m*n))[vox] - np.dot(T, K.T))**2)
    if aux is not None:
        aux['S'] = np.zeros(sig.shape)
        aux['S'].reshape((-1,m*n))[vox] = np.dot(T, K.T)  # predicted SH signal
        aux['dS'] = np.sqrt(np.einsum('...bj->...', (sig - aux['S'])**2))
        aux['dS'][~mask] = 0.0
        #aux['dSrel'] = aux['dS']/np.sqrt(np.einsum('...bj->...', sig**2))
        #aux['dSrel'][~mask] = 0.0
    return F, res


class _ParallelQP(object):
    def __init__(self, P, q, G, h):
        self.P = P
        self.q = q
        self.G = G
        self.h = h
    def __call__(self, j):
        return np.ravel(solvers.qp(self.P, self.q[:,j], self.G, self.h)['x'])


_pool = None
def _get_shared_pool():
    global _pool
    if _pool is None:
        _pool = Pool()
    return _pool


def cnsf(S, Z, H0, dirs, lmax=(8,0,0), rtol=5e-3, maxiters=50, printout=True):
    H = H0[:,:,:max(lmax)//2+1]
    residual = np.zeros((maxiters+1,))
    t0 = time.time()
    sol = None
    F, residual[0] = csdsig(S, H, dirs, lmax)
    for k in range(maxiters):
        H, W, sol = estimatekernelconvex(S[...,:_l2n(max(lmax))], Z[...,:max(lmax)//2+1], F, init=sol)
        F, residual[k+1] = csdsig(S, H, dirs, lmax)
        convergence = (abs(residual[k+1]-residual[k])/residual[k+1] < rtol)
        if printout:
            dt = int(time.time()-t0)+1
            print('{:d} iterations completed in {:d}h {:d}m {:d}s.'.format(k+1, dt//3600, (dt%36000)//60, dt%60))
            if convergence:
                print('Convergence reached.')
            sys.stdout.flush()
        if convergence:
            break
    return {'H': H, 'F': F, 'res': residual[:k+2], 'r0': residual[k+1], 'W': W, 'time': time.time()-t0, 'lmax': lmax, 'state': convergence}


def bestfitzonal(sig, dirs):
    '''
    Calculate the best fitting zonal harmonic in all directions.
    '''
    lmax = _n2l(sig.shape[-1])
    delta = modshbasiscart(lmax, dirs[:,0], dirs[:,1], dirs[:,2])
    R = [_rconvmat(d, lmax) for d in delta]
    pinvR = [np.linalg.pinv(r) for r in R]
    Z = np.einsum('ijk,...k->...ji', pinvR, sig)
    res = np.einsum('ijk,...ki->...ji', R, Z)
    res -= sig[...,np.newaxis]
    res *= res
    res = np.einsum('...jik->...k', res)    # Parseval accounted for above
    idx = np.argmin(res, axis=-1)
    vox = np.indices(idx.shape)
    if vox.ndim == 4:   # 3-d
        Z0 = Z[vox[0],vox[1],vox[2],:,:,idx]
    else:               # 1-d and flattened arrays
        Z0 = Z[vox,:,:,idx].squeeze()
    return Z0


def getzonal(sig):
    '''
    Calculate zonal harmonic from frequency content in each voxel.
    '''
    lmax = _n2l(sig.shape[-1])
    Z = np.zeros(sig.shape[:-1]+(lmax//2+1,))
    s = 1.
    for l in range(0, lmax+1, 2):
        j1, j2 = _l2n(l-2), _l2n(l)
        Sl = np.einsum('...ij,...kj->...ik', sig[...,j1:j2], sig[...,j1:j2])
        Ll, El = np.linalg.eigh(Sl)
        z = np.sqrt(Ll[...,-1,np.newaxis]) * El[...,:,-1]
        Z[...,:,l//2] = s * np.sign(np.sum(z, axis=-1)[...,np.newaxis]) * z
        s *= -1.
    return Z


def estimatekernelconvex(sig, zonal, fod, init=None):   #, w0=None, tau=10.):
    '''
    Response function estimation from data and ODFs, s.t. a convexity constraint in zonal harmonics.
    '''
    lmax = (zonal.shape[-1]-1)*2
    nshells, ncoefss = sig.shape[-2:]
    ntissues, ncoefsf = fod.shape[-2:]
    fodt = fod.reshape((-1, ntissues, ncoefsf))
    nvox = fodt.shape[0]
    # Build ODF convolution matrix
    idxrows = [v*nshells*ncoefss+k for v in range(nvox) for t in range(ntissues) for k in range(nshells*ncoefsf)]
    idxcols = [t*nshells*(lmax//2+1)+s*(lmax//2+1)+l//2 for v in range(nvox) for t in range(ntissues) for s in range(nshells) for l in range(0,lmax+1,2) for m in range(2*l+1)]
    fodt = fod * np.array([np.sqrt(4*np.pi/(2*l+1)) for l in range(0,lmax+1,2) for m in range(2*l+1)], ndmin=fod.ndim)
    fodt = np.tile(fodt, nshells)
    F = spmatrix(fodt.ravel(), idxrows, idxcols, size=(nvox*nshells*ncoefss, ntissues*nshells*(lmax//2+1)))
    # Build convexity matrix
    zonalt = zonal.reshape((-1,nshells*(lmax//2+1)))
    nvoxz, ncoefsz = zonalt.shape
    Z = matrix(0.0, (ntissues*ncoefsz, ntissues*nvoxz))
    for t in range(ntissues):
        Z[t*ncoefsz:(t+1)*ncoefsz, t*nvoxz:(t+1)*nvoxz] = zonalt.T
    # Construct matrices for QP problem
    H = Z.T * (F.T * F) * Z
    c = -Z.T * (F.T * matrix(sig.ravel()))
    #if w0 is not None:
    #    H[::ntissues*nvoxz+1] += tau
    #    c = c - matrix(w0.ravel() * tau)
    P = spmatrix(-1.0, range(ntissues*nvoxz), range(ntissues*nvoxz))
    z0 = matrix(0.0, (ntissues*nvoxz,1))
    A = spmatrix(1.0, [t for t in range(ntissues) for k in range(nvoxz)], range(ntissues*nvoxz))
    z1 = matrix(1.0, (ntissues,1))
    # Solve
    if init is not None:
        sol = solvers.qp(H, c, P, z0, A, z1, initvals=init)
    else:
        sol = solvers.qp(H, c, P, z0, A, z1)
    # Compute kernels
    w = np.reshape(sol['x'], (ntissues, nvoxz))
    w[w<0.] = 0.0                       # Correct numeric rounding errors that may
    w /= np.sum(w, 1)[:,np.newaxis]     # occurr if the input data is ill-conditioned.
    h = np.einsum('ijk,li->ljk', zonal, w)
    return h, w, sol


def estimatekernelconvex_full(sig, fod, dirs):
    '''
    Response function estimation from data and ODFs, s.t. a convexity constraint in spherical harmonics.
    '''
    nshells, ncoefss = sig.shape[-2:]
    ntissues, ncoefsf = fod.shape[-2:]
    fodt = fod.reshape((-1, ntissues, ncoefsf))
    nvox = fodt.shape[0]
    lmax = _n2l(ncoefsf)
    # Build ODF convolution matrix
    idxrows = [v*nshells*ncoefss+k for v in range(nvox) for t in range(ntissues) for k in range(nshells*ncoefsf)]
    idxcols = [t*nshells*(lmax//2+1)+s*(lmax//2+1)+l//2 for v in range(nvox) for t in range(ntissues) for s in range(nshells) for l in range(0,lmax+1,2) for m in range(2*l+1)]
    fodt = fod * np.array([np.sqrt(4*np.pi/(2*l+1)) for l in range(0,lmax+1,2) for m in range(2*l+1)], ndmin=fod.ndim)
    fodt = np.tile(fodt, nshells)
    F = spmatrix(fodt.ravel(), idxrows, idxcols, size=(nvox*nshells*ncoefss, ntissues*nshells*(lmax//2+1)))
    # Build convexity matrix
    idxrows = [s*(lmax//2+1)+l//2 for v in range(nvox) for s in range(nshells) for l in range(0,lmax+1,2) for m in range(2*l+1)]
    idxcols = [v*ncoefss+k for v in range(nvox) for s in range(nshells) for k in range(ncoefss)]
    sigt = spmatrix(sig.ravel(), idxrows, idxcols, (nshells*(lmax//2+1), nvox*ncoefss))
    S = spmatrix([], [], [], (ntissues*nshells*(lmax//2+1), ntissues*nvox*ncoefss))
    for t in range(ntissues):
        S[t*nshells*(lmax//2+1):(t+1)*nshells*(lmax//2+1), t*nvox*ncoefss:(t+1)*nvox*ncoefss] = sigt
    # Construct matrices for QP problem
    H = S.T * (F.T * F) * S
    c = -Z.T * (F.T * matrix(sig.ravel()))
    A = -1.0 * modshbasiscart(lmax, dirs[:,0], dirs[:,1], dirs[:,2])
    n, m = A.shape
    P = spmatrix([], [], [], (ntissues*nvox*n, ntissues*nvox*m))
    for x in range(ntissues*nvox):
        P[x*n:(x+1)*n, x*m:(x+1)*m] = A
    z0 = matrix(0.0, (ntissues*nvox*n,1))
    # Solve
    sol = solvers.qp(H, c, P, z0)
    # Compute kernels
    w = np.reshape(sol['x'], (ntissues, nvox, ncoefss))
    h = np.einsum('ijk,lik->ljk', sig, w)
    return h, w


def _rconvmat(f, lmax, out=None):
    if out is None:
        R = np.zeros((len(f), lmax//2+1))
    else:
        R = out
    for l in range(0,lmax+1,2):
        j0, j1 = _l2n(l-2), _l2n(l)
        R[j0:j1,l//2] = np.sqrt(4*np.pi/(2*l+1)) * f[j0:j1]
    return R


def initkernels(zonal, lmax=(8,0,0), max_iter=300, n_init=10, tol=1e-4):
    '''
    K-means initialization, constrained to subspaces.
    '''
    nb, nz = zonal.shape[-2:]
    Z = zonal.reshape((-1,nb,nz))
    nc, nv = len(lmax), Z.shape[0]
    C_x = np.zeros((n_init, nc, nb, nz))
    res_x = np.ones((n_init,)) * np.inf
    error = False
    for i in range(n_init):
        # initialize
        C0 = Z[np.random.randint(0,nv,nc),:,:]
        #C0 = _sortandproject(C0, lmax)
        for j in range(max_iter):
            # E-step
            D = _zonal_distance(C0, Z)
            labels = np.argmin(D, axis=0)
            res = np.sum(np.min(D, axis=0))
            if len(set(labels)) < nc:
                error = True
                break
            # M-step
            C = np.array([np.mean(Z[labels==k], axis=0) for k in range(nc)])
            C = _sortandproject(C, lmax)
            if np.sum((C-C0)**2) <= tol:
                break
            else:
                C0 = C
        if not error:
            C_x[i] = C
            res_x[i] = res
        else:
            error = False
    return C_x[np.argmin(res_x)]#, labels


def _zonal_distance(a, b):
    return cdist(a.reshape((a.shape[0], -1)), b.reshape((b.shape[0], -1)), 'cosine')


def _project(C, lmax=(8,0,0)):
    for i, l in enumerate(lmax):
        C[i,:,l//2+1:] = 0.0
    return C


def _sortandproject(C, lmax=(8,0,0)):
    bestC = C.copy()
    res0 = np.infty
    for p in itertools.permutations(range(len(lmax))):
        res = np.sum(np.diag(_zonal_distance(C[p,:,:], _project(np.copy(C[p,:,:]), lmax))))
        if res < res0:
            bestC = _project(np.copy(C[p,:,:]), lmax)
            res0 = res
    return bestC


