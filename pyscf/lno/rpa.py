#!/usr/bin/env python
# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

r''' RPA (direct ring CCD) amplitudes for LNO construction and external correction

    Closed-shell drCCD with real orbitals, written as matrices over the ov pair
    index (T_{ia,jb} = t2[i,j,a,b], V_{ia,jb} = (ia|jb) = sum_P J_{iaP} J_{jbP}):

        V + D o T + 2 V T + 2 T V + 4 T V T = 0,     D_{ia,jb} = e_a + e_b - e_i - e_j

    i.e. D o T = -(1+2T) V (1+2T) = -U U^T with U = (1+2T) J. Hence

        T = -D^{-1} o (U U^T),      D^{-1}_{ia,jb} ~= sum_w tau_{ia,w} tau_{jb,w}

    where tau is the pivoted Cholesky factor of the Cauchy matrix 1/(D_ia + D_jb).
    Only U (ov x naux) and tau (ov x ncd) are stored; U obeys the fixed-point equation

        U = J + 2 T J = J - 2 sum_w tau_w o (U M_w),    M_w = (tau_w o U)^T J

    which costs O(ncd * ov * naux^2) per iteration. The RPA correlation energy is

        E_RPA   = 2 sum T_{ia,jb} V_{ia,jb} = sum_{iaP} (U - J)_{iaP} J_{iaP}
        E_SOSEX = sum T_{ia,jb} (2 V_{ia,jb} - V_{ib,ja})

    Ref:
        Song, Gong, Bakry, Ye, arXiv:2601.00131 (RPA-based LNO-CC)
        Hesselmann, Phys. Rev. A 85, 012517 (2012)
        Kallay, J. Chem. Phys. 142, 204105 (2015)
'''

import sys
import numpy as np

from pyscf import lib
from pyscf.lib import logger

einsum = lib.einsum


def cholesky_denom(x, thresh=1e-5):
    r''' Pivoted Cholesky decomposition of the Cauchy matrix C_pq = 1/(x_p + x_q).

        Args:
            x : 1D array of positive excitation energies (e_a - e_i).
            thresh : stop when the largest remaining diagonal falls below thresh.

        Returns:
            tau (len(x), ncd) with C ~= tau tau^T.
    '''
    x = np.asarray(x, dtype=np.float64).ravel()
    if np.any(x <= 0):
        raise ValueError('RPA energy denominator requires e_a - e_i > 0 '
                         '(min = %.3g).' % x.min())
    d = 0.5 / x
    vecs = []
    while True:
        p = np.argmax(d)
        if d[p] < thresh or len(vecs) == x.size:
            break
        col = 1. / (x + x[p])
        if vecs:
            L = np.asarray(vecs).T
            col -= np.dot(L, L[p])
        l = col / np.sqrt(d[p])
        vecs.append(l)
        d -= l*l
    return np.asarray(vecs).T.reshape(x.size, len(vecs))

def get_TJ(U, J, tau):
    r''' (T J)_{iaP} = - sum_w tau_{ia,w} sum_Q U_{iaQ} [ sum_{jb} U_{jbQ} tau_{jb,w} J_{jbP} ]
    '''
    TJ = np.zeros_like(J)
    for w in range(tau.shape[1]):
        tw = tau[:,w,None]
        M = np.dot((U*tw).T, J)
        TJ -= tw * np.dot(U, M)
    return TJ

def kernel(J, eia, cd_thresh=1e-5, max_cycle=100, conv_tol=1e-8, conv_tol_normu=1e-6,
           diis_space=8, damp=0., verbose=None, log=None):
    r''' Solve factorized drCCD.

        Args:
            J : (nov, naux) real DF integrals, (ia|jb) = J J^T.
            eia : (nov,) e_a - e_i.

        Returns:
            converged, e_rpa, U, tau
    '''
    if log is None: log = logger.new_logger(lib.StreamObject, verbose)
    cput0 = (logger.process_clock(), logger.perf_counter())
    J = np.asarray(J)
    if np.iscomplexobj(J):
        raise NotImplementedError('drCCD is implemented for real orbitals only.')
    nov, naux = J.shape
    tau = cholesky_denom(eia, cd_thresh)
    log.debug('drCCD: nov = %d  naux = %d  ncd = %d (cd_thresh = %g)',
              nov, naux, tau.shape[1], cd_thresh)

    adiis = lib.diis.DIIS()
    adiis.space = diis_space

    U = J.copy()
    e_rpa = 0.
    converged = False
    for cycle in range(max_cycle):
        TJ = get_TJ(U, J, tau)
        e_new = 2. * np.dot(TJ.ravel(), J.ravel())
        Unew = J + 2.*TJ
        TJ = None
        err = Unew - U
        normu = np.linalg.norm(err)
        if damp > 0:
            Unew = (1.-damp)*Unew + damp*U
        de, e_rpa = e_new - e_rpa, e_new
        log.debug('drCCD cycle = %d  E_RPA = %.15g  dE = %.3g  norm(dU) = %.3g',
                  cycle+1, e_rpa, de, normu)
        if abs(de) < conv_tol and normu < conv_tol_normu:
            converged = True
            break
        U = adiis.update(Unew, xerr=err)
        Unew = err = None
    if not converged:
        log.warn('drCCD did not converge in %d cycles (dE = %.3g  norm(dU) = %.3g).',
                 max_cycle, de, normu)
    log.timer('drCCD', *cput0)
    return converged, e_rpa, U, tau

def energy(U, J):
    r''' E_RPA = sum_{iaP} (U - J)_{iaP} J_{iaP} '''
    return np.dot((U-J).ravel(), J.ravel())

def energy_sosex(U, J, tau, nocc, nvir, max_memory=2000):
    r''' E_SOSEX = E_RPA - sum T_{ia,jb} (ib|ja), evaluated in occupied blocks.
    '''
    naux = J.shape[1]
    ncd = tau.shape[1]
    U = U.reshape(nocc,nvir,naux)
    J = J.reshape(nocc,nvir,naux)
    tau = tau.reshape(nocc,nvir,ncd)
    e_rpa = energy(U, J)
    mem_avail = max(max_memory - lib.current_memory()[0], 1)
    # per occ in a block pair: ~3 copies of (o v ncd*naux) and 2 copies of (o v^2)
    blksize = max(1, min(nocc, int(mem_avail*1e6/8 /
                                   (3*nvir*ncd*naux + 2*nvir**2*nocc))))
    ex = 0.
    for i0,i1 in lib.prange(0,nocc,blksize):
        Ui = (tau[i0:i1,:,:,None] * U[i0:i1,:,None,:]).reshape(i1-i0,nvir,-1)
        for j0,j1 in lib.prange(0,nocc,blksize):
            Uj = (tau[j0:j1,:,:,None] * U[j0:j1,:,None,:]).reshape(j1-j0,nvir,-1)
            t2 = -einsum('iax,jbx->ijab', Ui, Uj)
            Uj = None
            ovvo = einsum('ibx,jax->ijab', J[i0:i1], J[j0:j1])
            ex += np.dot(t2.ravel(), ovvo.ravel())
            t2 = ovvo = None
        Ui = None
    return e_rpa - ex

def make_t2(U, tau, nocc, nvir):
    r''' Dense t2[i,j,a,b] = T_{ia,jb} = - sum_{w,P} (tau_w U)_{iaP} (tau_w U)_{jbP}
    '''
    Ut = (tau[:,:,None] * U[:,None,:]).reshape(nocc,nvir,-1)
    return -einsum('iax,jbx->ijab', Ut, Ut)


def get_ovJ(eris):
    r''' Load the full real DF integral J[i,a,P] from an LNO ERI object.

        For KLNO with real supercell orbitals, (ia|jb) = sum_P (R R + I I) and J is the
        concatenation of the real and imaginary parts along the auxiliary index.
    '''
    _check_real(eris)
    nocc, nvir = eris.nocc, eris.nvir
    blks = []
    blksize = max(1, nocc//4)
    for i0,i1 in lib.prange(0,nocc,blksize):
        blks.append(_as_real_J(eris.get_occ_blk(i0,i1)))
    return np.concatenate(blks, axis=0).reshape(nocc,nvir,-1)

def get_ovJ_act(eris, uocc, uvir):
    r''' J[I,A,P] = sum_{ia} uocc[i,I] uvir[a,A] J[i,a,P] for active (impurity) orbitals.
    '''
    _check_real(eris)
    OvJ = _as_real_J(eris.xform_occ(np.asarray(uocc, dtype=np.float64)))
    return einsum('Iax,aA->IAx', OvJ, uvir)

def _check_real(eris):
    if np.dtype(eris.dtype).kind == 'c':
        raise NotImplementedError('RPA-based LNO requires real orbitals (molecules, '
                                  'Gamma point, or time-reversal-symmetric k-meshes).')

def _as_real_J(blk):
    if isinstance(blk, tuple):
        return np.concatenate(blk, axis=-1)
    return np.asarray(blk)


class _RPAAmpERIS:
    r''' Drop-in replacement of the LNO DF ERI object for `make_lno_rdm1`.

        It exposes the three-index quantity Ut[i,a,(w,P)] = tau[i,a,w] U[i,a,P] as `ovL`,
        so that T_{ia,jb} = - sum_x Ut_{iax} Ut_{jbx}. The rdm1 functions recognize
        `rpa_amp = True` and use t2 = -num instead of num/denom.
    '''
    rpa_amp = True

    def __init__(self, U, tau, max_memory, verbose=None, stdout=None):
        self.nocc, self.nvir, self.naux_df = U.shape
        self.U = U
        self.tau = tau.reshape(self.nocc,self.nvir,-1)
        self.max_memory = max_memory
        self.verbose = logger.WARN if verbose is None else verbose
        self.stdout = sys.stdout if stdout is None else stdout
        self.dtype = np.float64
        self.dsize = 8

    @property
    def ncd(self):
        return self.tau.shape[2]
    @property
    def naux(self):
        return self.ncd * self.naux_df

    def get_occ_blk(self, i0,i1):
        return (self.tau[i0:i1,:,:,None] *
                self.U[i0:i1,:,None,:]).reshape(i1-i0,self.nvir,self.naux)

    def get_vir_blk(self, a0,a1):
        return (self.tau[:,a0:a1,:,None] *
                self.U[:,a0:a1,None,:]).reshape(self.nocc,a1-a0,self.naux)

    def _occblksize(self, nout):
        M = (self.max_memory - lib.current_memory()[0])*1e6 / self.dsize
        return min(self.nocc, max(1, int(np.floor(M*0.5/(self.nvir*self.naux) - nout))))

    def xform_occ(self, u):
        nOcc = u.shape[1]
        OvL = np.zeros((nOcc,self.nvir,self.naux), dtype=self.dtype)
        for i0,i1 in lib.prange(0,self.nocc,self._occblksize(nOcc)):
            OvL += lib.einsum('iax,iI->Iax', self.get_occ_blk(i0,i1), u[i0:i1])
        return OvL

    def xform_vir(self, u):
        nVir = u.shape[1]
        oVL = np.empty((self.nocc,nVir,self.naux), dtype=self.dtype)
        blksize = self._occblksize(self.nocc*nVir/float(self.nvir))
        for i0,i1 in lib.prange(0,self.nocc,blksize):
            oVL[i0:i1] = lib.einsum('iax,aA->iAx', self.get_occ_blk(i0,i1), u)
        return oVL


def get_fragment_energy_rpa(oovv, t2, uocc_loc):
    r''' Direct-ring fragment energy 2 sum_{Kjab} t2(Kjab) (Ka|jb) (no exchange). '''
    m = np.dot(uocc_loc, uocc_loc.T.conj())
    ed = (einsum('ijab,kjab,ik->', t2, oovv, m) * 2).real
    return lib.tag_array(ed, spin_comp=np.array((ed*0.5, ed*0.5)))
