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
# Author: Ardavan Farahvash
#

r'''
Direct ring CCD (direct RPA) with density fitting, closed shell

With real orbitals and matrices over the ov pair index (T_{ia,jb} = t2[i,j,a,b],
V_{ia,jb} = (ia|jb) = sum_P J_{iaP} J_{jbP}, D_{ia,jb} = e_a + e_b - e_i - e_j), the
drCCD amplitude equation is

    V + D o T + 2 V T + 2 T V + 4 T V T = 0

i.e. D o T = -U U^T with the dressed integrals U = (1 + 2T) J. Hence

    t2[i,j,a,b] = (U_ia . U_jb) / (e_i + e_j - e_a - e_b)

a drCCD amplitude is an MP2 amplitude with J replaced by U. Two solvers give U:

    'riccati' : fixed-point iteration of U = J + 2 T J with the factorized
                T_{ia,jb} = - sum_w tau_{ia,w} tau_{jb,w} (U U^T)_{ia,jb}, where tau is the
                pivoted Cholesky factor of 1/(D_ia + D_jb). O(ncd ov naux^2) per cycle.
    'eig'     : the RPA eigenproblem with A = D + 2V, B = 2V,

                    [A B; B A] (X; Y) = w [1 0; 0 -1] (X; Y),     T = 1/2 Y X^-1,

                solved in the Hermitian form D^1/2 (D + 4V) D^1/2 Z = w^2 Z with
                X + Y = D^1/2 Z w^-1/2 and X - Y = D^-1/2 Z w^1/2, so that
                U = (1 + Y X^-1) J = (X + Y) X^-1 J. O((ov)^3) time and O((ov)^2) memory.

The correlation energies are

    E_dRPA  = 2 sum T_{ia,jb} V_{ia,jb} = sum_{iaP} (U - J)_{iaP} J_{iaP}
    E_SOSEX = sum T_{ia,jb} (2 V_{ia,jb} - V_{ib,ja})

Ref:
    Scuseria, Henderson, Sorensen, J. Chem. Phys. 129, 231101 (2008)
    Song, Gong, Bakry, Ye, arXiv:2601.00131
'''

import numpy as np

from pyscf import lib
from pyscf.lib import logger
from pyscf.mp import dfmp2
from pyscf import __config__

einsum = lib.einsum

WITH_T2 = getattr(__config__, 'regmp_driccd_with_t2', True)


def kernel(mycc, J, eo, ev, verbose=None):
    r''' Dressed integrals U[i,a,P] of drCCD with the solver set in `mycc`.

    Args:
        J : (nocc, nvir, naux) real DF integrals in canonical orbitals.
        eo, ev : occupied and virtual orbital energies.

    Returns:
        converged, U
    '''
    log = logger.new_logger(mycc, verbose)
    nocc, nvir, naux = J.shape
    J = np.asarray(J).reshape(nocc*nvir,naux)
    if np.iscomplexobj(J):
        raise NotImplementedError('drCCD is implemented for real orbitals only.')
    eia = (ev[None,:] - eo[:,None]).ravel()
    solver = mycc.solver.lower()
    if solver == 'riccati':
        conv, U = kernel_riccati(J, eia, cd_thresh=mycc.cd_thresh, max_cycle=mycc.max_cycle,
                                 conv_tol=mycc.conv_tol, conv_tol_normu=mycc.conv_tol_normu,
                                 diis_space=mycc.diis_space, damp=mycc.damp, log=log)
    elif solver == 'eig':
        conv, U = True, kernel_eig(J, eia, log=log)
    else:
        raise ValueError('drCCD solver must be "riccati" or "eig" (got %s)' % mycc.solver)
    return conv, U.reshape(nocc,nvir,naux)

def cholesky_denom(x, thresh=1e-6):
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

def kernel_riccati(J, eia, cd_thresh=1e-6, max_cycle=100, conv_tol=1e-8, conv_tol_normu=1e-6,
                   diis_space=8, damp=0., verbose=None, log=None):
    r''' Iterative (factorized Riccati) drCCD solver.

    Args:
        J : (nov, naux) real DF integrals, (ia|jb) = J J^T.
        eia : (nov,) e_a - e_i.

    Returns:
        converged, U (nov, naux)
    '''
    if log is None: log = logger.new_logger(lib.StreamObject, verbose)
    cput0 = (logger.process_clock(), logger.perf_counter())
    nov, naux = J.shape
    tau = cholesky_denom(eia, cd_thresh)
    log.debug('drCCD (riccati): nov = %d  naux = %d  ncd = %d (cd_thresh = %g)',
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
    log.timer('drCCD (riccati)', *cput0)
    return converged, U

def kernel_eig(J, eia, verbose=None, log=None):
    r''' drCCD from the RPA eigenproblem (see the module docstring).

    Args:
        J : (nov, naux) real DF integrals, (ia|jb) = J J^T.
        eia : (nov,) e_a - e_i.

    Returns:
        U (nov, naux)
    '''
    if log is None: log = logger.new_logger(lib.StreamObject, verbose)
    cput0 = (logger.process_clock(), logger.perf_counter())
    nov = eia.size
    log.debug('drCCD (eig): nov = %d  naux = %d', nov, J.shape[1])
    d = np.sqrt(eia)
    dJ = d[:,None] * J
    M = 4. * np.dot(dJ, dJ.T)
    dJ = None
    M[np.diag_indices(nov)] += eia**2
    w2, Z = np.linalg.eigh(M)
    M = None
    if w2[0] <= 0:
        raise RuntimeError('RPA instability: smallest excitation energy^2 = %.3g' % w2[0])
    w = np.sqrt(w2)
    XpY = Z * (d[:,None] / np.sqrt(w))
    X = .5 * (XpY + Z * (np.sqrt(w) / d[:,None]))
    Z = None
    U = np.dot(XpY, np.linalg.solve(X, J))
    log.timer('drCCD (eig)', *cput0)
    return U

def energy(U, J):
    r''' E_dRPA = sum_{iaP} (U - J)_{iaP} J_{iaP}, tagged with its (equal) same- and
    opposite-spin parts.
    '''
    e = np.dot((U-J).ravel(), J.ravel())
    return lib.tag_array(e, e_corr_ss=e*.5, e_corr_os=e*.5)

def energy_sosex(U, J, eo, ev, max_memory=2000):
    r''' E_SOSEX = E_dRPA - sum_{ijab} t2[i,j,a,b] (ib|ja), with t2 from `make_t2`. '''
    nocc, nvir = eo.size, ev.size
    U = U.reshape(nocc,nvir,-1)
    J = J.reshape(nocc,nvir,-1)
    eia = eo[:,None] - ev
    mem_avail = max(max_memory - lib.current_memory()[0], 1)
    blksize = max(1, min(nocc, int(mem_avail*0.5e6/8 / (3*nocc*nvir**2))))
    ex = 0.
    for i0, i1 in lib.prange(0, nocc, blksize):
        t2 = einsum('iax,jbx->ijab', U[i0:i1], U)
        t2 /= lib.direct_sum('ia+jb->ijab', eia[i0:i1], eia)
        ovov = einsum('iax,jbx->ijab', J[i0:i1], J)
        ex += np.einsum('ijab,ijba', t2, ovov)
        t2 = ovov = None
    return float(energy(U, J)) - ex

def make_t2(U, eo, ev):
    r''' Dense drCCD amplitudes t2[i,j,a,b] = (U_ia . U_jb) / (e_i + e_j - e_a - e_b). '''
    eia = eo[:,None] - ev
    return einsum('iax,jbx->ijab', U, U) / lib.direct_sum('ia+jb->ijab', eia, eia)


class DRICCD(dfmp2.DFRMP2):
    r''' Direct ring CCD (direct RPA) with density fitting for an RHF reference.

    Attributes:
        solver : str
            'riccati' (iterative, default) or 'eig' (RPA eigenproblem).
        with_sosex : bool
            Whether to also compute the SOSEX energy. Default is True.
        cd_thresh, max_cycle, conv_tol, conv_tol_normu, diis_space, damp :
            Settings of the 'riccati' solver.

    Saved results:
        e_corr : dRPA correlation energy
        e_corr_sosex : SOSEX correlation energy (if with_sosex)
        converged
        U : dressed DF integrals (nocc, nvir, naux), U = (1 + 2T) J
        t2 : amplitudes (if with_t2)
    '''

    solver = getattr(__config__, 'regmp_driccd_DRICCD_solver', 'riccati')
    with_sosex = getattr(__config__, 'regmp_driccd_DRICCD_with_sosex', True)
    cd_thresh = getattr(__config__, 'regmp_driccd_DRICCD_cd_thresh', 1e-6)
    max_cycle = getattr(__config__, 'regmp_driccd_DRICCD_max_cycle', 100)
    conv_tol = getattr(__config__, 'regmp_driccd_DRICCD_conv_tol', 1e-8)
    conv_tol_normu = getattr(__config__, 'regmp_driccd_DRICCD_conv_tol_normu', 1e-6)
    diis_space = getattr(__config__, 'regmp_driccd_DRICCD_diis_space', 8)
    damp = getattr(__config__, 'regmp_driccd_DRICCD_damp', 0.)

    _keys = {'converged', 'e_corr_sosex', 'U'}

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None, mo_energy=None):
        if mf.istype('UHF'):
            raise NotImplementedError('drCCD is implemented for RHF references only.')
        dfmp2.DFRMP2.__init__(self, mf, frozen, mo_coeff, mo_occ, mo_energy)
        self.converged = False
        self.e_corr_sosex = None
        self.U = None

    def dump_flags(self, verbose=None):
        dfmp2.DFRMP2.dump_flags(self, verbose)
        log = logger.new_logger(self, verbose)
        log.info('solver = %s  with_sosex = %s', self.solver, self.with_sosex)
        if self.solver.lower() == 'riccati':
            log.info('cd_thresh = %g  max_cycle = %d  conv_tol = %g  conv_tol_normu = %g  '
                     'diis_space = %d  damp = %g', self.cd_thresh, self.max_cycle,
                     self.conv_tol, self.conv_tol_normu, self.diis_space, self.damp)
        return self

    def init_amps(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2):
        if eris is None:
            eris = self.ao2mo(mo_coeff)
        eo, ev = self.split_mo_energy()[1:3]
        J = eris.get_occ_blk(0, eris.nocc)
        self.converged, self.U = kernel(self, J, eo, ev)
        e_corr = energy(self.U, J)
        if self.with_sosex:
            self.e_corr_sosex = energy_sosex(self.U, J, eo, ev, self.max_memory)
        t2 = make_t2(self.U, eo, ev) if with_t2 else None
        return e_corr, t2

    @property
    def e_tot_sosex(self):
        return float(self.e_hf) + self.e_corr_sosex

    def _finalize(self):
        log = logger.new_logger(self)
        log.note('E(%s) = %.15g  E_corr(dRPA) = %.15g',
                 self.__class__.__name__, self.e_tot, self.e_corr)
        if self.e_corr_sosex is not None:
            log.note('E(%s-SOSEX) = %.15g  E_corr(SOSEX) = %.15g',
                     self.__class__.__name__, self.e_tot_sosex, self.e_corr_sosex)
        return self

    def make_rdm2(self, *args, **kwargs):
        raise NotImplementedError('drCCD rdm2')
