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
Direct ring CCD (direct RPA) with k-point sampling and density fitting, closed shell

Label the excitations P = (k_i, i, a) by their momentum transfer q = k_a - k_i and let
J_q[P,L] = (i a|L) / sqrt(N_k) (the GDF integrals of `pyscf.pbc.mp.kmp2`), so that
(ia|jb) = sum_L J_q[P,L] J_{-q}[Q,L] for P in q and Q in -q. The drCCD amplitudes
T_q[P,Q] = t2[i,j,a,b] (P in q, Q in -q) obey

    D o T_q = -U_q U_{-q}^T,      U_q = conj(J_q) + 2 T_q J_{-q}

with D_PQ = e_a + e_b - e_i - e_j: a drCCD amplitude is the MP2 amplitude with conj(J)
replaced by the dressed integrals U (`pyscf.regpt2.driccd` is the real, Gamma-point
case U = (1 + 2T) J). Two solvers give U:

    'riccati' : fixed-point iteration of U_q = conj(J_q) + 2 T_q J_{-q} for all q, with
                T_q = -C o (U_q U_{-q}^T) and C_PQ = 1/(D_P + D_Q) factorized by the
                pivoted Cholesky decomposition over the excitations of q and -q.
    'eig'     : for each pair (q, -q), the RPA eigenproblem

                    [A_q B_q; B_q^+ A_{-q}^*] (X; Y) = w [1 0; 0 -1] (X; Y),
                    A_q = D_q + 2 conj(J_q) J_q^T,    B_q = 2 conj(J_q J_{-q}^T),

                whose positive-w eigenvectors give T_{-q} = 1/2 conj(Y X^-1) and
                T_q = T_{-q}^T.

The correlation energies per unit cell are

    E_dRPA  = 1/N_k sum_q sum_{PL} (U_q - conj(J_q))_{PL} J_q[P,L]
    E_SOSEX = 1/N_k sum_{ki,kj,ka} sum_{ijab} t2[i,j,a,b] [2 (ia|jb) - (ib|ja)]
'''

import numpy as np
import scipy.linalg

from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc.mp import kmp2
from pyscf.regpt2 import driccd
from pyscf.pbc.regpt2.kkappamp2 import KKappaMP2
from pyscf import __config__

einsum = lib.einsum

WITH_T2 = getattr(__config__, 'pbc_regpt2_kdriccd_with_t2', True)


def kernel(mycc, J, D, minus=None, verbose=None):
    r''' Dressed integrals U_q of drCCD with the solver set in `mycc`.

    Args:
        J, D : per momentum transfer q, the DF integrals J_q (nexc, naux) and the
               excitation energies D_q = e_a - e_i (nexc), see `get_ovJ`.
        minus : the label of -q for each q. Default is from `mycc.khelper`.

    Returns:
        converged, U (list over q)
    '''
    log = logger.new_logger(mycc, verbose)
    if minus is None:
        minus = mycc.khelper.kconserv[0,:,0]      # q -> -q
    solver = mycc.solver.lower()
    if solver == 'riccati':
        return kernel_riccati(J, D, minus, cd_thresh=mycc.cd_thresh,
                              max_cycle=mycc.max_cycle, conv_tol=mycc.conv_tol,
                              conv_tol_normu=mycc.conv_tol_normu,
                              diis_space=mycc.diis_space, damp=mycc.damp, log=log)
    elif solver == 'eig':
        return True, kernel_eig(J, D, minus, log=log)
    raise ValueError('drCCD solver must be "riccati" or "eig" (got %s)' % mycc.solver)

def get_ovJ(mycc, mo_energy):
    r''' GDF integrals J_q[P,L] = (i a|L) / sqrt(N_k) and excitation energies
    D_q[P] = e_a - e_i of the excitations P = (k_i, i, a) with k_a = k_i + q, ordered by
    k_i. q is labelled by the k-point k_q with k_q - k_0 = q.
    '''
    nocc, nkpts = mycc.nocc, mycc.nkpts
    kconserv = mycc.khelper.kconserv
    Lov = kmp2._init_mp_df_eris(mycc)
    J, D = [], []
    for kq in range(nkpts):
        kas = [kconserv[ki,0,kq] for ki in range(nkpts)]
        J.append(np.vstack([Lov[ki,ka].reshape(Lov[ki,ka].shape[0],-1).T
                            for ki, ka in enumerate(kas)]) / np.sqrt(nkpts))
        D.append(np.concatenate([(mo_energy[ka][nocc:] - mo_energy[ki][:nocc,None]).ravel()
                                 for ki, ka in enumerate(kas)]))
    return J, D

def kernel_riccati(J, D, minus, cd_thresh=1e-6, max_cycle=100, conv_tol=1e-8,
                   conv_tol_normu=1e-6, diis_space=8, damp=0., verbose=None, log=None):
    r''' Iterative (factorized Riccati) drCCD for all momentum transfers q.

    Returns:
        converged, U (list over q)
    '''
    if log is None: log = logger.new_logger(lib.StreamObject, verbose)
    cput0 = (logger.process_clock(), logger.perf_counter())
    nq = len(J)
    sizes = [x.size for x in J]
    tau = []
    for q in range(nq):
        t = driccd.cholesky_denom(np.concatenate((D[q], D[minus[q]])), cd_thresh)
        tau.append((t[:D[q].size], t[D[q].size:]))
    log.debug('drCCD (riccati): nexc = %d  naux = %d  ncd = %s (cd_thresh = %g)',
              D[0].size, J[0].shape[1], [t[0].shape[1] for t in tau], cd_thresh)

    def get_TJ(U):
        r''' (T_q J_{-q}) for all q '''
        TJ = []
        for q in range(nq):
            mq = minus[q]
            tx, ty = tau[q]
            TJq = np.zeros(J[q].shape, dtype=np.complex128)
            for w in range(tx.shape[1]):
                M = np.dot((U[mq] * ty[:,w,None]).T, J[mq])
                TJq -= tx[:,w,None] * np.dot(U[q], M)
            TJ.append(TJq)
        return TJ

    adiis = lib.diis.DIIS()
    adiis.space = diis_space
    U = [x.conj() for x in J]
    e_rpa = 0.
    converged = False
    for cycle in range(max_cycle):
        TJ = get_TJ(U)
        e_new = 2 * sum(np.sum(TJq * Jq) for TJq, Jq in zip(TJ, J)).real / nq     # per cell
        Unew = [Jq.conj() + 2*TJq for TJq, Jq in zip(TJ, J)]
        TJ = None
        err = np.concatenate([(x - y).ravel() for x, y in zip(Unew, U)])
        normu = np.linalg.norm(err)
        Unew = np.concatenate([x.ravel() for x in Unew])
        if damp > 0:
            Unew = (1.-damp)*Unew + damp*np.concatenate([x.ravel() for x in U])
        de, e_rpa = e_new - e_rpa, e_new
        log.debug('drCCD cycle = %d  E_RPA = %.15g  dE = %.3g  norm(dU) = %.3g',
                  cycle+1, e_rpa, de, normu)
        if abs(de) < conv_tol and normu < conv_tol_normu:
            converged = True
            break
        Unew = adiis.update(Unew, xerr=err)
        U = [x.reshape(y.shape) for x, y in
             zip(np.split(Unew, np.cumsum(sizes)[:-1]), J)]
        Unew = err = None
    if not converged:
        log.warn('drCCD did not converge in %d cycles (dE = %.3g  norm(dU) = %.3g).',
                 max_cycle, de, normu)
    log.timer('drCCD (riccati)', *cput0)
    return converged, U

def kernel_eig(J, D, minus, verbose=None, log=None):
    r''' drCCD from the RPA eigenproblem of each pair (q, -q), see the module docstring.

    Returns:
        U (list over q)
    '''
    if log is None: log = logger.new_logger(lib.StreamObject, verbose)
    cput0 = (logger.process_clock(), logger.perf_counter())
    nq = len(J)
    U = [None] * nq
    for q in range(nq):
        mq = minus[q]
        if U[q] is not None:
            continue
        n = D[q].size
        A = 2 * np.dot(J[q].conj(), J[q].T)
        A[np.diag_indices(n)] += D[q]
        Am = 2 * np.dot(J[mq], J[mq].conj().T)
        Am[np.diag_indices(n)] += D[mq]
        B = 2 * np.dot(J[q], J[mq].T).conj()
        G = np.block([[A, B], [B.conj().T, Am]])
        A = Am = B = None
        S = np.append(np.ones(n), -np.ones(n))
        try:
            lam, v = scipy.linalg.eigh(np.diag(S), G)     # S v = (1/w) G v
        except np.linalg.LinAlgError:
            raise RuntimeError('RPA instability: the RPA matrix is not positive definite.')
        G = None
        v = v[:,lam > 0]
        assert v.shape[1] == n
        Tm = .5 * np.linalg.solve(v[:n].T, v[n:].T).T.conj()     # T_{-q} = 1/2 conj(Y X^-1)
        v = None
        U[mq] = J[mq].conj() + 2 * np.dot(Tm, J[q])
        U[q] = J[q].conj() + 2 * np.dot(Tm.T, J[mq])
        log.debug1('drCCD (eig): q = %d  nexc = %d', q, n)
    log.timer('drCCD (eig)', *cput0)
    return U

def energy(U, J, nkpts):
    r''' E_dRPA per cell, tagged with its (equal) same- and opposite-spin parts. '''
    e = sum(np.sum((Uq - Jq.conj()) * Jq) for Uq, Jq in zip(U, J)).real / nkpts
    return lib.tag_array(e, e_corr_ss=e*.5, e_corr_os=e*.5)

def energy_sosex_and_t2(U, J, eo, ev, kconserv, with_t2=WITH_T2):
    r''' E_SOSEX per cell and the amplitudes t2[ki,kj,ka] (if with_t2), for the occupied
    and virtual orbital energies eo[k], ev[k] and `kconserv` of
    `pyscf.pbc.lib.kpts_helper.get_kconserv`.
    '''
    nkpts = len(eo)
    nocc, nvir = len(eo[0]), len(ev[0])
    nov = nocc * nvir
    def blk(X, ki, ka):     # (nocc, nvir, naux) block of the excitations (ki,ka)
        return X[kconserv[ka,ki,0]][ki*nov:(ki+1)*nov].reshape(nocc,nvir,-1)
    t2 = np.zeros((nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir), dtype=complex) if with_t2 else None
    ex = 0.
    for ki in range(nkpts):
        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = kconserv[ki,ka,kj]
                t2_ijab = einsum('iax,jbx->ijab', blk(U,ki,ka), blk(U,kj,kb))
                t2_ijab /= lib.direct_sum('i-a+j-b->ijab', eo[ki], ev[ka], eo[kj], ev[kb])
                ovov_x = einsum('ibx,jax->ijab', blk(J,ki,kb), blk(J,kj,ka))    # (ib|ja)
                ex -= np.sum(t2_ijab * ovov_x).real
                if with_t2:
                    t2[ki,kj,ka] = t2_ijab
    return float(energy(U, J, nkpts)) + ex / nkpts, t2


class KDRICCD(kmp2.KMP2):
    r''' Direct ring CCD (direct RPA) with k-point sampling and GDF for a KRHF reference.

    Attributes:
        solver : str
            'riccati' (iterative, default) or 'eig' (RPA eigenproblem per q).
        with_sosex : bool
            Whether to also compute the SOSEX energy. Default is True.
        cd_thresh, max_cycle, conv_tol, conv_tol_normu, diis_space, damp :
            Settings of the 'riccati' solver (defaults of `pyscf.regpt2.DRICCD`).

    Saved results:
        e_corr : dRPA correlation energy per unit cell
        e_corr_sosex : SOSEX correlation energy per unit cell (if with_sosex)
        converged
        U : dressed DF integrals U_q (list over q, see `get_ovJ`)
        t2 : amplitudes t2[ki,kj,ka] (if with_t2)
    '''

    solver = getattr(__config__, 'pbc_regpt2_kdriccd_KDRICCD_solver', 'riccati')
    with_sosex = driccd.DRICCD.with_sosex
    cd_thresh = driccd.DRICCD.cd_thresh
    max_cycle = driccd.DRICCD.max_cycle
    conv_tol = driccd.DRICCD.conv_tol
    conv_tol_normu = driccd.DRICCD.conv_tol_normu
    diis_space = driccd.DRICCD.diis_space
    damp = driccd.DRICCD.damp

    _keys = {'converged', 'e_corr_sosex', 'U'}

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        if mf.istype('KUHF'):
            raise NotImplementedError('k-point drCCD is implemented for KRHF only.')
        kmp2.KMP2.__init__(self, mf, frozen, mo_coeff, mo_occ)
        if not self.with_df_ints:
            raise NotImplementedError('k-point drCCD requires GDF integrals.')
        self.converged = False
        self.e_corr_sosex = None
        self.U = None

    def dump_flags(self):
        kmp2.KMP2.dump_flags(self)
        logger.info(self, 'solver = %s  with_sosex = %s', self.solver, self.with_sosex)
        if self.solver.lower() == 'riccati':
            logger.info(self, 'cd_thresh = %g  max_cycle = %d  conv_tol = %g  '
                        'conv_tol_normu = %g  diis_space = %d  damp = %g', self.cd_thresh,
                        self.max_cycle, self.conv_tol, self.conv_tol_normu, self.diis_space,
                        self.damp)
        return self

    kernel = KKappaMP2.kernel

    def init_amps(self, mo_energy, mo_coeff, with_t2=WITH_T2):
        if (np.any(self.get_nocc(per_kpoint=True) != self.nocc) or
                np.any(self.get_nmo(per_kpoint=True) != self.nmo)):
            raise NotImplementedError('k-point drCCD needs the same number of occupied and '
                                      'virtual orbitals at every k-point.')
        J, D = get_ovJ(self, mo_energy)
        self.converged, self.U = kernel(self, J, D)
        e_corr = energy(self.U, J, self.nkpts)
        t2 = None
        if self.with_sosex or with_t2:
            nocc = self.nocc
            e_sosex, t2 = energy_sosex_and_t2(self.U, J, [e[:nocc] for e in mo_energy],
                                              [e[nocc:] for e in mo_energy],
                                              self.khelper.kconserv, with_t2)
            if self.with_sosex:
                self.e_corr_sosex = e_sosex
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
        raise NotImplementedError('k-point drCCD rdm2')
