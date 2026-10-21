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

r''' Size-consistent Brillouin-Wigner second-order perturbation theory, BW-s2(alpha)

    BW-s2 dresses the occupied orbitals with a regularizer W built from the
    amplitudes and solves self-consistently (closed shell, spatial orbitals):

        W_ij = 1/4 sum_{kab} [ (2 t_ik^ab - t_ik^ba) (ja|kb) + (i <-> j) ]

        (F_oo + alpha W) U = U e~                  (dressed occupied orbitals)

        t~_ij^ab = (i~a|j~b) / (e~_i + e~_j - e_a - e_b)

        E_c = sum_{ijab} t~_ij^ab [ 2 (i~a|j~b) - (i~b|j~a) ]

    W is the spin-orbital W_ij = 1/8 sum_kab [t_ik^ab <jk||ab> + t_jk^ab <ik||ab>]
    with tr W (both spins) = E_c, so alpha = 1 is the original BW-s2 (exact for
    minimal-basis H2 at dissociation) and alpha = 0 is MP2. This is the alpha
    convention of the BW-s2(alpha) paper (molecular optimum alpha = 4) and of the
    periodic paper (alpha = 2 for solids).

    A BW-s2 amplitude is an MP2 amplitude in the dressed occupied basis: only
    U and e~ (nocc x nocc) are stored, and t2 is never formed for the full system.

    Ref:
        Carter-Fenk, Head-Gordon, J. Chem. Phys. 158, 234108 (2023), arXiv:2303.06271
        Carter-Fenk, Shee, Head-Gordon, arXiv:2309.01376 (BW-s2(alpha))
        Chen, Zhang, Dinh, Rettig, Lee, arXiv:2508.15744 (solids; W prefactor 1/8)
'''

import sys
import numpy as np

from pyscf import lib
from pyscf.lib import logger

einsum = lib.einsum


def energy_and_W(JW, eoW, ev, max_memory=2000):
    r''' Correlation energy and regularizer W in the dressed occupied basis.

        Args:
            JW : (nocc, nvir, naux) real DF integrals in the dressed occupied basis.
            eoW, ev : dressed occupied and virtual orbital energies.

        Returns:
            e_corr, W (nocc, nocc)
    '''
    nocc, nvir, naux = JW.shape
    eov = eoW[:,None] - ev
    mem_avail = max(max_memory - lib.current_memory()[0], 1)
    # per occupied in a block: 3 copies of (o v^2) and 1 of (v naux)
    blksize = max(1, min(nocc, int(mem_avail*1e6/8 / (3*nocc*nvir**2 + nvir*naux))))
    e_corr = 0.
    X = np.empty((nocc,nocc))
    for i0,i1 in lib.prange(0,nocc,blksize):
        ovov = einsum('iax,kbx->ikab', JW[i0:i1], JW)
        t2 = ovov / lib.direct_sum('ia+kb->ikab', eov[i0:i1], eov)
        tt = 2*t2 - t2.transpose(0,1,3,2)
        t2 = None
        e_corr += np.dot(tt.ravel(), ovov.ravel())
        ovov = None
        Gamma = einsum('ikab,kbx->iax', tt, JW)
        tt = None
        X[i0:i1] = einsum('iax,jax->ij', Gamma, JW)
        Gamma = None
    return e_corr, 0.25 * (X + X.T)

def energy_and_W_dense(oovv, eoW, ev):
    r''' Same as `energy_and_W` from dense oovv[i,j,a,b] = (ia|jb) (dressed basis). '''
    eov = eoW[:,None] - ev
    t2 = oovv / lib.direct_sum('ia+jb->ijab', eov, eov)
    tt = 2*t2 - t2.transpose(0,1,3,2)
    e_corr = np.dot(tt.ravel(), oovv.ravel())
    X = einsum('ikab,jkab->ij', tt, oovv)
    return e_corr, 0.25 * (X + X.T), t2

def kernel(J, eo, ev, alpha=1., max_cycle=50, conv_tol=1e-8, max_memory=2000,
           verbose=None, log=None):
    r''' BW-s2(alpha) from DF integrals.

        Args:
            J : (nocc, nvir, naux) real DF integrals, (ia|jb) = sum_P J[i,a,P] J[j,b,P],
                in canonical orbitals (occupied Fock = diag(eo)).
            eo, ev : canonical occupied and virtual orbital energies.

        Returns:
            converged, e_corr, eoW, U
            eoW are the dressed occupied energies and U[i,I] the dressed occupied
            orbitals in the input occupied basis (orbocc_dressed = orbocc @ U).
    '''
    if log is None: log = logger.new_logger(lib.StreamObject, verbose)
    cput0 = (logger.process_clock(), logger.perf_counter())
    J = np.asarray(J)
    if np.iscomplexobj(J):
        raise NotImplementedError('BW-s2 is implemented for real orbitals only.')
    nocc = J.shape[0]
    fock_oo = np.diag(eo)

    eoW = np.asarray(eo, dtype=np.float64)
    U = np.eye(nocc)
    e_corr = None
    converged = False
    for cycle in range(max_cycle):
        JW = einsum('iax,iI->Iax', J, U)
        e_new, W = energy_and_W(JW, eoW, ev, max_memory=max_memory)
        JW = None
        de = np.inf if e_corr is None else e_new - e_corr
        e_corr = e_new
        log.debug('BW-s2 cycle = %d  E_corr = %.15g  dE = %.3g', cycle+1, e_corr, de)
        if abs(de) < conv_tol:
            converged = True
            break
        W = np.dot(U, np.dot(W, U.T))       # dressed -> input occupied basis
        eoW, U = np.linalg.eigh(fock_oo + alpha*W)
    if not converged:
        log.warn('BW-s2 did not converge in %d cycles (dE = %.3g).', max_cycle, de)
    log.timer('BW-s2', *cput0)
    return converged, e_corr, eoW, U

def kernel_dense(oovv, eo, ev, alpha=1., max_cycle=50, conv_tol=1e-8, verbose=None,
                 log=None):
    r''' BW-s2(alpha) from dense oovv[i,j,a,b] = (ia|jb) in canonical orbitals.

        Returns:
            converged, e_corr, eoW, U, t2
            t2 are the converged BW-s2 amplitudes rotated back to the input basis.
    '''
    if log is None: log = logger.new_logger(lib.StreamObject, verbose)
    nocc = oovv.shape[0]
    fock_oo = np.diag(eo)
    eoW = np.asarray(eo, dtype=np.float64)
    U = np.eye(nocc)
    e_corr = None
    converged = False
    for cycle in range(max_cycle):
        oovvW = einsum('ijab,iI,jJ->IJab', oovv, U, U)
        e_new, W, t2 = energy_and_W_dense(oovvW, eoW, ev)
        oovvW = None
        de = np.inf if e_corr is None else e_new - e_corr
        e_corr = e_new
        log.debug('BW-s2 cycle = %d  E_corr = %.15g  dE = %.3g', cycle+1, e_corr, de)
        if abs(de) < conv_tol:
            converged = True
            break
        W = np.dot(U, np.dot(W, U.T))
        eoW, U = np.linalg.eigh(fock_oo + alpha*W)
    if not converged:
        log.warn('BW-s2 did not converge in %d cycles (dE = %.3g).', max_cycle, de)
    t2 = einsum('IJab,iI,jJ->ijab', t2, U, U)
    return converged, e_corr, eoW, U, t2


class _DressedERIS:
    r''' In-core LNO DF ERI object (see `make_lno_rdm1`) for J[i,a,P] in the dressed
        occupied basis, so the MP2 rdm1 code yields BW-s2 density matrices.
    '''
    def __init__(self, ovL, max_memory, verbose=None, stdout=None):
        self.ovL = np.asarray(ovL, dtype=np.float64, order='C')
        self.nocc, self.nvir, self.naux = self.ovL.shape
        self.max_memory = max_memory
        self.verbose = logger.WARN if verbose is None else verbose
        self.stdout = sys.stdout if stdout is None else stdout
        self.dtype = np.float64
        self.dsize = 8

    def get_occ_blk(self, i0,i1):
        return self.ovL[i0:i1]

    def get_vir_blk(self, a0,a1):
        return np.asarray(self.ovL[:,a0:a1], order='C')

    def xform_occ(self, u):
        return einsum('iax,iI->Iax', self.ovL, u)

    def xform_vir(self, u):
        return einsum('iax,aA->iAx', self.ovL, u)
