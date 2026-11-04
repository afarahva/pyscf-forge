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
kappa-regularized MP2 (kappa-MP2) with k-point sampling, closed shell

The k-point MP2 of `pyscf.pbc.mp.kmp2` with every amplitude damped by

    g_ijab = 1 - exp(-kappa * (e_a + e_b - e_i - e_j))

so that the amplitudes carry one power of g and the energy (per unit cell) two, as in
the molecular `pyscf.regpt2.kappamp2`. kappa -> infinity recovers KMP2.

Ref:
    Chen, Zhang, Dinh, Rettig, Lee, arXiv:2508.15744
'''

import numpy as np

from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc.mp import kmp2
from pyscf.regpt2.kappamp2 import kappa_factor, KAPPA
from pyscf import __config__

einsum = lib.einsum

WITH_T2 = getattr(__config__, 'pbc_regpt2_kkappamp2_with_t2', True)


def kernel(mp, mo_energy, mo_coeff, verbose=logger.NOTE, with_t2=WITH_T2):
    r''' k-point kappa-MP2 energy per cell and amplitudes t2[ki,kj,ka]; the loop of
    `pyscf.pbc.mp.kmp2.kernel` (padded `mo_energy` and `mo_coeff`).
    '''
    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.new_logger(mp, verbose)
    nocc, nkpts = mp.nocc, mp.nkpts
    nvir = mp.nmo - nocc
    kconserv = mp.khelper.kconserv

    # e_i - e_a of the (ki,ka) pairs; LARGE_DENOM for the padding of frozen orbitals
    nonzero_opadding, nonzero_vpadding = kmp2.padding_k_idx(mp, kind="split")
    eia = np.full((nkpts,nkpts,nocc,nvir), kmp2.LARGE_DENOM)
    for ki in range(nkpts):
        for ka in range(nkpts):
            idx = np.ix_(nonzero_opadding[ki], nonzero_vpadding[ka])
            eia[ki,ka][idx] = (mo_energy[ki][:nocc,None] - mo_energy[ka][nocc:])[idx]

    if mp.with_df_ints:
        Lov = kmp2._init_mp_df_eris(mp)
        def get_oovv(ki, ka, kj, kb):
            return einsum('Lia,Ljb->ijab', Lov[ki,ka], Lov[kj,kb]) / nkpts
    else:
        fao2mo = mp._scf.with_df.ao2mo
        orbo, orbv = [c[:,:nocc] for c in mo_coeff], [c[:,nocc:] for c in mo_coeff]
        def get_oovv(ki, ka, kj, kb):
            ovov = fao2mo((orbo[ki], orbv[ka], orbo[kj], orbv[kb]),
                          (mp.kpts[ki], mp.kpts[ka], mp.kpts[kj], mp.kpts[kb]), compact=False)
            return ovov.reshape(nocc,nvir,nocc,nvir).transpose(0,2,1,3) / nkpts

    t2 = np.zeros((nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir), dtype=complex) if with_t2 else None
    emp2_ss = emp2_os = 0.
    for ki in range(nkpts):
        for kj in range(nkpts):
            oovv = [get_oovv(ki, ka, kj, kconserv[ki,ka,kj]) for ka in range(nkpts)]
            for ka in range(nkpts):
                kb = kconserv[ki,ka,kj]
                denom = lib.direct_sum('ia,jb->ijab', eia[ki,ka], eia[kj,kb])
                g = kappa_factor(np.minimum(denom, 0), mp.kappa)    # 0 on the padding
                t2_ijab = np.conj(oovv[ka] / denom) * g
                if with_t2:
                    t2[ki,kj,ka] = t2_ijab
                gt2 = g * t2_ijab
                edi = einsum('ijab,ijab', gt2, oovv[ka]).real * 2
                exi = -einsum('ijab,ijba', gt2, oovv[kb]).real
                emp2_ss += edi*0.5 + exi
                emp2_os += edi*0.5
            oovv = None

    log.timer('KKappaMP2', *cput0)
    emp2_ss /= nkpts
    emp2_os /= nkpts
    emp2 = lib.tag_array(emp2_ss+emp2_os, e_corr_ss=emp2_ss, e_corr_os=emp2_os)
    return emp2, t2


class KKappaMP2(kmp2.KMP2):
    r''' kappa-MP2 with k-point sampling for a KRHF reference (canonical orbitals).

    Attributes:
        kappa : float
            Regularization strength. Default is 1.1.

    Saved results:
        e_corr : correlation energy per unit cell
        t2 : amplitudes t2[ki,kj,ka] (if with_t2)
    '''

    kappa = KAPPA

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None, kappa=None):
        if mf.istype('KUHF'):
            raise NotImplementedError('k-point kappa-MP2 is implemented for KRHF only.')
        kmp2.KMP2.__init__(self, mf, frozen, mo_coeff, mo_occ)
        if kappa is not None:
            self.kappa = kappa

    def dump_flags(self):
        kmp2.KMP2.dump_flags(self)
        logger.info(self, 'kappa = %s', self.kappa)
        return self

    def kernel(self, mo_energy=None, mo_coeff=None, with_t2=WITH_T2):
        if mo_energy is None:
            mo_energy = self.mo_energy
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        self.dump_flags()
        self.e_hf = self.get_e_hf(mo_coeff=mo_coeff)
        mo_coeff, mo_energy = kmp2._add_padding(self, mo_coeff, mo_energy)
        self.e_corr, self.t2 = self.init_amps(mo_energy, mo_coeff, with_t2)
        self.e_corr_ss = getattr(self.e_corr, 'e_corr_ss', 0)
        self.e_corr_os = getattr(self.e_corr, 'e_corr_os', 0)
        self.e_corr = float(self.e_corr)
        self._finalize()
        return self.e_corr, self.t2

    def init_amps(self, mo_energy, mo_coeff, with_t2=WITH_T2):
        return kernel(self, mo_energy, mo_coeff, self.verbose, with_t2)

    def make_rdm2(self, *args, **kwargs):
        raise NotImplementedError('k-point kappa-MP2 rdm2')
