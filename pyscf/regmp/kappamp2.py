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
kappa-regularized MP2 (kappa-MP2) for canonical RHF and UHF references

The MP2 integrals are damped by the regularizer

    g_ijab = 1 - exp(-kappa * Delta_ijab),    Delta_ijab = e_a + e_b - e_i - e_j > 0

so that the amplitudes carry one power of g and the energy two:

    t_ijab = g_ijab (ia|jb) / (e_i + e_j - e_a - e_b)
    E_corr = sum_ijab g_ijab t_ijab [ 2 (ia|jb) - (ib|ja) ]          (closed shell)

kappa -> infinity recovers MP2. The one-particle density matrix is the (unrelaxed) MP2
one of the regularized amplitudes.

Ref:
    Chen, Zhang, Dinh, Rettig, Lee, arXiv:2508.15744
'''

import numpy as np

from pyscf import lib
from pyscf.lib import logger
from pyscf.mp import mp2, ump2, dfmp2, dfump2
from pyscf import __config__

WITH_T2 = getattr(__config__, 'regmp_kappamp2_with_t2', True)
KAPPA = getattr(__config__, 'regmp_kappamp2_kappa', 1.1)


def kappa_factor(denom, kappa):
    r''' Regularizer g = 1 - exp(kappa * denom) for MP2 denominators
    denom = e_i + e_j - e_a - e_b (< 0).
    '''
    return 1. - np.exp(kappa * denom)

def _blksize(mp, nocc, nvir, with_t2):
    mem_avail = mp.max_memory - lib.current_memory()[0]
    if with_t2:
        mem_avail -= nocc**2*nvir**2 * 8/1e6
    return min(nocc, max(1, int(mem_avail*0.5e6/8 / (4*nocc*nvir**2))))

def kernel(mp, mo_energy, get_ovov, with_t2=WITH_T2, verbose=None):
    r''' Restricted kappa-MP2 energy and amplitudes.

    Args:
        mo_energy : (occupied, virtual) orbital energies.
        get_ovov : function (i0, i1) -> ovov[i0:i1,j,a,b] = (ia|jb).
    '''
    log = logger.new_logger(mp, verbose)
    eo, ev = mo_energy
    nocc, nvir = eo.size, ev.size
    eia = eo[:,None] - ev

    t2 = np.empty((nocc,nocc,nvir,nvir)) if with_t2 else None
    blksize = _blksize(mp, nocc, nvir, with_t2)
    log.debug1('occ blksize for %s: %d/%d', mp.__class__.__name__, blksize, nocc)

    emp2_ss = emp2_os = 0
    for i0, i1 in lib.prange(0, nocc, blksize):
        ovov = get_ovov(i0, i1)
        denom = lib.direct_sum('ia+jb->ijab', eia[i0:i1], eia)
        g = kappa_factor(denom, mp.kappa)
        t2i = ovov.conj() * g / denom
        denom = None
        gt2 = g * t2i
        g = None
        ed = np.einsum('ijab,ijab', gt2, ovov) * 2
        ex = -np.einsum('ijab,ijba', gt2, ovov)
        gt2 = ovov = None
        emp2_ss += ed*.5 + ex
        emp2_os += ed*.5
        if with_t2:
            t2[i0:i1] = t2i
        t2i = None

    emp2_ss = emp2_ss.real
    emp2_os = emp2_os.real
    emp2 = lib.tag_array(emp2_ss+emp2_os, e_corr_ss=emp2_ss, e_corr_os=emp2_os)
    return emp2, t2

def kernel_uhf(mp, mo_energy, get_ovov, with_t2=WITH_T2, verbose=None):
    r''' Unrestricted kappa-MP2 energy and amplitudes (t2aa, t2ab, t2bb).

    Args:
        mo_energy : ((occ_a, vir_a), (occ_b, vir_b)) orbital energies.
        get_ovov : function (sa, sb, i0, i1) -> ovov[i0:i1,j,a,b] = (i_sa a_sa|j_sb b_sb).
    '''
    log = logger.new_logger(mp, verbose)
    eia = [eo[:,None] - ev for eo, ev in mo_energy]
    nocc = [x.shape[0] for x in eia]
    nvir = [x.shape[1] for x in eia]

    if with_t2:
        t2 = (np.empty((nocc[0],nocc[0],nvir[0],nvir[0])),
              np.empty((nocc[0],nocc[1],nvir[0],nvir[1])),
              np.empty((nocc[1],nocc[1],nvir[1],nvir[1])))
    else:
        t2 = None

    emp2_ss = emp2_os = 0
    for s, (sa, sb) in enumerate([(0,0), (0,1), (1,1)]):
        blksize = _blksize(mp, max(nocc), max(nvir), with_t2)
        log.debug1('occ blksize for %s (%d,%d): %d/%d',
                   mp.__class__.__name__, sa, sb, blksize, nocc[sa])
        for i0, i1 in lib.prange(0, nocc[sa], blksize):
            ovov = get_ovov(sa, sb, i0, i1)
            denom = lib.direct_sum('ia+jb->ijab', eia[sa][i0:i1], eia[sb])
            g = kappa_factor(denom, mp.kappa)
            t2i = ovov.conj() * g / denom
            denom = None
            gt2 = g * t2i
            g = None
            if sa == sb:
                emp2_ss += np.einsum('ijab,ijab', gt2, ovov) * .5
                emp2_ss -= np.einsum('ijab,ijba', gt2, ovov) * .5
                if with_t2:
                    t2[s][i0:i1] = t2i - t2i.transpose(0,1,3,2)
            else:
                emp2_os += np.einsum('ijab,ijab', gt2, ovov)
                if with_t2:
                    t2[s][i0:i1] = t2i
            gt2 = ovov = t2i = None

    emp2_ss = emp2_ss.real
    emp2_os = emp2_os.real
    emp2 = lib.tag_array(emp2_ss+emp2_os, e_corr_ss=emp2_ss, e_corr_os=emp2_os)
    return emp2, t2


class _KappaMP2:
    r''' Methods shared by the kappa-MP2 classes. Only canonical orbitals are supported. '''

    kappa = KAPPA

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None, kappa=None):
        super().__init__(mf, frozen, mo_coeff, mo_occ)
        if kappa is not None:
            self.kappa = kappa

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        logger.info(self, 'kappa = %s', self.kappa)
        return self

    def make_rdm1(self, t2=None, ao_repr=False, with_frozen=True):
        r''' Unrelaxed MP2 rdm1 of the kappa-regularized amplitudes. '''
        if t2 is None:
            t2 = self.t2
        if t2 is None:
            raise RuntimeError('kappa-MP2 rdm1 needs the amplitudes; run kernel with with_t2=True.')
        return super().make_rdm1(t2=t2, ao_repr=ao_repr, with_frozen=with_frozen)

    def make_rdm2(self, *args, **kwargs):
        raise NotImplementedError('kappa-MP2 rdm2')

    # For non-canonical MP2
    def update_amps(self, t2, eris):
        raise NotImplementedError('kappa-MP2 is implemented for canonical orbitals only.')

    def nuc_grad_method(self):
        raise NotImplementedError

    Gradients = NotImplemented


class KappaRMP2(_KappaMP2, mp2.RMP2):
    def init_amps(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2):
        if eris is None:
            eris = self.ao2mo(mo_coeff)
        if mo_energy is None:
            mo_energy = eris.mo_energy
        nocc = self.nocc
        nvir = self.nmo - nocc
        def get_ovov(i0, i1):
            ovov = np.asarray(eris.ovov[i0*nvir:i1*nvir])
            return ovov.reshape(i1-i0,nvir,nocc,nvir).transpose(0,2,1,3)
        return kernel(self, (mo_energy[:nocc], mo_energy[nocc:]), get_ovov, with_t2)

class KappaUMP2(_KappaMP2, ump2.UMP2):
    def init_amps(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2):
        if eris is None:
            eris = self.ao2mo(mo_coeff)
        if mo_energy is None:
            mo_energy = eris.mo_energy
        nocc = self.get_nocc()
        nvir = [nmo - no for nmo, no in zip(self.get_nmo(), nocc)]
        ovov = {(0,0): eris.ovov, (0,1): eris.ovOV, (1,1): eris.OVOV}
        def get_ovov(sa, sb, i0, i1):
            blk = np.asarray(ovov[sa,sb][i0*nvir[sa]:i1*nvir[sa]])
            return blk.reshape(i1-i0,nvir[sa],nocc[sb],nvir[sb]).transpose(0,2,1,3)
        moe = [(e[:no], e[no:]) for e, no in zip(mo_energy, nocc)]
        return kernel_uhf(self, moe, get_ovov, with_t2)

class KappaDFRMP2(_KappaMP2, dfmp2.DFRMP2):
    def init_amps(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2):
        if eris is None:
            eris = self.ao2mo(mo_coeff)
        ovL = eris.get_occ_blk(0, eris.nocc)
        def get_ovov(i0, i1):
            return lib.einsum('iax,jbx->ijab', ovL[i0:i1], ovL)
        return kernel(self, self.split_mo_energy()[1:3], get_ovov, with_t2)

class KappaDFUMP2(_KappaMP2, dfump2.DFUMP2):
    def init_amps(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2):
        if eris is None:
            eris = self.ao2mo(mo_coeff)
        ovL = [eris.get_occ_blk(s, 0, eris.nocc[s]) for s in (0,1)]
        def get_ovov(sa, sb, i0, i1):
            return lib.einsum('iax,jbx->ijab', ovL[sa][i0:i1], ovL[sb])
        moe = [x[1:3] for x in self.split_mo_energy()]
        return kernel_uhf(self, moe, get_ovov, with_t2)


def KappaMP2(mf, frozen=None, mo_coeff=None, mo_occ=None, kappa=None):
    r''' kappa-MP2 for an RHF or UHF reference (ROHF is converted to UHF), with density
    fitting if the SCF uses it.
    '''
    mf = mf.remove_soscf()
    if mf.istype('UHF') or mf.istype('ROHF'):
        if not mf.istype('UHF'):
            mf = mf.to_uhf()
        cls = KappaDFUMP2 if getattr(mf, 'with_df', None) else KappaUMP2
    elif mf.istype('RHF'):
        cls = KappaDFRMP2 if getattr(mf, 'with_df', None) else KappaRMP2
    else:
        raise NotImplementedError('kappa-MP2 for %s' % mf.__class__)
    return cls(mf, frozen, mo_coeff, mo_occ, kappa=kappa)
