#!/usr/bin/env python
# Copyright 2014-2025 The PySCF Developers. All Rights Reserved.
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
# Authors: Chenghan Li
#          Xing Zhang
#          Ardavan Farahvash
#

''' Make unrestricted MP2 rdm1 for ULNO (one localized hole, '1h')

    Args:
        eris :
            Unrestricted LNO ERI object (see `ulno.py`). The DF version provides
            get_occ_blk(i0,i1,spin) and xform_occ(u,spin); the non-DF version provides
            get_ivov, get_ivOV, get_IVov and get_IVOV.
        moeocc, moevir : [alpha, beta]
            MO energies of the occupied and virtual MOs used to obtain `eris`.
        uocc : [alpha, beta]
            Overlap between the canonical and localized occupied orbitals.
'''

import numpy as np

from pyscf import lib
from pyscf.lib import logger
from pyscf.lno.make_lno_rdm1 import _mp2_rdm1_occblksize, subspace_eigh, _regularize, DEBUG_BLKSIZE


def make_lo_rdm1_1h(eris, moeocc, moevir, uocc):
    r'''
    Create unrestricted MP2 density matrix with one localized hole

    Math:
        dmoo_a(i,j) = sum_{Kab} [ T(Kaib) * T(Kajb)]
                      + 2* sum_{K'ab'} [ t(K'aib') * t(K'ajb') ]

        dmoo_b(i,j) - reverse alpha/beta spin indices from dmoo_a

        dmvv_a(a,b) = sum_{Kic} [ T(Kaic) * T(Kbjc) ]
                      + sum_{K'ic'} [ t(K'c'ib) * t(K'c'ia) ]
                      + sum_{Ki'c'} [ t(Kai'c) * t(Kbi'c') ]

        dmvv_b(a,b) - reverse alpha/beta spin indices from dmvv_a

    Notation:
    i,j - canonical occupied orbitals
    a,b,c - canonical virtual orbitals
    K - local occupied orbitals
    indices with ' are beta spin, others are alpha.

    are canonical alpha; I,J,K are local alpha;
    A,B,C are canonical beta;

    t(ijab) = (ia|jb) / (e_i+e_j-e_a-e_b)
    T(ijab) = t(ijab) - t(ijba)

    #
    Args:
        eris : ERI object
            Provides `ovL` in the canonical MOs.
        moeocc : [moeocc,_a moeocc_b]
            Occ MO energies
        moeocc : [moevir,_a moevir_b]
            Vir MO energies
        uocc : [u_a, u_b]
            Overlap between the canonical and localized occupied orbitals.
            u_a(i,I) = <i|I> (alpha), u_b(i',I') = <i'|I'> (beta)
    '''
    log = logger.new_logger(eris)

    # Unpack spins
    moeocc_a, moeocc_b = moeocc
    moevir_a, moevir_b = moevir
    u_a, u_b = uocc

    nocca, nvira = eris.nocc[0], eris.nvir[0]
    noccb, nvirb = eris.nocc[1], eris.nvir[1]
    nOcca = u_a.shape[1]
    nOccb = u_b.shape[1]
    dtype = eris.dtype
    dsize = eris.dsize

    # Determine block sizes
    mem_avail = eris.max_memory - lib.current_memory()[0]
    M = mem_avail * 0.1 * 1e6 / dsize

    # Alpha block size (based on alpha canonical occ)
    occblksize_a, mem_peak_a = _mp2_rdm1_occblksize(
        nocca, nvira, 0, 0, 3, M/2, dsize)
    if DEBUG_BLKSIZE:
        occblksize_a = max(1, nocca // 2)

    # Beta block size (based on beta canonical occ)
    occblksize_b, mem_peak_b = _mp2_rdm1_occblksize(
        noccb, nvirb, 0, 0, 3, M/2, dsize)
    if DEBUG_BLKSIZE:
        occblksize_b = max(1, noccb // 2)

    #
    log.debug1('make_lo_rdm1_1h (alpha): nocc=%d nvir=%d nOcc=%d blksize=%d peak_mem=%.2f MB',
               nocca, nvira, nOcca, occblksize_a, mem_peak_a)

    log.debug1('make_lo_rdm1_1h (beta):  nocc=%d nvir=%d nOcc=%dblksize=%d peak_mem=%.2f MB',
               noccb, nvirb, nOccb, occblksize_b, mem_peak_b)

    # Localized MO energies
    moeI_a, u_a = subspace_eigh(np.diag(moeocc_a), u_a)
    moeI_b, u_b = subspace_eigh(np.diag(moeocc_b), u_b)

    # Energy denominators
    eov_a = moeocc_a[:, None] - moevir_a
    eIv_a = moeI_a[:, None] - moevir_a
    eov_b = moeocc_b[:, None] - moevir_b
    eIv_b = moeI_b[:, None] - moevir_b

    # Initialize RDMs
    dmoo_a = np.zeros((nocca, nocca), dtype=dtype)
    dmoo_b = np.zeros((noccb, noccb), dtype=dtype)
    dmvv_a = np.zeros((nvira, nvira), dtype=dtype)
    dmvv_b = np.zeros((nvirb, nvirb), dtype=dtype)

    # -- construct t2aa and contract
    for Kbatch, (K0, K1) in enumerate(lib.prange(0, nOcca, occblksize_a)):
        # fragment-occ DF energies
        eKv_a = eIv_a[K0:K1]

        for ibatch, (i0, i1) in enumerate(lib.prange(0, nocca, occblksize_a)):
            # full-occ DF energies
            eiv_a = eov_a[i0:i1]

            # form t2_ovov-block
            denom_aa = lib.direct_sum('Ka+ib->Kaib', eKv_a, eiv_a)
            t2aa_i = eris.get_ivov(u_a, K0, K1, i0, i1) / denom_aa
            t2aa_i = _regularize(eris, t2aa_i, denom_aa)
            t2aa_i = t2aa_i - t2aa_i.transpose(0, 3, 2, 1)
            denom_aa = None
            eiv_a = None

            for jbatch, (j0, j1) in enumerate(lib.prange(0, nocca, occblksize_a)):
                if ibatch == jbatch:
                    t2aa_j = t2aa_i
                else:
                    # full-occ DF energies
                    ejv_a = eov_a[j0:j1]

                    # form t2_ovov-block
                    denom_aa = lib.direct_sum('Ka+jb->Kajb', eKv_a, ejv_a)
                    t2aa_j = eris.get_ivov(u_a, K0, K1, j0, j1) / denom_aa
                    t2aa_j = _regularize(eris, t2aa_j, denom_aa)
                    t2aa_j = t2aa_j - t2aa_j.transpose(0, 3, 2, 1)
                    denom_aa = None
                    ejv_a = None

                # contract block to make occupied MP2-DM
                dmoo_a[i0:i1,
                       j0:j1] += lib.einsum('Kaib,Kajb->ij', t2aa_i, t2aa_j.conj())
                t2aa_j = None

            # contract block to make virtual MP2-DM
            dmvv_a += lib.einsum('Kaic,Kbic->ba', np.conj(t2aa_i), t2aa_i)
            t2aa_i = None

        eKv_a = None

    # -- construct t2bb and contract
    for Kbatch, (K0, K1) in enumerate(lib.prange(0, nOccb, occblksize_b)):
        # fragment-occ DF energies
        eKv_b = eIv_b[K0:K1]

        for ibatch, (i0, i1) in enumerate(lib.prange(0, noccb, occblksize_b)):
            # full-occ DF integrals/energies
            eiv_b = eov_b[i0:i1]

            # form t2_ovov-block
            denom_bb = lib.direct_sum('Ka+ib->Kaib', eKv_b, eiv_b)
            t2bb_i = eris.get_IVOV(u_b, K0, K1, i0, i1) / denom_bb
            t2bb_i = _regularize(eris, t2bb_i, denom_bb)
            t2bb_i = t2bb_i - t2bb_i.transpose(0, 3, 2, 1)
            denom_bb = None
            eiv_b = None

            for jbatch, (j0, j1) in enumerate(lib.prange(0, noccb, occblksize_b)):
                if ibatch == jbatch:
                    t2bb_j = t2bb_i
                else:
                    # full-occ DF integrals/energies
                    ejv_b = eov_b[j0:j1]

                    # form t2_ovov-block
                    denom_bb = lib.direct_sum('Ka+jb->Kajb', eKv_b, ejv_b)
                    t2bb_j = eris.get_IVOV(u_b, K0, K1, j0, j1) / denom_bb
                    t2bb_j = _regularize(eris, t2bb_j, denom_bb)
                    t2bb_j = t2bb_j - t2bb_j.transpose(0, 3, 2, 1)
                    denom_bb = None
                    ejv_b = None

                # contract block to make occupied MP2/DM
                dmoo_b[i0:i1,
                       j0:j1] += lib.einsum('Kaib,Kajb->ij', t2bb_i, t2bb_j.conj())
                t2bb_j = None

            # contract block to make virtual MP2/DM
            dmvv_b += lib.einsum('Kaic,Kbic->ba', t2bb_i.conj(), t2bb_i)
            t2bb_i = None
        eKv_b = None

    # -- construct t2ba and contract
    for Kbatch, (K0, K1) in enumerate(lib.prange(0, nOccb, occblksize_b)):
        # fragment-occ DF energies
        eKv_b = eIv_b[K0:K1]

        for ibatch, (i0, i1) in enumerate(lib.prange(0, nocca, occblksize_a)):
            # full-occ DF energies
            eiv_a = eov_a[i0:i1]

            # form t2_ovov-block
            denom_ba = lib.direct_sum('Ka+ib->Kaib', eKv_b, eiv_a)
            t2ba_i = eris.get_IVov(u_b, K0, K1, i0, i1) / denom_ba
            t2ba_i = _regularize(eris, t2ba_i, denom_ba)
            denom_ba = None
            eiv_a = None

            for jbatch, (j0, j1) in enumerate(lib.prange(0, nocca, occblksize_a)):
                if ibatch == jbatch:
                    t2ba_j = t2ba_i
                else:
                    # full-occ DF energies
                    ejv_a = eov_a[j0:j1]

                    # form t2_ovov-block
                    denom_ba = lib.direct_sum('Ka+jb->Kajb', eKv_b, ejv_a)
                    t2ba_j = eris.get_IVov(u_b, K0, K1, j0, j1) / denom_ba
                    t2ba_j = _regularize(eris, t2ba_j, denom_ba)
                    denom_ba = None
                    ejv_a = None

                # contract block to make occupied MP2/DM
                dmoo_a[i0:i1, j0:j1] += 2 * \
                    lib.einsum('Kaib,Kajb->ij', t2ba_i, t2ba_j.conj())
                t2ba_j = None

            # contract block to make virtual MP2/DM
            dmvv_a += lib.einsum('Kcia,Kcib->ba', t2ba_i.conj(), t2ba_i)
            dmvv_b += lib.einsum('Kaic,Kbic->ba', t2ba_i.conj(), t2ba_i)
            t2ba_i = None
        eKv_b = None

    # -- construct t2ab and contract
    for Kbatch, (K0, K1) in enumerate(lib.prange(0, nOcca, occblksize_a)):
        # fragment-occ DF energies
        eKv_a = eIv_a[K0:K1]

        for ibatch, (i0, i1) in enumerate(lib.prange(0, noccb, occblksize_b)):
            # full-occ DF energies
            eiv_b = eov_b[i0:i1]

            # form t2_ovov-block
            denom_ab = lib.direct_sum('Ka+ib->Kaib', eKv_a, eiv_b)
            t2ab_i = eris.get_ivOV(u_a, K0, K1, i0, i1) / denom_ab
            t2ab_i = _regularize(eris, t2ab_i, denom_ab)
            denom_ba = None
            eiv_b = None

            for jbatch, (j0, j1) in enumerate(lib.prange(0, noccb, occblksize_b)):
                if ibatch == jbatch:
                    t2ab_j = t2ab_i
                else:
                    # full-occ DF energies
                    ejv_b = eov_b[j0:j1]

                    # form t2_ovov-block
                    denom_ab = lib.direct_sum('Ka+jb->Kajb', eKv_a, ejv_b)
                    t2ab_j = eris.get_ivOV(u_a, K0, K1, j0, j1) / denom_ab
                    t2ab_j = _regularize(eris, t2ab_j, denom_ab)
                    denom_ba = None
                    ejv_b = None

                # contract block to make occupied MP2/DM
                dmoo_b[i0:i1, j0:j1] += 2 * \
                    lib.einsum('Kaib,Kajb->ij', t2ab_i, t2ab_j.conj())
                t2ab_j = None

            dmvv_a += lib.einsum('Kaic,Kbic->ba', t2ab_i.conj(), t2ab_i)
            dmvv_b += lib.einsum('Kcia,Kcib->ba', t2ab_i.conj(), t2ab_i)
            t2ab_i = None
        eKv_a = None

    return [dmoo_a, dmoo_b], [dmvv_a, dmvv_b]


def make_lo_rdm1_1h_df(eris, moeocc, moevir, uocc):
    r'''
    Create unrestricted MP2 density matrix with one localized hole
    Density-fitted version

    Math:
        dmoo_a(i,j) = sum_{Kab} [ T(Kaib) * T(Kajb)]
                      + 2* sum_{K'ab'} [ t(K'aib') * t(K'ajb') ]

        dmoo_b(i,j) - reverse alpha/beta spin indices from dmoo_a

        dmvv_a(a,b) = sum_{Kic} [ T(Kaic) * T(Kbjc) ]
                      + sum_{K'ic'} [ t(K'c'ib) * t(K'c'ia) ]
                      + sum_{Ki'c'} [ t(Kai'c) * t(Kbi'c') ]


        dmvv_b(a,b) - reverse alpha/beta spin indices from dmvv_a

    Notation:
    i,j - canonical occupied orbitals
    a,b,c - canonical virtual orbitals
    K - local occupied orbitals
    indices with ' are beta spin, others are alpha.

    are canonical alpha; I,J,K are local alpha;
    A,B,C are canonical beta;

    t(ijab) = (ia|jb) / (e_i+e_j-e_a-e_b)
    T(ijab) = t(ijab) - t(ijba)

    #
    Args:
        eris : ERI object
            Provides `ovL` in the canonical MOs.
        moeocc : [moeocc,_a moeocc_b]
            Occ MO energies
        moeocc : [moevir,_a moevir_b]
            Vir MO energies
        uocc : [u_a, u_b]
            Overlap between the canonical and localized occupied orbitals.
            u_a(i,I) = <i|I> (alpha), u_b(i',I') = <i'|I'> (beta)
    '''
    log = logger.new_logger(eris)

    # Unpack spins
    moeocc_a, moeocc_b = moeocc
    moevir_a, moevir_b = moevir
    u_a, u_b = uocc

    nocca, nvira, naux = eris.nocc[0], eris.nvir[0], eris.naux
    noccb, nvirb = eris.nocc[1], eris.nvir[1]
    nOcca = u_a.shape[1]
    nOccb = u_b.shape[1]
    dtype = eris.dtype
    dsize = eris.dsize

    # Determine block sizes
    mem_avail = eris.max_memory - lib.current_memory()[0]
    M = mem_avail * 0.7 * 1e6 / dsize

    # Alpha block size (based on alpha canonical occ)
    occblksize_a, mem_peak_a = _mp2_rdm1_occblksize(
        nocca, nvira, naux, 3, 3, M/2, dsize)
    if DEBUG_BLKSIZE:
        occblksize_a = max(1, nocca // 2)

    # Beta block size (based on beta canonical occ)
    occblksize_b, mem_peak_b = _mp2_rdm1_occblksize(
        noccb, nvirb, naux, 3, 3, M/2, dsize)
    if DEBUG_BLKSIZE:
        occblksize_b = max(1, noccb // 2)

    #
    log.debug1('make_lo_rdm1_1h (alpha): nocc=%d nvir=%d nOcc=%d naux=%d blksize=%d peak_mem=%.2f MB',
               nocca, nvira, nOcca, naux, occblksize_a, mem_peak_a)

    log.debug1('make_lo_rdm1_1h (beta):  nocc=%d nvir=%d nOcc=%d naux=%d blksize=%d peak_mem=%.2f MB',
               noccb, nvirb, nOccb, naux, occblksize_b, mem_peak_b)

    # Localized MO energies
    moeI_a, u_a = subspace_eigh(np.diag(moeocc_a), u_a)
    moeI_b, u_b = subspace_eigh(np.diag(moeocc_b), u_b)

    # Energy denominators
    eov_a = moeocc_a[:, None] - moevir_a
    eIv_a = moeI_a[:, None] - moevir_a
    eov_b = moeocc_b[:, None] - moevir_b
    eIv_b = moeI_b[:, None] - moevir_b

    # Initialize RDMs
    dmoo_a = np.zeros((nocca, nocca), dtype=dtype)
    dmoo_b = np.zeros((noccb, noccb), dtype=dtype)
    dmvv_a = np.zeros((nvira, nvira), dtype=dtype)
    dmvv_b = np.zeros((nvirb, nvirb), dtype=dtype)

    # -- construct t2aa and contract
    for Kbatch, (K0, K1) in enumerate(lib.prange(0, nOcca, occblksize_a)):

        # fragment-occ DF integrals/energies
        KvL_a = eris.xform_occ(u_a[:, K0:K1], spin='a')
        eKv_a = eIv_a[K0:K1]

        for ibatch, (i0, i1) in enumerate(lib.prange(0, nocca, occblksize_a)):

            # full-occ DF integrals/energies
            ivL_a = eris.get_occ_blk(i0, i1, spin='a')
            eiv_a = eov_a[i0:i1]

            # form t2-block
            denom_aa = lib.direct_sum('Ka+ib->Kaib', eKv_a, eiv_a)
            t2aa_i = lib.einsum('Kax,ibx->Kaib', KvL_a, ivL_a) / denom_aa
            t2aa_i = _regularize(eris, t2aa_i, denom_aa)
            t2aa_i = t2aa_i - t2aa_i.transpose(0, 3, 2, 1)
            denom_aa = None
            ivL_a = None
            eiv_a = None

            for jbatch, (j0, j1) in enumerate(lib.prange(0, nocca, occblksize_a)):
                if jbatch == ibatch:
                    t2aa_j = t2aa_i
                else:
                    # full-occ DF integrals/energies
                    jvL_a = eris.get_occ_blk(j0, j1, spin='a')
                    ejv_a = eov_a[j0:j1]

                    # form t2-block
                    denom_aa = lib.direct_sum('Ka+jb->Kajb', eKv_a, ejv_a)
                    t2aa_j = lib.einsum(
                        'Kax,jbx->Kajb', KvL_a, jvL_a) / denom_aa
                    t2aa_j = _regularize(eris, t2aa_j, denom_aa)
                    t2aa_j = t2aa_j - t2aa_j.transpose(0, 3, 2, 1)
                    denom_aa = None
                    jvL_a = None
                    ejv_a = None

                # contract block to make occupied MP2-DM
                dmoo_a[i0:i1,
                       j0:j1] += lib.einsum('Kaib,Kajb->ij', t2aa_i, t2aa_j.conj())
                t2aa_j = None

            # contract block to make virtual MP2-DM
            dmvv_a += lib.einsum('Kaic,Kbic->ba', np.conj(t2aa_i), t2aa_i)
            t2aa_i = None

        KvL_a = None
        eKv_a = None

    # -- construct t2bb and contract
    for Kbatch, (K0, K1) in enumerate(lib.prange(0, nOccb, occblksize_b)):
        # fragment-occ DF integrals/energies
        KvL_b = eris.xform_occ(u_b[:, K0:K1], spin='b')
        eKv_b = eIv_b[K0:K1]

        for ibatch, (i0, i1) in enumerate(lib.prange(0, noccb, occblksize_b)):
            # full-occ DF integrals/energies
            ivL_b = eris.get_occ_blk(i0, i1, spin='b')
            eiv_b = eov_b[i0:i1]

            # form t2-block
            denom_bb = lib.direct_sum('Ka+ib->Kaib', eKv_b, eiv_b)
            t2bb_i = lib.einsum('Kax,ibx->Kaib', KvL_b, ivL_b) / denom_bb
            t2bb_i = _regularize(eris, t2bb_i, denom_bb)
            t2bb_i = t2bb_i - t2bb_i.transpose(0, 3, 2, 1)
            denom_bb = None
            ivL_b = None
            eiv_b = None

            for jbatch, (j0, j1) in enumerate(lib.prange(0, noccb, occblksize_b)):
                if jbatch == ibatch:
                    t2bb_j = t2bb_i
                else:
                    # full-occ DF integrals/energies
                    jvL_b = eris.get_occ_blk(j0, j1, spin='b')
                    ejv_b = eov_b[j0:j1]

                    # form t2-block
                    denom_bb = lib.direct_sum('Ka+jb->Kajb', eKv_b, ejv_b)
                    t2bb_j = lib.einsum(
                        'Kax,jbx->Kajb', KvL_b, jvL_b) / denom_bb
                    t2bb_j = _regularize(eris, t2bb_j, denom_bb)
                    t2bb_j = t2bb_j - t2bb_j.transpose(0, 3, 2, 1)
                    denom_bb = None
                    jvL_b = None
                    ejv_b = None

                # contract block to make occupied MP2-DM
                dmoo_b[i0:i1,
                       j0:j1] += lib.einsum('Kaib,Kajb->ij', t2bb_i, t2bb_j.conj())
                t2bb_j = None

            # contract block to make virtual MP2-DM
            dmvv_b += lib.einsum('Kaic,Kbic->ba', t2bb_i.conj(), t2bb_i)
            t2bb_i = None

        KvL_b = None
        eKv_b = None

    # -- construct t2ba and contract
    for Kbatch, (K0, K1) in enumerate(lib.prange(0, nOccb, occblksize_b)):
        # fragment-occ DF integrals/energies
        KvL_b = eris.xform_occ(u_b[:, K0:K1], spin='b')
        eKv_b = eIv_b[K0:K1]

        for ibatch, (i0, i1) in enumerate(lib.prange(0, nocca, occblksize_a)):
            # full-occ DF integrals/energies
            ivL_a = eris.get_occ_blk(i0, i1, spin='a')
            eiv_a = eov_a[i0:i1]

            # form t2-block
            denom_ba = lib.direct_sum('Ka+ib->Kaib', eKv_b, eiv_a)
            t2ba_i = lib.einsum('Kax,ibx->Kaib', KvL_b, ivL_a) / denom_ba
            t2ba_i = _regularize(eris, t2ba_i, denom_ba)
            ivL_a = None
            eiv_a = None
            denom_ba = None

            for jbatch, (j0, j1) in enumerate(lib.prange(0, nocca, occblksize_a)):
                if jbatch == ibatch:
                    t2ba_j = t2ba_i

                else:
                    # full-occ DF integrals/energies
                    jvL_a = eris.get_occ_blk(j0, j1, spin='a')
                    ejv_a = eov_a[j0:j1]

                    # form t2-block
                    denom_ba = lib.direct_sum('Ka+jb->Kajb', eKv_b, ejv_a)
                    t2ba_j = lib.einsum(
                        'Kax,jbx->Kajb', KvL_b, jvL_a) / denom_ba
                    t2ba_j = _regularize(eris, t2ba_j, denom_ba)
                    jvL_a = None
                    ejv_a = None
                    denom_ba = None

                # contract block to make occupied MP2-DM
                dmoo_a[i0:i1, j0:j1] += 2 * \
                    lib.einsum('Kaib,Kajb->ij', t2ba_i, t2ba_j.conj())
                t2ba_j = None

            # contract block to make virtual MP2-DM
            dmvv_a += lib.einsum('Kcia,Kcib->ba', t2ba_i.conj(), t2ba_i)
            dmvv_b += lib.einsum('Kaic,Kbic->ba', t2ba_i.conj(), t2ba_i)
            t2ba_i = None

        KvL_b = None
        eKv_b = None

    # -- construct t2ab and contract
    for Kbatch, (K0, K1) in enumerate(lib.prange(0, nOcca, occblksize_a)):
        # fragment-occ DF integrals/energies
        KvL_a = eris.xform_occ(u_a[:, K0:K1], spin='a')
        eKv_a = eIv_a[K0:K1]

        for ibatch, (i0, i1) in enumerate(lib.prange(0, noccb, occblksize_b)):
            # full-occ DF integrals/energies
            ivL_b = eris.get_occ_blk(i0, i1, spin='b')
            eiv_b = eov_b[i0:i1]

            # form t2-block
            denom_ab = lib.direct_sum('Ka+ib->Kaib', eKv_a, eiv_b)
            t2ab_i = lib.einsum('Kax,ibx->Kaib', KvL_a, ivL_b) / denom_ab
            t2ab_i = _regularize(eris, t2ab_i, denom_ab)
            ivL_b = None
            eiv_b = None
            denom_ba = None

            for jbatch, (j0, j1) in enumerate(lib.prange(0, noccb, occblksize_b)):
                if jbatch == ibatch:
                    t2ab_j = t2ab_i
                else:
                    # full-occ DF integrals/energies
                    jvL_b = eris.get_occ_blk(j0, j1, spin='b')
                    ejv_b = eov_b[j0:j1]

                    # form t2-block
                    denom_ab = lib.direct_sum('Ka+jb->Kajb', eKv_a, ejv_b)
                    t2ab_j = lib.einsum(
                        'Kax,jbx->Kajb', KvL_a, jvL_b) / denom_ab
                    t2ab_j = _regularize(eris, t2ab_j, denom_ab)
                    jvL_b = None
                    ejv_b = None
                    denom_ba = None

                # contract block to make occupied MP2-DM
                dmoo_b[i0:i1, j0:j1] += 2 * \
                    lib.einsum('Kaib,Kajb->ij', t2ab_i, t2ab_j.conj())
                t2ba_j = None

            # contract block to make virtual MP2-DM
            dmvv_a += lib.einsum('Kaic,Kbic->ba', t2ab_i.conj(), t2ab_i)
            dmvv_b += lib.einsum('Kcia,Kcib->ba', t2ab_i.conj(), t2ab_i)
            t2ab_i = None

        KvL_a = None
        eKv_a = None

    return [dmoo_a, dmoo_b], [dmvv_a, dmvv_b]
