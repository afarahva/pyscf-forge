''' Make MP2 rdm1 of different flavors

    Args:
        eris :
            Provide access to the MO basis DF integral `ovL` through the following methods:
                get_occ_blk(i0,i1) -> ([i0:i1]v|L)
                get_vir_blk(a0,a1) -> (o[a0:a1]|L)
                xform_occ(u)       -> einsum('iaL,iI->IaL', ovL, u)
                xform_vir(u)       -> einsum('iaL,aA->iAL', ovL, u)
        orbact_data : tuple
            moeocc, moevir, uoccact, uviract = orbact_data
            where
                - `moeocc` and `moevir` are the MO energy for the occupied and virtual MOs
                  used to obtain `ovL`
                - `uoccact` and `uviract` are the overlap matrix between canonical and active
                  orbitals, i.e., uoccact[i,I] = <i|I> and uviract[a,A] = <a|A>.
        dm_type : str
            '1h'/'1p'/'2p' for occ and '1p'/'1h'/'2h' for vir.
'''
import sys
import numpy as np
from functools import reduce

from pyscf import lib
from pyscf.lib import logger
from pyscf import __config__

from pyscf.lno.make_lno_rdm1 import (subspace_eigh, _mp2_rdm1_occblksize, _mp2_rdm1_virblksize,
                                     _make_t2)
from pyscf.pbc.lno.tools import zdotCNtoR

DEBUG_BLKSIZE = getattr(__config__, 'lno_base_make_rdm1_k2s_DEBUG_BLKSIZE', False)


def make_lo_rdm1_occ(eris, moeocc, moevir, uocc, uvir, dm_type):
    isreal = eris.dtype == np.float64
    if dm_type == '1h':
        if isreal:
            dm = make_lo_rdm1_occ_1h_real(eris, moeocc, moevir, uocc)
        else:
            dm = make_lo_rdm1_occ_1h_complex(eris, moeocc, moevir, uocc)
    elif dm_type == '1p':
        if isreal:
            dm = make_lo_rdm1_occ_1p_real(eris, moeocc, moevir, uvir)
        else:
            dm = make_lo_rdm1_occ_1p_complex(eris, moeocc, moevir, uvir)
    elif dm_type == '2p':
        if isreal:
            dm = make_lo_rdm1_occ_2p_real(eris, moeocc, moevir, uvir)
        else:
            dm = make_lo_rdm1_occ_2p_complex(eris, moeocc, moevir, uvir)
    else:
        raise RuntimeError('Requested occ LNO type "%s" is unknown.' % dm_type)
    dm = _check_dm_imag(eris, dm)
    return dm

def make_lo_rdm1_vir(eris, moeocc, moevir, uocc, uvir, dm_type):
    isreal = eris.dtype == np.float64
    if dm_type == '1p':
        if isreal:
            dm = make_lo_rdm1_vir_1p_real(eris, moeocc, moevir, uvir)
        else:
            dm = make_lo_rdm1_vir_1p_complex(eris, moeocc, moevir, uvir)
    elif dm_type == '1h':
        if isreal:
            dm = make_lo_rdm1_vir_1h_real(eris, moeocc, moevir, uocc)
        else:
            dm = make_lo_rdm1_vir_1h_complex(eris, moeocc, moevir, uocc)
    elif dm_type == '2h':
        if isreal:
            dm = make_lo_rdm1_vir_2h_real(eris, moeocc, moevir, uocc)
        else:
            dm = make_lo_rdm1_vir_2h_complex(eris, moeocc, moevir, uocc)
    else:
        raise RuntimeError('Requested vir LNO type "%s" is unknown.' % dm_type)
    dm = _check_dm_imag(eris, dm)
    return dm

''' make lo rdm1 for real orbitals
'''
def make_full_rdm1(eris, moeocc, moevir, with_occ=True, with_vir=True):
    r''' Occ-occ and vir-vir blocks of MP2 density matrix

        Math:
            dm(i,j)
                = 2 * \sum_{kab} t2(ikab).conj() * ( 2*t2(jkab) - t2(jkba) )
            dm(a,b)
                = 2 * \sum_{ijc} t2(ijac) * ( 2*t2(ijbc) - t2(ijcb) ).conj()
    '''
    assert( with_occ or with_vir )

    nocc, nvir, naux = eris.nocc, eris.nvir, eris.Naux_ibz
    REAL = np.float64
    dsize = 8

    # determine occblksize
    mem_avail = eris.max_memory - lib.current_memory()[0]
    M = mem_avail * 0.7 * 1e6/dsize
    occblksize, mem_peak = _mp2_rdm1_occblksize(nocc,nvir,naux, 4, 3, M, dsize)
    if DEBUG_BLKSIZE: occblksize = max(1,nocc//2)
    logger.debug1(eris, 'make_full_rdm1  :  nocc = %d  nvir = %d  naux = %d  occblksize = %d  '
                  'peak mem = %.2f MB', nocc, nvir, naux, occblksize, mem_peak)
    bufsize = occblksize*min(occblksize,nocc)*nvir**2
    buf = np.empty(bufsize, dtype=REAL)

    eov = moeocc[:,None] - moevir

    dmoo = np.zeros((nocc,nocc), dtype=REAL) if with_occ else None
    dmvv = np.zeros((nvir,nvir), dtype=REAL) if with_vir else None
    for ibatch,(i0,i1) in enumerate(lib.prange(0,nocc,occblksize)):
        ivLR, ivLI = eris.get_occ_blk(i0,i1)
        ivLR = ivLR.reshape(-1,naux)
        ivLI = ivLI.reshape(-1,naux)
        eiv = eov[i0:i1]
        for jbatch,(j0,j1) in enumerate(lib.prange(0,nocc,occblksize)):
            if jbatch == ibatch:
                jvLR, jvLI = ivLR, ivLI
                ejv = eiv
            else:
                jvLR, jvLI = eris.get_occ_blk(j0,j1)
                jvLR = jvLR.reshape(-1,naux)
                jvLI = jvLI.reshape(-1,naux)
                ejv = eov[j0:j1]
            denom = lib.direct_sum('ia+jb->iajb', eiv, ejv)
            t2ijvv = np.ndarray((ivLR.shape[0],jvLR.shape[0]), dtype=REAL, buffer=buf)
            zdotCNtoR(ivLR, ivLI, jvLR.T, jvLI.T, cR=t2ijvv)
            t2ijvv = _make_t2(eris, t2ijvv.reshape(*denom.shape), denom)
            jvLR = jvLI = None
            denom = None
            if with_occ:
                dmoo[i0:i1,j0:j1]  = 4*lib.einsum('iakb,jakb->ij', t2ijvv, t2ijvv)
                dmoo[i0:i1,j0:j1] -= 2*lib.einsum('iakb,jbka->ij', t2ijvv, t2ijvv)
            if with_vir:
                dmvv  = 4*lib.einsum('iajc,ibjc->ab', t2ijvv, t2ijvv)
                dmvv -= 2*lib.einsum('iajc,icjb->ab', t2ijvv, t2ijvv)
            t2ijvv = None
        ivLR = ivLI = None
    buf = None

    return dmoo, dmvv

def make_lo_rdm1_occ_1h_real(eris, moeocc, moevir, u):
    r''' Occupied MP2 density matrix with one localized hole

        Math:
            dm(i,j)
                = 2 * \sum_{k'ab} t2(ik'ab) ( 2*t2(jk'ab) - t2(jk'ba) )

        Args:
            eris : ERI object
                Provides `ovL` in the canonical MOs.
            u : np.ndarray
                Overlap between the canonical and localized occupied orbitals, i.e.,
                    u(i,i') = <i|i'>
    '''
    nocc, nvir, naux = eris.nocc, eris.nvir, eris.Naux_ibz
    REAL = np.float64
    dsize = 8
    assert(u.dtype == REAL)
    nOcc = u.shape[1]

    # determine occblksize
    mem_avail = eris.max_memory - lib.current_memory()[0]
    M = mem_avail * 0.7 * 1e6/dsize
    occblksize, mem_peak = _mp2_rdm1_occblksize(nocc,nvir,naux, 4, 3, M, dsize)
    if DEBUG_BLKSIZE: occblksize = max(1,nocc//2)
    logger.debug1(eris, 'make_lo_rdm1_occ_1h :  nocc = %d  nvir = %d  nOcc = %d  naux = %d  '
                  'occblksize = %d  peak mem = %.2f MB',
                  nocc, nvir, nOcc, naux, occblksize, mem_peak)
    bufsize = occblksize*min(occblksize,nOcc)*nvir**2
    buf1 = np.empty(bufsize, dtype=REAL)
    buf2 = np.empty(bufsize, dtype=REAL)

    moeOcc, u = subspace_eigh(np.diag(moeocc), u)
    eov = moeocc[:,None] - moevir
    eOv = moeOcc[:,None] - moevir

    dm = np.zeros((nocc,nocc), dtype=REAL)
    for Kbatch,(K0,K1) in enumerate(lib.prange(0,nOcc,occblksize)):
        KvLR, KvLI = eris.xform_occ(u[:,K0:K1])
        KvLR = KvLR.reshape(-1,naux)
        KvLI = KvLI.reshape(-1,naux)
        eKv = eOv[K0:K1]
        for ibatch,(i0,i1) in enumerate(lib.prange(0,nocc,occblksize)):
            eiv = eov[i0:i1]
            eivKv = lib.direct_sum('ia+Kb->iaKb', eiv, eKv)
            ivLR, ivLI = eris.get_occ_blk(i0,i1)
            ivLR = ivLR.reshape(-1,naux)
            ivLI = ivLI.reshape(-1,naux)
            t2ivKv = np.ndarray((ivLR.shape[0],KvLR.shape[0]), dtype=REAL, buffer=buf1)
            zdotCNtoR(ivLR, ivLI, KvLR.T, KvLI.T, cR=t2ivKv)
            t2ivKv = _make_t2(eris, t2ivKv.reshape(*eivKv.shape), eivKv)
            ivLR = ivLI = None
            eivKv = None
            for jbatch,(j0,j1) in enumerate(lib.prange(0,nocc,occblksize)):
                if jbatch == ibatch:
                    t2jvKv = t2ivKv
                else:
                    ejv = eov[j0:j1]
                    ejvKv = lib.direct_sum('ia+Kb->iaKb', ejv, eKv)
                    jvLR, jvLI = eris.get_occ_blk(j0,j1)
                    jvLR = jvLR.reshape(-1,naux)
                    jvLI = jvLI.reshape(-1,naux)
                    t2jvKv = np.ndarray((jvLR.shape[0],KvLR.shape[0]), dtype=REAL, buffer=buf2)
                    zdotCNtoR(jvLR, jvLI, KvLR.T, KvLI.T, cR=t2jvKv)
                    t2jvKv = _make_t2(eris, t2jvKv.reshape(*ejvKv.shape), ejvKv)
                    jvLR = jvLI = None
                    ejvKv = None

                dm[i0:i1,j0:j1] -= 4 * lib.einsum('iaKb,jaKb->ij', t2ivKv, t2jvKv)
                dm[i0:i1,j0:j1] += 2 * lib.einsum('iaKb,jbKa->ij', t2ivKv, t2jvKv)

                t2jvKv = None
            t2ivKv = None
        KvLR = KvLI = None
    buf1 = buf2 = None

    return dm

def make_lo_rdm1_occ_1p_real(eris, moeocc, moevir, u):
    r''' Occupied MP2 density matrix with one localized particle

        Math:
            dm(i,j)
                = 2 * \sum_{k'ab} t2(ik'ab) ( 2*t2(jk'ab) - t2(jk'ba) )

        Args:
            eris : ERI object
                Provides `ovL` in the canonical MOs.
            u : np.ndarray
                Overlap between the canonical and localized virtual orbitals, i.e.,
                    u(a,a') = <a|a'>
    '''
    nocc, nvir, naux = eris.nocc, eris.nvir, eris.Naux_ibz
    REAL = np.float64
    dsize = 8
    assert(u.dtype == REAL)
    nVir = u.shape[1]

    # determine occblksize
    mem_avail = eris.max_memory - lib.current_memory()[0]
    M = mem_avail * 0.7 * 1e6/dsize
    virblksize, mem_peak = _mp2_rdm1_virblksize(nocc,nvir,naux, 4, 3, M, dsize)
    if DEBUG_BLKSIZE: virblksize = max(1,nvir//2)
    logger.debug1(eris, 'make_lo_rdm1_occ_1p :  nocc = %d  nvir = %d  nVir = %d  naux = %d  '
                  'virblksize = %d  peak mem = %.2f MB',
                  nocc, nvir, nVir, naux, virblksize, mem_peak)
    bufsize = virblksize*min(virblksize,nVir)*nocc**2
    buf = np.empty(bufsize, dtype=REAL)

    moeVir, u = subspace_eigh(np.diag(moevir), u)
    eov = moeocc[:,None] - moevir
    eoV = moeocc[:,None] - moeVir

    dm = np.zeros((nocc,nocc), dtype=REAL)
    for Abatch,(A0,A1) in enumerate(lib.prange(0,nVir,virblksize)):
        oALR, oALI = eris.xform_vir(u[:,A0:A1])
        oALR = oALR.reshape(-1,naux)
        oALI = oALI.reshape(-1,naux)
        eoA = eoV[:,A0:A1]
        for bbatch,(b0,b1) in enumerate(lib.prange(0,nvir,virblksize)):
            eob = eov[:,b0:b1]
            eoAob = lib.direct_sum('iA+jb->iAjb', eoA, eob)

            obLR, obLI = eris.get_vir_blk(b0,b1)
            obLR = obLR.reshape(-1,naux)
            obLI = obLI.reshape(-1,naux)
            t2oAob = np.ndarray((oALR.shape[0],obLR.shape[0]), dtype=REAL, buffer=buf)
            zdotCNtoR(oALR, oALI, obLR.T, obLI.T, cR=t2oAob)
            t2oAob = _make_t2(eris, t2oAob.reshape(*eoAob.shape), eoAob)
            eoAob = None
            obLR = obLI = None

            dm -= 2 * lib.einsum('iAkb,jAkb->ij', t2oAob, t2oAob)
            dm +=     lib.einsum('iAkb,kAjb->ij', t2oAob, t2oAob)
            dm +=     lib.einsum('kAib,jAkb->ij', t2oAob, t2oAob)
            dm -= 2 * lib.einsum('kAib,kAjb->ij', t2oAob, t2oAob)

            t2oAob = None
        oALR = oALI = None
    buf = None

    return dm

def make_lo_rdm1_occ_2p_real(eris, moeocc, moevir, u):
    r''' Occupied MP2 density matrix with two localized particles

        Math:
            dm(i,j)
                = 2 * \sum_{ka'b'} t2(ika'b') ( 2*t2(jka'b') - t2(jkb'a') )

        Args:
            eris : ERI object
                Provides `ovL` in the canonical MOs.
            u : np.ndarray
                Overlap between the canonical and localized virtual orbitals, i.e.,
                    u(a,a') = <a|a'>
    '''
    nocc, nvir, naux = eris.nocc, eris.nvir, eris.Naux_ibz
    REAL = np.float64
    dsize = 8
    assert(u.dtype == REAL)
    nVir = u.shape[1]

    # determine Occblksize
    mem_avail = eris.max_memory - lib.current_memory()[0]
    M = mem_avail * 0.7 * 1e6/dsize
    Virblksize, mem_peak = _mp2_rdm1_virblksize(nocc,nVir,naux, 4, 3, M, dsize)
    if DEBUG_BLKSIZE: Virblksize = max(1,nVir//2)
    logger.debug1(eris, 'make_lo_rdm1_occ_2p:  nocc = %d  nvir = %d  nVir = %d  naux = %d  '
                  'Virblksize = %d  peak mem = %.2f MB',
                  nocc, nvir, nVir, naux, Virblksize, mem_peak)
    bufsize = (Virblksize*nocc)**2
    buf = np.empty(bufsize, dtype=REAL)

    moeVir, u = subspace_eigh(np.diag(moevir), u)
    eoV = moeocc[:,None] - moeVir

    dm = np.zeros((nocc,nocc), dtype=REAL)
    for Abatch,(A0,A1) in enumerate(lib.prange(0,nVir,Virblksize)):
        oALR, oALI = eris.xform_vir(u[:,A0:A1])
        oALR = oALR.reshape(-1,naux)
        oALI = oALI.reshape(-1,naux)
        eoA = eoV[:,A0:A1]
        for Bbatch,(B0,B1) in enumerate(lib.prange(0,nVir,Virblksize)):
            if Bbatch == Abatch:
                eoB = eoA
                oBLR = oALR
                oBLI = oALI
            else:
                eoB = eoV[:,B0:B1]
                oBLR, oBLI = eris.xform_vir(u[:,B0:B1])
                oBLR = oBLR.reshape(-1,naux)
                oBLI = oBLI.reshape(-1,naux)

            eoAoB = lib.direct_sum('iA+jB->iAjB', eoA, eoB)
            t2oAoB = np.ndarray((oALR.shape[0], oBLR.shape[0]), dtype=REAL, buffer=buf)
            zdotCNtoR(oALR, oALI, oBLR.T, oBLI.T, cR=t2oAoB)
            t2oAoB = _make_t2(eris, t2oAoB.reshape(*eoAoB.shape), eoAoB)
            eoAoB = None
            oBLR = oBLI = None

            dm -= 4 * lib.einsum('iAkB,jAkB->ij', t2oAoB, t2oAoB)
            dm += 2 * lib.einsum('iAkB,kAjB->ij', t2oAoB, t2oAoB)

            t2oAoB = None
        oALR = oALI = None
    buf = None

    return dm

def make_lo_rdm1_vir_1p_real(eris, moeocc, moevir, u):
    r''' Virtual MP2 density matrix with one localized particle

        Math:
            dm(a,b)
                = \sum_{ijc'} 2 * t2(ijac') * ( 2 * t2(ijbc') - t2(jibc') )

        Args:
            eris : ERI object
                Provides `ovL` in the canonical MOs.
            u : np.ndarray
                Overlap between the canonical and localized occupied orbitals, i.e.,
                    u(a,a') = <a|a'>
    '''
    nocc, nvir, naux = eris.nocc, eris.nvir, eris.Naux_ibz
    REAL = np.float64
    dsize = 8
    assert(u.dtype == REAL)
    nVir = u.shape[1]

    # determine occblksize
    mem_avail = eris.max_memory - lib.current_memory()[0]
    M = mem_avail * 0.7 * 1e6/dsize
    virblksize, mem_peak = _mp2_rdm1_virblksize(nocc,nvir,naux, 4, 3, M, dsize)
    if DEBUG_BLKSIZE: virblksize = max(1,nvir//2)
    logger.debug1(eris, 'make_lo_rdm1_vir_1p :  nocc = %d  nvir = %d  nVir = %d  naux = %d  '
                  'virblksize = %d  peak mem = %.2f MB',
                  nocc, nvir, nVir, naux, virblksize, mem_peak)
    bufsize = nvir*min(nvir,virblksize)*nocc**2
    buf1 = np.empty(bufsize, dtype=REAL)
    buf2 = np.empty(bufsize, dtype=REAL)

    moeVir, u = subspace_eigh(np.diag(moevir), u)
    eov = moeocc[:,None] - moevir
    eoV = moeocc[:,None] - moeVir

    # TODO: can we batch over occ index?
    dm = np.zeros((nvir,nvir), dtype=REAL)
    for Abatch,(A0,A1) in enumerate(lib.prange(0,nVir,virblksize)):
        oALR, oALI = eris.xform_vir(u[:,A0:A1])
        oALR = oALR.reshape(-1,naux)
        oALI = oALI.reshape(-1,naux)
        eoA = eoV[:,A0:A1]
        for abatch,(a0,a1) in enumerate(lib.prange(0,nvir,virblksize)):
            eoa = eov[:,a0:a1]
            eoAoa = lib.direct_sum('iA+jb->iAjb', eoA, eoa)
            oaLR, oaLI = eris.get_vir_blk(a0,a1)
            oaLR = oaLR.reshape(-1,naux)
            oaLI = oaLI.reshape(-1,naux)
            t2oAoa = np.ndarray((oALR.shape[0], oaLR.shape[0]), dtype=REAL, buffer=buf1)
            zdotCNtoR(oALR, oALI, oaLR.T, oaLI.T, cR=t2oAoa)
            t2oAoa = _make_t2(eris, t2oAoa.reshape(*eoAoa.shape), eoAoa)
            eoAoa = None
            oaLR = oaLI = None
            for bbatch,(b0,b1) in enumerate(lib.prange(0,nvir,virblksize)):
                if abatch == bbatch:
                    t2oAob = t2oAoa
                else:
                    eob = eov[:,b0:b1]
                    eoAob = lib.direct_sum('iA+jb->iAjb', eoA, eob)
                    obLR, obLI = eris.get_vir_blk(b0,b1)
                    obLR = obLR.reshape(-1,naux)
                    obLI = obLI.reshape(-1,naux)
                    t2oAob = np.ndarray((oALR.shape[0], obLR.shape[0]), dtype=REAL, buffer=buf2)
                    zdotCNtoR(oALR, oALI, obLR.T, obLI.T, cR=t2oAob)
                    t2oAob = _make_t2(eris, t2oAob.reshape(*eoAob.shape), eoAob)
                    eoAob = None
                    obLR = obLI = None

                dm[a0:a1,b0:b1] += 4 * lib.einsum('iAja,iAjb->ab', t2oAoa, t2oAob)
                dm[a0:a1,b0:b1] -= 2 * lib.einsum('iAja,jAib->ab', t2oAoa, t2oAob)

                t2oAob = None
            t2oAoa = None
        oALR = oALI = None
    buf1 = buf2 = None

    return dm

def make_lo_rdm1_vir_1h_real(eris, moeocc, moevir, u):
    r''' Occupied MP2 density matrix with one localized hole

        Math:
            dm(a,b)
                = \sum_{i'jc} 2 * t2(i'jac) * t2(i'jbc) + 2 * t2(i'jca) * t2(i'jcb)
                            - t2(i'jac) * t2(i'jcb) - t2(i'jca) * t2(i'jbc)

        Args:
            eris : ERI object
                Provides `ovL` in the canonical MOs.
            u : np.ndarray
                Overlap between the canonical and localized occupied orbitals, i.e.,
                    u(i,i') = <i|i'>
    '''
    nocc, nvir, naux = eris.nocc, eris.nvir, eris.Naux_ibz
    REAL = np.float64
    dsize = 8
    assert(u.dtype == REAL)
    nOcc = u.shape[1]

    # determine Occblksize
    mem_avail = eris.max_memory - lib.current_memory()[0]
    M = mem_avail * 0.7 * 1e6/dsize
    occblksize, mem_peak = _mp2_rdm1_occblksize(nocc,nvir,naux, 4, 3, M, dsize)
    if DEBUG_BLKSIZE: occblksize = max(1,nocc//2)
    logger.debug1(eris, 'make_lo_rdm1_vir_1h :  nocc = %d  nvir = %d  nOcc = %d  naux = %d  '
                  'occblksize = %d  peak mem = %.2f MB',
                  nocc, nvir, nOcc, naux, occblksize, mem_peak)
    bufsize = min(nOcc,occblksize)*min(nocc,occblksize)*nvir**2
    buf = np.empty(bufsize, dtype=REAL)

    moeOcc, u = subspace_eigh(np.diag(moeocc), u)
    eOv = moeOcc[:,None] - moevir
    eov = moeocc[:,None] - moevir

    dm = np.zeros((nvir,nvir), dtype=REAL)
    for Ibatch,(I0,I1) in enumerate(lib.prange(0,nOcc,occblksize)):
        IvLR, IvLI = eris.xform_occ(u[:,I0:I1])
        IvLR = IvLR.reshape(-1,naux)
        IvLI = IvLI.reshape(-1,naux)
        eIv = eOv[I0:I1]
        for jbatch,(j0,j1) in enumerate(lib.prange(0,nocc,occblksize)):
            ejv = eov[j0:j1]
            eIvjv = lib.direct_sum('Ia+jb->Iajb', eIv, ejv)
            jvLR, jvLI = eris.get_occ_blk(j0,j1)
            jvLR = jvLR.reshape(-1,naux)
            jvLI = jvLI.reshape(-1,naux)
            t2Ivjv = np.ndarray((IvLR.shape[0], jvLR.shape[0]), dtype=REAL, buffer=buf)
            zdotCNtoR(IvLR, IvLI, jvLR.T, jvLI.T, cR=t2Ivjv)
            t2Ivjv = _make_t2(eris, t2Ivjv.reshape(*eIvjv.shape), eIvjv)
            eIvjv = None
            jvLR = jvLI = None

            dm += 2 * lib.einsum('Iajc,Ibjc->ab', t2Ivjv, t2Ivjv)
            dm -=     lib.einsum('Iajc,Icjb->ab', t2Ivjv, t2Ivjv)
            dm -=     lib.einsum('Icja,Ibjc->ab', t2Ivjv, t2Ivjv)
            dm += 2 * lib.einsum('Icja,Icjb->ab', t2Ivjv, t2Ivjv)

            t2Ivjv = None
        IvLR = IvLI = None
    buf = None

    return dm

def make_lo_rdm1_vir_2h_real(eris, moeocc, moevir, u):
    r''' Occupied MP2 density matrix with two localized holes

        Math:
            dm(a,b)
                = 2 * \sum_{i'j'c} t2(i'j'ac) ( 2*t2(i'j'bc) - t2(i'j'cb) )

        Args:
            eris : ERI object
                Provides `ovL` in the canonical MOs.
            u : np.ndarray
                Overlap between the canonical and localized occupied orbitals, i.e.,
                    u(i,i') = <i|i'>
    '''
    nocc, nvir, naux = eris.nocc, eris.nvir, eris.Naux_ibz
    REAL = np.float64
    dsize = 8
    assert(u.dtype == REAL)
    nOcc = u.shape[1]

    # determine Occblksize
    mem_avail = eris.max_memory - lib.current_memory()[0]
    M = mem_avail * 0.7 * 1e6/dsize
    Occblksize, mem_peak = _mp2_rdm1_occblksize(nOcc,nvir,naux, 4, 3, M, dsize)
    if DEBUG_BLKSIZE: Occblksize = max(1,nOcc//2)
    logger.debug1(eris, 'make_lo_rdm1_vir_2h:  nocc = %d  nvir = %d  nOcc = %d  naux = %d  '
                  'Occblksize = %d  peak mem = %.2f MB',
                  nocc, nvir, nOcc, naux, Occblksize, mem_peak)
    bufsize = (Occblksize*nvir)**2
    buf = np.empty(bufsize, dtype=REAL)

    moeOcc, u = subspace_eigh(np.diag(moeocc), u)
    eOv = moeOcc[:,None] - moevir

    dm = np.zeros((nvir,nvir), dtype=REAL)
    for Ibatch,(I0,I1) in enumerate(lib.prange(0,nOcc,Occblksize)):
        IvLR, IvLI = eris.xform_occ(u[:,I0:I1])
        IvLR = IvLR.reshape(-1,naux)
        IvLI = IvLI.reshape(-1,naux)
        eIv = eOv[I0:I1]
        for Jbatch,(J0,J1) in enumerate(lib.prange(0,nOcc,Occblksize)):
            if Jbatch == Ibatch:
                eJv = eIv
                JvLR = IvLR
                JvLI = IvLI
            else:
                eJv = eOv[J0:J1]
                JvLR, JvLI = eris.xform_occ(u[:,J0:J1])
                JvLR = JvLR.reshape(-1,naux)
                JvLI = JvLI.reshape(-1,naux)

            eIvJv = lib.direct_sum('Ia+Jb->IaJb', eIv, eJv)
            t2IvJv = np.ndarray((IvLR.shape[0], JvLR.shape[0]), dtype=REAL, buffer=buf)
            zdotCNtoR(IvLR, IvLI, JvLR.T, JvLI.T, cR=t2IvJv)
            t2IvJv = _make_t2(eris, t2IvJv.reshape(*eIvJv.shape), eIvJv)
            eIvJv = None
            JvLR = JvLI = None

            dm += 4 * lib.einsum('IaJc,IbJc->ab', t2IvJv, t2IvJv)
            dm -= 2 * lib.einsum('IaJc,IcJb->ab', t2IvJv, t2IvJv)

            t2IvJv = None
        IvLR = IvLI = None
    buf = None

    return dm


''' make lo rdm1 for complex orbitals
'''
def make_lo_rdm1_occ_1h_complex(eris, moeocc, moevir, u):
    r''' Occupied MP2 density matrix with one localized hole

        Math:
            dm(i,j)
                = 2 * \sum_{k'ab} t2(ik'ab) ( 2*t2(jk'ab) - t2(jk'ba) )

        Args:
            eris : ERI object
                Provides `ovL` in the canonical MOs.
            u : np.ndarray
                Overlap between the canonical and localized occupied orbitals, i.e.,
                    u(i,i') = <i|i'>
    '''
    nocc, nvir, naux = eris.nocc, eris.nvir, eris.naux
    nqpts = len(eris.qpts)
    dtype = eris.dtype
    dsize = eris.dsize
    nOcc = u.shape[1]

    # determine occblksize
    mem_avail = eris.max_memory - lib.current_memory()[0]
    M = mem_avail * 0.7 * 1e6/dsize
    occblksize, mem_peak = _mp2_rdm1_occblksize(nocc,nvir,naux, nqpts+2, 3, M, dsize)
    if DEBUG_BLKSIZE: occblksize = max(1,nocc//2)
    logger.debug1(eris, 'make_lo_rdm1_occ_1h :  nocc = %d  nvir = %d  nOcc = %d  naux = %d  '
                  'occblksize = %d  peak mem = %.2f MB',
                  nocc, nvir, nOcc, naux, occblksize, mem_peak)

    moeOcc, u = subspace_eigh(np.diag(moeocc), u)
    eov = moeocc[:,None] - moevir
    eOv = moeOcc[:,None] - moevir

    dm = np.zeros((nocc,nocc), dtype=dtype)
    for Kbatch,(K0,K1) in enumerate(lib.prange(0,nOcc,occblksize)):
        KvL = [eris.xform_occ(q, u[:,K0:K1]) for q in range(nqpts)]
        eKv = eOv[K0:K1]
        for ibatch,(i0,i1) in enumerate(lib.prange(0,nocc,occblksize)):
            eiv = eov[i0:i1]
            eiKvv = lib.direct_sum('ia+Kb->iKab', eiv, eKv)
            t2iKvv = 0
            for q1,q2 in enumerate(eris.qconserv):
                t2iKvv += lib.einsum('iax,Kbx->iKab', eris.get_occ_blk(q1,i0,i1), KvL[q2])
            t2iKvv = _make_t2(eris, t2iKvv, eiKvv)
            eiKvv = None
            for jbatch,(j0,j1) in enumerate(lib.prange(0,nocc,occblksize)):
                if jbatch == ibatch:
                    t2jKvv = t2iKvv
                else:
                    ejv = eov[j0:j1]
                    ejKvv = lib.direct_sum('ia+Kb->iKab', ejv, eKv)
                    t2jKvv = 0
                    for q1,q2 in enumerate(eris.qconserv):
                        t2jKvv += lib.einsum('iax,Kbx->iKab', eris.get_occ_blk(q1,j0,j1), KvL[q2])
                    t2jKvv = _make_t2(eris, t2jKvv, ejKvv)
                    ejKvv = None

                dm[i0:i1,j0:j1] -= 4 * lib.einsum('iKab,jKab->ij', np.conj(t2iKvv), t2jKvv)
                dm[i0:i1,j0:j1] += 2 * lib.einsum('iKab,jKba->ij', np.conj(t2iKvv), t2jKvv)

                t2jKvv = None
            t2iKvv = None
        KvL = None

    return dm

def make_lo_rdm1_occ_1p_complex(eris, moeocc, moevir, u):
    r''' Occupied MP2 density matrix with one localized particle

        Math:
            dm(i,j)
                = 2 * \sum_{k'ab} t2(ik'ab) ( 2*t2(jk'ab) - t2(jk'ba) )

        Args:
            eris : ERI object
                Provides `ovL` in the canonical MOs.
            u : np.ndarray
                Overlap between the canonical and localized virtual orbitals, i.e.,
                    u(a,a') = <a|a'>
    '''
    nocc, nvir, naux = eris.nocc, eris.nvir, eris.naux
    nqpts = len(eris.qpts)
    dtype = eris.dtype
    dsize = eris.dsize
    nVir = u.shape[1]

    # determine occblksize
    mem_avail = eris.max_memory - lib.current_memory()[0]
    M = mem_avail * 0.7 * 1e6/dsize
    virblksize, mem_peak = _mp2_rdm1_virblksize(nocc,nvir,naux, nqpts+1, 2, M, dsize)
    if DEBUG_BLKSIZE: virblksize = max(1,nvir//2)
    logger.debug1(eris, 'make_lo_rdm1_occ_1p :  nocc = %d  nvir = %d  nVir = %d  naux = %d  '
                  'virblksize = %d  peak mem = %.2f MB',
                  nocc, nvir, nVir, naux, virblksize, mem_peak)

    moeVir, u = subspace_eigh(np.diag(moevir), u)
    eov = moeocc[:,None] - moevir
    eoV = moeocc[:,None] - moeVir

    dm = np.zeros((nocc,nocc), dtype=dtype)
    for Abatch,(A0,A1) in enumerate(lib.prange(0,nVir,virblksize)):
        oAL = [eris.xform_vir(q,u[:,A0:A1]) for q in range(nqpts)]
        eoA = eoV[:,A0:A1]
        for bbatch,(b0,b1) in enumerate(lib.prange(0,nvir,virblksize)):
            eob = eov[:,b0:b1]
            eooAb = lib.direct_sum('iA+jb->ijAb', eoA, eob)
            t2ooAb = 0
            for q1,q2 in enumerate(eris.qconserv):
                t2ooAb += lib.einsum('iAx,jbx->ijAb', oAL[q1], eris.get_vir_blk(q2,b0,b1))
            t2ooAb = _make_t2(eris, t2ooAb, eooAb)
            eooAb = None

            dm -= 2 * lib.einsum('ikAb,jkAb->ij', np.conj(t2ooAb), t2ooAb)
            dm +=     lib.einsum('ikAb,kjAb->ij', np.conj(t2ooAb), t2ooAb)
            dm +=     lib.einsum('kiAb,jkAb->ij', np.conj(t2ooAb), t2ooAb)
            dm -= 2 * lib.einsum('kiAb,kjAb->ij', np.conj(t2ooAb), t2ooAb)

            t2ooAb = None
        oAL = None

    return dm

def make_lo_rdm1_occ_2p_complex(eris, moeocc, moevir, u):
    r''' Occupied MP2 density matrix with two localized particles

        Math:
            dm(i,j)
                = 2 * \sum_{ka'b'} t2(ika'b') ( 2*t2(jka'b') - t2(jkb'a') )

        Args:
            eris : ERI object
                Provides `ovL` in the canonical MOs.
            u : np.ndarray
                Overlap between the canonical and localized virtual orbitals, i.e.,
                    u(a,a') = <a|a'>
    '''
    nocc, nvir, naux = eris.nocc, eris.nvir, eris.naux
    nqpts = len(eris.qpts)
    dtype = eris.dtype
    dsize = eris.dsize
    nVir = u.shape[1]

    # determine Occblksize
    mem_avail = eris.max_memory - lib.current_memory()[0]
    M = mem_avail * 0.7 * 1e6/dsize
    Virblksize, mem_peak = _mp2_rdm1_virblksize(nocc,nVir,naux, nqpts+1, 2, M, dsize)
    if DEBUG_BLKSIZE: Virblksize = max(1,nVir//2)
    logger.debug1(eris, 'make_lo_rdm1_occ_2p:  nocc = %d  nvir = %d  nVir = %d  naux = %d  '
                  'Virblksize = %d  peak mem = %.2f MB',
                  nocc, nvir, nVir, naux, Virblksize, mem_peak)

    moeVir, u = subspace_eigh(np.diag(moevir), u)
    eoV = moeocc[:,None] - moeVir

    dm = np.zeros((nocc,nocc), dtype=dtype)
    for Abatch,(A0,A1) in enumerate(lib.prange(0,nVir,Virblksize)):
        oAL = [eris.xform_vir(q,u[:,A0:A1]) for q in range(nqpts)]
        eoA = eoV[:,A0:A1]
        for Bbatch,(B0,B1) in enumerate(lib.prange(0,nVir,Virblksize)):
            if Bbatch == Abatch:
                eoB = eoA
            else:
                eoB = eoV[:,B0:B1]
            eooAB = lib.direct_sum('iA+jB->ijAB', eoA, eoB)

            t2ooAB = 0
            if Bbatch == Abatch:
                for q1,q2 in enumerate(eris.qconserv):
                    t2ooAB += lib.einsum('iAx,jBx->ijAB', oAL[q1], oAL[q2])
            else:
                for q1,q2 in enumerate(eris.qconserv):
                    t2ooAB += lib.einsum('iAx,jBx->ijAB', oAL[q1], eris.xform_vir(q2, u[:,B0:B1]))
            t2ooAB = _make_t2(eris, t2ooAB, eooAB)

            eooAB = None

            dm -= 4 * lib.einsum('ikAB,jkAB->ij', np.conj(t2ooAB), t2ooAB)
            dm += 2 * lib.einsum('ikAB,kjAB->ij', np.conj(t2ooAB), t2ooAB)

            t2ooAB = None
        oAL = None

    return dm

def make_lo_rdm1_vir_1p_complex(eris, moeocc, moevir, u):
    r''' Virtual MP2 density matrix with one localized particle

        Math:
            dm(a,b)
                = \sum_{ijc'} 2 * t2(ijac') * ( 2 * t2(ijbc') - t2(jibc') )

        Args:
            eris : ERI object
                Provides `ovL` in the canonical MOs.
            u : np.ndarray
                Overlap between the canonical and localized occupied orbitals, i.e.,
                    u(a,a') = <a|a'>
    '''
    nocc, nvir, naux = eris.nocc, eris.nvir, eris.naux
    nqpts = len(eris.qpts)
    dtype = eris.dtype
    dsize = eris.dsize
    nVir = u.shape[1]

    # determine occblksize
    mem_avail = eris.max_memory - lib.current_memory()[0]
    M = mem_avail * 0.7 * 1e6/dsize
    virblksize, mem_peak = _mp2_rdm1_virblksize(nocc,nvir,naux, nqpts+2, 3, M, dsize)
    if DEBUG_BLKSIZE: virblksize = max(1,nvir//2)
    logger.debug1(eris, 'make_lo_rdm1_vir_1p :  nocc = %d  nvir = %d  nVir = %d  naux = %d  '
                  'virblksize = %d  peak mem = %.2f MB',
                  nocc, nvir, nVir, naux, virblksize, mem_peak)

    moeVir, u = subspace_eigh(np.diag(moevir), u)
    eov = moeocc[:,None] - moevir
    eoV = moeocc[:,None] - moeVir

    # TODO: can we batch over occ index?
    dm = np.zeros((nvir,nvir), dtype=dtype)
    for Abatch,(A0,A1) in enumerate(lib.prange(0,nVir,virblksize)):
        oAL = [eris.xform_vir(q, u[:,A0:A1]) for q in range(nqpts)]
        eoA = eoV[:,A0:A1]
        for abatch,(a0,a1) in enumerate(lib.prange(0,nvir,virblksize)):
            eoa = eov[:,a0:a1]
            eooAa = lib.direct_sum('iA+jb->ijAb', eoA, eoa)
            t2ooAa = 0
            for q1,q2 in enumerate(eris.qconserv):
                t2ooAa += lib.einsum('iAx,jbx->ijAb', oAL[q1], eris.get_vir_blk(q2,a0,a1))
            t2ooAa = _make_t2(eris, t2ooAa, eooAa)
            eooAa = None
            for bbatch,(b0,b1) in enumerate(lib.prange(0,nvir,virblksize)):
                if abatch == bbatch:
                    t2ooAb = t2ooAa
                else:
                    eob = eov[:,b0:b1]
                    eooAb = lib.direct_sum('iA+jb->ijAb', eoA, eob)
                    t2ooAb = 0
                    for q1,q2 in enumerate(eris.qconserv):
                        t2ooAb += lib.einsum('iAx,jbx->ijAb', oAL[q1], eris.get_vir_blk(q2,b0,b1))
                    t2ooAb = _make_t2(eris, t2ooAb, eooAb)
                    eooAb = None

                dm[a0:a1,b0:b1] += 4 * lib.einsum('ijAa,ijAb->ab', t2ooAa, np.conj(t2ooAb))
                dm[a0:a1,b0:b1] -= 2 * lib.einsum('ijAa,jiAb->ab', t2ooAa, np.conj(t2ooAb))

                t2ooAb = None
            t2ooAa = None
        oAL = None

    return dm

def make_lo_rdm1_vir_1h_complex(eris, moeocc, moevir, u):
    r''' Occupied MP2 density matrix with one localized hole

        Math:
            dm(a,b)
                = \sum_{i'jc} 2 * t2(i'jac) * t2(i'jbc) + 2 * t2(i'jca) * t2(i'jcb)
                            - t2(i'jac) * t2(i'jcb) - t2(i'jca) * t2(i'jbc)

        Args:
            eris : ERI object
                Provides `ovL` in the canonical MOs.
            u : np.ndarray
                Overlap between the canonical and localized occupied orbitals, i.e.,
                    u(i,i') = <i|i'>
    '''
    nocc, nvir, naux = eris.nocc, eris.nvir, eris.naux
    nqpts = len(eris.qpts)
    dtype = eris.dtype
    dsize = eris.dsize
    nOcc = u.shape[1]

    # determine Occblksize
    mem_avail = eris.max_memory - lib.current_memory()[0]
    M = mem_avail * 0.7 * 1e6/dsize
    occblksize, mem_peak = _mp2_rdm1_occblksize(nocc,nvir,naux, nqpts+1, 2, M, dsize)
    if DEBUG_BLKSIZE: occblksize = max(1,nocc//2)
    logger.debug1(eris, 'make_lo_rdm1_vir_1h :  nocc = %d  nvir = %d  nOcc = %d  naux = %d  '
                  'occblksize = %d  peak mem = %.2f MB',
                  nocc, nvir, nOcc, naux, occblksize, mem_peak)

    moeOcc, u = subspace_eigh(np.diag(moeocc), u)
    eOv = moeOcc[:,None] - moevir
    eov = moeocc[:,None] - moevir

    dm = np.zeros((nvir,nvir), dtype=dtype)
    for Ibatch,(I0,I1) in enumerate(lib.prange(0,nOcc,occblksize)):
        IvL = [eris.xform_occ(q,u[:,I0:I1]) for q in range(nqpts)]
        eIv = eOv[I0:I1]
        for jbatch,(j0,j1) in enumerate(lib.prange(0,nocc,occblksize)):
            ejv = eov[j0:j1]
            eIjvv = lib.direct_sum('Ia+jb->Ijab', eIv, ejv)
            t2Ijvv = 0
            for q1,q2 in enumerate(eris.qconserv):
                t2Ijvv += lib.einsum('Iax,jbx->Ijab', IvL[q1], eris.get_occ_blk(q2,j0,j1))
            t2Ijvv = _make_t2(eris, t2Ijvv, eIjvv)
            eIjvv = None

            dm += 2 * lib.einsum('Ijac,Ijbc->ab', t2Ijvv, np.conj(t2Ijvv))
            dm -=     lib.einsum('Ijac,Ijcb->ab', t2Ijvv, np.conj(t2Ijvv))
            dm -=     lib.einsum('Ijca,Ijbc->ab', t2Ijvv, np.conj(t2Ijvv))
            dm += 2 * lib.einsum('Ijca,Ijcb->ab', t2Ijvv, np.conj(t2Ijvv))

            t2Ijvv = None
        IvL = None

    return dm

def make_lo_rdm1_vir_2h_complex(eris, moeocc, moevir, u):
    r''' Occupied MP2 density matrix with two localized holes

        Math:
            dm(a,b)
                = 2 * \sum_{i'j'c} t2(i'j'ac) ( 2*t2(i'j'bc) - t2(i'j'cb) )

        Args:
            eris : ERI object
                Provides `ovL` in the canonical MOs.
            u : np.ndarray
                Overlap between the canonical and localized occupied orbitals, i.e.,
                    u(i,i') = <i|i'>
    '''
    nocc, nvir, naux = eris.nocc, eris.nvir, eris.naux
    nqpts = len(eris.qpts)
    dtype = eris.dtype
    dsize = eris.dsize
    nOcc = u.shape[1]

    # determine Occblksize
    mem_avail = eris.max_memory - lib.current_memory()[0]
    M = mem_avail * 0.7 * 1e6/dsize
    Occblksize, mem_peak = _mp2_rdm1_occblksize(nOcc,nvir,naux, nqpts+1, 2, M, dsize)
    if DEBUG_BLKSIZE: Occblksize = max(1,nOcc//2)
    logger.debug1(eris, 'make_lo_rdm1_vir_2h:  nocc = %d  nvir = %d  nOcc = %d  naux = %d  '
                  'Occblksize = %d  peak mem = %.2f MB',
                  nocc, nvir, nOcc, naux, Occblksize, mem_peak)

    moeOcc, u = subspace_eigh(np.diag(moeocc), u)
    eOv = moeOcc[:,None] - moevir

    dm = np.zeros((nvir,nvir), dtype=dtype)
    for Ibatch,(I0,I1) in enumerate(lib.prange(0,nOcc,Occblksize)):
        IvL = [eris.xform_occ(q, u[:,I0:I1]) for q in range(nqpts)]
        eIv = eOv[I0:I1]
        for Jbatch,(J0,J1) in enumerate(lib.prange(0,nOcc,Occblksize)):
            if Jbatch == Ibatch:
                eJv = eIv
            else:
                eJv = eOv[J0:J1]
            eIJvv = lib.direct_sum('Ia+Jb->IJab', eIv, eJv)

            t2IJvv = 0
            if Jbatch == Ibatch:
                for q1,q2 in enumerate(eris.qconserv):
                    t2IJvv += lib.einsum('Iax,Jbx->IJab', IvL[q1], IvL[q2])
            else:
                for q1,q2 in enumerate(eris.qconserv):
                    t2IJvv += lib.einsum('Iax,Jbx->IJab', IvL[q1], eris.xform_occ(q2, u[:,J0:J1]))
            t2IJvv = _make_t2(eris, t2IJvv, eIJvv)

            eIJvv = None

            dm += 4 * lib.einsum('IJac,IJbc->ab', t2IJvv, np.conj(t2IJvv))
            dm -= 2 * lib.einsum('IJac,IJcb->ab', t2IJvv, np.conj(t2IJvv))

            t2IJvv = None
        IvL = None

    return dm

''' make lo rdm1 from k-point integrals
'''
def make_lo_rdm1_1h_occ_k(eris, u):
    r''' Occupied '1h' MP2 density matrix from k-point integrals (see `make_lo_rdm1_1h_k`). '''
    return make_lo_rdm1_1h_k(eris, u, with_vir=False)[0]

def make_lo_rdm1_1h_vir_k(eris, u):
    r''' Virtual '1h' MP2 density matrix from k-point integrals (see `make_lo_rdm1_1h_k`). '''
    return make_lo_rdm1_1h_k(eris, u, with_occ=False)[1]

def make_lo_rdm1_1h_k(eris, u, with_occ=True, with_vir=True):
    r''' Occupied and virtual MP2 density matrices with one localized hole, from the k-point
    DF integrals of the Bloch orbitals (`pyscf.pbc.lno.klno._KLNODFKERIS`)

        Math:
            dm(i,j) = \sum_{Kac} t2(Kiac).conj() * ( 2*t2(Kjca) - 4*t2(Kjac) )
            dm(a,b) = \sum_{Ijc} t2(Ijac) * X(Ijbc).conj() + t2(Ijca) * X(Ijcb).conj()
            X(Ijac) = 2*t2(Ijac) - t2(Ijca)
            t2(Ijac) = (Ia|jc).conj() / (e_I + e_j - e_a - e_c)

        as `make_lo_rdm1_occ_1h_real` and `make_lo_rdm1_vir_1h_real`, with j, a, c Bloch
        orbitals and I = \sum_{ki} |k i> C(k i,I) the semicanonical fragment orbitals.
        Momentum conservation fixes k_I = k_a - k_j + k_c, so each block of (Ia|jc) needs
        the integrals Lov[kI,ka] and Lov[kj,kc] of one momentum transfer q = k_a - k_I.

        The amplitudes are made for a block of I and one k_c at a time, as the slabs
            A(Ijac) = t2(Ijac),  B(Ijca) = t2(Ijca)     (c in k_c; j, a in all k)
        which hold every amplitude the density matrices need for c in k_c. Each amplitude
        is thus made twice (in the slab of k_c and in that of k_a), but no amplitude array
        is larger than (nI, Nk*nocc, Nk*nvir, nvir).

        Args:
            eris : _KLNODFKERIS
                Provides the k-point integrals `Lov`, Bloch orbital energies
                `mo_energy_occ`/`mo_energy_vir` (Nk,nocc)/(Nk,nvir), and the overlaps
                `Wocc`/`Wvir` between the Bloch and the supercell canonical orbitals.
            u : np.ndarray
                Overlap between the supercell canonical and localized occupied orbitals,
                i.e., u(i,i') = <i|i'>

        Returns:
            dm_occ, dm_vir in the supercell canonical orbitals (None if not requested).
    '''
    Lov, kconserv = eris.Lov, eris.kconserv
    eo, ev = eris.mo_energy_occ, eris.mo_energy_vir
    nkpts, nocc = eo.shape
    nvir = ev.shape[1]
    nko, nkv = nkpts*nocc, nkpts*nvir
    naux = max(L.shape[0] for L in Lov.ravel())
    COMPLEX = np.complex128
    dsize = 16

    # semicanonical fragment orbitals in the Bloch basis
    eI, cI = subspace_eigh(np.diag(eo.ravel()), np.dot(eris.Wocc, u))
    cI = cI.reshape(nkpts,nocc,-1)
    nOcc = eI.size
    evall = ev.ravel()

    # determine Occblksize: per I, the (I a|L) of all q and ~6 slabs (A, B, temporaries)
    mem_avail = eris.max_memory - lib.current_memory()[0]
    mem_fix = (nko**2*with_occ + nkv**2*with_vir) * dsize/1e6
    mem_I = (nkpts*naux*nkv + 6*nko*nkv*nvir) * dsize/1e6
    Occblksize = max(1, min(nOcc, int(np.floor((mem_avail*0.7 - mem_fix) / mem_I))))
    if DEBUG_BLKSIZE: Occblksize = max(1,nOcc//2)
    logger.debug1(eris, 'make_lo_rdm1_1h_k :  nkpts = %d  nocc = %d  nvir = %d  nOcc = %d  '
                  'naux = %d  Occblksize = %d  peak mem = %.2f MB', nkpts, nocc, nvir, nOcc,
                  naux, Occblksize, mem_fix + Occblksize*mem_I)

    dmoo = np.zeros((nko,nko), dtype=COMPLEX) if with_occ else None
    dmvv = np.zeros((nkv,nkv), dtype=COMPLEX) if with_vir else None
    for I0,I1 in lib.prange(0,nOcc,Occblksize):
        nI = I1 - I0
        eIv = eI[I0:I1]
        # IvL[q][ka] = (I a|L), a in k_a, of momentum q = k_a - k_I: (naux_q, nI*nvir)
        IvL = [[lib.einsum('Lia,iI->LIa', Lov[kconserv[ka,q,0],ka],
                           cI[kconserv[ka,q,0],:,I0:I1].conj()).reshape(-1,nI*nvir)
                for ka in range(nkpts)] for q in range(nkpts)]
        A = np.empty((nI,nkpts,nocc,nkpts,nvir,nvir), dtype=COMPLEX)
        B = np.empty((nI,nkpts,nocc,nvir,nkpts,nvir), dtype=COMPLEX)
        for kc in range(nkpts):
            for kj in range(nkpts):
                # A: (I a|j c) with q = k_j - k_c;  B: (I c|j a) with q = k_j - k_a
                ovL = Lov[kj,kc].reshape(-1,nocc*nvir)
                for ka in range(nkpts):
                    num = lib.dot(IvL[kconserv[kj,kc,0]][ka].T, ovL)
                    A[:,kj,:,ka] = num.reshape(nI,nvir,nocc,nvir).transpose(0,2,1,3)
                    num = lib.dot(IvL[kconserv[kj,ka,0]][kc].T, Lov[kj,ka].reshape(-1,nocc*nvir))
                    B[:,kj,:,:,ka] = num.reshape(nI,nvir,nocc,nvir).transpose(0,2,1,3)
                num = ovL = None
                # amplitudes in place (A[:,kj] and B[:,kj] are views)
                denom = lib.direct_sum('I+j-a-c->Ijac', eIv, eo[kj], evall, ev[kc])
                _make_t2(eris, A[:,kj], denom.reshape(A.shape[:1]+A.shape[2:]))
                denom = lib.direct_sum('I+j-c-a->Ijca', eIv, eo[kj], ev[kc], evall)
                _make_t2(eris, B[:,kj], denom.reshape(B.shape[:1]+B.shape[2:]))
                denom = None
            a = A.reshape(nI,nko,nkv,nvir)
            b = B.reshape(nI,nko,nvir,nkv)
            if with_vir:
                X = (2*a - b.transpose(0,1,3,2)).conj()
                dmvv += lib.einsum('Ijac,Ijbc->ab', a, X)
                X = (2*b - a.transpose(0,1,3,2)).conj()
                dmvv += lib.einsum('Ijca,Ijcb->ab', b, X)
            if with_occ:
                X = 2*b.transpose(0,1,3,2) - 4*a
                dmoo += lib.einsum('Kiac,Kjac->ij', a.conj(), X)
            X = a = b = None
        A = B = IvL = None

    # Bloch -> supercell canonical orbitals
    if with_occ:
        dmoo = _check_dm_imag(eris, reduce(np.dot, (eris.Wocc.T.conj(), dmoo, eris.Wocc)))
    if with_vir:
        dmvv = _check_dm_imag(eris, reduce(np.dot, (eris.Wvir.T.conj(), dmvv, eris.Wvir)))
    return dmoo, dmvv

def conj_(a):
    # in-place conjugate
    np.conj(a, out=a)

def _check_dm_imag(eris, dm):
    if eris.dtype_eri == np.float64:
        dmi = abs(dm.imag).max()
        if dmi > 1e-4:
            logger.warn(eris, 'Discard large imag part in DM (%s). '
                        'This may lead to error.', dmi)
        dm = dm.real
    return dm
