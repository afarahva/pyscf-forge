''' MP2 density matrices for the LNOs of KLNO

    Each kind of DF integrals of `pyscf.pbc.lno.klno` has its own routines:
        real supercell orbitals      `*_real`      (`ovLR`, `ovLI`)
        complex supercell orbitals   `*_complex`   (`ovL` of each q)
        k-point Bloch orbitals       `*_k`         (`Lov[ki,ka]`; '1h' and '1p' only)
    and `make_lo_rdm1` and `make_lo_rdm1_k` pick the routine for `lno_type`.

    Args:
        eris :
            The supercell integrals provide `ovL` in the canonical MOs through
                get_occ_blk(i0,i1) -> ([i0:i1]v|L)
                get_vir_blk(a0,a1) -> (o[a0:a1]|L)
                xform_occ(u)       -> einsum('iaL,iI->IaL', ovL, u)
                xform_vir(u)       -> einsum('iaL,aA->iAL', ovL, u)
            (real and imaginary parts for real orbitals; with q as first argument for
            complex orbitals).
        moeocc, moevir : np.ndarray
            Energies of the canonical occupied and virtual MOs of the integrals.
        u : np.ndarray
            Overlap between the canonical and the localized orbitals of the fragment,
            u[i,I] = <i|I> for '1h'/'2h' and u[a,A] = <a|A> for '1p'/'2p'.
        dm_type : str
            '1h'/'1p'/'2p' for occ and '1p'/'1h'/'2h' for vir.
'''
import numpy as np
from functools import reduce

from pyscf import lib
from pyscf.lib import logger
from pyscf import __config__

from pyscf.lno.make_lno_rdm1 import (subspace_eigh, _mp2_rdm1_occblksize, _mp2_rdm1_virblksize,
                                     _make_t2)
from pyscf.pbc.lno.tools import zdotCNtoR

DEBUG_BLKSIZE = getattr(__config__, 'lno_base_make_rdm1_k2s_DEBUG_BLKSIZE', False)
# upper bound (MB) on the blocks of ovL rows read at once when streaming ovL
MAX_ROWBLK_MB = getattr(__config__, 'lno_base_make_rdm1_k2s_max_rowblk_mb', 4000)


def make_lo_rdm1(eris, moeocc, moevir, uocc, uvir, lno_type):
    r''' Occupied and virtual LNO density matrices from the supercell integrals, for
    lno_type = [occ_type, vir_type] (a type that is None is not made). With real orbitals,
    ['1h','1h'] and ['1p','1p'] make both from one set of amplitudes.
    '''
    occ_type, vir_type = lno_type
    if occ_type == vir_type and eris.dtype == np.float64:
        if occ_type == '1h':
            return make_lo_rdm1_1h_real(eris, moeocc, moevir, uocc)
        return make_lo_rdm1_1p_real(eris, moeocc, moevir, uvir)
    dmoo = make_lo_rdm1_occ(eris, moeocc, moevir, uocc, uvir, occ_type) if occ_type else None
    dmvv = make_lo_rdm1_vir(eris, moeocc, moevir, uocc, uvir, vir_type) if vir_type else None
    return dmoo, dmvv

def make_lo_rdm1_occ(eris, moeocc, moevir, uocc, uvir, dm_type):
    # dm_type: (routine for real orbitals, routine for complex orbitals, u)
    routines = {'1h': (make_lo_rdm1_occ_1h_real, make_lo_rdm1_occ_1h_complex, uocc),
                '1p': (make_lo_rdm1_occ_1p_real, make_lo_rdm1_occ_1p_complex, uvir),
                '2p': (make_lo_rdm1_occ_2p_real, make_lo_rdm1_occ_2p_complex, uvir)}
    if dm_type not in routines:
        raise RuntimeError('Requested occ LNO type "%s" is unknown.' % dm_type)
    real_routine, complex_routine, u = routines[dm_type]
    routine = real_routine if eris.dtype == np.float64 else complex_routine
    return _check_dm_imag(eris, routine(eris, moeocc, moevir, u))

def make_lo_rdm1_vir(eris, moeocc, moevir, uocc, uvir, dm_type):
    # dm_type: (routine for real orbitals, routine for complex orbitals, u)
    routines = {'1p': (make_lo_rdm1_vir_1p_real, make_lo_rdm1_vir_1p_complex, uvir),
                '1h': (make_lo_rdm1_vir_1h_real, make_lo_rdm1_vir_1h_complex, uocc),
                '2h': (make_lo_rdm1_vir_2h_real, make_lo_rdm1_vir_2h_complex, uocc)}
    if dm_type not in routines:
        raise RuntimeError('Requested vir LNO type "%s" is unknown.' % dm_type)
    real_routine, complex_routine, u = routines[dm_type]
    routine = real_routine if eris.dtype == np.float64 else complex_routine
    return _check_dm_imag(eris, routine(eris, moeocc, moevir, u))

''' make lo rdm1 for real orbitals
'''
def make_full_rdm1(eris, moeocc, moevir, with_occ=True, with_vir=True):
    r''' Occ-occ and vir-vir blocks of MP2 density matrix

        Math:
            dm(i,j)
                = 2 * \sum_{kab} t2(ikab).conj() * ( 2*t2(jkab) - t2(jkba) )
            dm(a,b)
                = 2 * \sum_{ijc} t2(ijac) * ( 2*t2(ijbc) - t2(ijcb) ).conj()

        These are the '1h' density matrices with every occupied orbital localized (the
        occupied one with the opposite sign).
    '''
    dmoo, dmvv = make_lo_rdm1_1h_real(eris, moeocc, moevir, np.eye(eris.nocc),
                                      with_occ=with_occ, with_vir=with_vir)
    if with_occ:
        dmoo = -dmoo
    return dmoo, dmvv

def make_lo_rdm1_occ_1h_real(eris, moeocc, moevir, u):
    r''' Occupied MP2 density matrix with one localized hole (see `make_lo_rdm1_1h_real`)
    '''
    return make_lo_rdm1_1h_real(eris, moeocc, moevir, u, with_vir=False)[0]

def make_lo_rdm1_vir_1h_real(eris, moeocc, moevir, u):
    r''' Virtual MP2 density matrix with one localized hole. Alone, the blocked algorithm
    makes each amplitude once in two passes over `ovL` too, with less memory than
    `make_lo_rdm1_1h_real`, which is used when both density matrices are needed.
    '''
    return _make_lo_rdm1_vir_1h_real_blocked(eris, moeocc, moevir, u)

def make_lo_rdm1_1h_real(eris, moeocc, moevir, u, with_occ=True, with_vir=True):
    r''' Occupied and virtual MP2 density matrices with one localized hole

        Math:
            T(jcKa) = t2(Kjac),  K the semicanonical localized occupied orbitals
            dm(i,j) = \sum_{aKb} -4 T(iaKb) T(jaKb) + 2 T(iaKb) T(jbKa)
            dm(a,b) = \sum_{jcK} 2 T(jcKa) T(jcKb) - T(jcKa) T(jbKc) - T(jaKc) T(jcKb)
                                 + 2 T(jaKc) T(jbKc)

        For a block of K, all amplitudes T(j c K a) (nocc nvir^2 per K) are kept in memory.
        They are made in one pass over `ovL` in blocks of occupied rows j (`get_occ_blk`),
        after the pass of `xform_occ`, so each amplitude is made once and each K block reads
        `ovL` twice, contiguously, for both density matrices. (The blocked algorithm
        `_make_lo_rdm1_occ_1h_real_blocked`/`_make_lo_rdm1_vir_1h_real_blocked`, which needs
        less memory but rebuilds the amplitudes of the occupied DM for every occupied block,
        is used if one K of amplitudes does not fit.)

        Args:
            eris : ERI object
                Provides `ovL` in the canonical MOs.
            u : np.ndarray
                Overlap between the canonical and localized occupied orbitals, i.e.,
                    u(i,i') = <i|i'>

        Returns:
            dm_occ, dm_vir (None if not requested).
    '''
    assert( with_occ or with_vir )
    nocc, nvir, naux = eris.nocc, eris.nvir, eris.Naux_ibz
    REAL = np.float64
    dsize = 8
    assert(u.dtype == REAL)
    nOcc = u.shape[1]

    # Memory (in numbers): per K the amplitudes T (nocc nvir^2) and KvL (2 nvir naux); per
    # occupied row of a block the integrals (2 nvir naux) and ~4 amplitude-sized temporaries.
    mem_avail = eris.max_memory - lib.current_memory()[0]
    M = mem_avail * 0.7 * 1e6/dsize
    mem_K = nocc*nvir**2 + 2*nvir*naux
    Occblksize = min(nOcc, int(np.floor(0.6*M / mem_K)))
    if DEBUG_BLKSIZE: Occblksize = min(Occblksize, max(1,nOcc//2))
    if Occblksize < 1:
        logger.debug1(eris, 'make_lo_rdm1_1h :  not enough memory for the amplitudes of one '
                      'localized occupied (%.2f MB); using the blocked algorithm',
                      mem_K*dsize/1e6)
        dmoo = dmvv = None
        if with_occ:
            dmoo = _make_lo_rdm1_occ_1h_real_blocked(eris, moeocc, moevir, u)
        if with_vir:
            dmvv = _make_lo_rdm1_vir_1h_real_blocked(eris, moeocc, moevir, u)
        return dmoo, dmvv
    mem_row = 2*nvir*naux + 4*nvir*Occblksize*nvir
    occblksize = int(np.floor((M - Occblksize*mem_K) / mem_row))
    occblksize = max(1, min(nocc, occblksize, int(MAX_ROWBLK_MB*1e6/dsize / mem_row)))
    if DEBUG_BLKSIZE: occblksize = max(1,nocc//2)
    logger.debug1(eris, 'make_lo_rdm1_1h :  nocc = %d  nvir = %d  nOcc = %d  naux = %d  '
                  'Occblksize = %d  occblksize = %d  peak mem = %.2f MB',
                  nocc, nvir, nOcc, naux, Occblksize, occblksize,
                  (Occblksize*mem_K + occblksize*mem_row)*dsize/1e6)

    moeOcc, u = subspace_eigh(np.diag(moeocc), u)
    eov = moeocc[:,None] - moevir
    eOv = moeOcc[:,None] - moevir

    dmoo = np.zeros((nocc,nocc), dtype=REAL) if with_occ else None
    dmvv = np.zeros((nvir,nvir), dtype=REAL) if with_vir else None
    for K0,K1 in lib.prange(0,nOcc,Occblksize):
        nK = K1 - K0
        KvLR, KvLI = eris.xform_occ(u[:,K0:K1])
        KvLR = KvLR.reshape(nK*nvir,naux)
        KvLI = KvLI.reshape(nK*nvir,naux)
        eKv = eOv[K0:K1]
        # T(j c K a) for all j, c, a: one pass over the rows j of ovL
        T = np.empty((nocc,nvir,nK,nvir), dtype=REAL)
        for j0,j1 in lib.prange(0,nocc,occblksize):
            jcLR, jcLI = eris.get_occ_blk(j0,j1)
            t2 = zdotCNtoR(jcLR.reshape(-1,naux), jcLI.reshape(-1,naux), KvLR.T, KvLI.T)
            jcLR = jcLI = None
            ejcKa = lib.direct_sum('jc+Ka->jcKa', eov[j0:j1], eKv)
            T[j0:j1] = _make_t2(eris, t2.reshape(ejcKa.shape), ejcKa)
            t2 = ejcKa = None
        KvLR = KvLI = None
        T2 = T.reshape(nocc,-1)
        if with_occ:
            dmoo -= 4 * lib.dot(T2, T2.T)
        # the remaining contractions in blocks of rows j
        for j0,j1 in lib.prange(0,nocc,occblksize):
            Tj = T[j0:j1]
            if with_occ:
                Tjx = np.ascontiguousarray(Tj.transpose(0,3,2,1)).reshape(j1-j0,-1)
                dmoo[:,j0:j1] += 2 * lib.dot(T2, Tjx.T)
                Tjx = None
            if with_vir:
                dmvv += 2 * lib.einsum('jcKa,jcKb->ab', Tj, Tj)
                tmp = lib.einsum('jcKa,jbKc->ab', Tj, Tj)
                dmvv -= tmp
                dmvv -= tmp.T
                dmvv += 2 * lib.einsum('jaKc,jbKc->ab', Tj, Tj)
                tmp = None
            Tj = None
        T = T2 = None

    return dmoo, dmvv

def _make_lo_rdm1_occ_1h_real_blocked(eris, moeocc, moevir, u):
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
    r''' Occupied MP2 density matrix with one localized particle (see `make_lo_rdm1_1p_real`)
    '''
    return make_lo_rdm1_1p_real(eris, moeocc, moevir, u, with_vir=False)[0]

def make_lo_rdm1_vir_1p_real(eris, moeocc, moevir, u):
    r''' Virtual MP2 density matrix with one localized particle (see `make_lo_rdm1_1p_real`)
    '''
    return make_lo_rdm1_1p_real(eris, moeocc, moevir, u, with_occ=False)[1]

def make_lo_rdm1_1p_real(eris, moeocc, moevir, u, with_occ=True, with_vir=True):
    r''' Occupied and virtual MP2 density matrices with one localized particle

        Math:
            X(iAkb) = t2(ikAb),  A the semicanonical localized virtuals
            dm(i,j) = \sum_{Akb} -2 X(iAkb) X(jAkb) + X(iAkb) X(kAjb) + X(kAib) X(jAkb)
                                 - 2 X(kAib) X(kAjb)
            dm(a,b) = \sum_{iAk} 4 X(iAka) X(iAkb) - 2 X(iAka) X(kAib)

        For a block of A, all amplitudes X(i A k b) (nocc^2 nvir per A) are kept in memory.
        They are made in one pass over `ovL` in blocks of occupied rows (`get_occ_blk`),
        after the pass of `xform_vir`, so each amplitude is made once and each A block
        reads `ovL` twice, contiguously, for both density matrices. (The blocked algorithm
        `_make_lo_rdm1_occ_1p_real_blocked`/`_make_lo_rdm1_vir_1p_real_blocked`, which needs
        less memory, rebuilds the amplitudes for every virtual block and reads `ovL` by
        virtual columns; it is used only if one A of amplitudes does not fit.)

        Args:
            eris : ERI object
                Provides `ovL` in the canonical MOs.
            u : np.ndarray
                Overlap between the canonical and localized virtual orbitals, i.e.,
                    u(a,a') = <a|a'>

        Returns:
            dm_occ, dm_vir (None if not requested).
    '''
    assert( with_occ or with_vir )
    nocc, nvir, naux = eris.nocc, eris.nvir, eris.Naux_ibz
    REAL = np.float64
    dsize = 8
    assert(u.dtype == REAL)
    nVir = u.shape[1]

    # Memory (in numbers): per A the amplitudes X (nocc^2 nvir) and oAL (2 nocc naux); per
    # occupied row of a block the integrals (2 nvir naux) and ~4 amplitude-sized temporaries.
    # The row blocks are also bounded so that streaming `ovL` does not take the memory the OS
    # page cache could use to hold it.
    mem_avail = eris.max_memory - lib.current_memory()[0]
    M = mem_avail * 0.7 * 1e6/dsize
    mem_A = nocc**2*nvir + 2*nocc*naux
    Virblksize = min(nVir, int(np.floor(0.6*M / mem_A)))
    if DEBUG_BLKSIZE: Virblksize = min(Virblksize, max(1,nVir//2))
    if Virblksize < 1:
        logger.debug1(eris, 'make_lo_rdm1_1p :  not enough memory for the amplitudes of one '
                      'localized virtual (%.2f MB); using the blocked algorithm',
                      mem_A*dsize/1e6)
        dmoo = dmvv = None
        if with_occ:
            dmoo = _make_lo_rdm1_occ_1p_real_blocked(eris, moeocc, moevir, u)
        if with_vir:
            dmvv = _make_lo_rdm1_vir_1p_real_blocked(eris, moeocc, moevir, u)
        return dmoo, dmvv
    mem_row = 2*nvir*naux + 4*nocc*Virblksize*nvir
    occblksize = int(np.floor((M - Virblksize*mem_A) / mem_row))
    occblksize = max(1, min(nocc, occblksize, int(MAX_ROWBLK_MB*1e6/dsize / mem_row)))
    if DEBUG_BLKSIZE: occblksize = max(1,nocc//2)
    logger.debug1(eris, 'make_lo_rdm1_1p :  nocc = %d  nvir = %d  nVir = %d  naux = %d  '
                  'Virblksize = %d  occblksize = %d  peak mem = %.2f MB',
                  nocc, nvir, nVir, naux, Virblksize, occblksize,
                  (Virblksize*mem_A + occblksize*mem_row)*dsize/1e6)

    moeVir, u = subspace_eigh(np.diag(moevir), u)
    eov = moeocc[:,None] - moevir
    eoV = moeocc[:,None] - moeVir

    dmoo = np.zeros((nocc,nocc), dtype=REAL) if with_occ else None
    dmvv = np.zeros((nvir,nvir), dtype=REAL) if with_vir else None
    for A0,A1 in lib.prange(0,nVir,Virblksize):
        nA = A1 - A0
        oALR, oALI = eris.xform_vir(u[:,A0:A1])
        oALR = oALR.reshape(nocc*nA,naux)
        oALI = oALI.reshape(nocc*nA,naux)
        eoA = eoV[:,A0:A1]
        # X(i A k b) for all k, b: one pass over the rows k of ovL
        X = np.empty((nocc,nA,nocc,nvir), dtype=REAL)
        for k0,k1 in lib.prange(0,nocc,occblksize):
            kbLR, kbLI = eris.get_occ_blk(k0,k1)
            t2 = zdotCNtoR(oALR, oALI, kbLR.reshape(-1,naux).T, kbLI.reshape(-1,naux).T)
            kbLR = kbLI = None
            eoAkb = lib.direct_sum('iA+kb->iAkb', eoA, eov[k0:k1])
            X[:,:,k0:k1] = _make_t2(eris, t2.reshape(eoAkb.shape), eoAkb)
            t2 = eoAkb = None
        oALR = oALI = None
        # contractions, in blocks of the third index k: Xs = X(i A k b) and Xr = X(k A i b)
        for k0,k1 in lib.prange(0,nocc,occblksize):
            Xs = np.ascontiguousarray(X[:,:,k0:k1])
            Xr = X[k0:k1]
            if with_occ:
                dmoo -= 2 * lib.einsum('iAkb,jAkb->ij', Xs, Xs)
                tmp = lib.einsum('iAkb,kAjb->ij', Xs, Xr)
                dmoo += tmp
                dmoo += tmp.T
                dmoo -= 2 * lib.einsum('kAib,kAjb->ij', Xr, Xr)
                tmp = None
            if with_vir:
                dmvv += 4 * lib.einsum('iAka,iAkb->ab', Xs, Xs)
                dmvv -= 2 * lib.einsum('iAka,kAib->ab', Xs, Xr)
            Xs = Xr = None
        X = None

    return dmoo, dmvv

def _make_lo_rdm1_occ_1p_real_blocked(eris, moeocc, moevir, u):
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

def _make_lo_rdm1_vir_1p_real_blocked(eris, moeocc, moevir, u):
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
    # t2oAoa/t2oAob are (nocc, <=virblksize of nVir, nocc, <=virblksize of nvir)
    bufsize = min(nVir,virblksize)*min(nvir,virblksize)*nocc**2
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

def _make_lo_rdm1_vir_1h_real_blocked(eris, moeocc, moevir, u):
    r''' Virtual MP2 density matrix with one localized hole

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
def make_lo_rdm1_k(eris, uocc, uvir, lno_type):
    r''' Occupied and virtual '1h'/'1p' LNO density matrices from the k-point integrals,
    for lno_type = [occ_type, vir_type] (a type that is None is not made). The same type
    for both makes both from one set of amplitudes.
    '''
    def make(dm_type, with_occ, with_vir):
        if dm_type == '1h':
            return make_lo_rdm1_1h_k(eris, uocc, with_occ=with_occ, with_vir=with_vir)
        return make_lo_rdm1_1p_k(eris, uvir, with_occ=with_occ, with_vir=with_vir)
    occ_type, vir_type = lno_type
    if occ_type == vir_type:
        return make(occ_type, True, True)
    dmoo = make(occ_type, True, False)[0] if occ_type else None
    dmvv = make(vir_type, False, True)[1] if vir_type else None
    return dmoo, dmvv

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
        the integrals of one momentum transfer q = k_a - k_I = k_j - k_c only.

        Both density matrices are sums over c, so the amplitudes are made for a block of I
        and one k_c at a time, as the slabs
            t_ac = t2(I j a c),  t_ca = t2(I j c a)     (c in k_c; j, a in all k)
        which hold every amplitude the density matrices need for those c. Each amplitude is
        thus made twice (in t_ac of k_c and in t_ca of k_a), but no amplitude array is
        larger than (nI, Nk*nocc, Nk*nvir, nvir).

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

    # determine Occblksize: per I, the (I a|L) of all q and ~6 slabs (t_ac, t_ca,
    # temporaries); fixed, the density matrices and jaL
    mem_avail = eris.max_memory - lib.current_memory()[0]
    mem_fix = (nko**2*with_occ + nkv**2*with_vir + nkpts*naux*nko*nvir) * dsize/1e6
    mem_I = (nkpts*naux*nkv + 6*nko*nkv*nvir) * dsize/1e6
    Occblksize = max(1, min(nOcc, int(np.floor((mem_avail*0.7 - mem_fix) / mem_I))))
    if DEBUG_BLKSIZE: Occblksize = max(1,nOcc//2)
    logger.debug1(eris, 'make_lo_rdm1_1h_k :  nkpts = %d  nocc = %d  nvir = %d  nOcc = %d  '
                  'naux = %d  Occblksize = %d  peak mem = %.2f MB', nkpts, nocc, nvir, nOcc,
                  naux, Occblksize, mem_fix + Occblksize*mem_I)

    # jaL[q] = (j a|L) of all k_j, with k_a = k_j - q: (naux_q, Nk, nocc, nvir)
    kjs = np.arange(nkpts)
    jaL = [np.stack([Lov[kj,kconserv[kj,q,0]] for kj in kjs], axis=1) for q in range(nkpts)]

    dmoo = np.zeros((nko,nko), dtype=COMPLEX) if with_occ else None
    dmvv = np.zeros((nkv,nkv), dtype=COMPLEX) if with_vir else None
    eov, evv = eo.ravel(), ev.ravel()
    for I0,I1 in lib.prange(0,nOcc,Occblksize):
        nI = I1 - I0
        eIv = eI[I0:I1]
        # IaL[q] = (I a|L) of all k_a, with k_I = k_a - q: (naux_q, Nk, nI, nvir)
        IaL = [np.stack([lib.einsum('Lia,iI->LIa', Lov[kconserv[ka,q,0],ka],
                                    cI[kconserv[ka,q,0],:,I0:I1].conj())
                         for ka in range(nkpts)], axis=1) for q in range(nkpts)]
        t_ac = np.empty((nI,nkpts,nocc,nkpts,nvir,nvir), dtype=COMPLEX)
        t_ca = np.empty((nI,nkpts,nocc,nvir,nkpts,nvir), dtype=COMPLEX)
        for kc in range(nkpts):
            # (I a|j c): q = k_j - k_c, one product per k_j for all k_a
            for kj in range(nkpts):
                t_ac[:,kj] = lib.einsum('LkIa,Ljc->Ijkac', IaL[kconserv[kj,kc,0]], Lov[kj,kc])
            # (I c|j a): q = k_j - k_a, one product per q for all k_j (and k_a = k_j - q)
            for q in range(nkpts):
                t_ca[:,kjs,:,:,kconserv[kjs,q,0]] = lib.einsum('LIc,Lkja->kIjca',
                                                               IaL[q][:,kc], jaL[q])
            # amplitudes in place
            tac = _make_t2(eris, t_ac.reshape(nI,nko,nkv,nvir),
                           lib.direct_sum('I+j-a-c->Ijac', eIv, eov, evv, ev[kc]))
            tca = _make_t2(eris, t_ca.reshape(nI,nko,nvir,nkv),
                           lib.direct_sum('I+j-c-a->Ijca', eIv, eov, ev[kc], evv))
            if with_vir:
                x = 2*tac - tca.transpose(0,1,3,2)
                dmvv += lib.einsum('Ijac,Ijbc->ab', tac, x.conj())
                x = 2*tca - tac.transpose(0,1,3,2)
                dmvv += lib.einsum('Ijca,Ijcb->ab', tca, x.conj())
            if with_occ:
                x = 2*tca.transpose(0,1,3,2) - 4*tac
                dmoo += lib.einsum('Kiac,Kjac->ij', tac.conj(), x)
            x = tac = tca = None
        t_ac = t_ca = IaL = None
    jaL = None

    # Bloch -> supercell canonical orbitals
    if with_occ:
        dmoo = _check_dm_imag(eris, reduce(np.dot, (eris.Wocc.T.conj(), dmoo, eris.Wocc)))
    if with_vir:
        dmvv = _check_dm_imag(eris, reduce(np.dot, (eris.Wvir.T.conj(), dmvv, eris.Wvir)))
    return dmoo, dmvv

def make_lo_rdm1_1p_k(eris, u, with_occ=True, with_vir=True):
    r''' Occupied and virtual MP2 density matrices with one localized particle, from the
    k-point DF integrals of the Bloch orbitals (`pyscf.pbc.lno.klno._KLNODFKERIS`). This is
    `make_lo_rdm1_1h_k` with the roles of occupied and virtual orbitals exchanged.

        Math:
            dm(i,j) = \sum_{kAb} t2(ikAb).conj() * ( t2(kjAb) - 2*t2(jkAb) )
                               + t2(kiAb).conj() * ( t2(jkAb) - 2*t2(kjAb) )
            dm(a,b) = \sum_{ikA} t2(ikAa) * ( 4*t2(ikAb) - 2*t2(kiAb) ).conj()
            t2(ikAb) = (iA|kb).conj() / (e_i + e_k - e_A - e_b)

        as `make_lo_rdm1_1p_real`, with i, k, b Bloch orbitals and
        A = \sum_{ka} |k a> C(k a,A) the semicanonical fragment virtuals. Momentum
        conservation fixes k_A = k_i + k_k - k_b, so each block of (iA|kb) needs the integrals
        of one momentum transfer q = k_A - k_i = k_k - k_b only.

        Both density matrices are sums over k, so the amplitudes are made for a block of A
        and one k_k at a time, as the slabs
            t_ik = t2(i k A b),  t_ki = t2(k i A b)     (k in k_k; i, b in all k)
        which hold every amplitude the density matrices need for those k. Each amplitude is
        thus made twice (in t_ik of k_k and in t_ki of k_i), but no amplitude array is
        larger than (nA, Nk*nocc, nocc, Nk*nvir).

        Args:
            eris : _KLNODFKERIS
                Provides the k-point integrals `Lov`, Bloch orbital energies
                `mo_energy_occ`/`mo_energy_vir` (Nk,nocc)/(Nk,nvir), and the overlaps
                `Wocc`/`Wvir` between the Bloch and the supercell canonical orbitals.
            u : np.ndarray
                Overlap between the supercell canonical and localized virtual orbitals,
                i.e., u(a,a') = <a|a'>

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

    # semicanonical fragment virtuals in the Bloch basis
    eA, cA = subspace_eigh(np.diag(ev.ravel()), np.dot(eris.Wvir, u))
    cA = cA.reshape(nkpts,nvir,-1)
    nVir = eA.size

    # determine Virblksize: per A, the (i A|L) of all q and ~7 slabs (t_ik, t_ki,
    # denominators, temporaries); fixed, the density matrices and ibL
    mem_avail = eris.max_memory - lib.current_memory()[0]
    mem_fix = (nko**2*with_occ + nkv**2*with_vir + nkpts*naux*nko*nvir) * dsize/1e6
    mem_A = (nkpts*naux*nko + 7*nko*nocc*nkv) * dsize/1e6
    Virblksize = max(1, min(nVir, int(np.floor((mem_avail*0.7 - mem_fix) / mem_A))))
    if DEBUG_BLKSIZE: Virblksize = max(1,nVir//2)
    logger.debug1(eris, 'make_lo_rdm1_1p_k :  nkpts = %d  nocc = %d  nvir = %d  nVir = %d  '
                  'naux = %d  Virblksize = %d  peak mem = %.2f MB', nkpts, nocc, nvir, nVir,
                  naux, Virblksize, mem_fix + Virblksize*mem_A)

    # ibL[q] = (i b|L) of all k_i, with k_b = k_i - q: (naux_q, Nk, nocc, nvir)
    kis = np.arange(nkpts)
    ibL = [np.stack([Lov[ki,kconserv[ki,q,0]] for ki in kis], axis=1) for q in range(nkpts)]

    dmoo = np.zeros((nko,nko), dtype=COMPLEX) if with_occ else None
    dmvv = np.zeros((nkv,nkv), dtype=COMPLEX) if with_vir else None
    eov, evv = eo.ravel(), ev.ravel()
    for A0,A1 in lib.prange(0,nVir,Virblksize):
        nA = A1 - A0
        eAv = eA[A0:A1]
        # iAL[q] = (i A|L) of all k_i, with k_A = k_i + q: (naux_q, Nk, nocc, nA)
        iAL = [np.stack([lib.einsum('Lia,aA->LiA', Lov[ki,kconserv[ki,0,q]],
                                    cA[kconserv[ki,0,q],:,A0:A1])
                         for ki in kis], axis=1) for q in range(nkpts)]
        t_ik = np.empty((nA,nkpts,nocc,nocc,nkpts,nvir), dtype=COMPLEX)
        t_ki = np.empty((nA,nocc,nkpts,nocc,nkpts,nvir), dtype=COMPLEX)
        for kk in range(nkpts):
            # (i A|k b): q = k_k - k_b, one product per k_b for all k_i
            for kb in range(nkpts):
                t_ik[:,:,:,:,kb] = lib.einsum('LmiA,Lkb->Amikb', iAL[kconserv[kk,kb,0]],
                                              Lov[kk,kb])
            # (k A|i b): q = k_i - k_b, one product per q for all k_i (and k_b = k_i - q)
            for q in range(nkpts):
                t_ki[:,:,kis,:,kconserv[kis,q,0]] = lib.einsum('LkA,Lmib->mAkib',
                                                               iAL[q][:,kk], ibL[q])
            # amplitudes in place
            tik = _make_t2(eris, t_ik.reshape(nA,nko,nocc,nkv),
                           lib.direct_sum('i+k-A-b->Aikb', eov, eo[kk], eAv, evv))
            tki = _make_t2(eris, t_ki.reshape(nA,nocc,nko,nkv),
                           lib.direct_sum('k+i-A-b->Akib', eo[kk], eov, eAv, evv))
            if with_occ:
                x = tki.transpose(0,2,1,3) - 2*tik
                dmoo += lib.einsum('Aikb,Ajkb->ij', tik.conj(), x)
                x = tik.transpose(0,2,1,3) - 2*tki
                dmoo += lib.einsum('Akib,Akjb->ij', tki.conj(), x)
            if with_vir:
                x = 4*tik - 2*tki.transpose(0,2,1,3)
                dmvv += lib.einsum('Aika,Aikb->ab', tik, x.conj())
            x = tik = tki = None
        t_ik = t_ki = iAL = None
    ibL = None

    # Bloch -> supercell canonical orbitals
    if with_occ:
        dmoo = _check_dm_imag(eris, reduce(np.dot, (eris.Wocc.T.conj(), dmoo, eris.Wocc)))
    if with_vir:
        dmvv = _check_dm_imag(eris, reduce(np.dot, (eris.Wvir.T.conj(), dmvv, eris.Wvir)))
    return dmoo, dmvv

def _check_dm_imag(eris, dm):
    if eris.dtype_eri == np.float64:
        dmi = abs(dm.imag).max()
        if dmi > 1e-4:
            logger.warn(eris, 'Discard large imag part in DM (%s). '
                        'This may lead to error.', dmi)
        dm = dm.real
    return dm
