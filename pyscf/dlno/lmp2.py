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
# Author: Ardavan Farahvash
#

r''' Fragment MP2 energies in a domain (local MP2).

    The fragment energy uses the same LO projector as the LNO fragment energies
    (`pyscf.lno.lnoccsd.get_fragment_energy`): with u[i,I] = <i|I> between the
    (semicanonical) domain occupied orbitals i and the fragment LOs I,

        E_frag = sum_I sum_{ik} u[i,I] u[k,I] sum_{jab} t2[i,j,a,b] (2 (ka|jb) - (kb|ja))

    so that the fragment energies of a complete set of LOs add up to the MP2 energy of the
    domain orbitals. Only the integrals of the LOs, (Ia|jb), and the amplitudes contracted
    with u are formed, in blocks of the occupied orbitals.

    `eris` is an LNO ERI object (`pyscf.lno.lno._LNODFINCOREERIS` and friends, including
    the k-point ones with real and imaginary DF integrals).
'''

import numpy as np
from pyscf import lib
from pyscf.lib import logger

einsum = lib.einsum


def _naux(eris):
    return getattr(eris, 'Naux_ibz', None) or eris.naux

def _ovov(a, b):
    r''' (ia|jb) of the DF integrals a = (i,a,L) and b = (j,b,L) (or their real and
    imaginary parts, (ia|jb) = Re sum_L a[i,a,L]^* b[j,b,L]).
    '''
    if isinstance(a, tuple):
        ni, nv = a[0].shape[:2]
        nj, nw = b[0].shape[:2]
        out = lib.ddot(a[0].reshape(ni*nv,-1), b[0].reshape(nj*nw,-1).T)
        out = lib.ddot(a[1].reshape(ni*nv,-1), b[1].reshape(nj*nw,-1).T, 1, out, 1)
    else:
        ni, nv = a.shape[:2]
        nj, nw = b.shape[:2]
        out = np.dot(a.reshape(ni*nv,-1).conj(), b.reshape(nj*nw,-1).T)
        if np.iscomplexobj(out):
            out = out.real
    return out.reshape(ni,nv,nj,nw)

def _occblksize(nocc, nvir, nlo, mem_avail, nblk_arr=3):
    r''' Occupied block size for arrays of size (blk, nvir, blk, nvir). '''
    M = max(mem_avail, 100) * 0.6e6 / 8
    fixed = 2 * nlo * nvir * nocc * nvir
    blk = int(np.sqrt(max(M - fixed, nvir**2) / (nblk_arr * nvir**2)))
    return max(1, min(nocc, blk))

def get_frag_mp2(eris, moeocc, moevir, u, max_memory=None, verbose=None):
    r''' Restricted fragment MP2 energy (see module doc).

    Args:
        moeocc, moevir : orbital energies of the canonical (semicanonical) MOs of `eris`.
        u : (nocc, nlo) array, <i|I>.

    Returns:
        The fragment energy tagged with its same-spin and opposite-spin components
        (`spin_comp`), as in `get_fragment_energy`.
    '''
    nocc, nvir = eris.nocc, eris.nvir
    nlo = u.shape[1]
    if nocc == 0 or nvir == 0 or nlo == 0:
        return lib.tag_array(0., spin_comp=np.zeros(2))
    if max_memory is None:
        max_memory = eris.max_memory
    u = np.asarray(u)
    mem_avail = max_memory - lib.current_memory()[0]
    blk = _occblksize(nocc, nvir, nlo, mem_avail)

    eov = moeocc[:,None] - moevir
    LvL = eris.xform_occ(u)     # (I a | L)
    ed = ex = 0.
    for j0, j1 in lib.prange(0, nocc, blk):
        jvL = eris.get_occ_blk(j0, j1)
        G = _ovov(LvL, jvL)                     # (I a | j b)
        T = np.zeros_like(G)                    # sum_i u[i,I] t2[i,a,j,b]
        ejv = eov[j0:j1]
        for i0, i1 in lib.prange(0, nocc, blk):
            ivL = jvL if (i0, i1) == (j0, j1) else eris.get_occ_blk(i0, i1)
            t2 = _ovov(ivL, jvL)
            t2 /= lib.direct_sum('ia+jb->iajb', eov[i0:i1], ejv)
            T += einsum('iI,iajb->Iajb', u[i0:i1].conj(), t2)
            t2 = ivL = None
        ed += 2 * einsum('Iajb,Iajb->', T, G).real
        ex -= einsum('Iajb,Ibja->', T, G).real
        jvL = G = T = None
    ess = ed*0.5 + ex
    eos = ed*0.5
    return lib.tag_array(ess+eos, spin_comp=np.array((ess, eos)))

def get_frag_ump2(eris, moeocc, moevir, u, max_memory=None, verbose=None):
    r''' Unrestricted fragment MP2 energy with the projector of
    `pyscf.lno.ulnoccsd.get_fragment_energy`.

    Args:
        eris : `pyscf.lno.ulno._ULNO_DF_ERIs` (alpha `ovL`, beta `OVL`).
        moeocc, moevir : per-spin orbital energies.
        u : per-spin (nocc, nlo) arrays, <i|I>.
    '''
    if max_memory is None:
        max_memory = eris.max_memory
    nocc, nvir = eris.nocc, eris.nvir
    spins = ('a', 'b')
    eov = [moeocc[s][:,None] - moevir[s] for s in range(2)]
    e = 0.
    for s in range(2):
        nlo = u[s].shape[1]
        if nlo == 0 or nocc[s] == 0 or nvir[s] == 0:
            continue
        LvL = eris.xform_occ(u[s], spin=spins[s])
        for t in range(2):     # spin of the second electron
            if nocc[t] == 0 or nvir[t] == 0:
                continue
            mem_avail = max_memory - lib.current_memory()[0]
            blk = _occblksize(max(nocc), max(nvir), nlo, mem_avail)
            for j0, j1 in lib.prange(0, nocc[t], blk):
                jvL = eris.get_occ_blk(j0, j1, spin=spins[t])
                G = _ovov(LvL, jvL)            # (I a | j b)
                T = np.zeros_like(G)
                ejv = eov[t][j0:j1]
                for i0, i1 in lib.prange(0, nocc[s], blk):
                    ivL = eris.get_occ_blk(i0, i1, spin=spins[s])
                    g = _ovov(ivL, jvL)
                    if s == t:     # antisymmetrized same-spin amplitudes
                        g = g - g.transpose(0,3,2,1)
                    g /= lib.direct_sum('ia+jb->iajb', eov[s][i0:i1], ejv)
                    T += einsum('iI,iajb->Iajb', u[s][i0:i1].conj(), g)
                    g = ivL = None
                if s == t:
                    # 1/4 sum t2(ijab) [(Ia|jb) - (Ib|ja)]
                    e += 0.25 * (einsum('Iajb,Iajb->', T, G) -
                                 einsum('Iajb,Ibja->', T, G)).real
                else:
                    e += 0.5 * einsum('Iajb,Iajb->', T, G).real
                jvL = G = T = None
        LvL = None
    return lib.tag_array(e, spin_comp=np.zeros(2))
