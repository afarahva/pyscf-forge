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

r''' Atomic domains for domain-based LNO (DLNO) methods.

    The domains follow the local MP2 of Pinski, Riplinger, Valeev, Neese
    (J. Chem. Phys. 143, 034108 (2015)) and Werner and co-workers, as implemented in
    https://github.com/fishjojo/dlno:

    - Boughton-Pulay (BP) domain of an orbital: the smallest set of atoms (grown by
      population, then by distance) whose AOs reproduce the orbital to a completeness
      `bp_thr`, i.e., 1 - min_a ||phi - sum_{mu in D} a_mu chi_mu||^2 >= bp_thr.
    - Projected atomic orbitals (PAOs): the AOs projected onto the active virtual space.
    - Primary domain of an LMO: its BP domain extended by the BP domains of the PAOs
      centered on its BP-domain atoms.

    Domains are stored as sorted integer arrays of atom indices.
'''

from functools import reduce

import numpy as np
import scipy.linalg

from pyscf import gto, lib
from pyscf.scf.addons import canonical_orth_


def ao_index_by_atom(mol, atmlst, aoslices=None):
    r''' Sorted AO indices of the atoms `atmlst`. '''
    if aoslices is None:
        aoslices = mol.aoslice_by_atom()
    atmlst = np.asarray(atmlst, dtype=int).ravel()
    if atmlst.size == 0:
        return np.zeros(0, dtype=int)
    return np.sort(np.concatenate([np.arange(aoslices[a,2], aoslices[a,3])
                                   for a in atmlst]))

def atom_population(mol, orbs, s1e, aoslices=None):
    r''' Mulliken gross population of each orbital on each atom, (natm, norb). '''
    if aoslices is None:
        aoslices = mol.aoslice_by_atom()
    gop = (orbs.conj() * np.dot(s1e, orbs)).real
    starts = aoslices[:,2]
    return np.add.reduceat(gop, starts, axis=0)

def bp_completeness(s1e, orbs, ao_idx):
    r''' Boughton-Pulay completeness <phi|P_D|phi> of the (normalized) orbitals `orbs`
    in the space of the AOs `ao_idx`, where P_D is the projector onto that AO space.
    '''
    orbs = np.asarray(orbs)
    if orbs.ndim == 1:
        orbs = orbs.reshape(-1,1)
    if len(ao_idx) == 0:
        return np.zeros(orbs.shape[1])
    v = np.dot(s1e[ao_idx], orbs)
    s22 = s1e[np.ix_(ao_idx, ao_idx)]
    try:
        a = scipy.linalg.solve(s22, v, assume_a='pos')
    except np.linalg.LinAlgError:
        a = np.linalg.lstsq(s22, v, rcond=None)[0]
    return lib.einsum('ij,ij->j', a.conj(), v).real

def get_bp_domain(mol, orbs, s1e, bp_thr, dist, q_thr=None, aoslices=None):
    r''' BP domains of the orbitals (columns of `orbs`).

    For each orbital, the atoms with absolute Mulliken population above `q_thr` are taken
    first, then the remaining atoms by increasing distance from the atom of largest
    population, until the BP completeness reaches `bp_thr`. The completeness grows
    monotonically along this list, so the shortest sufficient prefix is found by
    a doubling search followed by bisection.

    Args:
        dist : (natm, natm) array
            Interatomic distances (minimum-image distances for periodic systems).

    Returns:
        list of sorted atom-index arrays.
    '''
    if q_thr is None:
        q_thr = min(0.05, 5*(1-bp_thr))
    if aoslices is None:
        aoslices = mol.aoslice_by_atom()
    orbs = np.asarray(orbs)
    if orbs.ndim == 1:
        orbs = orbs.reshape(-1,1)
    natm = mol.natm
    pops = abs(atom_population(mol, orbs, s1e, aoslices))

    def complete(order, n, orb):
        ao_idx = ao_index_by_atom(mol, order[:n], aoslices)
        return bp_completeness(s1e, orb, ao_idx)[0] >= bp_thr - 1e-10

    domains = []
    for i in range(orbs.shape[1]):
        q = pops[:,i]
        orb = orbs[:,i:i+1]
        first = np.where(q > q_thr)[0]
        first = first[np.argsort(-q[first])]
        center = np.argmax(q)
        rest = np.setdiff1d(np.arange(natm), first)
        rest = rest[np.argsort(dist[center][rest], kind='stable')]
        order = np.concatenate([first, rest]).astype(int)
        n0 = max(len(first), 1)
        if complete(order, n0, orb):
            n = n0
        else:
            # doubling search for a sufficient prefix (all atoms always are), then
            # bisection between the last insufficient (nlo) and sufficient (nhi) prefix
            nlo, step = n0, 1
            nhi = min(natm, n0 + step)
            while nhi < natm and not complete(order, nhi, orb):
                nlo, step = nhi, step*2
                nhi = min(natm, n0 + step)
            while nhi - nlo > 1:
                nmid = (nlo + nhi) // 2
                if complete(order, nmid, orb):
                    nhi = nmid
                else:
                    nlo = nmid
            n = nhi
        domains.append(np.sort(order[:n]))
    return domains

def make_pao(s1e, orbvir, norm_thr=1e-4):
    r''' Normalized PAOs: the AOs projected onto the active virtual space, |C_v><C_v| S.

    Projecting onto the virtual MOs (rather than projecting the occupied MOs out of the
    AOs) keeps the PAOs in the span of the mean-field MOs when near-linearly dependent AO
    combinations were removed from it.

    Returns:
        pao : (nao, npao) array
        ao2pao : (nao,) int array mapping an AO to its PAO (-1 if discarded for having
            a norm below `norm_thr`).
    '''
    nao = s1e.shape[0]
    pao = np.dot(orbvir, np.dot(orbvir.T.conj(), s1e))
    norm = np.sqrt(abs(lib.einsum('ui,ui->i', pao.conj(), np.dot(s1e, pao))))
    keep = np.where(norm > norm_thr)[0]
    ao2pao = np.full(nao, -1, dtype=int)
    ao2pao[keep] = np.arange(keep.size)
    pao = pao[:,keep] / norm[keep]
    return pao, ao2pao

def pao_index_by_atom(mol, ao2pao, atmlst, aoslices=None):
    r''' Indices of the PAOs centered on the atoms `atmlst`. '''
    idx = ao2pao[ao_index_by_atom(mol, atmlst, aoslices)]
    return idx[idx >= 0]

def get_primary_domain(mol, lmo_bp_domain, pao_bp_domain, ao2pao, aoslices=None):
    r''' Primary domain of each LMO: its BP domain and the BP domains of the PAOs centered
    on its BP-domain atoms.
    '''
    out = []
    for atms in lmo_bp_domain:
        pidx = pao_index_by_atom(mol, ao2pao, atms, aoslices)
        dom = reduce(np.union1d, [atms] + [pao_bp_domain[p] for p in pidx])
        out.append(np.asarray(dom, dtype=int))
    return out

def domain_pao(mol, pao, ao2pao, s1e, atmlst, ovlp_atmlst=None, ovlp_thr=None,
               orth_thr=1e-6, aoslices=None):
    r''' Orthonormal virtual orbitals spanning the PAOs centered on `atmlst`.

    The PAOs are canonically orthogonalized (discarding overlap eigenvalues below
    `orth_thr`). If `ovlp_thr` is given, only the combinations with an overlap larger than
    `ovlp_thr` with the AO space of `ovlp_atmlst` (default `atmlst`) are kept, i.e., the
    eigenvectors of the projected overlap <a| P_D |b> with eigenvalues > `ovlp_thr`.
    '''
    pidx = pao_index_by_atom(mol, ao2pao, atmlst, aoslices)
    p = pao[:,pidx]
    if p.shape[1] == 0:
        return p
    x = canonical_orth_(np.dot(p.T.conj(), np.dot(s1e, p)), thr=orth_thr)
    p = np.dot(p, x)
    if ovlp_thr is not None:
        if ovlp_atmlst is None:
            ovlp_atmlst = atmlst
        ao_idx = ao_index_by_atom(mol, ovlp_atmlst, aoslices)
        t = np.dot(s1e[ao_idx], p)
        s22 = s1e[np.ix_(ao_idx, ao_idx)]
        ov = np.dot(t.T.conj(), scipy.linalg.solve(s22, t, assume_a='pos'))
        w, v = scipy.linalg.eigh(ov)
        p = np.dot(p, v[:,w > ovlp_thr])
    return p

def project_mo(orb, s21, s22):
    r''' Least-squares projection of `orb` onto the AO subspace with overlap `s22` and
    cross overlap `s21` = S[sub, :].
    '''
    return scipy.linalg.solve(s22, np.dot(s21, orb), assume_a='pos')

def orth_against(orbocc, orb, s):
    r''' Project the orthonormal orbitals `orbocc` out of `orb`. '''
    return orb - np.dot(orbocc, np.dot(orbocc.T.conj(), np.dot(s, orb)))

def lowdin(orb, s):
    r''' Lowdin orthonormalization of `orb` in the metric `s`. '''
    if orb.shape[1] == 0:
        return orb
    sm = np.dot(orb.T.conj(), np.dot(s, orb))
    e, u = scipy.linalg.eigh(sm)
    return np.dot(orb, np.dot(u*e**-0.5, u.T.conj()))

def canonical(orb, s, thr=1e-6):
    r''' Canonical orthonormalization of `orb` in the metric `s`, dropping overlap
    eigenvalues (of the normalized vectors) below `thr`.
    '''
    if orb.shape[1] == 0:
        return orb
    return np.dot(orb, canonical_orth_(np.dot(orb.T.conj(), np.dot(s, orb)), thr=thr))

def semicanonicalize(orb, fock):
    r''' Eigenvalues and eigenvectors of `fock` in the orthonormal orbitals `orb`. '''
    if orb.shape[1] == 0:
        return np.zeros(0), orb
    f = np.dot(orb.T.conj(), np.dot(fock, orb))
    e, u = scipy.linalg.eigh(f)
    return e, np.dot(orb, u)

def fake_mol_by_atom(mol, atmlst, coords=None):
    r''' A Mole holding the atoms `atmlst` of `mol` with the same basis (at the positions
    `coords` in Bohr if given). The AOs are in the order of `ao_index_by_atom` when
    `atmlst` is sorted. ECPs/pseudopotentials are dropped: the fake molecule is only used
    for AO overlap, multipole and density-fitting integrals.
    '''
    if hasattr(mol, 'to_mol'):   # pbc Cell
        mol = mol.to_mol()
    atmlst = np.asarray(atmlst, dtype=int)
    fake = mol.copy(deep=False)
    if coords is None:
        fake._atom = [mol._atom[a] for a in atmlst]
    else:
        fake._atom = [(mol._atom[a][0], list(coords[k])) for k,a in enumerate(atmlst)]
    fake._ecp = {}
    fake._pseudo = {}
    fake._ecpbas = np.zeros((0,8), dtype=np.int32)
    fake._atm, fake._bas, fake._env = fake.make_env(fake._atom, fake._basis,
                                                    mol._env[:gto.PTR_ENV_START])
    fake._built = True
    return fake

def atom_distance(mol, lattice=None):
    r''' Interatomic distances; minimum-image distances if `lattice` (rows are lattice
    vectors in Bohr) is given.
    '''
    coords = mol.atom_coords()
    d = coords[:,None,:] - coords[None,:,:]
    if lattice is not None:
        d = minimum_image(d, lattice)
    return np.linalg.norm(d, axis=-1)

def minimum_image(d, lattice):
    r''' Minimum-image representatives of the displacement vectors `d` (..., 3). '''
    lattice = np.asarray(lattice)
    shape = d.shape
    d = d.reshape(-1,3)
    frac = np.linalg.solve(lattice.T, d.T).T
    d = d - np.dot(np.round(frac), lattice)
    shifts = np.asarray([(i,j,k) for i in (-1,0,1) for j in (-1,0,1) for k in (-1,0,1)])
    cand = d[:,None,:] + np.dot(shifts, lattice)[None,:,:]
    best = np.argmin(np.linalg.norm(cand, axis=-1), axis=1)
    return cand[np.arange(d.shape[0]), best].reshape(shape)
