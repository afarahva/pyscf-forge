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

r''' Multipole approximation to the MP2 pair energies of LMOs (for pair screening).

    The direct (Coulomb) part of the MP2 pair energy of two distant LMOs i and j,

        E_ij = - c sum_{ab} (ai|bj)^2 / (e_a - e_i + e_b - e_j),

    with the exchange integral neglected and (ai|bj) expanded in multipoles of the
    transition densities a*i and b*j about the LMO centroids up to `order` 2 (dipole-
    dipole), 3 (+ dipole-quadrupole) or 4 (+ dipole-octupole and quadrupole-quadrupole),
    following https://github.com/fishjojo/dlno. Here a (b) are semicanonical PAOs of the
    primary domain of i (j). The prefactor c is 4 for doubly occupied spatial orbitals and
    1 for spin orbitals, so that the energy of the pairs is 1/2 sum_{i != j} E_ij.
'''

import numpy as np
from pyscf import lib

einsum = lib.einsum


def dipole_op(mol):
    nao = mol.nao
    return mol.intor('int1e_r').reshape(3,nao,nao)

def quadrupole_op(mol, origin):
    nao = mol.nao
    with mol.with_common_origin(origin):
        rr = mol.intor('int1e_rr').reshape(3,3,nao,nao)
    r2 = np.trace(rr)
    rr *= 3
    for x in range(3):
        rr[x,x] -= r2
    rr *= .5
    return rr

def octupole_op(mol, origin):
    nao = mol.nao
    with mol.with_common_origin(origin):
        rrr = mol.intor('int1e_rrr').reshape(3,3,3,nao,nao)
    r2r_0 = np.trace(rrr, axis1=1, axis2=2)
    r2r_1 = np.trace(rrr, axis1=2, axis2=0)
    r2r_2 = np.trace(rrr, axis1=0, axis2=1)
    rrr *= 5
    for x in range(3):
        rrr[:,x,x] -= r2r_0
        rrr[x,:,x] -= r2r_1
        rrr[x,x,:] -= r2r_2
    rrr *= .5
    return rrr


class LMOMoments:
    r''' Centroid and transition multipole moments of an LMO i to its virtual orbitals a.

    Args:
        mol : Mole
            A molecule holding the AOs in which `orbocc` and `orbvir` are expanded.
        e_occ : float
        orbocc : (nao,) array
        e_vir : (nvir,) array
        orbvir : (nao, nvir) array
    '''
    def __init__(self, mol, e_occ, orbocc, e_vir, orbvir, order=4):
        orbocc = np.asarray(orbocc).ravel()
        d = dipole_op(mol)
        self.center = einsum('u,xuv,v->x', orbocc.conj(), d, orbocc).real
        self.e_vo = np.asarray(e_vir) - e_occ
        self.mu = einsum('ua,xuv,v->xa', orbvir.conj(), d, orbocc).real
        self.theta = self.omega = None
        if order > 2:
            q = quadrupole_op(mol, self.center)
            self.theta = einsum('ua,xyuv,v->xya', orbvir.conj(), q, orbocc).real
        if order > 3:
            o = octupole_op(mol, self.center)
            self.omega = einsum('ua,xyzuv,v->xyza', orbvir.conj(), o, orbocc).real
        self.order = order


def pair_energy(mi, mj, rvec, prefac=4.):
    r''' Multipole MP2 pair energy E_ij of two LMOs with moments `mi`, `mj` whose
    centroids are separated by `rvec` = R_j - R_i.
    '''
    order = min(mi.order, mj.order)
    R = np.linalg.norm(rvec)
    if R < 1e-6:
        return -np.inf
    R_bar = rvec / R

    mu_ai = mi.mu
    mu_bj = mj.mu
    aibj = np.dot(mu_ai.T, mu_bj)
    tmp_ai = np.dot(R_bar, mu_ai)
    tmp_bj = np.dot(R_bar, mu_bj)
    aibj -= np.outer(tmp_ai, tmp_bj * 3)
    aibj /= R**3

    if order > 2:
        theta_ai = mi.theta
        theta_bj = mj.theta
        RR = np.outer(R_bar, R_bar)
        tmp1_ai = np.dot(RR.ravel(), theta_ai.reshape(9,-1))
        tmp1_bj = np.dot(RR.ravel(), theta_bj.reshape(9,-1))
        aibj_3  = np.outer(tmp_ai, tmp1_bj * 5)
        aibj_3 -= np.outer(tmp1_ai, tmp_bj * 5)
        mu_R_ai = einsum('xa,y->xya', mu_ai, R_bar).reshape(9,-1)
        mu_R_bj = einsum('xb,y->xyb', mu_bj, R_bar).reshape(9,-1)
        aibj_3 -= np.dot(2 * mu_R_ai.T, theta_bj.reshape(9,-1))
        aibj_3 += np.dot(theta_ai.reshape(9,-1).T, mu_R_bj * 2)
        aibj += aibj_3 / R**4

    if order > 3:
        omega_ai = mi.omega
        omega_bj = mj.omega
        RRR = einsum('x,y,z->xyz', R_bar, R_bar, R_bar)
        aibj_4  = einsum('xa,xyzb,yz->ab', mu_ai, omega_bj, RR * 9)
        aibj_4 += einsum('xy,xyza,zb->ab', RR * 9, omega_ai, mu_bj)
        omega_R3_ai = np.dot(RRR.ravel(), omega_ai.reshape(27,-1))
        omega_R3_bj = np.dot(RRR.ravel(), omega_bj.reshape(27,-1))
        aibj_4 -= np.outer(tmp_ai, omega_R3_bj * 21)
        aibj_4 -= np.outer(omega_R3_ai, tmp_bj * 21)
        aibj_4 += np.outer(tmp1_ai, tmp1_bj * 35)
        tmp2_ai = einsum('xya,y->xa', theta_ai, R_bar)
        tmp2_bj = einsum('xyb,y->xb', theta_bj, R_bar)
        aibj_4 -= np.dot(tmp2_ai.T, tmp2_bj * 20)
        aibj_4 += np.dot(theta_ai.reshape(9,-1).T, theta_bj.reshape(9,-1) * 2)
        aibj += aibj_4 / (3 * R**5)

    denom = mi.e_vo[:,None] + mj.e_vo[None,:]
    return -prefac * np.sum(aibj * aibj / denom)
