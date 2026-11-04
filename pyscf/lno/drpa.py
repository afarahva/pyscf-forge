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

r''' Direct ring CCD (direct RPA) amplitudes for LNO (closed shell, real orbitals)

The canonical drCCD of `pyscf.regmp.driccd`, applied to given DF integrals and orbital
energies rather than to an SCF object:

    full system : the dressed integrals U = (1 + 2T) J, whose MP2 amplitudes are the drCCD
                  ones, give the LNO density matrices (see `lno._DressedERIS`), and the
                  full-system dRPA / SOSEX energies;
    impurity    : the active-space amplitudes t2 seed the impurity CCSD and give the
                  fragment dRPA / SOSEX energy.

Ref:
    Song, Gong, Bakry, Ye, arXiv:2601.00131 (RPA-based LNO-CC)
'''

import numpy as np

from pyscf import lib
from pyscf.lib import logger
from pyscf.regmp import driccd
from pyscf import __config__

einsum = lib.einsum


class DRPA(lib.StreamObject):
    r''' drCCD for given DF integrals J[i,a,P] and orbital energies.

    Attributes:
        solver : str
            'eig' (RPA eigenproblem, default) or 'riccati' (iterative).
        with_sosex : bool
            Whether `kernel` also computes the SOSEX energy. Default is True.

    The remaining settings ('riccati' solver) are the defaults of
    `pyscf.regmp.driccd.DRICCD`.

    Saved results:
        e_corr, e_corr_sosex, converged
        U : dressed DF integrals (nocc, nvir, naux)
    '''

    solver = getattr(__config__, 'lno_drpa_DRPA_solver', 'eig')
    with_sosex = driccd.DRICCD.with_sosex
    cd_thresh = driccd.DRICCD.cd_thresh
    max_cycle = driccd.DRICCD.max_cycle
    conv_tol = driccd.DRICCD.conv_tol
    conv_tol_normu = driccd.DRICCD.conv_tol_normu
    diis_space = driccd.DRICCD.diis_space
    damp = driccd.DRICCD.damp

    _keys = {'max_memory', 'verbose', 'stdout', 'converged', 'e_corr', 'e_corr_sosex', 'U'}

    def __init__(self, solver=None, max_memory=4000, verbose=logger.NOTE, stdout=None):
        if solver is not None:
            self.solver = solver
        self.max_memory = max_memory
        self.verbose = verbose
        self.stdout = stdout
        self.converged = False
        self.e_corr = None
        self.e_corr_sosex = None
        self.U = None

    def kernel(self, J, eo, ev):
        r''' drCCD for the real DF integrals J (nocc, nvir, naux) and the canonical
        occupied and virtual orbital energies eo and ev.
        '''
        log = logger.new_logger(self)
        self.converged, self.U = driccd.kernel(self, J, eo, ev)
        if not self.converged:
            log.warn('drCCD did not converge, please be careful of the results.')
        self.e_corr = float(driccd.energy(self.U, J))
        if self.with_sosex:
            self.e_corr_sosex = driccd.energy_sosex(self.U, J, eo, ev, self.max_memory)
        return self.e_corr

    def make_t2(self, eo, ev):
        r''' Dense amplitudes t2[i,j,a,b] of the last `kernel`. '''
        return driccd.make_t2(self.U, eo, ev)


def get_ovJ(eris):
    r''' Load the full real DF integrals J[i,a,P] from an LNO ERI object.

    For KLNO with real supercell orbitals, (ia|jb) = sum_P (R R + I I) and J is the
    concatenation of the real and imaginary parts along the auxiliary index.
    '''
    if np.dtype(eris.dtype).kind == 'c':
        raise NotImplementedError('RPA-based LNO requires real orbitals (molecules, '
                                  'Gamma point, or time-reversal-symmetric k-meshes).')
    nocc, nvir = eris.nocc, eris.nvir
    blks = []
    for i0,i1 in lib.prange(0, nocc, max(1, nocc//4)):
        blk = eris.get_occ_blk(i0,i1)
        if isinstance(blk, tuple):
            blk = np.concatenate(blk, axis=-1)
        blks.append(np.asarray(blk))
    return np.concatenate(blks, axis=0).reshape(nocc,nvir,-1)

def get_fragment_energy_rpa(oovv, t2, uocc_loc):
    r''' Direct-ring fragment energy 2 sum_{Kjab} t2(Kjab) (Ka|jb) (no exchange). '''
    m = np.dot(uocc_loc, uocc_loc.T.conj())
    ed = (einsum('ijab,kjab,ik->', t2, oovv, m) * 2).real
    return lib.tag_array(ed, spin_comp=np.array((ed*0.5, ed*0.5)))
