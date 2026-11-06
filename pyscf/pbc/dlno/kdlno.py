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

r''' Domain-based LNO with a k-point mean-field reference (KDLNO).

    The KLNO methods (`pyscf.pbc.lno`) work in the BvK supercell of the k-point mesh, with
    supercell LMOs (`lo_coeff`, all of them, sorted by cell) and fragments in the
    reference cell. KDLNO builds the LNOs of each fragment from the MP2 of its domain
    (see `pyscf.dlno.dlno`) instead of the full supercell:

    - Domains are built in the supercell with minimum-image interatomic distances; the
      multipole pair energies use the primary domains unwrapped around each LMO.
    - The domain orbitals (the strong-pair LMOs and the PAOs of the extended domain,
      semicanonicalized with the supercell Fock matrix) are expanded in the full
      supercell AO basis (`domain_basis = 'full'`), and their integrals are evaluated
      from the k-point DF integrals as in KLNO, so no extra density fitting is involved.
    - The fragment, local MP2 and LNO energies are per unit cell, as in KLNO.
'''

import numpy as np

from pyscf.lib import logger
from pyscf.dlno.dlno import DLNO
from pyscf.dlno.domain import atom_distance, minimum_image, fake_mol_by_atom


class KDLNO(DLNO):
    r''' Domain-based KLNO, a mixin to put in front of a KLNO method class, e.g.

        class KDLNOCCSD(KDLNO, KLNOCCSD): ...

    See `pyscf.dlno.dlno.DLNO` for the domain parameters; `domain_basis` must be 'full'.
    The multipole moments for the pair screening use the BP domains of the LMOs
    (`multipole_domain = 'bp'`): the primary domains of solids with diffuse basis sets can
    span most of the supercell.
    '''
    domain_basis = 'full'
    multipole_domain = 'bp'
    _mol_multipole = None

    @property
    def _lattice(self):
        return self.mol.lattice_vectors()

    @property
    def _domain_basis_full(self):
        if self.domain_basis.lower() != 'full':
            raise NotImplementedError('KDLNO only supports domain_basis = "full".')
        return True

    def _atom_distance(self):
        return atom_distance(self.mol, lattice=self._lattice)

    def _multipole_mol(self, atmlst, center_atom):
        r''' The atoms `atmlst` as a molecule, unwrapped to their minimum images about the
        atom `center_atom`.
        '''
        if self._mol_multipole is None:
            self._mol_multipole = self.mol.to_mol()
        coords = self.mol.atom_coords()
        c = coords[center_atom]
        unwrapped = c + minimum_image(coords[atmlst] - c, self._lattice)
        return fake_mol_by_atom(self._mol_multipole, atmlst, coords=unwrapped)

    def _pair_vector(self, ri, rj):
        return minimum_image(rj - ri, self._lattice)

    def _occ_energy_shift(self):
        mf = self._scf
        if getattr(mf, 'exxdiv', None) == 'ewald':
            from pyscf.pbc import tools
            return -tools.madelung(mf.cell, mf.kpt)
        return 0.

    def reset_domains(self):
        self._mol_multipole = None
        return DLNO.reset_domains(self)
