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

import unittest
import numpy as np
from pyscf.pbc import gto, scf
from pyscf.pbc import regpt2

# Two H2 per cell, 6-31G, 3 k-points: Gamma-centered (pairs q != -q, real supercell
# orbitals) and shifted (complex supercell orbitals). Reference values were checked
# against KMP2, dense supercell calculations and, for drCCD, the full complex RPA
# eigenproblem.
def setUpModule():
    global cell, kmf, kmf_shifted
    cell = gto.Cell()
    cell.verbose = 5
    cell.output = '/dev/null'
    cell.atom = 'H 0 0 0; H 0 0 0.74; H 1.85 0 0; H 1.85 0 0.74'
    cell.a = np.diag([3.7, 4.0, 4.0])
    cell.basis = '6-31g'
    cell.precision = 1e-10
    cell.build()
    kmf, kmf_shifted = [
        scf.KRHF(cell, kpts=cell.make_kpts([3,1,1], scaled_center=center)).density_fit()
        .set(exxdiv=None).run(conv_tol=1e-11) for center in ([0,0,0], [0.1,0,0])]

def tearDownModule():
    global cell, kmf, kmf_shifted
    cell.stdout.close()
    del cell, kmf, kmf_shifted

def nelec_per_cell(dm):
    return sum(np.trace(d) for d in dm).real / len(dm)


class KnownValues(unittest.TestCase):
    def test_eig(self):
        for mf, e_ref, esx_ref in ((kmf, -0.059447335309, -0.029972187648),
                                   (kmf_shifted, -0.059458326841, -0.029985067152)):
            m = regpt2.KDRICCD(mf).set(solver='eig').run()
            self.assertAlmostEqual(m.e_corr, e_ref, 9)
            self.assertAlmostEqual(m.e_corr_sosex, esx_ref, 9)
        self.assertAlmostEqual(nelec_per_cell(m.make_rdm1()), cell.nelectron, 9)
        self.assertRaises(NotImplementedError, m.make_rdm2)

    def test_riccati(self):
        m = regpt2.KDRICCD(kmf_shifted).run()
        self.assertTrue(m.converged)
        self.assertAlmostEqual(m.e_corr, -0.059458326841, 7)
        self.assertAlmostEqual(m.e_corr_sosex, -0.029985067152, 7)

    def test_frozen(self):
        m = regpt2.KDRICCD(kmf, frozen=1).set(solver='eig').run()
        self.assertAlmostEqual(m.e_corr, -0.020688483637, 9)

    def test_uneven_kpoints(self):
        # different numbers of active occupied orbitals per k-point (as in metals): padded
        # orbitals decouple; reference from the dense complex RPA eigenproblem of the
        # physical excitations, E_c = 1/2 (sum of the RPA frequencies - Tr A)
        frozen = [[0],[0],[]]
        m = regpt2.KDRICCD(kmf, frozen=frozen).set(solver='eig').run()
        self.assertAlmostEqual(m.e_corr, -0.031656552033, 9)
        self.assertAlmostEqual(m.e_corr_sosex, -0.015953490932, 9)
        m = regpt2.KDRICCD(kmf, frozen=frozen).run()
        self.assertTrue(m.converged)
        self.assertAlmostEqual(m.e_corr, -0.031656552033, 7)

    def test_without_sosex(self):
        m = regpt2.KDRICCD(kmf).set(solver='eig', with_sosex=False)
        m.kernel(with_t2=False)
        self.assertTrue(m.e_corr_sosex is None)
        self.assertTrue(m.t2 is None)
        self.assertAlmostEqual(m.e_corr, -0.059447335309, 9)


if __name__ == "__main__":
    print("Full Tests for k-point drCCD (direct RPA)")
    unittest.main()
