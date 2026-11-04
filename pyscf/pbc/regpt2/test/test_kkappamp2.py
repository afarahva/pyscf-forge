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
    def test_kappa_mp2(self):
        m = regpt2.KKappaMP2(kmf, kappa=1.1).run()
        self.assertAlmostEqual(m.e_corr, -0.031844196892, 9)
        self.assertAlmostEqual(m.e_corr_ss + m.e_corr_os, m.e_corr, 12)
        m = regpt2.KKappaMP2(kmf_shifted, kappa=1.1).run()
        self.assertAlmostEqual(m.e_corr, -0.031857799315, 9)

    def test_large_kappa_is_kmp2(self):
        self.assertAlmostEqual(regpt2.KKappaMP2(kmf, kappa=1e6).run().e_corr,
                               -0.037335509567, 9)     # KMP2
        self.assertAlmostEqual(regpt2.KKappaMP2(kmf_shifted, kappa=1e6).run().e_corr,
                               -0.037347713292, 9)
        # frozen orbitals that differ between k-points (padding)
        self.assertAlmostEqual(regpt2.KKappaMP2(kmf, frozen=[[0],[0],[]], kappa=1e6).run().e_corr,
                               -0.018791546477, 9)

    def test_rdm1(self):
        m = regpt2.KKappaMP2(kmf, kappa=1.1).run()
        self.assertAlmostEqual(nelec_per_cell(m.make_rdm1()), cell.nelectron, 9)
        self.assertRaises(NotImplementedError, m.make_rdm2)


if __name__ == "__main__":
    print("Full Tests for k-point kappa-MP2")
    unittest.main()
