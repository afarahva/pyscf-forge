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
from pyscf import gto, scf, lib
from pyscf import regmp

# HF dimer (A24 database), aug-cc-pVDZ, frozen F 1s; DF correlation with aug-cc-pVDZ-RI
def setUpModule():
    global mol, mf, frozen
    mol = gto.Mole()
    mol.verbose = 5
    mol.output = '/dev/null'
    mol.atom = '''
    H  0.00000000  0.80267982  1.69529329
    F  0.00000000 -0.04596666  1.34034818
    H  0.00000000 -0.12040787 -0.49082840
    F  0.00000000  0.00976945 -1.40424978
    '''
    mol.basis = 'aug-cc-pvdz'
    mol.build()
    mf = scf.RHF(mol).run(conv_tol=1e-10)
    frozen = 2

def tearDownModule():
    global mol, mf
    mol.stdout.close()
    del mol, mf


class KnownValues(unittest.TestCase):
    def test_eig(self):
        m = regmp.DRICCD(mf, frozen=frozen)
        m.solver = 'eig'
        m.run()
        self.assertAlmostEqual(m.e_corr, -0.4934770122, 8)
        self.assertAlmostEqual(m.e_corr_sosex, -0.3262294408, 8)
        dm = m.make_rdm1()
        self.assertAlmostEqual(np.trace(dm), mol.nelectron, 9)
        self.assertAlmostEqual(lib.fp(dm), 2.1161757689, 7)
        self.assertRaises(NotImplementedError, m.make_rdm2)

    def test_riccati(self):
        m = regmp.DRICCD(mf, frozen=frozen).run()
        self.assertTrue(m.converged)
        self.assertAlmostEqual(m.e_corr, -0.4934770122, 6)
        self.assertAlmostEqual(m.e_corr_sosex, -0.3262294408, 6)

    def test_without_sosex(self):
        m = regmp.DRICCD(mf, frozen=frozen)
        m.with_sosex = False
        m.kernel(with_t2=False)
        self.assertTrue(m.e_corr_sosex is None)
        self.assertTrue(m.t2 is None)
        self.assertAlmostEqual(m.e_corr, -0.4934770122, 6)


if __name__ == "__main__":
    print("Full Tests for drCCD (direct RPA)")
    unittest.main()
