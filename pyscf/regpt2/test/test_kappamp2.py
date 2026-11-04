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
from pyscf import regpt2

# HF dimer (A24 database), aug-cc-pVDZ, frozen F 1s; DF correlation with aug-cc-pVDZ-RI
def setUpModule():
    global mol, mf, umf, frozen
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
    umf = scf.UHF(mol).run(conv_tol=1e-10)
    frozen = 2

def tearDownModule():
    global mol, mf, umf
    mol.stdout.close()
    del mol, mf, umf


class KnownValues(unittest.TestCase):
    def test_kappa_mp2(self):
        for cls, m_f, e_ref in ((regpt2.KappaRMP2,   mf,  -0.4335624783),
                                (regpt2.KappaUMP2,   umf, -0.4335624756),
                                (regpt2.KappaDFRMP2, mf,  -0.4335620371),
                                (regpt2.KappaDFUMP2, umf, -0.4335620345)):
            m = cls(m_f, frozen=frozen, kappa=1.1).run()
            self.assertAlmostEqual(m.e_corr, e_ref, 8)
        self.assertAlmostEqual(m.e_corr_ss, -0.1162741817, 8)
        self.assertAlmostEqual(m.e_corr_os, -0.3172878527, 8)

    def test_large_kappa_is_mp2(self):
        for cls, m_f, e_ref in ((regpt2.KappaRMP2,   mf,  -0.4460257996),    # MP2
                                (regpt2.KappaUMP2,   umf, -0.4460257958),
                                (regpt2.KappaDFRMP2, mf,  -0.4460262474),    # DF-MP2
                                (regpt2.KappaDFUMP2, umf, -0.4460262437)):
            m = cls(m_f, frozen=frozen, kappa=1e6).run()
            self.assertAlmostEqual(m.e_corr, e_ref, 8)

    def test_rdm1(self):
        m = regpt2.KappaRMP2(mf, frozen=frozen, kappa=1.1).run()
        dm = m.make_rdm1()
        self.assertAlmostEqual(np.trace(dm), mol.nelectron, 9)
        self.assertAlmostEqual(lib.fp(dm), 2.1135037154, 7)
        self.assertRaises(NotImplementedError, m.make_rdm2)
        m = regpt2.KappaDFUMP2(umf, frozen=frozen, kappa=1.1).run()
        dm = m.make_rdm1()
        self.assertAlmostEqual(np.trace(dm[0]) + np.trace(dm[1]), mol.nelectron, 9)
        self.assertAlmostEqual(lib.fp(dm), 2.1296767336, 6)

    def test_factory(self):
        self.assertTrue(isinstance(regpt2.KappaMP2(mf), regpt2.KappaRMP2))
        self.assertTrue(isinstance(regpt2.KappaMP2(umf), regpt2.KappaUMP2))
        self.assertTrue(isinstance(regpt2.KappaMP2(mf.density_fit()), regpt2.KappaDFRMP2))
        self.assertTrue(isinstance(regpt2.KappaMP2(umf.density_fit()), regpt2.KappaDFUMP2))
        self.assertEqual(regpt2.KappaMP2(mf, kappa=2.).kappa, 2.)

    def test_noncanonical(self):
        mf1 = mf.copy()
        mf1.converged = False
        self.assertRaises(NotImplementedError, regpt2.KappaRMP2(mf1, frozen=frozen).kernel)


if __name__ == "__main__":
    print("Full Tests for kappa-MP2")
    unittest.main()
