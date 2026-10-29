#!/usr/bin/env python
# Copyright 2026 The PySCF Developers. All Rights Reserved.
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
from pyscf import gto, scf, mp, cc, lo, ao2mo, lib
from pyscf.lno import LNOCCSD, bws2, rpa
from pyscf.lno import make_lno_rdm1
from pyscf.lno.test.test_lnorpa import ref_rdm1


class WaterDimerBWs2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.verbose = 4
        mol.output = '/dev/null'
        mol.atom = '''
        O   -1.485163346097   -0.114724564047    0.000000000000
        H   -1.868415346097    0.762298435953    0.000000000000
        H   -0.533833346097    0.040507435953    0.000000000000
        O    1.416468653903    0.111264435953    0.000000000000
        H    1.746241653903   -0.373945564047   -0.758561000000
        H    1.746241653903   -0.373945564047    0.758561000000
        '''
        mol.basis = 'cc-pvdz'
        mol.precision = 1e-10
        mol.build()
        mf = scf.RHF(mol).density_fit().run()
        frozen = 2

        mycc = cc.CCSD(mf, frozen=frozen)
        mycc.kernel()

        orbocc = mf.mo_coeff[:,frozen:np.count_nonzero(mf.mo_occ)]
        lo_coeff = lo.Boys(mol, orbocc).kernel()

        cls.mol = mol
        cls.mf = mf
        cls.frozen = frozen
        cls.lo_coeff = lo_coeff
        cls.efull_ccsd = mycc.e_corr
        cls.efull_mp2 = mp.MP2(mf, frozen=frozen).run().e_corr

    @classmethod
    def tearDownClass(cls):
        cls.mol.stdout.close()
        del cls.mol, cls.mf, cls.frozen, cls.lo_coeff, cls.efull_ccsd, cls.efull_mp2

    def _lno(self, **kwargs):
        frag_lolist = [[i] for i in range(self.lo_coeff.shape[1])]
        return LNOCCSD(self.mf, self.lo_coeff, frag_lolist, frozen=self.frozen).set(**kwargs)

    def _ints(self, mlno):
        eris = mlno.ao2mo()
        moeocc, moevir = mlno.split_mo_energy()[1:3]
        return eris, rpa.get_ovJ(eris), moeocc, moevir

    def test_solver(self):
        eris, J, eo, ev = self._ints(self._lno())
        oovv = np.einsum('iax,jbx->ijab', J, J)
        # alpha = 0 is MP2
        self.assertAlmostEqual(bws2.kernel(J, eo, ev, alpha=0.)[1], self.efull_mp2, 10)
        for alpha in [1., 4.]:
            conv, e_df, eoW, U = bws2.kernel(J, eo, ev, alpha=alpha, conv_tol=1e-11)
            conv_d, e_dense, eoW_d, U_d, t2 = bws2.kernel_dense(oovv, eo, ev, alpha=alpha,
                                                                conv_tol=1e-11)
            self.assertTrue(conv and conv_d)
            self.assertAlmostEqual(e_df, e_dense, 10)
            self.assertAlmostEqual(abs(eoW - eoW_d).max(), 0, 8)
            # regularization reduces the magnitude of the correlation energy
            self.assertLess(abs(e_df), abs(self.efull_mp2))
            # energy from the amplitudes rotated back to the canonical basis
            e_t2 = np.sum(t2 * (2*oovv - oovv.transpose(0,1,3,2)))
            self.assertAlmostEqual(e_t2, e_dense, 10)

    def test_h2_two_level(self):
        # minimal-basis H2: BW-s2(alpha=1) is the 2x2 FCI formula with the Fock gap,
        # E = d/2 - sqrt(d^2/4 + K^2), exact at dissociation (d -> 0).
        mol = gto.M(atom='H 0 0 0; H 0 0 8.0', basis='sto-3g', verbose=0)
        mf = scf.RHF(mol).run()
        eri = ao2mo.kernel(mol, mf.mo_coeff, compact=False).reshape(2,2,2,2)
        oovv = eri[:1,1:,:1,1:].transpose(0,2,1,3).copy()
        eo, ev = mf.mo_energy[:1], mf.mo_energy[1:]
        K, d = oovv[0,0,0,0], 2*(ev[0] - eo[0])
        e = bws2.kernel_dense(oovv, eo, ev, alpha=1., conv_tol=1e-14, max_cycle=5000)[1]
        self.assertAlmostEqual(e, d/2 - np.sqrt(d**2/4 + K**2), 12)

    def test_rdm1_bws2(self):
        # force blocking in the rdm1 code (module-level flag, restored afterwards)
        with lib.temporary_env(make_lno_rdm1, DEBUG_BLKSIZE=True):
            self._test_rdm1_bws2()

    def _test_rdm1_bws2(self):
        # The rdm1 code uses semicanonical energies for the localized orbitals, which
        # are exact when these are eigenvectors of the (dressed) Fock matrix: dressed
        # occupied orbitals (columns of U) and canonical virtuals. Then every dm type
        # equals the brute-force dm of the BW-s2 amplitudes.
        mlno = self._lno(lno_amp='bws2', alpha=2., bws2_conv_tol=1e-11)
        eris, J, eo, ev = self._ints(mlno)
        nvir = ev.size
        oovv = np.einsum('iax,jbx->ijab', J, J)
        t2 = bws2.kernel_dense(oovv, eo, ev, alpha=2., conv_tol=1e-11)[4]
        U = mlno.get_bws2(eris)[1]
        uocc = U[:,[1,4]]
        uvir = np.eye(nvir)[:,[0,3,5]]
        for occ_type, u in [('1h',uocc), ('1p',uvir), ('2p',uvir)]:
            dm = mlno.make_lo_rdm1_occ(eris, eo, ev, uocc, uvir, occ_type)
            self.assertAlmostEqual(abs(dm - ref_rdm1(t2, u, 'occ'+occ_type)).max(), 0, 8)
        for vir_type, u in [('1p',uvir), ('1h',uocc), ('2h',uocc)]:
            dm = mlno.make_lo_rdm1_vir(eris, eo, ev, uocc, uvir, vir_type)
            self.assertAlmostEqual(abs(dm - ref_rdm1(t2, u, 'vir'+vir_type)).max(), 0, 8)

    def test_alpha0_is_mp2(self):
        m_mp2 = self._lno(lno_thresh=[1e-4,1e-5])
        m_mp2.kernel()
        m_bw = self._lno(lno_thresh=[1e-4,1e-5], lno_amp='bws2', alpha=0.)
        m_bw.kernel()
        self.assertAlmostEqual(m_bw.e_corr_pt2, m_mp2.e_corr_pt2, 10)
        self.assertAlmostEqual(m_bw.e_corr_ccsd, m_mp2.e_corr_ccsd, 10)

    def test_complete_space(self):
        for alpha in [1., 4.]:
            mlno = self._lno(lno_thresh=[1e-14,1e-14], lno_amp='bws2', alpha=alpha)
            mlno.kernel()
            self.assertAlmostEqual(mlno.e_corr_pt2, mlno.e_corr_bws2_full, 9)
            self.assertAlmostEqual(mlno.e_corr_ccsd, self.efull_ccsd, 6)

    def test_by_thresh(self):
        errs = []
        for thresh in [1e-4, 1e-5]:
            mlno = self._lno(lno_thresh=[thresh*10, thresh], lno_amp='bws2', alpha=4.)
            mlno.kernel()
            ecc = mlno.e_corr_ccsd_pt2corrected(mlno.e_corr_bws2_full)
            errs.append(abs(ecc - self.efull_ccsd))
        self.assertLess(errs[1], 3e-4)
        self.assertLess(errs[1], errs[0])

    def test_restart(self):
        # the full-system BW-s2 energy is restored from the chkfile, so a restart with all
        # LNOs saved needs neither the integrals nor the full-system BW-s2
        from unittest import mock
        from pyscf.lno.test.test_lnoccsd import kernel_counted
        thresh = [1e-4, 1e-5]
        m1 = self._lno(lno_thresh=thresh, lno_amp='bws2', alpha=4.)
        m1.kernel()
        m2 = self._lno(lno_thresh=thresh, lno_amp='bws2', alpha=4.).load_chk(m1.chkfile)
        m2.frag_eners[0] = None
        with mock.patch.object(m2, 'solve_bws2', side_effect=AssertionError('BW-s2 solved')):
            calls = kernel_counted(m2)
        self.assertEqual(calls, {'ao2mo': 0, 'make_las': 0, 'impurity_solve': 1})
        self.assertEqual(m2.e_corr_bws2_full, m1.e_corr_bws2_full)
        for key in ['efrag_pt2', 'efrag_cc']:
            self.assertAlmostEqual(abs(getattr(m2, key) - getattr(m1, key)).max(), 0, 9)


if __name__ == "__main__":
    print("Full Tests for LNO-CCSD with BW-s2(alpha) amplitudes")
    unittest.main()
