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
from pyscf import gto, scf, cc, lo, lib
from pyscf.lno import LNOCCSD, ULNOCCSD, drpa
from pyscf.lno import make_lno_rdm1
from pyscf.lno.lno import _DressedERIS
from pyscf.regpt2 import driccd


def exact_drccd(V, eia):
    r''' Exact closed-shell drCCD amplitudes from the RPA eigenproblem.

        Riccati form B + A T + T A + T B T = 0 with A = D + 2V, B = 2V and t = T/2.
    '''
    n = eia.size
    A = np.diag(eia) + 2*V
    B = 2*V
    w, v = np.linalg.eig(np.block([[A, B], [-B, -A]]))
    idx = np.argsort(w.real)[n:]
    T = np.linalg.solve(v[:n,idx].T, v[n:,idx].T).T.real
    return T * 0.5

def ref_rdm1(t2, u, dm_type):
    r''' Brute-force LNO rdm1 from dense t2[i,j,a,b] (see make_lno_rdm1 docstrings). '''
    e = np.einsum
    if dm_type == 'occ1h':
        tK = e('ikab,kK->iKab', t2, u)
        return -4*e('iKab,jKab->ij', tK, tK) + 2*e('iKab,jKba->ij', tK, tK)
    if dm_type == 'occ1p':
        tA = e('ijab,aA->ijAb', t2, u)
        return (-2*e('ikAb,jkAb->ij', tA, tA) + e('ikAb,kjAb->ij', tA, tA)
                + e('kiAb,jkAb->ij', tA, tA) - 2*e('kiAb,kjAb->ij', tA, tA))
    if dm_type == 'occ2p':
        tAB = e('ijab,aA,bB->ijAB', t2, u, u)
        return -4*e('ikAB,jkAB->ij', tAB, tAB) + 2*e('ikAB,kjAB->ij', tAB, tAB)
    if dm_type == 'vir1p':
        tA = e('ijcb,cA->ijAb', t2, u)
        return 4*e('ijAa,ijAb->ab', tA, tA) - 2*e('ijAa,jiAb->ab', tA, tA)
    if dm_type == 'vir1h':
        tI = e('ijab,iI->Ijab', t2, u)
        return (2*e('Ijac,Ijbc->ab', tI, tI) - e('Ijac,Ijcb->ab', tI, tI)
                - e('Ijca,Ijbc->ab', tI, tI) + 2*e('Ijca,Ijcb->ab', tI, tI))
    if dm_type == 'vir2h':
        tIJ = e('ijab,iI,jJ->IJab', t2, u, u)
        return 4*e('IJac,IJbc->ab', tIJ, tIJ) - 2*e('IJac,IJcb->ab', tIJ, tIJ)
    raise RuntimeError


class WaterDimerRPA(unittest.TestCase):
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

    @classmethod
    def tearDownClass(cls):
        cls.mol.stdout.close()
        del cls.mol, cls.mf, cls.frozen, cls.lo_coeff, cls.efull_ccsd

    def _lno(self, frag_lolist=None, **kwargs):
        if frag_lolist is None:
            frag_lolist = [[i] for i in range(self.lo_coeff.shape[1])]
        mlno = LNOCCSD(self.mf, self.lo_coeff, frag_lolist, frozen=self.frozen)
        mlno.set(**kwargs)
        return mlno

    def _full_drccd(self, mlno, eris):
        moeocc, moevir = mlno.split_mo_energy()[1:3]
        nocc, nvir = moeocc.size, moevir.size
        J = drpa.get_ovJ(eris).reshape(nocc*nvir,-1)
        eia = (moevir[None,:] - moeocc[:,None]).ravel()
        return J, eia, moeocc, moevir

    def test_drccd_vs_exact(self):
        mlno = self._lno()
        eris = mlno.ao2mo()
        J, eia, moeocc, moevir = self._full_drccd(mlno, eris)
        nocc, nvir = moeocc.size, moevir.size
        V = np.dot(J, J.T)
        T = exact_drccd(V, eia)
        e_ref = 2*np.sum(T*V)
        Vx = V.reshape(nocc,nvir,nocc,nvir).transpose(0,3,2,1).reshape(nocc*nvir,-1)
        esx_ref = np.sum(T*(2*V-Vx))

        conv, U_ric = driccd.kernel_riccati(J, eia, cd_thresh=1e-9, conv_tol=1e-11,
                                            conv_tol_normu=1e-8)
        self.assertTrue(conv)
        for U in (U_ric, driccd.kernel_eig(J, eia)):
            self.assertAlmostEqual(driccd.energy(U, J), e_ref, 8)
            t2 = driccd.make_t2(U.reshape(nocc,nvir,-1), moeocc, moevir)
            self.assertAlmostEqual(abs(t2.transpose(0,2,1,3).reshape(nocc*nvir,-1)-T).max(), 0, 7)
            self.assertAlmostEqual(driccd.energy_sosex(U, J, moeocc, moevir), esx_ref, 8)

    def test_rdm1_dressed_eris(self):
        # force blocking in the rdm1 code (module-level flag, restored afterwards)
        with lib.temporary_env(make_lno_rdm1, DEBUG_BLKSIZE=True):
            self._test_rdm1_dressed_eris()

    def _test_rdm1_dressed_eris(self):
        # With rotations of the whole occupied/virtual space, the LNO density matrices of
        # the dressed integrals are those of the drCCD / kappa-MP2 amplitudes.
        mlno = self._lno()
        eris = mlno.ao2mo()
        J, eia, moeocc, moevir = self._full_drccd(mlno, eris)
        nocc, nvir = moeocc.size, moevir.size
        U = driccd.kernel_eig(J, eia).reshape(nocc,nvir,-1)
        t2_rpa = driccd.make_t2(U, moeocc, moevir)
        kappa = 1.1
        oovv = np.einsum('iax,jbx->ijab', J.reshape(nocc,nvir,-1), J.reshape(nocc,nvir,-1))
        denom = -lib.direct_sum('ia+jb->ijab', eia.reshape(nocc,nvir), eia.reshape(nocc,nvir))
        t2_kappa = oovv / denom * (1 - np.exp(kappa*denom))

        rng = np.random.default_rng(1)
        uocc = np.linalg.qr(rng.standard_normal((nocc,nocc)))[0]
        uvir = np.linalg.qr(rng.standard_normal((nvir,nvir)))[0]
        for amp, t2 in [(_DressedERIS(eris, ovL=U), t2_rpa),
                        (_DressedERIS(eris, kappa=kappa), t2_kappa)]:
            for occ_type, u in [('1h',uocc), ('1p',uvir), ('2p',uvir)]:
                dm = make_lno_rdm1.make_lo_rdm1_occ(amp, moeocc, moevir, uocc, uvir, occ_type)
                self.assertAlmostEqual(abs(dm - ref_rdm1(t2, u, 'occ'+occ_type)).max(), 0, 10)
            for vir_type, u in [('1p',uvir), ('1h',uocc), ('2h',uocc)]:
                dm = make_lno_rdm1.make_lo_rdm1_vir(amp, moeocc, moevir, uocc, uvir, vir_type)
                self.assertAlmostEqual(abs(dm - ref_rdm1(t2, u, 'vir'+vir_type)).max(), 0, 10)

    def test_lno_rpa_complete_space(self):
        # Without truncation, fragment RPA/SOSEX energies sum to the full-system ones
        # and LNO-CCSD reproduces canonical CCSD.
        for amp in ['rpa', 'sosex']:
            mlno = self._lno(lno_thresh=[1e-14,1e-14], lno_amp=amp)
            mlno.kernel()
            efull = mlno.e_corr_rpa_full if amp == 'rpa' else mlno.e_corr_sosex_full
            self.assertAlmostEqual(mlno.e_corr_pt2, efull, 7)
            self.assertAlmostEqual(mlno.e_corr_ccsd, self.efull_ccsd, 6)

    def test_amp_settings(self):
        mlno = self._lno()
        mo_coeff, mo_occ = self.mf.mo_coeff, self.mf.mo_occ
        maskact = np.arange(mo_occ.size) >= self.frozen
        self.assertEqual((mlno._lno_amp, mlno._pt2_amp), ('mp2', 'mp2'))
        mlno.lno_amp = 'RPA'
        self.assertEqual((mlno._lno_amp, mlno._pt2_amp), ('rpa', 'rpa'))
        self.assertEqual(mlno.DRPA().solver, 'eig')
        mlno.rpa_solver = 'riccati'
        self.assertEqual(mlno.DRPA().solver, 'riccati')
        kwargs = mlno._pt2_impurity_kwargs(mo_coeff, mo_occ, maskact)
        nocc = np.count_nonzero(mo_occ)
        self.assertEqual(kwargs['ovJ'].shape[:2], (nocc-self.frozen, mo_occ.size-nocc))
        mlno.pt2_amp = 'mp2'
        self.assertEqual((mlno._lno_uses_rpa, mlno._pt2_uses_rpa), (True, False))
        self.assertEqual(mlno._pt2_impurity_kwargs(mo_coeff, mo_occ, maskact),
                         {'pt2_amp': 'mp2', 'kappa': mlno.kappa})
        mlno.pt2_amp = 'ccsd'
        with self.assertRaises(ValueError):
            mlno._pt2_amp
        mlno.lno_amp, mlno.pt2_amp, mlno.prune_lno_basis = 'mp2', 'sosex', True
        with self.assertRaises(NotImplementedError):
            mlno._pt2_amp
        # no RPA/SOSEX amplitudes for ULNO
        umlno = ULNOCCSD(scf.UHF(self.mol).density_fit(), [self.lo_coeff]*2, [[[0],[]]])
        umlno.lno_amp = 'rpa'
        with self.assertRaises(NotImplementedError):
            umlno._lno_amp

    def test_rpa_lno_mp2_energy(self):
        # RPA LNOs with the MP2 fragment energy: same active spaces as lno_amp='rpa',
        # so the CCSD energy is identical while the PT2 energy is MP2.
        thresh = [1e-4, 1e-5]
        m_rpa = self._lno(lno_thresh=thresh, lno_amp='rpa')
        m_rpa.kernel()
        m_mix = self._lno(lno_thresh=thresh, lno_amp='rpa', pt2_amp='mp2')
        m_mix.kernel()
        m_mp2 = self._lno(lno_thresh=thresh)
        m_mp2.kernel()
        self.assertAlmostEqual(m_mix.e_corr_ccsd, m_rpa.e_corr_ccsd, 6)
        self.assertNotAlmostEqual(m_mix.e_corr_pt2, m_rpa.e_corr_pt2, 3)
        self.assertNotAlmostEqual(m_mix.e_corr_pt2, m_mp2.e_corr_pt2, 6)

    def test_lno_rpa_by_thresh(self):
        frag_lolist = [[i] for i in range(self.lo_coeff.shape[1])]
        gamma = 10
        errs = []
        for thresh in [1e-4, 1e-5]:
            mlno = self._lno(frag_lolist, lno_thresh=[thresh*gamma, thresh], lno_amp='rpa')
            mlno.kernel()
            ecc = mlno.e_corr_ccsd_pt2corrected(mlno.e_corr_rpa_full)
            errs.append(abs(ecc - self.efull_ccsd))
        self.assertLess(errs[1], 5e-4)
        self.assertLess(errs[1], errs[0])

    def test_lno_sosex_restart(self):
        # the full-system drCCD energies are restored from the chkfile, so a restart with
        # all LNOs saved neither solves drCCD again nor builds the full-system integrals:
        # the SOSEX energy of the fragment whose energies were removed only needs those of
        # its active space
        from unittest import mock
        from pyscf.lno.test.test_lnoccsd import kernel_counted
        thresh = [1e-4, 1e-5]
        m1 = self._lno(lno_thresh=thresh, lno_amp='sosex')
        m1.kernel()
        m2 = self._lno(lno_thresh=thresh, lno_amp='sosex').load_chk(m1.chkfile)
        m2.frag_eners[0] = None
        with mock.patch.object(m2, 'solve_rpa', side_effect=AssertionError('drCCD solved')):
            calls = kernel_counted(m2)
        self.assertEqual(calls, {'ao2mo': 0, 'make_las': 0, 'impurity_solve': 1})
        self.assertEqual(m2.e_corr_rpa_full, m1.e_corr_rpa_full)
        self.assertEqual(m2.e_corr_sosex_full, m1.e_corr_sosex_full)
        for key in ['efrag_pt2', 'efrag_cc']:
            self.assertAlmostEqual(abs(getattr(m2, key) - getattr(m1, key)).max(), 0, 9)


if __name__ == "__main__":
    print("Full Tests for LNO-CCSD with RPA (drCCD) amplitudes")
    unittest.main()
