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
from pyscf import __config__
setattr(__config__, 'lno_base_make_rdm1_DEBUG_BLKSIZE', True)    # debug blocking
from pyscf import gto, scf, cc, lo
from pyscf.lno import LNOCCSD, rpa
from pyscf.lno import make_lno_rdm1


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
        J = rpa.get_ovJ(eris).reshape(nocc*nvir,-1)
        eia = (moevir[None,:] - moeocc[:,None]).ravel()
        return J, eia, nocc, nvir

    def test_drccd_vs_exact(self):
        mlno = self._lno()
        eris = mlno.ao2mo()
        J, eia, nocc, nvir = self._full_drccd(mlno, eris)
        V = np.dot(J, J.T)
        T = exact_drccd(V, eia)
        e_ref = 2*np.sum(T*V)
        Vx = V.reshape(nocc,nvir,nocc,nvir).transpose(0,3,2,1).reshape(nocc*nvir,-1)
        esx_ref = np.sum(T*(2*V-Vx))

        conv, e_rpa, U, tau = rpa.kernel(J, eia, cd_thresh=1e-9, conv_tol=1e-11,
                                         conv_tol_normu=1e-8)
        self.assertTrue(conv)
        self.assertAlmostEqual(e_rpa, e_ref, 8)
        self.assertAlmostEqual(rpa.energy(U, J), e_ref, 8)
        t2 = rpa.make_t2(U, tau, nocc, nvir)
        self.assertAlmostEqual(abs(t2.transpose(0,2,1,3).reshape(nocc*nvir,-1)-T).max(), 0, 7)
        self.assertAlmostEqual(rpa.energy_sosex(U, J, tau, nocc, nvir), esx_ref, 8)

    def test_rdm1_rpa_amp(self):
        mlno = self._lno()
        eris = mlno.ao2mo()
        J, eia, nocc, nvir = self._full_drccd(mlno, eris)
        conv, e_rpa, U, tau = rpa.kernel(J, eia)
        t2 = rpa.make_t2(U, tau, nocc, nvir)
        amp = rpa._RPAAmpERIS(U.reshape(nocc,nvir,-1), tau, mlno.max_memory)
        moeocc, moevir = mlno.split_mo_energy()[1:3]

        rng = np.random.default_rng(1)
        uocc = np.linalg.qr(rng.standard_normal((nocc,2)))[0]
        uvir = np.linalg.qr(rng.standard_normal((nvir,3)))[0]
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
        self.assertEqual((mlno._lno_amp, mlno._pt2_amp), ('mp2', 'mp2'))
        mlno.lno_amp = 'RPA'
        self.assertEqual((mlno._lno_amp, mlno._pt2_amp), ('rpa', 'rpa'))
        mlno.pt2_amp = 'mp2'
        self.assertEqual((mlno._lno_uses_rpa, mlno._pt2_uses_rpa), (True, False))
        self.assertEqual(mlno._pt2_impurity_kwargs['rpa'], None)
        mlno.lno_amp, mlno.pt2_amp = 'mp2', 'kappa-mp2'
        self.assertEqual(mlno._pt2_impurity_kwargs['kappa'], mlno.kappa)
        mlno.pt2_amp = 'ccsd'
        with self.assertRaises(ValueError):
            mlno._pt2_amp

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


if __name__ == "__main__":
    print("Full Tests for LNO-CCSD with RPA (drCCD) amplitudes")
    unittest.main()
