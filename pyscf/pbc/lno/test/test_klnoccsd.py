#!/usr/bin/env python
# Copyright 2021 The PySCF Developers. All Rights Reserved.
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
from pyscf.pbc import gto, scf, tools as pbctools
from pyscf.lno import LNOCCSD, LNOCCSD_T
from pyscf.pbc.lno import KLNOCCSD, KLNOCCSD_T
from pyscf.lno.tools import autofrag_iao
from pyscf.pbc.lno.tools import k2s_scf, k2s_iao, is_trs_kmesh
from pyscf import lo
from pyscf.lno.test.test_lnoccsd import kernel_counted


class WaterDimer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cell = gto.Cell()
        cell.verbose = 4
        cell.output = '/dev/null'
        cell.atom = '''
        O   -1.485163346097   -0.114724564047    0.000000000000
        H   -1.868415346097    0.762298435953    0.000000000000
        H   -0.533833346097    0.040507435953    0.000000000000
        O    1.416468653903    0.111264435953    0.000000000000
        H    1.746241653903   -0.373945564047   -0.758561000000
        H    1.746241653903   -0.373945564047    0.758561000000
        '''
        cell.a = np.eye(3) * 5
        cell.basis = 'def2-svp'
        cell.precision = 1e-10
        cell.build()

        kmesh = [2,1,1]
        kpts = cell.make_kpts(kmesh)
        nkpts = len(kpts)
        scell = pbctools.super_cell(cell, kmesh)

        kmf = scf.KRHF(cell, kpts=kpts).density_fit().run()
        smf = k2s_scf(kmf)

        cls.cell = cell
        cls.kmf = kmf
        cls.scell = scell
        cls.smf = smf
        cls.frozen = 2 * nkpts

    @classmethod
    def tearDownClass(cls):
        cls.cell.stdout.close()
        cls.scell.stdout.close()
        del cls.cell, cls.kmf, cls.frozen
        del cls.scell, cls.smf

    def test_lno_iao_by_thresh(self):
        cell = self.cell
        kmf = self.kmf
        smf = self.smf
        scell = smf.cell
        frozen = self.frozen
        kpts = kmf.kpts
        nkpts = len(kpts)

        # IAO localization in supercell
        kocc_coeff = [kmf.mo_coeff[k][:,kmf.mo_occ[k]>1e-6] for k in range(nkpts)]
        lo_coeff = k2s_iao(cell, kocc_coeff, kpts, orth=True)

        # k-point LNO: only need to compute fragments within a unit cell
        cell_iao = lo.iao.reference_mol(cell)
        frag_lolist = autofrag_iao(cell_iao)

        # Supercell LNO:
        ''' In principle, one needs to treat all fragments within the supercell. But here
            the supercell SCF object is from `k2s_scf`. As a result, the MOs, the IAOs and
            the fragments are all translationally invariant. We only need to treat fragments
            from the first unit cell, i.e., same as k-point LNO.
        '''
        sfrag_lolist = frag_lolist

        gamma = 10
        threshs = [1e-5,1e-6,1e-100]
        for thresh in threshs:
            # mcc = LNOCCSD_T(mf, lo_coeff, frag_lolist, frozen=frozen).set(verbose=5)
            kmcc = KLNOCCSD(kmf, lo_coeff, frag_lolist, mf=smf, frozen=frozen).set(verbose=5)
            kmcc.lno_thresh = [thresh*gamma,thresh]
            kmcc.kernel()
            kemp2 = kmcc.e_corr_pt2
            keccsd = kmcc.e_corr_ccsd
            keccsd_t = kmcc.e_corr_ccsd_t

            mcc = LNOCCSD(smf, lo_coeff, frag_lolist, frozen=frozen).set(verbose=5)
            mcc.lno_thresh = [thresh*gamma,thresh]
            mcc.kernel()

            semp2 = mcc.e_corr_pt2
            seccsd = mcc.e_corr_ccsd
            seccsd_t = mcc.e_corr_ccsd_t
            # print('[%s],' % (','.join([f'{x:.10f}' for x in [semp2,seccsd,seccsd_t]])))
            self.assertAlmostEqual(kemp2, semp2, 6)
            self.assertAlmostEqual(keccsd, seccsd, 6)
            # self.assertAlmostEqual(keccsd_t, seccsd_t, 6)


class ShiftedKmesh(unittest.TestCase):
    ''' KLNO on k-point meshes closed under k -> -k but without Gamma (half-shifted along
        some reciprocal axes). The supercell is then at a time-reversal invariant twist with
        real orbitals, and KLNO-CCSD(T) with all LNOs kept must equal canonical KCCSD(T).
    '''
    @classmethod
    def setUpClass(cls):
        cell = gto.Cell()
        cell.verbose = 0
        cell.atom = 'H 0 0 0; H 0 0 0.74; H 1.85 0 0; H 1.85 0 0.74'
        cell.a = np.diag([3.7, 4.0, 4.0])
        cell.basis = '6-31g'
        cell.precision = 1e-10
        cell.build()
        cls.cell = cell

    @classmethod
    def tearDownClass(cls):
        del cls.cell

    def _kmf(self, kmesh, scaled_center, exxdiv='ewald'):
        kpts = self.cell.make_kpts(kmesh, scaled_center=scaled_center)
        kmf = scf.KRHF(self.cell, kpts=kpts).density_fit()
        kmf.exxdiv = exxdiv
        return kmf.run(conv_tol=1e-11)

    def _klno(self, kmf, smf=None, ccsd_t=True, run=None, **kwargs):
        if smf is None: smf = k2s_scf(kmf)
        kocc_coeff = [c[:,o>1e-6] for c,o in zip(kmf.mo_coeff, kmf.mo_occ)]
        lo_coeff = k2s_iao(self.cell, kocc_coeff, kmf.kpts, minao='sto-3g', orth=True)
        frag_lolist = autofrag_iao(lo.iao.reference_mol(self.cell, 'sto-3g'))
        mcc = (KLNOCCSD_T if ccsd_t else KLNOCCSD)(kmf, lo_coeff, frag_lolist, mf=smf)
        mcc.lno_thresh = [1e-100, 1e-100]
        mcc.kwargs_imp = {'conv_tol': 1e-10, 'conv_tol_normt': 1e-8}
        mcc.set(**kwargs)
        if run is None:
            mcc.kernel()
        else:
            run(mcc)
        return mcc, lo_coeff, frag_lolist

    def test_is_trs_kmesh(self):
        cell = self.cell
        for kmesh, center, ref in (([2,2,1], [0,0,0], True),
                                   ([2,2,1], [0.25,0.25,0], True),
                                   ([3,1,1], [1/6,0,0], True),
                                   ([2,2,1], [0.25,0,0], True),
                                   ([2,1,1], [0.1,0,0], False)):
            kpts = cell.make_kpts(kmesh, scaled_center=center)
            self.assertEqual(is_trs_kmesh(cell, kpts), ref)

    def test_k2s_scf_real(self):
        kmf = self._kmf([2,1,1], [0.25,0,0], exxdiv=None)
        smf = k2s_scf(kmf)
        self.assertTrue(np.isrealobj(smf.mo_coeff))
        self.assertTrue(np.isrealobj(smf.get_ovlp()))
        s_twist = smf.cell.pbc_intor('int1e_ovlp', hermi=1, kpts=kmf.kpts[0])
        self.assertAlmostEqual(abs(smf.get_ovlp() - s_twist).max(), 0, 9)
        self.assertIsNone(smf.exxdiv)

    def test_klnoccsd_t_canonical(self):
        from pyscf.pbc import cc
        from pyscf.pbc.cc import kccsd_t_rhf
        for kmesh, center in (([2,1,1], [0.25,0,0]),
                              ([3,1,1], [1/6,0,0])):
            kmf = self._kmf(kmesh, center)
            kcc = cc.KRCCSD(kmf).set(conv_tol=1e-10, conv_tol_normt=1e-8).run()
            et = kccsd_t_rhf.kernel(kcc, kcc.ao2mo())
            mcc = self._klno(kmf)[0]
            self.assertAlmostEqual(mcc.e_corr_ccsd, kcc.e_corr, 7)
            self.assertAlmostEqual(mcc.e_corr_ccsd_t, kcc.e_corr+et, 7)

    def test_klnoccsd_t_matches_supercell(self):
        kmf = self._kmf([2,1,1], [0.25,0,0])
        smf = k2s_scf(kmf)
        kmcc, lo_coeff, frag_lolist = self._klno(kmf, smf)
        nkpts = len(kmf.kpts)
        nlo = lo_coeff.shape[1] // nkpts
        sfrag_lolist = [[r*nlo+i for i in f] for r in range(nkpts) for f in frag_lolist]
        mcc = LNOCCSD_T(smf, lo_coeff, sfrag_lolist)
        mcc.lno_thresh = [1e-100, 1e-100]
        mcc.kwargs_imp = {'conv_tol': 1e-10, 'conv_tol_normt': 1e-8}
        mcc.kernel()
        self.assertAlmostEqual(kmcc.e_corr_ccsd, mcc.e_corr_ccsd/nkpts, 7)
        self.assertAlmostEqual(kmcc.e_corr_ccsd_t, mcc.e_corr_ccsd_t/nkpts, 7)

    def test_exxdiv_independent(self):
        ''' The Madelung shift is removed from the k2s Fock iff it is in kmf.mo_energy. '''
        ecc = [self._klno(self._kmf([2,1,1], [0.25,0,0], exxdiv=exxdiv),
                          ccsd_t=False)[0].e_corr_ccsd
               for exxdiv in ('ewald', None)]
        self.assertAlmostEqual(ecc[0], ecc[1], 7)

    def test_klnoccsd_t_outcore(self):
        ''' Outcore ovL for the LNO step, alone and with the DF impurity CCSD integrals
            (vvL), against incore energies. The aux blocks are split and every q block is
            zero padded (naux > naux_by_q) so the blocked and padded writes are exercised.
        '''
        from pyscf import lib
        from pyscf.pbc.lno import klno, klnoccsd
        from pyscf.pbc.lno import tools as ktools
        kmf = self._kmf([2,1,1], [0.25,0,0])
        # incore KLNO-CCSD(T): e_corr_pt2, e_corr_ccsd, e_corr_ccsd_t
        ref = [-0.0334853473, -0.0482735652, -0.0483090353]
        naux_pad = property(lambda k2sdf: k2sdf.get_naoaux().max() + 3)
        for force_dfkcc in (False, True):
            with lib.temporary_env(klno, DEBUG_BLKSIZE=True), \
                 lib.temporary_env(klnoccsd, DEBUG_BLKSIZE=True, FORCE_DFKCC=force_dfkcc), \
                 lib.temporary_env(ktools.K2SDF, naux=naux_pad):
                mcc = self._klno(kmf, lno_thresh=[1e-5, 1e-6], force_outcore_ao2mo=True)[0]
            self.assertAlmostEqual(mcc.e_corr_pt2, ref[0], 8)
            self.assertAlmostEqual(mcc.e_corr_ccsd, ref[1], 8)
            self.assertAlmostEqual(mcc.e_corr_ccsd_t, ref[2], 8)

    def test_klnoccsd_t_restart(self):
        ''' Restart from the chkfile of an outcore run: fragment 0 reuses its LNOs,
            fragment 1 is redone from scratch and the others are restored.
        '''
        kmf = self._kmf([2,1,1], [0.25,0,0])
        smf = k2s_scf(kmf)
        ref = [-0.0334853473, -0.0482735652, -0.0483090353]
        calls = []
        m1 = self._klno(kmf, smf, lno_thresh=[1e-5, 1e-6], force_outcore_ao2mo=True,
                        run=lambda mcc: calls.append(kernel_counted(mcc)))[0]
        nfrag = m1.nfrag
        self.assertEqual(calls[0], {'ao2mo': 1, 'make_las': nfrag, 'impurity_solve': nfrag})
        self.assertAlmostEqual(m1.e_corr_pt2, ref[0], 8)
        self.assertAlmostEqual(m1.e_corr_ccsd, ref[1], 8)
        self.assertAlmostEqual(m1.e_corr_ccsd_t, ref[2], 8)

        def restart(mcc):
            mcc.load_chk(m1.chkfile)
            mcc.frag_eners[0] = None
            mcc.frag_lno[1] = mcc.frag_eners[1] = None
            calls.append(kernel_counted(mcc))
        m2 = self._klno(kmf, smf, lno_thresh=[1e-5, 1e-6], run=restart)[0]
        self.assertEqual(calls[1], {'ao2mo': 1, 'make_las': 1, 'impurity_solve': 2})
        for key in ['efrag_pt2', 'efrag_cc', 'efrag_cc_t']:
            self.assertAlmostEqual(abs(getattr(m2, key) - getattr(m1, key)).max(), 0, 9)



if __name__ == "__main__":
    print("Full Tests for LNO-CCSD and LNO-CCSD(T)")
    unittest.main()
