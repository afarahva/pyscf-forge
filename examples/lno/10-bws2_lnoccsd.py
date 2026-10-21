#!/usr/bin/env python
r'''
BW-s2(alpha)-based LNO-CCSD (Carter-Fenk & Head-Gordon, arXiv:2303.06271;
Carter-Fenk, Shee & Head-Gordon, arXiv:2309.01376).

lno_amp = 'bws2' builds the LNOs from BW-s2 amplitudes; with the default
pt2_amp = None the same amplitudes also give the fragment low-level energy and the
impurity CCSD guess. `alpha` is the regularizer strength: alpha = 0 is MP2, alpha = 1
the original BW-s2, alpha = 4 the molecular optimum and alpha = 2 suggested for solids.

BW-s2 dresses the occupied orbitals self-consistently, (F_oo + alpha W) U = U e~, and
a BW-s2 amplitude is an MP2 amplitude in the dressed occupied basis, so the LNO
density matrices use the MP2 code on the rotated DF integrals. The full-system
BW-s2 energy is stored in `e_corr_bws2_full` and passed to the external correction
like the full-system MP2 energy.
'''

import numpy as np
from pyscf import gto, scf, lo, mp
from pyscf.lno import LNOCCSD

mol = gto.M(atom='''
O   -1.485163346097   -0.114724564047    0.000000000000
H   -1.868415346097    0.762298435953    0.000000000000
H   -0.533833346097    0.040507435953    0.000000000000
O    1.416468653903    0.111264435953    0.000000000000
H    1.746241653903   -0.373945564047   -0.758561000000
H    1.746241653903   -0.373945564047    0.758561000000
''', basis='cc-pvdz', verbose=4)
mf = scf.RHF(mol).density_fit().run()
frozen = 2

orbocc = mf.mo_coeff[:,frozen:np.count_nonzero(mf.mo_occ)]
lo_coeff = lo.PipekMezey(mol, orbocc).kernel()
frag_lolist = [[i] for i in range(lo_coeff.shape[1])]
lno_thresh = [1e-4, 1e-5]

# MP2-based LNO-CCSD
mlno = LNOCCSD(mf, lo_coeff, frag_lolist, frozen=frozen)
mlno.lno_thresh = lno_thresh
mlno.kernel()
emp2 = mp.MP2(mf, frozen=frozen).run().e_corr
ecorr_mp2 = mlno.e_corr_ccsd_pt2corrected(emp2)

# BW-s2(alpha)-based LNO-CCSD
ecorr_bw = {}
for alpha in [1., 4.]:
    mbw = LNOCCSD(mf, lo_coeff, frag_lolist, frozen=frozen)
    mbw.lno_thresh = lno_thresh
    mbw.lno_amp = 'bws2'
    mbw.alpha = alpha
    mbw.kernel()
    ecorr_bw[alpha] = mbw.e_corr_ccsd_pt2corrected(mbw.e_corr_bws2_full)

print('LNO-CCSD E_corr  MP2 LNO + dMP2              = %.8f' % ecorr_mp2)
for alpha, e in ecorr_bw.items():
    print('LNO-CCSD E_corr  BW-s2(%g) LNO + dBW-s2(%g)   = %.8f' % (alpha, alpha, e))
