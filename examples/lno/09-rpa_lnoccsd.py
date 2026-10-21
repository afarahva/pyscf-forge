#!/usr/bin/env python
r'''
RPA-based LNO-CCSD(T) (Song, Gong, Bakry, Ye, arXiv:2601.00131).

lno_amp = 'rpa' builds the LNOs from direct-ring CCD (RPA) amplitudes; with the
default pt2_amp = None the same amplitudes also give the fragment low-level energy
and the impurity CCSD guess. 'rpa' uses the direct RPA energy 2 sum T V; 'sosex'
uses the same drCCD amplitudes with the SOSEX energy sum T (2V - V_x).

The full-system drCCD is solved once with a factorized O(N^4) algorithm
(T = -D^{-1} o (U U^T), Cholesky-decomposed denominator). Its energy is stored in
`e_corr_rpa_full` (and `e_corr_sosex_full` for 'sosex') and is passed to the
external correction like the full-system MP2 energy:

    E_corr(LNO-CCSD + dRPA) = mlno.e_corr_ccsd_pt2corrected(mlno.e_corr_rpa_full)

For periodic systems, use KLNOCCSD with real orbitals (Gamma point or time-reversal
symmetric k-meshes); see examples 03 and 08 for the setup.
'''

import numpy as np
from pyscf import gto, scf, lo, cc
from pyscf.lno import LNOCCSD_T

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

gamma = 10
thresh = 1e-5

# MP2-based LNO-CCSD(T)
mlno = LNOCCSD_T(mf, lo_coeff, frag_lolist, frozen=frozen)
mlno.lno_thresh = [thresh*gamma, thresh]
mlno.kernel()

# RPA-based LNO-CCSD(T): RPA LNOs + RPA external correction
mrpa = LNOCCSD_T(mf, lo_coeff, frag_lolist, frozen=frozen)
mrpa.lno_thresh = [thresh*gamma, thresh]
mrpa.lno_amp = 'rpa'
mrpa.kernel()
ecorr_rpa = mrpa.e_corr_ccsd_t_pt2corrected(mrpa.e_corr_rpa_full)

# RPA LNOs + SOSEX external correction
msx = LNOCCSD_T(mf, lo_coeff, frag_lolist, frozen=frozen)
msx.lno_thresh = [thresh*gamma, thresh]
msx.lno_amp = 'sosex'
msx.kernel()
ecorr_sx = msx.e_corr_ccsd_t_pt2corrected(msx.e_corr_sosex_full)

from pyscf import mp
emp2 = mp.MP2(mf, frozen=frozen).run(with_t2=False).e_corr
ecorr_mp2 = mlno.e_corr_ccsd_t_pt2corrected(emp2)

print('LNO-CCSD(T) E_corr  MP2 LNO + dMP2     = %.8f' % ecorr_mp2)
print('LNO-CCSD(T) E_corr  RPA LNO + dRPA     = %.8f' % ecorr_rpa)
print('LNO-CCSD(T) E_corr  RPA LNO + dSOSEX   = %.8f' % ecorr_sx)
