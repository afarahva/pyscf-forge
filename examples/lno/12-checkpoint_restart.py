#!/usr/bin/env python

r'''
Checkpoint / restart for LNO calculations.

`kernel()` writes a chkfile as it goes:
  * after a fragment's LNO active space is built  -> 'lno/frag<i>/las'
  * after its impurity problem is solved          -> 'lno/frag<i>/energy'
plus any full-system low-level energies (RPA / BW-s2).  A later run reads this file with
`load_chk()`, which populates `frag_lno` (LNOs that are built) and `frag_eners` (fragments
that are finished).  The next `kernel()` then only computes what is missing:

  * a fragment with a stored energy is skipped entirely;
  * a fragment with stored LNOs but no energy skips the (expensive) LNO build and only
    solves its impurity problem;
  * everything else is computed from scratch.

By default `chkfile` is a temporary file removed with the object.  Assign a filename to
keep it for restart.
'''

import os
import numpy as np
from pyscf import gto, scf, lo
from pyscf.lno import LNOCCSD

mol = gto.M(atom='''
O   0.000000 0.000000  0.000000
H   0.758602 0.000000  0.504284
H   0.260455 0.000000 -0.872893
O   3.000000 0.500000  0.000000
H   3.758602 0.500000  0.504284
H   3.260455 0.500000 -0.872893
''', basis='cc-pvdz', verbose=4, max_memory=8000)
mf = scf.RHF(mol).density_fit().run()

orbocc = mf.mo_coeff[:, mf.mo_occ > 1e-6]
nocc = orbocc.shape[1]
lo_coeff = lo.PipekMezey(mol, orbocc).kernel()
frag_lolist = [[i] for i in range(nocc)]
lno_thresh = [1e-5, 1e-6]

CHK = 'lno_restart_demo.chk'
if os.path.exists(CHK):
    os.remove(CHK)

#
# 1. A full run that keeps its chkfile.
#
mcc = LNOCCSD(mf, lo_coeff, frag_lolist, lno_thresh=lno_thresh)
mcc.chkfile = CHK            # assign a name so the file is kept
mcc.kernel()
e_ref = mcc.e_corr_ccsd
print('\n[1] full run       E_corr(LNO-CCSD) = % .12f' % e_ref)

#
# 2. Restart from the chkfile in a fresh object. load_chk() fills frag_lno and frag_eners,
#    so every fragment is "restored" and nothing is recomputed.
#
mcc2 = LNOCCSD(mf, lo_coeff, frag_lolist, lno_thresh=lno_thresh)
mcc2.chkfile = CHK           # same file -> the restarted run keeps writing to it
mcc2.load_chk()
print('    restored LNOs    : %d / %d fragments'
      % (sum(x is not None for x in mcc2.frag_lno), nocc))
print('    restored energies: %d / %d fragments'
      % (sum(x is not None for x in mcc2.frag_eners), nocc))
mcc2.kernel()                # all fragments restored; no CCSD is run
print('[2] restart (full) E_corr(LNO-CCSD) = % .12f' % mcc2.e_corr_ccsd)
assert abs(mcc2.e_corr_ccsd - e_ref) < 1e-9

#
# 3. Interrupted run: LNOs are on disk but the impurity solves did not finish. Here we
#    simulate it by keeping frag_lno and clearing frag_eners. The restart reuses the LNOs
#    (no rebuild) and only reruns the impurity problems.
#
mcc3 = LNOCCSD(mf, lo_coeff, frag_lolist, lno_thresh=lno_thresh)
mcc3.load_chk(CHK)
mcc3.frag_eners = [None] * nocc      # pretend the CCSD solves were lost
print('\n[3] restart with LNOs only: impurity problems rerun, LNOs reused')
mcc3.kernel()
print('    E_corr(LNO-CCSD) = % .12f' % mcc3.e_corr_ccsd)
assert abs(mcc3.e_corr_ccsd - e_ref) < 1e-9

os.remove(CHK)
print('\nAll restart paths reproduce the full-run energy.')
