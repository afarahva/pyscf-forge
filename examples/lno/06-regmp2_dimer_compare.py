#!/usr/bin/env python
r'''
kappa-regularized MP2 (kappa-MP2) in LNO-CCSD: water dimer comparison.

kappa-MP2 (arXiv:2508.15744) damps each MP2 amplitude by a single factor
    g_ijab = 1 - exp(-kappa * (e_a + e_b - e_i - e_j)),    kappa = 1.1 (default)
The LNO base class exposes three switches (see LNO.__init__):
    regmp2_lno : use kappa-MP2 amplitudes only in the LNO density matrices
                 (i.e. when choosing the local natural orbital active space)
    regmp2_cc  : use kappa-MP2 amplitudes only for the fragment MP2 energy and
                 the CCSD initial guess
    regmp2     : use kappa-MP2 everywhere (supersedes the two above)

This script runs LNO-CCSD on a water dimer at lno_thresh = [1e-5, 1e-6] with
PM-localized orbitals and compares three calculations:

  1. plain MP2 guess / plain LNO-DM   (baseline)
  2. kappa-MP2 CC guess (regmp2_cc)    -- expected within ~1 mEh of baseline:
     the converged CCSD energy should barely depend on the starting amplitudes.
  3. kappa-MP2 LNO-DM  (regmp2_lno)    -- expected close, but possibly a little
     further from baseline, since the selected active space is not identical.
'''

import numpy as np
from pyscf import gto, scf, lno, lo

atom = '''
O   0.000000 0.000000  0.000000
H   0.758602 0.000000  0.504284
H   0.260455 0.000000 -0.872893
O   3.000000 0.500000  0.000000
H   3.758602 0.500000  0.504284
H   3.260455 0.500000 -0.872893
'''
mol = gto.M(atom=atom, basis='cc-pvdz', spin=0, verbose=3, max_memory=8000)
mf = scf.RHF(mol).density_fit().run()

# PM-localized occupied orbitals; each LO is its own fragment
orbocc = mf.mo_coeff[:, mf.mo_occ > 1e-6]
nocc = orbocc.shape[1]
lo_coeff = lo.PipekMezey(mol, orbocc).kernel()
frag_lolist = [[i] for i in range(nocc)]

lno_thresh = [1e-5, 1e-6]
kappa = 1.1

def run(**kw):
    mcc = lno.LNOCCSD(mf, lo_coeff, frag_lolist, lno_thresh=lno_thresh)
    mcc.kappa = kappa
    for k, v in kw.items():
        setattr(mcc, k, v)
    mcc.kernel()
    return mcc.e_corr_ccsd

e_base = run()
e_cc   = run(regmp2_cc=True)   # kappa-MP2 only for the CC guess / MP2 energy
e_lno  = run(regmp2_lno=True)  # kappa-MP2 only for the LNO density matrices

print()
print('LNO-CCSD correlation energy (water dimer, lno_thresh=%s, kappa=%.2f)'
      % (lno_thresh, kappa))
print('  plain MP2 guess / plain LNO-DM : % .10f' % e_base)
print('  kappa-MP2 CC guess (regmp2_cc) : % .10f  (dE = % .3e Eh)'
      % (e_cc,  e_cc  - e_base))
print('  kappa-MP2 LNO-DM  (regmp2_lno) : % .10f  (dE = % .3e Eh)'
      % (e_lno, e_lno - e_base))
print()
print('Expectation: |dE(regmp2_cc)| < 1 mEh (CCSD is insensitive to its guess);')
print('             regmp2_lno is also close but may differ a bit more, since')
print('             the selected LNO active space is not exactly the same.')

assert abs(e_cc - e_base) < 1e-3, 'kappa-MP2 CC guess changed CCSD by > 1 mEh'
