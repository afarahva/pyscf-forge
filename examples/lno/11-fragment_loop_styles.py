#!/usr/bin/env python

r'''
Two equivalent ways to run an LNO calculation over many fragments, and their very
different memory profiles.

Style A -- one combined `kernel()` call (the default).
    All fragment LNOs are built first, then all impurity problems are solved. The
    full-system integrals are transformed only once, but every fragment's LNOs are
    held in memory at the same time.  Peak memory ~ (integrals) + (all LNOs).

Style B -- one fragment at a time, sharing a single in-core integral transform.
    The full-system integrals are built once and passed to each single-fragment
    `kernel(eris=eris)` call, so they are not rebuilt; only one fragment's LNOs live
    at a time.  Peak memory ~ (integrals) + (one fragment's LNOs).  This reproduces
    the memory profile of the pre-two-phase code.

Because the per-fragment energies are strictly additive (the total correlation energy
is just their sum, with unit `frag_wghtlist`), the two styles give identical energies.
This example checks that.
'''

import numpy as np
from pyscf import gto, scf, lo
from pyscf.lno import LNOCCSD

atom = '''
O   0.000000 0.000000  0.000000
H   0.758602 0.000000  0.504284
H   0.260455 0.000000 -0.872893
O   3.000000 0.500000  0.000000
H   3.758602 0.500000  0.504284
H   3.260455 0.500000 -0.872893
'''
mol = gto.M(atom=atom, basis='cc-pvdz', verbose=4, max_memory=8000)
mf = scf.RHF(mol).density_fit().run()

# Pipek-Mezey LOs, one fragment per occupied LO.
orbocc = mf.mo_coeff[:, mf.mo_occ > 1e-6]
nocc = orbocc.shape[1]
lo_coeff = lo.PipekMezey(mol, orbocc).kernel()
frag_lolist = [[i] for i in range(nocc)]
lno_thresh = [1e-5, 1e-6]

#
# Style A: a single combined kernel() call.
#
mcc_A = LNOCCSD(mf, lo_coeff, frag_lolist, lno_thresh=lno_thresh)
mcc_A.kernel()
e_ccsd_A = mcc_A.e_corr_ccsd
e_pt2_A = mcc_A.e_corr_pt2

#
# Style B: one fragment at a time, sharing a single in-core integral transform.
#
nfrag = len(frag_lolist)
efrag_cc = np.zeros(nfrag)
efrag_pt2 = np.zeros(nfrag)

eris = None
for i, loidx in enumerate(frag_lolist):
    mcc_i = LNOCCSD(mf, lo_coeff, [loidx], lno_thresh=lno_thresh).set(verbose=4)
    if eris is None:
        # Built once.  ao2mo() chooses in-core automatically when the integrals fit in
        # `max_memory`; with enough memory the same in-core object serves every fragment.
        eris = mcc_i.ao2mo()
    mcc_i.kernel(eris=eris)          # reused, not rebuilt or destroyed
    efrag_cc[i] = mcc_i.e_corr_ccsd  # single-fragment total == this fragment's energy
    efrag_pt2[i] = mcc_i.e_corr_pt2
    del mcc_i                        # frees only this fragment's LNOs

e_ccsd_B = efrag_cc.sum()
e_pt2_B = efrag_pt2.sum()

print()
print('                       E_corr(LNO-CCSD)        E_corr(PT2)')
print('Style A (combined)  % .12f    % .12f' % (e_ccsd_A, e_pt2_A))
print('Style B (per-frag)  % .12f    % .12f' % (e_ccsd_B, e_pt2_B))
print('|A - B|             % .3e        % .3e'
      % (abs(e_ccsd_A - e_ccsd_B), abs(e_pt2_A - e_pt2_B)))
assert abs(e_ccsd_A - e_ccsd_B) < 1e-9
assert abs(e_pt2_A - e_pt2_B) < 1e-9
print()
print('Both styles agree. Use the delta-PT2 correction the same way in either case:')
print('    e_ccsd_corrected = e_corr_ccsd - e_corr_pt2 + efull_mp2')
