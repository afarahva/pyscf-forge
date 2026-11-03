#!/usr/bin/env python

r'''
Two equivalent ways to run an LNO calculation over many fragments, and their memory
profiles.

Style A -- one combined `kernel()` call (the default).
    Fragments are done one at a time: each fragment's LNOs are built from the
    full-system integrals and its impurity problem is solved before the next fragment.
    The full-system integrals are transformed once and kept for all fragments.
    Peak memory ~ (full-system integrals) + (one impurity problem).

Style B -- one job per fragment, with `delete_full_eris = True`.
    Each job only treats its own fragment (here: a single-fragment `frag_lolist`; with
    a shared chkfile, `frag_eners` / `load_chk` can be used instead). The full-system
    integrals are only needed for the LNOs, so `delete_full_eris` releases them before
    the impurity problem, leaving all of `max_memory` to the impurity solver.
    Peak memory ~ max(full-system integrals, one impurity problem). The jobs are
    independent and can run in parallel, e.g. as separate cluster jobs.

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
# Style B: one independent job per fragment, releasing the full-system integrals before
# the impurity problem.
#
nfrag = len(frag_lolist)
efrag_cc = np.zeros(nfrag)
efrag_pt2 = np.zeros(nfrag)

for i, loidx in enumerate(frag_lolist):
    mcc_i = LNOCCSD(mf, lo_coeff, [loidx], lno_thresh=lno_thresh)
    mcc_i.delete_full_eris = True
    mcc_i.kernel()
    efrag_cc[i] = mcc_i.e_corr_ccsd  # single-fragment total == this fragment's energy
    efrag_pt2[i] = mcc_i.e_corr_pt2
    del mcc_i

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
