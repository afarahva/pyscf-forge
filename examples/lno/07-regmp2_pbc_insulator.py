#!/usr/bin/env python
r'''
kappa-regularized MP2 (kappa-MP2) in periodic LNO-CCSD -- INSULATOR (diamond).

Companion of examples/lno/08-regmp2_pbc_metal.py.  Both run k-point LNO-CCSD
(KLNOCCSD) on a 2x2x2 mesh with a small double-zeta basis and compare three
calculations:
    baseline    : plain MP2 guess / plain LNO density matrices
    regmp2_cc   : kappa-MP2 only for the fragment MP2 energy and CCSD guess
    regmp2_lno  : kappa-MP2 only for the LNO density matrices (active space)

For a wide-gap insulator the MP2 denominators are large, so the damping factor
g = 1 - exp(-kappa*Delta) is close to 1 everywhere.  Regularization therefore
barely changes anything: the MP2 energy shifts only a little, and both regmp2_cc
and regmp2_lno give essentially the baseline LNO-CCSD energy.  Contrast this with
the metal in example 08.
'''

import numpy as np
from pyscf.pbc import gto, scf
from pyscf import lo
from pyscf.pbc.lno.tools import k2s_scf, sort_orb_by_cell
from pyscf.pbc.lno import KLNOCCSD

# --- diamond (2-atom fcc primitive cell), small DZ basis + GTH pseudopotential
cell = gto.Cell()
cell.atom = 'C 0 0 0; C 0.8917 0.8917 0.8917'
cell.a = np.array([[0.,      1.7834, 1.7834],
                   [1.7834,  0.,     1.7834],
                   [1.7834,  1.7834, 0.    ]])
cell.basis = 'gth-dzv'
cell.pseudo = 'gth-pade'
cell.verbose = 3
cell.build()

kmesh = [2, 2, 2]
kpts = cell.make_kpts(kmesh)
kmf = scf.KRHF(cell, kpts=kpts).density_fit()
kmf.kernel()

# supercell mean-field + PM-localized occupied orbitals (BvK supercell)
mf = k2s_scf(kmf)
orbocc = mf.mo_coeff[:, mf.mo_occ > 1e-6]
mlo = lo.PipekMezey(mf.cell, orbocc)
lo_coeff = mlo.kernel()
while True:  # jacobi sweeps to avoid a local minimum
    lo_coeff1 = mlo.stability_jacobi()[1]
    if lo_coeff1 is lo_coeff:
        break
    mlo = lo.PipekMezey(mf.mol, lo_coeff1)
    mlo.init_guess = None
    lo_coeff = mlo.kernel()

# sort LOs by unit cell; one fragment per LO in the reference cell
s1e = mf.get_ovlp()
Nk = len(kpts)
nlo = lo_coeff.shape[1] // Nk
lo_coeff = sort_orb_by_cell(mf.cell, lo_coeff, Nk, s=s1e)
frag_lolist = [[i] for i in range(nlo)]

def run(**kw):
    mcc = KLNOCCSD(kmf, lo_coeff, frag_lolist, mf=mf)
    mcc.lno_thresh = [1e-5, 1e-6]
    mcc.kappa = 1.1
    for k, v in kw.items():
        setattr(mcc, k, v)
    mcc.kernel()
    return mcc.e_corr_pt2, mcc.e_corr_ccsd   # per unit cell

base = run()
cc   = run(regmp2_cc=True)
lno  = run(regmp2_lno=True)

print()
print('Diamond (insulator), KLNO-CCSD per cell, 2x2x2, gth-dzv, kappa=1.1')
print('                          E_MP2            E_CCSD')
print('baseline            % .10f  % .10f' % base)
print('regmp2_cc           % .10f  % .10f' % cc)
print('regmp2_lno          % .10f  % .10f' % lno)
print()
print('CCSD change vs baseline:  regmp2_cc % .2e Eh   regmp2_lno % .2e Eh'
      % (cc[1] - base[1], lno[1] - base[1]))
print('Wide gap -> weak damping: MP2 barely shifts and both regmp2_cc and')
print('regmp2_lno reproduce the baseline LNO-CCSD energy. See example 08 (metal).')
