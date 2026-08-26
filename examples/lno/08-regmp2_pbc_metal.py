#!/usr/bin/env python
r'''
kappa-regularized MP2 (kappa-MP2) in periodic LNO-CCSD -- METAL (bcc lithium).

Companion of examples/lno/07-regmp2_pbc_insulator.py.  Same 2x2x2 / small-DZ
setup, comparing baseline vs regmp2_cc vs regmp2_lno.

Unlike the insulator, a metal has small occupied-virtual gaps, so some MP2
denominators are tiny and the plain MP2 amplitudes are large.  The damping
factor g = 1 - exp(-kappa*Delta) then acts strongly, and the choice of where
kappa-MP2 is applied matters a lot:

  * regmp2_cc  strongly changes the fragment MP2 correlation energy (the CCSD
    guess).  For the small 2x2x2 mesh below the impurity CCSD still converges
    and lands near the baseline CCSD, but the guess it started from is very
    different.  For denser meshes / smaller gaps the un-regularized MP2
    amplitudes can grow so large that seeding CCSD with them (or with the
    heavily-damped kappa guess) makes the impurity CCSD hard to converge --
    watch for "Impurity CCSD did not converge" warnings in the log.
  * regmp2_lno changes which local natural orbitals are selected, i.e. the
    active space itself, so its effect does not simply cancel in CCSD.

Compare the size of the MP2 shift here with the insulator in example 07.
'''

import numpy as np
from pyscf.pbc import gto, scf
from pyscf import lo
from pyscf.pbc.lno.tools import k2s_scf, sort_orb_by_cell
from pyscf.pbc.lno import KLNOCCSD

# --- bcc lithium (2-atom cubic cell), small DZ basis + GTH pseudopotential
a0 = 3.51
cell = gto.Cell()
cell.atom = 'Li 0 0 0; Li %f %f %f' % (a0/2, a0/2, a0/2)
cell.a = np.eye(3) * a0
cell.basis = 'gth-dzv'
cell.pseudo = 'gth-pade'
cell.verbose = 3
cell.build()

kmesh = [2, 2, 2]
kpts = cell.make_kpts(kmesh)
kmf = scf.KRHF(cell, kpts=kpts).density_fit()
kmf.kernel()

mf = k2s_scf(kmf)
orbocc = mf.mo_coeff[:, mf.mo_occ > 1e-6]
mlo = lo.PipekMezey(mf.cell, orbocc)
lo_coeff = mlo.kernel()
while True:
    lo_coeff1 = mlo.stability_jacobi()[1]
    if lo_coeff1 is lo_coeff:
        break
    mlo = lo.PipekMezey(mf.mol, lo_coeff1)
    mlo.init_guess = None
    lo_coeff = mlo.kernel()

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
print('bcc Li (metal), KLNO-CCSD per cell, 2x2x2, gth-dzv, kappa=1.1')
print('                          E_MP2            E_CCSD')
print('baseline            % .10f  % .10f' % base)
print('regmp2_cc           % .10f  % .10f' % cc)
print('regmp2_lno          % .10f  % .10f' % lno)
print()
print('MP2 shift (regmp2_cc):  % .3e Eh  (%.0f%% of the baseline MP2 energy)'
      % (cc[0] - base[0], 100*abs((cc[0]-base[0])/base[0])))
print('Small gaps -> strong damping: the MP2 energy / CCSD guess changes far')
print('more than for the insulator in example 07.')
