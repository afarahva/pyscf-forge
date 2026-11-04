# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
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

'''
Regularized and renormalized second-order correlation methods (canonical orbitals)

    BWS2     : size-consistent Brillouin-Wigner PT2, BW-s2(alpha) (RHF, DF)
    KappaMP2 : kappa-regularized MP2 (RHF/UHF, conventional or DF)
    DRICCD   : direct ring CCD / direct RPA, with SOSEX (RHF, DF)
'''

from pyscf.regmp import bws2, kappamp2, driccd
from pyscf.regmp.bws2 import BWS2
from pyscf.regmp.kappamp2 import KappaMP2, KappaRMP2, KappaUMP2, KappaDFRMP2, KappaDFUMP2
from pyscf.regmp.driccd import DRICCD
