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
Regularized and renormalized second-order correlation methods with k-point sampling
(canonical orbitals, energies per unit cell)

    KKappaMP2 : kappa-regularized MP2 (KRHF)
    KDRICCD   : direct ring CCD / direct RPA, with SOSEX (KRHF, GDF)
'''

from pyscf.pbc.regpt2 import kkappamp2, kdriccd
from pyscf.pbc.regpt2.kkappamp2 import KKappaMP2
from pyscf.pbc.regpt2.kdriccd import KDRICCD
