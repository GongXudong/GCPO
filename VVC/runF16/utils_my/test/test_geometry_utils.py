import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
from runF16.utils_my import geometry_utils
from functools import partial

my_all_close = partial(np.allclose, atol=1.e-8)


class Test(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()

    def test_v_mu_chi_2_enh_1(self):
        my_all_close(geometry_utils.v_mu_chi_2_enh(20, 30, 30), (5*np.sqrt(3), 15, 10.))

    def test_v_mu_chi_2_enh_2(self):
        my_all_close(geometry_utils.v_mu_chi_2_enh(20, 45, 45), (10*np.sqrt(2), 10, 10.))


if __name__ == "__main__":
    unittest.main()