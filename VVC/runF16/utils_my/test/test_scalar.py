import unittest
import numpy as np
import sys
import os
from collections import namedtuple
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from scalar import get_min_max_scalar
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
from env.guidenv import GuideEnv

from functools import partial

my_all_close = partial(np.allclose, atol=1.e-8)


class Test(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.scalar = get_min_max_scalar(
            feature_range=[0., 1.],
           mins=np.array([-10., -5.]),
           maxs=np.array([10., 5.])
        )

    def test_transform_1(self):
        data = np.array([5., 2.])
        res = self.scalar.transform(data.reshape((1, -1)))
        self.assertTrue(np.allclose(res, np.array([[0.75, 0.7]]), atol=1.e-8))

    def test_inverse_transform_1(self):
        data = np.array([0.75, 0.7])
        res = self.scalar.inverse_transform(data.reshape((1, -1)))
        self.assertTrue(np.allclose(res, np.array([[5., 2.]]), atol=1.e-8))

    def test_bound_1(self):
        data = np.array([5., 6.])
        res = self.scalar.transform(data.reshape((1, -1)))
        self.assertTrue(np.allclose(res, np.array([[0.75, 1.]]), atol=1.e-8))

    def test_bound_2(self):
        data = np.array([-15., 6.])
        res = self.scalar.transform(data.reshape((1, -1)))
        self.assertTrue(np.allclose(res, np.array([[0, 1.]]), atol=1.e-8))


class Test_Action_Data(unittest.TestCase):
    
    def setUp(self) -> None:
        super().setUp()
        self.scalar = get_min_max_scalar(
            mins=GuideEnv.get_action_lower_bounds(),
            maxs=GuideEnv.get_action_higher_bounds(),
            feature_range=[0., 1.]
        )
    
    def test_1(self):
        data = np.array([0., 1., 1.])
        res = self.scalar.inverse_transform(data.reshape((1, -1)))
        self.assertTrue(np.allclose(res, np.array([[-300., 9., 1.]]), atol=1.e-8))
    
    def test_2(self):
        data = np.array([0.5, 0., 0.5])
        res = self.scalar.inverse_transform(data.reshape((1, -1)))
        self.assertTrue(np.allclose(res, np.array([[0., -3., 0.5]]), atol=1.e-8))


class Test_Observation_Data(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.scalar = get_min_max_scalar(
            mins=GuideEnv.get_state_lower_bounds(),
            maxs=GuideEnv.get_state_higher_bounds(),
            feature_range=[0., 1.]
        )
    
    def test_in_bound_1(self):
        state_vars = GuideEnv.get_state_vars()
        data = np.array(state_vars(phi=-90., theta=-90., psi=-180., v=0., mu=-90., chi=-180., p=-300., h=0., target_v=0., target_mu=-90., target_chi=-180.))
        res = self.scalar.transform(data.reshape((1, -1)))
        self.assertTrue(np.allclose(res, np.array([0.25] + [0.]*10).reshape((1, -1)), atol=1.e-8))

    def test_on_bound_1(self):
        state_vars = GuideEnv.get_state_vars()
        data = np.array(state_vars(phi=-180., theta=-90., psi=-180., v=0., mu=-90., chi=-180., p=-300., h=0., target_v=0., target_mu=-90., target_chi=-180.))
        res = self.scalar.transform(data.reshape((1, -1)))
        self.assertTrue(np.allclose(res, np.array([0.]*11).reshape((1, -1)), atol=1.e-8))
    
    def test_exceed_bound_1(self):
        state_vars = GuideEnv.get_state_vars()
        data = np.array(state_vars(phi=-190., theta=-91., psi=-180., v=0., mu=-90., chi=-180., p=-300., h=0., target_v=0., target_mu=-90., target_chi=-180.))
        res = self.scalar.transform(data.reshape((1, -1)))
        self.assertTrue(np.allclose(res, np.array([0.]*11), atol=1.e-8))


if __name__ == "__main__":
    unittest.main()