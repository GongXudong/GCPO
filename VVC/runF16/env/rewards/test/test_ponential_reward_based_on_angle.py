import unittest
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

from runF16.env.rewards.ponential_reward_based_on_angle import PonentialRewardBasedOnAngle
from runF16.env.guidenv import GuideEnv
from runF16.utils_my.geometry_utils import angle_of_2_3d_vectors

class PonentialRewardBasedOnAngleTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()

        state_min = GuideEnv.get_state_lower_bounds()
        state_max = GuideEnv.get_state_higher_bounds()
        self.state_var_type = GuideEnv.get_state_vars()
        self.ponential_reward = PonentialRewardBasedOnAngle(
            b=1.,
            # gamma=0.9999,
            gamma=0.99,
        )
    
    def test_1(self):
        state = self.state_var_type(phi=0., theta=0., psi=0., v=170, mu=10., chi=20., p=0., h=0., target_v=200., target_mu=20., target_chi=30.)
        next_state = self.state_var_type(phi=0., theta=0., psi=0., v=180, mu=15., chi=25., p=0., h=0., target_v=200., target_mu=20., target_chi=30.)
        reward = self.ponential_reward.get_reward(state=state, next_state=next_state, done=False, ve=1., vn=0., vh=0., next_ve=2., next_vn=0., next_vh=0.)
        
        target_vector = [
            200 * np.cos(np.deg2rad(20)) * np.sin(np.deg2rad(30)),
            200 * np.cos(np.deg2rad(20)) * np.cos(np.deg2rad(30)),
            200 * np.sin(np.deg2rad(20))
        ]
        angle_current = angle_of_2_3d_vectors(target_vector, [1, 0, 0])
        phi_current = -angle_current / 180.
        angle_next = angle_of_2_3d_vectors(target_vector, [2, 0, 0])
        phi_next = -angle_next / 180.
        print("target vector: ", target_vector)
        print("current phi: ", phi_current)
        print("next phi: ", phi_next)
        reward_calc = 0.99 * phi_next - phi_current
        print(f"ponential reward: {reward}, 手工算的reward: {reward_calc}")
        self.assertAlmostEqual(reward, reward_calc)
    
    def test_2(self):
        state = self.state_var_type(phi=0., theta=0., psi=0., v=170, mu=10., chi=20., p=0., h=0., target_v=200., target_mu=45., target_chi=45.)
        next_state = self.state_var_type(phi=0., theta=0., psi=0., v=180, mu=15., chi=25., p=0., h=0., target_v=200., target_mu=45., target_chi=45.)
        reward = self.ponential_reward.get_reward(state=state, next_state=next_state, done=False, ve=1., vn=1., vh=0., next_ve=2., next_vn=0., next_vh=0.)

        target_vector = [
            200 * np.cos(np.deg2rad(45)) * np.sin(np.deg2rad(45)),
            200 * np.cos(np.deg2rad(45)) * np.cos(np.deg2rad(45)),
            200 * np.sin(np.deg2rad(45))
        ]
        angle_current = angle_of_2_3d_vectors(target_vector, [1, 1, 0])
        phi_current = -angle_current / 180.
        angle_next = angle_of_2_3d_vectors(target_vector, [2, 0, 0])
        phi_next = -angle_next / 180.
        print("target vector: ", target_vector)
        print("current phi: ", phi_current)
        print("next phi: ", phi_next)
        reward_calc = 0.99 * phi_next - phi_current
        print(f"ponential reward: {reward}, 手工算的reward: {reward_calc}")
        self.assertAlmostEqual(reward, reward_calc)
    
    def test_3(self):
        # s_phi,s_theta,s_psi,s_v,s_mu,s_chi,s_p,s_h,s_ve,s_vn,s_vh,a_p,a_nz,a_pla,reward,target_v,target_mu,target_chi
        # -34.82288360595703,-17.509508728981018,-29.585763216018677,266.82206988334656,-35.338425636291504,-24.951635599136353,3.860235214233398,1986.4974915981293,-105.43815494548781,220.25934770312574,-155.05821480514692,-300.0,-1.3537157,1.0,-0.005470006898565805,180.0,0.0,75.00000000000001
        # -35.501686334609985,-18.976736068725586,-26.16441249847412,265.2636766433716,-35.77076554298401,-25.58044195175171,-56.01253509521484,1970.9916412830353,-107.47108208194305,214.36659713400599,-141.14642079431675,-10.004711,-1.3871429,1.0,-0.005240202140804229,180.0,0.0,75.00000000000001
        # -41.987900733947754,-26.214328408241272,-23.997970819473267,265.47208428382874,-32.11909353733063,-26.626557111740112,-26.259148120880127,1956.876963376999,-111.08085403502754,211.06516940922683,-136.6607040574005,-9.111768,0.61908925,1.0,0.0015927842183176555,180.0,0.0,75.00000000000001
        
        self.ponential_reward2 = PonentialRewardBasedOnAngle(
            b=1.,
            # gamma=0.9999,
            gamma=0.999,
        )

        state = self.state_var_type(phi=-34.82288360595703, theta=-17.509508728981018, psi=-29.585763216018677, v=266.82206988334656, mu=-35.338425636291504, chi=-24.951635599136353, p=3.860235214233398, h=1986.4974915981293, target_v=180., target_mu=0., target_chi=75.)
        next_state = self.state_var_type(phi=-35.501686334609985, theta=-18.976736068725586, psi=-26.16441249847412, v=265.2636766433716, mu=-35.77076554298401, chi=-25.58044195175171, p=-56.01253509521484, h=1970.9916412830353, target_v=180., target_mu=0., target_chi=75.)
        next_nexte_state = self.state_var_type(phi=-41.987900733947754, theta=-26.214328408241272, psi=-23.997970819473267, v=265.47208428382874, mu=-32.11909353733063, chi=-26.626557111740112, p=-26.259148120880127, h=1956.876963376999, target_v=180., target_mu=0., target_chi=75.)
        reward = self.ponential_reward2.get_reward(
            state=state, next_state=next_state, done=False, 
            ve=-105.43815494548781, vn=220.25934770312574, vh=-155.05821480514692, 
            next_ve=-107.47108208194305, next_vn=214.36659713400599, next_vh=-141.14642079431675
        )
        self.assertAlmostEqual(reward, -0.005470006898565805)

        reward2 = self.ponential_reward2.get_reward(
            state=next_state, next_state=next_nexte_state, done=False,
            ve=-107.47108208194305, vn=214.36659713400599, vh=-141.14642079431675,
            next_ve=-111.08085403502754, next_vn=211.06516940922683, next_vh=-136.6607040574005
        )
        self.assertAlmostEqual(reward2, -0.005240202140804229)

    def test_4(self):
        # 71.16406917572021,1.2550699710845947,22.964107990264893,192.59043037891388,-1.8747460842132568,14.91070032119751,0.08336305618286133,4988.3270263671875,51.57005997950969,185.4602379507713,-6.5259077746502,-0.08032322,2.3194427,0.004726596,0.004058037898238331,130.0,10.000000000000004,145.0
        # 71.1858057975769,1.1704623699188232,23.59659433364868,192.32948124408722,-1.9444674253463745,15.53941011428833,-0.01460909843444824,4987.674653530121,53.5252457519204,184.6245441776862,-6.754978873489469,-0.15987754,2.3115842,0.0046903268,0.004047272201099683,130.0,10.000000000000004,145.0

        self.ponential_reward2 = PonentialRewardBasedOnAngle(
            b=1.,
            # gamma=0.9999,
            gamma=0.999,
        )

        state = self.state_var_type(phi=-34.82288360595703, theta=-17.509508728981018, psi=-29.585763216018677, v=266.82206988334656, mu=-35.338425636291504, chi=-24.951635599136353, p=3.860235214233398, h=1986.4974915981293, target_v=130., target_mu=10., target_chi=145.)
        next_state = self.state_var_type(phi=-35.501686334609985, theta=-18.976736068725586, psi=-26.16441249847412, v=265.2636766433716, mu=-35.77076554298401, chi=-25.58044195175171, p=-56.01253509521484, h=1970.9916412830353, target_v=130., target_mu=10., target_chi=145.)
        reward = self.ponential_reward2.get_reward(
            state=state, next_state=next_state, done=False, 
            ve=51.57005997950969, vn=185.4602379507713, vh=-6.5259077746502, 
            next_ve=53.5252457519204, next_vn=184.6245441776862, next_vh=-6.754978873489469
        )
        self.assertAlmostEqual(reward, 0.004058037898238331)


if __name__ == "__main__":
    unittest.main()