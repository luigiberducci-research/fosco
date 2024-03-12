import unittest

import torch
from fosco.learner import make_learner
from fosco.systems import make_system


class TestLearner(unittest.TestCase):
    def test_save_load(self):
        from fosco.common.consts import TimeDomain
        from fosco.common.consts import CertificateType

        f = make_system("SingleIntegrator")()
        learner_type = make_learner(system=f, time_domain=TimeDomain.CONTINUOUS)
        learner1 = learner_type(
            state_size=f.n_vars,
            learn_method=CertificateType.CBF,
            hidden_sizes=(5, 5),
            activation=("relu", "relu"),
            optimizer="adam",
            lr=3e-4,
            weight_decay=1e-5,
        )

        model_path = "tmp/models/barrier.pt"
        learner1.save(model_path=model_path)

        learner2 = learner_type(
            state_size=f.n_vars,
            learn_method=CertificateType.CBF,
            hidden_sizes=(5, 5),
            activation=("relu", "relu"),
            optimizer="adam",
            lr=3e-4,
            weight_decay=1e-5,
        )

        # test before loading
        state1 = learner1.state_dict()
        state2 = learner2.state_dict()
        self.assertTrue(
            all([k in state2 for k in state1]),
            f"expected same modules, got {state1.keys()} and {state2.keys()}",
        )
        self.assertTrue(
            all([k in state1 for k in state2]),
            f"expected same modules, got {state1.keys()} and {state2.keys()}",
        )
        for k in state1:
            self.assertFalse(
                torch.allclose(state1[k], state2[k]),
                f"expected not the same parameters before loading, got {learner1.parameters()} and {learner2.parameters()}",
            )

        # test loading
        learner2.load(model_path=model_path)

        state1 = learner1.state_dict()
        state2 = learner2.state_dict()
        self.assertTrue(
            all([k in state2 for k in state1]),
            f"expected same modules, got {state1.keys()} and {state2.keys()}",
        )
        self.assertTrue(
            all([k in state1 for k in state2]),
            f"expected same modules, got {state1.keys()} and {state2.keys()}",
        )
        for k in state1:
            self.assertTrue(
                torch.allclose(state1[k], state2[k]),
                f"expected the same parameters before loading, got {learner1.parameters()} and {learner2.parameters()}",
            )
