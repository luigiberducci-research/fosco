import unittest

import torch
from fosco.learner import make_learner
from fosco.systems import make_system
from fosco.systems.uncertainty import add_uncertainty


class TestLearner(unittest.TestCase):
    def test_learner_cbf_save_load(self):
        f = make_system("SingleIntegrator")()
        learner_type = make_learner(system=f)
        learner1 = learner_type(
            state_size=f.n_vars,
            hidden_sizes=(5, 5),
            activation=("relu", "relu"),
            epochs=100,
            optimizer="adam",
            lr=3e-4,
            weight_decay=1e-5,
            loss_margins=0.0,
            loss_weights=1.0,
            loss_relu="relu",
        )

        outdir = "tmp/models"
        model_name = "learner"
        config_path = learner1.save(outdir=outdir, model_name=model_name)

        learner2 = learner_type.load(config_path=config_path)

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
                f"expected the same parameters after loading, got {list(learner1.parameters())} and {list(learner2.parameters())}",
            )

        # remove tmp dir
        import shutil

        shutil.rmtree("tmp")

    def test_learner_rcbf_save_load(self):
        f = make_system("SingleIntegrator")()
        f = add_uncertainty("AdditiveBounded", system=f)

        learner_type = make_learner(system=f)
        learner1 = learner_type(
            state_size=f.n_vars,
            hidden_sizes=(5, 5),
            activation=("relu", "relu"),
            epochs=100,
            optimizer="adam",
            lr=3e-4,
            weight_decay=1e-5,
            loss_margins=0.0,
            loss_weights=1.0,
            loss_relu="relu",
        )

        outdir = "tmp/models"
        model_name = "learner"
        config_path = learner1.save(outdir=outdir, model_name=model_name)

        learner2 = learner_type.load(config_path=config_path)

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
                f"expected the same parameters after loading, got {list(learner1.parameters())} and {list(learner2.parameters())}",
            )

        # remove tmp dir
        import shutil

        shutil.rmtree("tmp")
