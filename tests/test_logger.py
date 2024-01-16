import unittest

import matplotlib.pyplot as plt
import numpy as np


class TestAimLogger(unittest.TestCase):
    def test_make_run(self):
        from logger.logger_aim import AimLogger

        config = {"foo": "bar"}
        logger = AimLogger(config=config, experiment="test")

        a, b = np.random.rand(2) * 10
        for t in range(10):
            # log some scalar
            sinval = a * np.sin(t) + b
            cosval = a * np.cos(t) + b
            logger.log_scalar("sin", value=sinval, step=t)
            logger.log_scalar("cos", value=cosval, step=t)

            # define matplotlib figure
            fig = plt.figure()
            plt.imshow(np.random.rand(10, 10, 3))
            plt.close(fig)
            logger.log_image(tag="random", image=fig, step=t)

        from aim.storage.context import Context
        for metric_name in ["sin", "cos"]:
            metric = logger._run.get_metric(metric_name, context=Context({}))
            self.assertTrue(metric is not None)

        # check existance of image
        image = logger._run.get_image_sequence("random", context=Context({}))
        self.assertTrue(image is not None)

    def test_factory(self):
        from logger import LoggerType
        from logger import make_logger
        from logger.logger_aim import AimLogger

        # logger from type
        logger = make_logger(logger_type=LoggerType.AIM, config={"foo": "bar"})
        self.assertTrue(isinstance(logger, AimLogger))

        # logger from string
        logger = make_logger(logger_type="aim", config={"foo": "bar"})
        self.assertTrue(isinstance(logger, AimLogger))

        # logger wto config
        logger = make_logger(logger_type="aim")
        self.assertTrue(isinstance(logger, AimLogger))