import logging

import torch

from fosco.common.timing import timed
from fosco.logger import LOGGING_LEVELS


class Consolidator:
    # todo: parameterize data augmentation with sampling in the neighbourhood of the cex
    # todo: parameterize neighborhood size
    def __init__(self, resampling_n: int, resampling_stddev: float, verbose: int = 0):
        self.cex_n = resampling_n
        self.cex_stddev = resampling_stddev

        self._assert_state()

        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(LOGGING_LEVELS[verbose])
        self._logger.debug("Consolidator initialized")

    def _assert_state(self) -> None:
        assert (
                self.cex_n > 0
        ), f"Nr of counterexamples must be greater than 0, got {self.cex_n}"
        assert (
                self.cex_stddev > 0
        ), f"Standard deviation must be greater than 0, got {self.cex_stddev}"

    @timed
    def get(self, cex, datasets, **kwargs):
        datasets = self.add_ces_to_data(cex, datasets)
        # todo: return logging info about data augmentation

        self._logger.info(
            f"Dataset sizes: {', '.join([f'{k}: {v.shape[0]}' for k, v in datasets.items()])}"
        )

        return {"datasets": datasets}

    def add_ces_to_data(self, cex, datasets):
        for lab, cex in cex.items():
            if cex != []:
                x = self.randomise_counterex(point=cex)
                datasets[lab] = torch.cat([datasets[lab], x], dim=0).detach()
        return datasets

    def randomise_counterex(self, point):
        """
        Given one ctx, useful to sample around it to increase data set
        these points might *not* be real ctx, but probably close to invalidity condition

        :param point: tensor
        :return: list of ctx
        """
        C = []
        shape = (1, max(point.shape[0], point.shape[1]))
        point = point.reshape(shape)
        for i in range(self.cex_n):
            random_point = point + self.cex_stddev * torch.randn(
                shape
            )
            C.append(random_point)
        C.append(point)
        return torch.stack(C, dim=1)[0, :, :]


def make_consolidator(**kwargs) -> Consolidator:
    """
    Factory method for consolidator.

    :param kwargs:
    :return:
    """
    return Consolidator(**kwargs)
