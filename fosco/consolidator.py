import logging

import torch

from logger import LOGGING_LEVELS


class Consolidator:
    # todo: parameterize data augmentation with sampling in the neighbourhood of the cex
    # todo: parameterize neighborhood size
    def __init__(self, verbose: int = 0):
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(LOGGING_LEVELS[verbose])
        self._logger.debug("Consolidator initialized")

    def get(self, cex, datasets, **kwargs):
        datasets = self.add_ces_to_data(cex, datasets)
        # todo: return logging info about data augmentation

        self._logger.info(f"Dataset sizes: {', '.join([f'{k}: {v.shape[0]}' for k, v in datasets.items()])}")

        return {"datasets": datasets}

    def add_ces_to_data(self, cex, datasets):
        for lab, cex in cex.items():
            if cex != []:
                x = cex
                datasets[lab] = torch.cat([datasets[lab], x], dim=0).detach()
        return datasets


def make_consolidator(**kwargs) -> Consolidator:
    """
    Factory method for consolidator.

    :param kwargs:
    :return:
    """
    return Consolidator(**kwargs)
