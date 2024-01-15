import torch


class Consolidator:
    # todo: parameterize data augmentation with sampling in the neighbourhood of the cex
    # todo: parameterize neighborhood size
    def get(self, cex, datasets, **kwargs):
        datasets = self.add_ces_to_data(cex, datasets)
        # todo: return logging info about data augmentation
        return {"datasets": datasets}

    def add_ces_to_data(self, cex, datasets):
        for lab, cex in cex.items():
            if cex != []:
                x = cex
                datasets[lab] = torch.cat([datasets[lab], x], dim=0).detach()
            print(lab, datasets[lab].shape)
        return datasets


def make_consolidator(**kwargs) -> Consolidator:
    """
    Factory method for consolidator.

    :param kwargs:
    :return:
    """
    return Consolidator(**kwargs)
