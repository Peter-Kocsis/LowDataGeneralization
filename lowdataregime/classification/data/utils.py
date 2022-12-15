from _operator import add

import torch
from PIL import ImageStat
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from tqdm.auto import tqdm


class Stats(ImageStat.Stat):
    def __add__(self, other):
        return Stats(list(map(add, self.h, other.h)))


def dataset_statistics(dataset: Dataset, num_workers=4):
    loader = DataLoader(dataset, batch_size=100, num_workers=num_workers)

    statistics = None
    toPIL = transforms.ToPILImage()
    for data, _ in tqdm(loader):
        for b in range(data.shape[0]):
            if statistics is None:
                statistics = Stats(toPIL(data[b]))
            else:
                statistics += Stats(toPIL(data[b]))
    return statistics
