import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


class GTSRBDataset(Dataset):
  """The German Traffic Sign Recognition Benchmark dataset."""

  def __init__(self, img_file_path, label_file_path, transform=None, device = 'cuda:0'):
    """
    Args:
    img_file_path (string): Path to the input image file (a `.pt` file).
    label_file_path (string): Path to the corresponding labels of the input image file (a `.pt` file)..
    transform (callable, optional): Optional transform to be applied on a sample.
    """
    self.X = torch.load(img_file_path, map_location=torch.device(device)).squeeze(1)
    self.y = torch.load(label_file_path, map_location=torch.device(device)).squeeze(1)
    self.transform = transform

  def __len__(self):
    return self.X.size(0)

  def __getitem__(self, idx):
    if self.transform:
      return self.transform(self.X[idx]), self.y[idx]
    else:
      return self.X[idx], self.y[idx]


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    https://github.com/ufoym/imbalanced-dataset-sampler
    """

    def __init__(self, dataset, indices=None, num_samples=None):

        # if indices are not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)] for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        return dataset[idx][1].item()

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples

random_transforms = [
    transforms.RandomApply([], p=0),
    transforms.CenterCrop(48),
    transforms.ColorJitter(contrast=5),
    transforms.ColorJitter(hue=0.3),
    transforms.RandomRotation(30),
    transforms.RandomAffine(0, translate=(0.2, 0.2), interpolation=InterpolationMode('bilinear')),
    transforms.RandomAffine(0, shear=20, interpolation=InterpolationMode('bilinear')),
    transforms.RandomAffine(0, scale=(0.8, 1.2), interpolation=InterpolationMode('bilinear')),
]

train_data_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomChoice(random_transforms),
    transforms.Resize((48, 48)),
])

data_transforms = transforms.Compose([
    transforms.Resize((48, 48)),
])


"""
Cut mix implementation.
https://github.com/hysts/pytorch_cutmix/blob/master/cutmix.py
"""

def cutmix(batch, alpha):
    data, targets = batch
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)

    image_h, image_w = data.shape[2:]
    cx = np.random.uniform(0, image_w)
    cy = np.random.uniform(0, image_h)
    w = image_w * np.sqrt(1 - lam)
    h = image_h * np.sqrt(1 - lam)
    x0 = int(np.round(max(cx - w / 2, 0)))
    x1 = int(np.round(min(cx + w / 2, image_w)))
    y0 = int(np.round(max(cy - h / 2, 0)))
    y1 = int(np.round(min(cy + h / 2, image_h)))

    data[:, :, y0:y1, x0:x1] = shuffled_data[:, :, y0:y1, x0:x1]
    targets = (targets, shuffled_targets, lam)

    return data, targets


class CutMixCollator:
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, batch):
        batch = torch.utils.data.dataloader.default_collate(batch)
        batch = cutmix(batch, self.alpha)
        return batch


class CutMixCriterion:
    def __init__(self, reduction):
        self.criterion = nn.CrossEntropyLoss(reduction=reduction)

    def __call__(self, preds, targets):
        targets1, targets2, lam = targets
        return lam * self.criterion(
            preds, targets1) + (1 - lam) * self.criterion(preds, targets2)
