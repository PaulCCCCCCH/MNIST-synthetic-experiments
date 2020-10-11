import pickle
import os
from torch.utils.data import DataLoader, Dataset
import torch
from torchvision.datasets import MNIST
import numpy as np
from PIL import Image
from torchvision import transforms


class MyMNIST(Dataset):
    def __init__(self, inputs, labels, transform=None):
        self.inputs = inputs
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        inputs = torch.tensor(self.inputs[index])
        if self.transform:
            inputs = Image.fromarray(np.array(inputs.data), mode='RGB')
            inputs = self.transform(inputs)
        else:
            inputs = inputs.view(-1, 28, 28)
        labels = torch.tensor(self.labels[index]).squeeze()
        print(inputs)
        print(labels)

        return inputs, labels

    def __len__(self):
        return len(self.inputs)


# Adapted from https://github.com/clovaai/rebias/blob/master/datasets/colour_mnist.py
class BiasedMNIST(MNIST):
    """A base class for Biased-MNIST.
    We manually select ten colours to synthetic colour bias. (See `COLOUR_MAP` for the colour configuration)
    Usage is exactly same as torchvision MNIST dataset class.
    You have two paramters to control the level of bias.
    Parameters
    ----------
    root : str
        path to MNIST dataset.
    data_label_correlation : float, default=1.0
        Here, each class has the pre-defined colour (bias).
        data_label_correlation, or `rho` controls the level of the dataset bias.
        A sample is coloured with
            - the pre-defined colour with probability `rho`,
            - coloured with one of the other colours with probability `1 - rho`.
              The number of ``other colours'' is controlled by `n_confusing_labels` (default: 9).
        Note that the colour is injected into the background of the image (see `_binary_to_colour`).
        Hence, we have
            - Perfectly biased dataset with rho=1.0
            - Perfectly unbiased with rho=0.1 (1/10) ==> our ``unbiased'' setting in the test time.
        In the paper, we explore the high correlations but with small hints, e.g., rho=0.999.
    n_confusing_labels : int, default=9
        In the real-world cases, biases are not equally distributed, but highly unbalanced.
        We mimic the unbalanced biases by changing the number of confusing colours for each class.
        In the paper, we use n_confusing_labels=9, i.e., during training, the model can observe
        all colours for each class. However, you can make the problem harder by setting smaller n_confusing_labels, e.g., 2.
        We suggest to researchers considering this benchmark for future researches.
    """

    COLOUR_MAP = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [225, 225, 0], [225, 0, 225],
                  [0, 255, 255], [255, 128, 0], [255, 0, 128], [128, 0, 255], [128, 128, 128]]

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, data_label_correlation=1.0, n_confusing_labels=9, do_shuffle=True):
        super().__init__(root, train=train, transform=transform,
                         target_transform=target_transform,
                         download=download)
        self.data_label_correlation = data_label_correlation
        self.n_confusing_labels = n_confusing_labels
        self.do_shuffle = do_shuffle
        self.data, self.targets, self.biased_targets = self.build_biased_mnist()

        indices = np.arange(len(self.data))

        self._shuffle(indices)
        self.data = self.data[indices].numpy()
        self.targets = self.targets[indices]
        self.biased_targets = self.biased_targets[indices]

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    def _shuffle(self, iteratable):
        if self.do_shuffle:
            np.random.shuffle(iteratable)

    def _update_bias_indices(self, bias_indices, label):
        """
        Modifies bias_indices.
        Args:
            bias_indices: dict {label: [indices]}.

        """
        if self.n_confusing_labels > 9 or self.n_confusing_labels < 1:
            raise ValueError(self.n_confusing_labels)

        indices = np.where((self.targets == label).numpy())[0]
        self._shuffle(indices)
        indices = torch.LongTensor(indices)

        n_samples = len(indices)
        n_correlated_samples = int(n_samples * self.data_label_correlation)
        n_decorrelated_per_class = int(np.ceil((n_samples - n_correlated_samples) / self.n_confusing_labels))

        correlated_indices = indices[:n_correlated_samples]
        bias_indices[label] = torch.cat([bias_indices[label], correlated_indices])

        decorrelated_indices = torch.split(indices[n_correlated_samples:], n_decorrelated_per_class)

        other_labels = [_label % 10 for _label in range(label + 1, label + 1 + self.n_confusing_labels)]
        self._shuffle(other_labels)

        for idx, _indices in enumerate(decorrelated_indices):
            _label = other_labels[idx]
            bias_indices[_label] = torch.cat([bias_indices[_label], _indices])

    def _binary_to_colour(self, data, colour):
        """
        Args:
            data: grey-scale image of shape (N, 28, 28)
        Returns:
            RGB image of shape (N, 28, 28, 3)
        """

        fg_data = torch.zeros_like(data)
        fg_data[data != 0] = 255
        fg_data[data == 0] = 0
        fg_data = torch.stack([fg_data, fg_data, fg_data], dim=1)

        bg_data = torch.zeros_like(data)
        bg_data[data == 0] = 1
        bg_data[data != 0] = 0
        bg_data = torch.stack([bg_data, bg_data, bg_data], dim=3)
        bg_data = bg_data * torch.ByteTensor(colour)
        bg_data = bg_data.permute(0, 3, 1, 2)

        data = fg_data + bg_data
        data = data.permute(0, 2, 3, 1)
        return data

    def _make_biased_mnist(self, indices, label):
        """
        Args:
            indices: indices to be turned into biased images, shaped (N)
            label: a scalar index specifying a target colour to be used
        Returns:
            a tuple (images, labels)
        """
        return self._binary_to_colour(self.data[indices], self.COLOUR_MAP[label]), self.targets[indices]

    def build_biased_mnist(self):
        """
        Returns:
            data: batch of images of shape (N, 28, 28, 3)
            targets: labels of shape (N) corresponding to data
            biased_targets: index of background color to be used, shaped (N)
        """
        n_labels = self.targets.max().item() + 1
        bias_indices = {label: torch.LongTensor() for label in range(n_labels)}

        for label in range(n_labels):
            self._update_bias_indices(bias_indices, label)

        data = torch.ByteTensor()
        targets = torch.LongTensor()
        biased_targets = []

        for bias_label, indices in bias_indices.items():
            _data, _targets = self._make_biased_mnist(indices, bias_label)
            data = torch.cat([data, _data])
            targets = torch.cat([targets, _targets])
            biased_targets.extend([bias_label] * len(indices))

        biased_targets = torch.LongTensor(biased_targets)
        return data, targets, biased_targets

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        return img, target, int(self.biased_targets[index])


def _load_mnist(path):

    with open(path, 'rb') as f:
        """
        size of `data` is:
        [
         3      (train, dev, test)
         2      (img, label)
         size   (10000 for dev and test, 50000 for train)
         dim    (784 for data, 1 for label)
        ]
        """
        data = pickle.load(f, encoding='bytes')
        train = data[0]
        dev = data[1]
        test = data[2]

    return train, dev, test


def _load_mnist_test_only(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def get_mnist_dataset_test_only(args):
    s = _load_mnist_test_only(args.test_only_data_path)
    return DataLoader(MyMNIST(s[0], s[1]), batch_size=args.batch_size)


def get_mnist_dataset(args, paired=False):

    train, dev, test = _load_mnist(args.paired_data_path if paired else args.data_path)
    if args.first_n_samples is not None:
        train = train[:args.first_n_samples]
    if args.is_rgb_data:
        transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                             std=(0.5, 0.5, 0.5))])
    else:
        transform = None

    return [DataLoader(MyMNIST(s[0], s[1], transform), batch_size=args.batch_size) for s in [train, dev, test]]


def __test_loader(path):
    train, dev, test = _load_mnist(path)
    return [DataLoader(MyMNIST(s[0], s[1]), batch_size=50) for s in [train, dev, test]]


def get_standard_mnist(args):
    mnist = MNIST(os.path.join('data', 'new'),
                  train=True,
                  download=True)
    train = MyMNIST(mnist.train_data[:40000], mnist.targets[:40000])
    dev = MyMNIST(mnist.train_data[40000:50000], mnist.targets[40000:50000])
    test = MyMNIST(mnist.train_data[50000:60000], mnist.targets[50000: 60000])

    return [DataLoader(s, batch_size=args.batch_size) for s in [train, dev, test]]


def get_colored_mnist(args):
    train = True
    data_label_correlation = 1
    do_shuffle = True
    # TODO: Find out the effect of this parameter
    n_confusing_labels = 9

    dataset = BiasedMNIST(os.path.join('data', 'new'),
                          train=train,
                          download=True,
                          data_label_correlation=data_label_correlation,
                          n_confusing_labels=n_confusing_labels,
                          do_shuffle=do_shuffle)

    """
    train = MyMNIST(dataset.train_data[:40000], dataset.targets[:40000])
    dev = MyMNIST(dataset.train_data[40000:50000], dataset.targets[40000:50000])
    test = MyMNIST(dataset.train_data[50000:60000], dataset.targets[50000: 60000])

    return [DataLoader(s, batch_size=args.batch_size) for s in [train, dev, test]]
    """
    train = (dataset.data[:40000], dataset.targets[:40000])
    dev = (dataset.data[40000:50000], dataset.targets[40000:50000])
    test = (dataset.data[50000:60000], dataset.targets[50000: 60000])
    return train, dev, test


if __name__ == '__main__':
    train_set, dev_set, test_set = __test_loader(os.path.join('data', 'mnist.pkl'))
    a, b, c = _load_mnist(os.path.join('data', 'mnist.pkl'))


