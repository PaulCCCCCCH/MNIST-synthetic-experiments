import pickle
import os
from torch.utils.data import DataLoader, Dataset
import torch
from torchvision.datasets import MNIST
import numpy as np
from PIL import Image
from torchvision import transforms


class MyMNIST(Dataset):
    def __init__(self, inputs, labels, transform=None, mode: str=None):
        self.inputs = inputs
        self.labels = labels
        self.transform = transform
        self.mode = mode

    def __getitem__(self, index):
        inputs = torch.tensor(self.inputs[index])
        # Data need to be in RGB format if transform is provided
        if self.transform:
            # mode is used to specify how to handle augmented data
            if self.mode is None:
                data = inputs.data
            elif self.mode == 'pick_first':
                data = inputs.data[0]
            elif self.mode == 'pick_random':
                data = inputs.data[np.random.randint(inputs.shape[0])]
            else:
                # Expect mode to be an integer
                assert self.mode.isdigit() and self.mode < inputs.shape[0], \
                    'mode has to be an integer between 0 and {}'.format(inputs.shape[0])
                data = inputs.data[int(self.mode)]

            inputs = Image.fromarray(np.array(data), mode='RGB')
            inputs = self.transform(inputs)
            labels = self.labels[index].squeeze()
        else:
            inputs = inputs.view(-1, 28, 28)
            labels = torch.tensor(self.labels[index]).squeeze()

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

    # COLOUR_MAP = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [225, 225, 0], [225, 0, 225],
    #               [0, 255, 255], [255, 128, 0], [255, 0, 128], [128, 0, 255], [128, 128, 128]]

    COLOUR_MAP = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [225, 225, 0], [225, 0, 225],
                  [0, 255, 255], [255, 195, 0], [255, 0, 195], [195, 0, 255], [195, 195, 195]]

    def __init__(self, root, args, train=True, transform=None, target_transform=None,
                 download=False, data_label_correlation=1.0, n_confusing_labels=9, do_shuffle=True):

        super().__init__(root, train=train, transform=transform,
                         target_transform=target_transform,
                         download=download)

        self.args = args
        self.held_out_data = self.data[50000:]
        self.held_out_targets = self.targets[50000:]
        self.data = self.data[:50000]
        self.targets = self.targets[:50000]

        # Set shuffler so that dataset will be the same order across different runs
        # This is to make sure that training set is in the same order as the paired set.
        self.shuffler = np.random.default_rng(seed=1)

        if args.bias_mode == 'none':
            self.BIASED_LABELS = set()
        elif args.bias_mode == 'partial':
            self.BIASED_LABELS = {5, 6, 7, 8, 9}
        elif args.bias_mode == 'all':
            self.BIASED_LABELS = {n for n in range(10)}
        else:
            raise NotImplementedError

        self.data_label_correlation = data_label_correlation
        self.n_confusing_labels = n_confusing_labels
        self.do_shuffle = do_shuffle

        self.data, self.targets, self.biased_targets = self.build_mnist()
        indices = np.arange(len(self.data))
        self._shuffle(indices)
        self.data = self.data[indices].numpy()
        self.targets = self.targets[indices]
        self.biased_targets = self.biased_targets[indices]

        self.held_out_data, self.held_out_targets, self.held_out_biased_targets = self.build_mnist(held_out=True)
        indices = np.arange(len(self.held_out_data))
        self._shuffle(indices)
        self.held_out_data = self.held_out_data[indices].numpy()
        self.held_out_targets = self.held_out_targets[indices]
        self.biased_targets = self.held_out_biased_targets[indices]

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    def _shuffle(self, iteratable):
        if self.do_shuffle:
            self.shuffler.shuffle(iteratable)
            # np.random.shuffle(iteratable)

    def _update_bias_indices(self, bias_indices, label, held_out=False):
        """
        Modifies bias_indices.
        Args:
            bias_indices: dict {label: [indices]}.

        """
        if self.n_confusing_labels > 9 or self.n_confusing_labels < 1:
            raise ValueError(self.n_confusing_labels)
        targets = self.held_out_targets if held_out else self.targets
        indices = np.where((targets == label).numpy())[0]
        self._shuffle(indices)
        indices = torch.LongTensor(indices)

        n_samples = len(indices)
        n_correlated_samples = int(n_samples * self.data_label_correlation)

        correlated_indices = indices[:n_correlated_samples]
        bias_indices[label] = torch.cat([bias_indices[label], correlated_indices])

        # The following has no effect when data_label_correlation is 1
        ##########################
        """
        n_decorrelated_per_class = int(np.ceil((n_samples - n_correlated_samples) / self.n_confusing_labels))
        decorrelated_indices = torch.split(indices[n_correlated_samples:], n_decorrelated_per_class)

        other_labels = [_label % 10 for _label in range(label + 1, label + 1 + self.n_confusing_labels)]
        self._shuffle(other_labels)

        for idx, _indices in enumerate(decorrelated_indices):
            _label = other_labels[idx]
            bias_indices[_label] = torch.cat([bias_indices[_label], _indices])
        """
        ##########################

    def _binary_to_colour(self, data, colour, augment=False):
        """
        Args:
            data: a batch of grey-scale images of shape (N, 28, 28)
            colour: a colour to be used to fill the background for each image.
                    If colour is None, then fill the background with a random
                    colour from the colour pool instead.
        Returns:
            RGB image of shape (N, 28, 28, 3)
        """

        fg_data = torch.zeros_like(data)
        fg_data[data != 0] = 255
        fg_data[data == 0] = 0
        fg_data = torch.stack([fg_data, fg_data, fg_data], dim=1)
        # fg_data: (N, 3, 28, 28)

        bg_data = torch.zeros_like(data)
        bg_data[data == 0] = 1
        bg_data[data != 0] = 0
        bg_data = torch.stack([bg_data, bg_data, bg_data], dim=3)
        # bg_data: (N, 28, 28, 3)

        if not augment:
            # Dealing with biased label
            if colour:
                # Use the same color as backgrounds for all the images
                bg_data = bg_data * torch.ByteTensor(colour)

            # Dealing with unbiased label
            else:
                # Use random color for each sample
                color_indices = np.random.randint(10, size=data.shape[0])
                colors = torch.ByteTensor(np.array(self.COLOUR_MAP)[color_indices])
                colors = colors.unsqueeze(1).unsqueeze(2)
                bg_data = bg_data * colors

            bg_data = bg_data.permute(0, 3, 1, 2)

            # data: (N, 3, 28, 28)
            data = fg_data + bg_data
            data = data.permute(0, 2, 3, 1)
            # data: (N, 28, 28, 3)

        else:
            # Dealing with biased labels
            if colour:
                bg_data = bg_data.unsqueeze(1)  # (N, 1, 28, 28, 3)
                colour_map = torch.tensor(self.COLOUR_MAP).unsqueeze(1).unsqueeze(2)  # (10, 1, 1, 3)
                bg_data = bg_data * colour_map  # (N, 10, 28, 28, 3)

            # Dealing with unbiased labels
            else:
                color_indices = np.random.randint(10, size=data.shape[0])
                colors = torch.ByteTensor(np.array(self.COLOUR_MAP)[color_indices])
                colors = colors.unsqueeze(1).unsqueeze(2)
                bg_data = bg_data * colors
                bg_data = bg_data.unsqueeze(1)  # (N, 1, 28, 28, 3)
                # Copy itself
                # bg_data = torch.stack([bg_data, bg_data], dim=1)  # (N, 2, 28, 28, 3)
                bg_data = bg_data.repeat((1, 10, 1, 1, 1))  # (N, 10, 28, 28, 3)

            bg_data = bg_data.permute(0, 4, 1, 2, 3)
            fg_data = fg_data.unsqueeze(2)  # (N, 3, 1, 28, 28)
            data = fg_data + bg_data  # (N, 3, 10, 28, 28)
            data = data.permute(0, 2, 3, 4, 1)

        return data

    def _make_biased_mnist(self, indices, label, held_out=False):
        """
        Args:
            indices: indices to be turned into biased images, shaped (N)
            label: a scalar index specifying a target colour to be used
        Returns:
            a tuple (images, labels)
        """
        data = self.held_out_data[indices] if held_out else self.data[indices]
        targets = self.held_out_targets[indices] if held_out else self.targets[indices]
        augment = not held_out and self.args.augment_mode != 'none'
        return self._binary_to_colour(data, self.COLOUR_MAP[label], augment=augment), targets

    def _make_unbiased_mnist(self, indices, label, held_out=False):
        """
        Args:
            indices: indices to be turned into biased images, shaped (N)
            label: a scalar index specifying a target colour to be used
        Returns:
            a tuple (images, labels)
        """
        data = self.held_out_data[indices] if held_out else self.data[indices]
        targets = self.held_out_targets[indices] if held_out else self.targets[indices]
        augment = not held_out and self.args.augment_mode != 'none'
        return self._binary_to_colour(data, None, augment=augment), targets

    def build_mnist(self, held_out=False):
        """
        Returns:
            data: batch of images of shape (N, 28, 28, 3)
            targets: labels of shape (N) corresponding to data
            biased_targets: index of background color to be used, shaped (N)
        """
        if (self.args.augment_mode == 'clipped') or self.args.clipped and not held_out:
            all_labels = range(5, 10)
        else:
            all_labels = range(self.targets.max().item() + 1)
        bias_indices = {label: torch.LongTensor() for label in all_labels}

        for label in all_labels:
            self._update_bias_indices(bias_indices, label, held_out=held_out)

        data = torch.ByteTensor()
        targets = torch.LongTensor()
        biased_targets = []

        for bias_label, indices in bias_indices.items():
            if bias_label not in self.BIASED_LABELS or held_out:
                _data, _targets = self._make_unbiased_mnist(indices, bias_label, held_out)
            else:
                _data, _targets = self._make_biased_mnist(indices, bias_label, held_out)

            print(_data.shape)
            print(_data.dtype)

            #############################################
            # TODO: Why is this step taking 10 times larger space than when test on jupyter notebook?
            # They have exactly the same size and dtype in both settings.
            #############################################
            data = torch.cat([data, _data])
            print(data.shape)
            print(data.dtype)
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

    if args.augment_data_mode and paired:
        train_loader = DataLoader(MyMNIST(train[0], train[1], transform, args.augment_data_mode), batch_size=args.batch_size)
        dev_loader = DataLoader(MyMNIST(dev[0], dev[1], transform, args.augment_data_mode), batch_size=args.batch_size)
        test_loader = DataLoader(MyMNIST(test[0], test[1], transform), batch_size=args.batch_size)
        return train_loader, dev_loader, test_loader
    else:
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
    do_shuffle = not args.ordered
    # TODO: Find out the effect of this parameter
    n_confusing_labels = 9

    dataset = BiasedMNIST(os.path.join('data', 'new'),
                          args,
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
    data_size = dataset.data.shape[0]
    train_size = int(data_size * 0.8)
    dev_size = int(data_size * 0.2)
    train = (dataset.data[0:train_size].astype(np.uint8), dataset.targets[:train_size])

    dev = (dataset.data[train_size:train_size + dev_size].astype(np.uint8), dataset.targets[train_size:train_size + dev_size])

    test = (dataset.held_out_data.astype(np.uint8), dataset.held_out_targets)

    print(dataset.targets[0:40000])
    print(dataset.targets[40000:50000])
    print(dataset.targets[50000:60000])
    print(dataset.held_out_targets)

    print("Training set size: {}".format(train_size))
    print("Dev set size: {}".format(dev_size))
    print("Test set size: {}".format(dataset.held_out_data.shape[0]))
    return train, dev, test


if __name__ == '__main__':
    train_set, dev_set, test_set = __test_loader(os.path.join('data', 'mnist.pkl'))
    a, b, c = _load_mnist(os.path.join('data', 'mnist.pkl'))


