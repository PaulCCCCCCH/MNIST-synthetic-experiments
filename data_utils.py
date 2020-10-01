import pickle
import os
from torch.utils.data import DataLoader, Dataset
import torch


class MNIST(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __getitem__(self, index):
        inputs = torch.tensor(self.inputs[index]).view(1, 28, 28)
        labels = torch.tensor(self.labels[index]).squeeze()
        return inputs, labels

    def __len__(self):
        return len(self.inputs)


def load_standard_mnist(path):

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


def load_adv_mnist(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def get_adv_mnist_dataset(args):
    s = load_adv_mnist(args.adv_data_path)
    return DataLoader(MNIST(s[0], s[1]), batch_size=args.batch_size)


def get_standard_mnist_dataset(args):
    train, dev, test = load_standard_mnist(os.path.join(args.data_dir, 'mnist.pkl'))
    if args.first_n_samples is not None:
        train = train[:args.first_n_samples]
    return [DataLoader(MNIST(s[0], s[1]), batch_size=args.batch_size) for s in [train, dev, test]]


if __name__ == '__main__':
    train_set, dev_set, test_set = get_standard_mnist_dataset(os.path.join('data', 'mnist.pkl'), 10)
    a, b, c = load_standard_mnist(os.path.join('data', 'mnist.pkl'))

