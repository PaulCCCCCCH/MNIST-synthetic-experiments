import pickle
from models import LeNet
import torch
from args import *
import numpy as np

lenet = LeNet()
state_dict = torch.load(save_path)
lenet.load_state_dict(state_dict)
lenet.to(device)

with open(path_mnist_standard, 'rb') as f:
    # size of mnist_standard is:
    #  3 (train, dev, test)
    # * 2 (img, label)
    # * size (10000 for dev and test, 50000 for train)
    # * dim (784 for data, 1 for label)
    mnist_standard = pickle.load(f, encoding='bytes')

test_s = mnist_standard[2]


# mnist_perturbed: [(10000, 784)]
mnist_perturbed = []
for s in paths_mnist_perturbed:
    mnist_perturbed.append(np.load(s))

test_size = mnist_perturbed[0].shape[0]

# Testing

for idx, test_p in enumerate(mnist_perturbed):
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, test_size, batch_size):
            data = torch.tensor(test_p[i:i+batch_size], device=device)
            data = data.view((batch_size, 1, 28, 28))
            label = torch.tensor(test_s[1][i:i+batch_size], device=device)
            label = label.squeeze()
            outputs = lenet(data)
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

    print('Result of file {}'.format(files_mnist_perturbed[idx]))
    print('Correct predictions: {} of {}'.format(correct, total))
    print('Accuracy of the network on the 10000 test images: {}'.format(correct / total))
    print()
