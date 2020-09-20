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

train_s = mnist_standard[0]
dev_s = mnist_standard[1]
test_s = mnist_standard[2]

train_size = len(train_s[0])
test_size = len(dev_s[0])

# Testing
correct = 0
total = 0
with torch.no_grad():
    for i in range(0, test_size, batch_size):
        data = torch.tensor(test_s[0][i:i+batch_size], device=device)
        data = data.view((batch_size, 1, 28, 28))
        label = torch.tensor(test_s[1][i:i+batch_size], device=device)
        label = label.squeeze()
        outputs = lenet(data)
        _, predicted = torch.max(outputs.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

print('Correct predictions: {} of {}'.format(correct, total))
print('Accuracy of the network on 10000 standard test images: {}'.format(correct / total))
