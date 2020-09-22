import pickle
import numpy as np
from models import LeNet
import torch.optim as optim
import torch.nn as nn
from args import *


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


test_size = len(dev_s[0])


# mnist_perturbed: [(10000, 784)]
mnist_perturbed = []
for s in paths_mnist_perturbed:
    mnist_perturbed.append(np.load(s))

# Start training
lenet = LeNet()
lenet.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(lenet.parameters(), lr=learning_rate, momentum=0.9)


def train_on_set(inputs, labels):

    min_loss = float('inf')
    train_size = len(inputs)
    for epoch in range(num_epoch):
        epoch_loss = 0
        total = 0
        correct = 0
        # Train
        for i in range(0, train_size, batch_size):
            # Format input data
            data = torch.tensor(inputs[i:i+batch_size], device=device)
            data = data.view((batch_size, 1, 28, 28))
            label = torch.tensor(labels[i:i+batch_size], device=device)
            label = label.squeeze()

            # Training step
            optimizer.zero_grad()
            outputs = lenet(data)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            # Calculate statistics
            epoch_loss += loss.data
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

        print("Epoch {}, train loss: {}".format(epoch, epoch_loss))
        print("Epoch {}, train accuracy: {}".format(epoch, correct / total))

        # Evaluate on dev set
        with torch.no_grad():
            dev_loss = 0
            total = 0
            correct = 0
            for i in range(0, test_size, batch_size):
                data = torch.tensor(dev_s[0][i:i+batch_size], device=device)
                data = data.view((batch_size, 1, 28, 28))
                label = torch.tensor(dev_s[1][i:i+batch_size], device=device)
                label = label.squeeze()
                outputs = lenet(data)
                loss = criterion(outputs, label)

                dev_loss += loss.data

                _, predicted = torch.max(outputs.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

            # Save the model if it is the best so far
            # if dev_loss < min_loss:
            #     min_loss = dev_loss
            #     print("Best model at epoch {}, model saved.".format(epoch))
            torch.save(lenet.state_dict(), save_path)

            print("Epoch {}, dev loss: {}".format(epoch, dev_loss))
            print("Epoch {}, dev accuracy : {}".format(epoch, correct / total))
            print()


train_on_set(train_s[0], train_s[1])
train_on_set(mnist_perturbed[0], test_s[1])
