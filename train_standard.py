import pickle
import numpy as np
from models import LeNet, save_model
import torch.optim as optim
import torch.nn as nn
from args import *
from data_utils import get_standard_mnist_dataset
from models import load_model


train, dev, test = get_standard_mnist_dataset(os.path.join(args.data_dir, 'mnist.pkl'), args.batch_size)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initializing model
lenet = LeNet()
lenet.to(device)

state_dict = load_model(args)
lenet.load_state_dict(state_dict)

min_loss = float('inf')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(lenet.parameters(), lr=args.learning_rate, momentum=args.momentum)

for epoch in range(args.epoch):
    epoch_loss = 0
    total = 0
    correct = 0
    # Train
    for inputs_batch, labels_batch in train:
        # Format input data
        inputs = inputs_batch.to(device)
        labels = labels_batch.to(device)

        # Training step
        optimizer.zero_grad()
        outputs = lenet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Calculate statistics
        epoch_loss += loss.data
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print("Epoch {}, train loss: {}".format(epoch, epoch_loss))
    print("Epoch {}, train accuracy: {}".format(epoch, correct / total))

    # Evaluate on dev set
    with torch.no_grad():
        dev_loss = 0
        total = 0
        correct = 0
        for inputs_batch, labels_batch in dev:
            inputs = inputs_batch.to(device)
            labels = labels_batch.to(device)
            outputs = lenet(inputs)
            loss = criterion(outputs, labels)
            dev_loss += loss.data
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Save the model if it is the best so far
        if dev_loss < min_loss:
            min_loss = dev_loss
            print("Best model at epoch {}, model saved.".format(epoch))
            save_model(lenet.state_dict(), args)

        print("Epoch {}, dev loss: {}".format(epoch, dev_loss))
        print("Epoch {}, dev accuracy : {}".format(epoch, correct / total))
        print()

