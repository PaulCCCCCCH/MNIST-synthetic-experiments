from models import LeNetWithReg, save_model
import torch
import torch.optim as optim
import torch.nn as nn
from args import *
from data_utils import get_mnist_dataset, get_mnist_dataset_test_only
from models import load_model
import logging
from utils import set_logger
# TODO: There are only 40000 training samples in the original training set
if ARGS.first_n_samples is not None:
    ARGS.first_n_samples = max(ARGS.first_n_samples, 40000)

set_logger(ARGS)

paired_data = get_mnist_dataset_test_only(ARGS)
if ARGS.test_only_data_path:
    train = get_mnist_dataset_test_only(ARGS)
else:
    train, dev, test = get_mnist_dataset(ARGS)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.info("Using device: " + str(device))

# Initializing model
lenet = LeNetWithReg(ARGS)
lenet.to(device)

state_dict = load_model(ARGS)
if state_dict is not None:
    print("Loading previous model with name: '{}'".format(ARGS.model_name))
    lenet.load_state_dict(state_dict)
else:
    print("No previous model found with name '{}', training a new one.".format(ARGS.model_name))

min_loss = float('inf')
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(lenet.parameters(), lr=ARGS.learning_rate, momentum=ARGS.momentum)

for epoch in range(ARGS.epoch):
    epoch_loss = 0
    total = 0
    correct = 0
    # Train
    for (inputs_batch, labels_batch), (inputs_batch_paired, labels_batch_paired) in zip(train, paired_data):
        # Format input data
        inputs = inputs_batch.to(device)
        labels = labels_batch.to(device)
        inputs_paired = inputs_batch_paired.to(device)
        labels_paired = labels_batch_paired.to(device)

        # Training step
        optimizer.zero_grad()

        # Calculate loss1 as usual
        outputs1 = lenet(inputs)
        loss1 = criterion(outputs1, labels)

        # Calculate loss2 as the loss from paired training data
        outputs2 = lenet(inputs_paired)
        loss2 = criterion(outputs2, labels_paired)

        loss = loss1 + loss2
        loss.backward()
        optimizer.step()

        # Calculate statistics
        epoch_loss += loss.data
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print("Epoch {}, train loss: {}".format(epoch, epoch_loss))
    print("Epoch {}, train accuracy: {}".format(epoch, correct / total))

    # Skip evaluation on dev set when training on test_only MNIST and save
    if ARGS.test_only_data_path:
        save_model(lenet.state_dict(), ARGS)

    # Evaluate on dev set when training on standard MNIST
    else:
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

