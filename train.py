import torch
from models import LeNet, save_model
import torch.optim as optim
import torch.nn as nn
from args import *
from data_utils import get_mnist_dataset, get_mnist_dataset_test_only
from utils import set_logger
from models import load_model
import logging

set_logger(ARGS)
logging.info("Called train.py with args:\n" + ARGS.toString())

logging.info("Loading standard dataset")
train, dev, test = get_mnist_dataset(ARGS)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.info("Using device: " + str(device))

# Initializing model
lenet = LeNet()
lenet.to(device)

state_dict = load_model(ARGS)
if state_dict is not None:
    logging.info("Loading previous model with name: '{}'".format(ARGS.model_name))
    lenet.load_state_dict(state_dict)
else:
    logging.info("No previous model found with name '{}', training a new one.".format(ARGS.model_name))

min_loss = float('inf')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(lenet.parameters(), lr=ARGS.learning_rate, momentum=ARGS.momentum)

logging.info("Training...")
for epoch in range(ARGS.epoch):
    epoch_loss = 0
    total = 0
    correct = 0
    # Train
    for inputs_batch, labels_batch in train:
        lenet.train()
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

    logging.info("Epoch {}, total train loss: {}".format(epoch, epoch_loss))
    logging.info("Epoch {}, train accuracy: {}".format(epoch, correct / total))

    # Skip evaluation on dev set when training on adv MNIST and save
    if ARGS.test_only_data_path:
        save_model(lenet.state_dict(), ARGS, ARGS.saveAsNew)

    # Evaluate on dev set when training on standard MNIST
    else:
        with torch.no_grad():
            lenet.evel()
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
                logging.info("Best model at epoch {}, model saved.".format(epoch))
                save_model(lenet.state_dict(), ARGS, ARGS.saveAsNew)

            logging.info("Epoch {}, total dev loss: {}".format(epoch, dev_loss))
            logging.info("Epoch {}, dev accuracy : {}".format(epoch, correct / total))

logging.info("Done")

