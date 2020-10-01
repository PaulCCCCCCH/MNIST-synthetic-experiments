from models import LeNet, load_model
import torch
from args import *
from data_utils import get_mnist_dataset, get_mnist_dataset_test_only
import logging
from utils import set_logger

set_logger(ARGS)

if ARGS.test_only_data_path:
    test = get_mnist_dataset_test_only(ARGS)
else:
    _, _, test = get_mnist_dataset(ARGS)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.info("Using device: " + str(device))


lenet = LeNet()
lenet.to(device)

state_dict = load_model(ARGS)
assert state_dict is not None, "Model with name '{}' not found".format(ARGS.model_name)
print("Loading previous model with name: '{}'".format(ARGS.model_name))
lenet.load_state_dict(state_dict)

# Testing
correct = 0
total = 0
with torch.no_grad():
    for inputs_batch, labels_batch in test:
        inputs = inputs_batch.to(device)
        labels = labels_batch.to(device)

        outputs = lenet(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Correct predictions: {} of {}'.format(correct, total))
print('Accuracy of the network on 10000 test images: {}'.format(correct / total))