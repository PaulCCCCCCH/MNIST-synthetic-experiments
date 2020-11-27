from models import LeNet, LeNetWithReg, load_model
import torch
from args import *
from data_utils import get_mnist_dataset, get_mnist_dataset_test_only
import logging
from utils import set_logger, plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


set_logger(ARGS)

if ARGS.test_only_data_path:
    test = get_mnist_dataset_test_only(ARGS)
    logging.info("Evaluating model {} on dataset {}".format(ARGS.model_name, ARGS.test_only_data_path))
else:
    _, _, test = get_mnist_dataset(ARGS)
    logging.info("Evaluating model {} on dataset {}".format(ARGS.model_name, ARGS.data_path))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.info("Using device: " + str(device))

if ARGS.use_reg_model:
    lenet = LeNetWithReg(ARGS)
else:
    lenet = LeNet(ARGS)
lenet.to(device)

state_dict = load_model(ARGS)
assert state_dict is not None, "Model with name '{}' not found".format(ARGS.model_name)
logging.info("Loading previous model with name: '{}'".format(ARGS.model_name))
lenet.load_state_dict(state_dict)
lenet.eval()

correct = 0
total = 0

all_predictions = []
all_labels = []

with torch.no_grad():
    for inputs_batch, labels_batch in test:
        inputs = inputs_batch.to(device)
        labels = labels_batch.to(device)

        if ARGS.use_reg_model:
            outputs, _, _ = lenet(inputs)
        else:
            outputs = lenet(inputs)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


cm = confusion_matrix(all_labels, all_predictions)

logging.info('Correct predictions: {} of {}'.format(correct, total))
logging.info('Accuracy of the network on 10000 test images: {}'.format(correct / total))

logging.info('Showing confusion matrix: \n ' + str(cm))
plot_confusion_matrix(cm, [str(x) for x in range(10)], 'Confusion Matrix for model {}'.format(ARGS.model_name))
test_method = ''
try:
    test_method = os.path.split(ARGS.data_path)[1].split('.')[0].split('_test_')[1]
except IndexError:
    pass

plt.savefig(os.path.join(ARGS.save_dir, 'confusion_matrix_test_{}.png'.format(test_method)), format='png')

