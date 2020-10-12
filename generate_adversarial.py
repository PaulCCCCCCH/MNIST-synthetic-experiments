import torch
import torch.nn as nn
from attacks import fgsm_attack, cw_l2_attack
from data_utils import get_mnist_dataset, get_colored_mnist
from args import *
from models import LeNet, load_model
import pickle
from utils import set_logger
import logging

def attack_fgsm(model, loader, device, epsilon):
    if loader is None:
        return None, None, None
    count = 0
    correct = 0
    adv_inputs = []
    adv_labels = []
    batch_size = loader.batch_size
    for inputs_batch, labels_batch in loader:

        inputs = inputs_batch.to(device)
        labels = labels_batch.to(device)
        inputs.requires_grad = True
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        # If already wrong, don't bother to attack. Put the original image.
        if predicted.item() != labels.item():
            inputs_to_put = inputs.view(28*28, 1).squeeze().detach().cpu().numpy()
        else:
            loss = criterion(outputs, labels)
            model.zero_grad()
            loss.backward()

            input_grad = inputs.grad.data
            perturbed_data = fgsm_attack(inputs, epsilon, input_grad)
            inputs_to_put = perturbed_data.view(28*28, 1).squeeze().detach().cpu().numpy()

            # Evaluate on the adversarial data
            new_outputs = model(perturbed_data)
            _, new_pred = torch.max(new_outputs, 1)
            if new_pred.item() == labels.item():
                correct += 1

        adv_inputs.append(inputs_to_put)
        adv_labels.append(labels_batch.squeeze().numpy())

        count += 1
        if count % (len(loader) // 10) == 0:
            ("Finished {} / {}".format(count * batch_size, len(loader)*batch_size))

    final_acc = correct / (len(loader) * batch_size)
    logging.info("Epsilon: {}\tTest Accuracy = {} / {} = {}\t".format(epsilon, correct, len(loader)*batch_size, final_acc))

    return final_acc, adv_inputs, adv_labels


def attack_cw(model, loader, device, c_value):
    if loader is None:
        return None, None, None
    count = 0
    correct = 0
    adv_inputs = []
    adv_labels = []
    batch_size = loader.batch_size
    for inputs_batch, labels_batch in loader:

        inputs = inputs_batch.to(device)
        labels = labels_batch.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        # If already wrong, don't bother to attack. Put the original image.
        # if predicted.item() != labels.item():
        #     inputs_to_put = inputs.view(-1, 28*28).detach().cpu().numpy()
        if False:
            pass

        else:
            # adv is of size (Batch, 784)
            perturbed_data = cw_l2_attack(model, inputs, labels, device, c=c_value)
            new_outputs = model(perturbed_data)
            _, new_preds = torch.max(new_outputs.data, 1)
            correct += (new_preds == labels).sum().item()
            inputs_to_put = perturbed_data.view(-1, 28*28).detach().cpu().numpy()

        count += 1
        if count % (len(loader) // 10) == 0:
            logging.info("Finished {} / {}".format(count * batch_size, len(loader) * batch_size))

        adv_inputs.extend(inputs_to_put)
        adv_labels.extend(labels_batch.numpy())

    final_acc = correct / (len(loader) * batch_size)
    logging.info("Test Accuracy = {} / {} = {}\t".format(correct, len(loader)*batch_size, final_acc))

    return final_acc, adv_inputs, adv_labels


if __name__ == '__main__':
    set_logger(ARGS)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: " + str(device))

    if not ARGS.attack_name == 'colored':
        logging.info('Loading model ' + ARGS.model_name)
        lenet = LeNet(ARGS)
        lenet.to(device)
        state_dict = load_model(ARGS)
        lenet.load_state_dict(state_dict)
        criterion = nn.CrossEntropyLoss()
        lenet.eval()
    else:
        logging.info('Color-biased data generation does not require a model. Skipping model loading.')
        lenet = None

    if ARGS.attack_name == 'fgsm':
        adversarial_dir = ARGS.adversarial_dir
        train, dev, test = get_mnist_dataset(ARGS)
        datasets = [None, None, test] if ARGS.test_data_only else [train, dev, test]
        for epsilon in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        # for epsilon in [0.35]:
            logging.info("Generating fgsm data set for epsilon {}".format(epsilon))
            to_dump = []
            for dataset in datasets:
                acc, adv_inputs, adv_labels = attack_fgsm(lenet, dataset, device, epsilon)
                to_dump.append((adv_inputs, adv_labels))
            with open(os.path.join(adversarial_dir, "{}_epsilon_{}.pkl".format(ARGS.attack_name, epsilon)), 'wb') as f:
                pickle.dump(to_dump, f)

    elif ARGS.attack_name == 'cw':
        adversarial_dir = ARGS.adversarial_dir
        train, dev, test = get_mnist_dataset(ARGS)
        datasets = [None, None, test] if ARGS.test_data_only else [train, dev, test]
        for c_value in [0.1, 0.5, 1.0, 5.0]:
        # for c_value in [0.1]:
            logging.info("Generating cw data set for c = {}".format(c_value))
            to_dump = []
            for dataset in datasets:
                acc, adv_inputs, adv_labels = attack_cw(lenet, dataset, device, c_value)
                to_dump.append((adv_inputs, adv_labels))
            with open(os.path.join(adversarial_dir, "{}_c_{}.pkl".format(ARGS.attack_name, c_value)), 'wb') as f:
                pickle.dump(to_dump, f)

    # This is not really an attack, but it is put here for convenience, as it shares most of the logic
    # with the attack methods.
    elif ARGS.attack_name == 'colored':
        adversarial_dir = ARGS.adversarial_dir
        train, dev, test = get_colored_mnist(ARGS)
        datasets = [None, None, test] if ARGS.test_data_only else [train, dev, test]
        logging.info("Generating dataset with color bias")
        to_dump = []
        for dataset in datasets:
            to_dump.append(dataset)
        with open(os.path.join(adversarial_dir, 'colored_{}.pkl'.format(ARGS.bias_mode)), 'wb') as f:
            pickle.dump(to_dump, f)

    else:
        raise NotImplementedError


