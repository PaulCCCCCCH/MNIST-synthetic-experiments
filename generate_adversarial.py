import torch
import torch.nn as nn
from attacks import fgsm_attack, cw_l2_attack
from data_utils import get_standard_mnist_dataset
from args import *
from models import LeNet, load_model
import pickle
import matplotlib as plt


def attack_fgsm(model, device, testloader, epsilon):
    count = 0
    correct = 0
    adv_inputs = []
    adv_labels = []
    batch_size = testloader.batch_size
    for inputs_batch, labels_batch in testloader:

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
        if count % (len(testloader) // 10) == 0:
            print("Finished {} / {}".format(count * batch_size, len(testloader)*batch_size))

    final_acc = correct / (len(testloader) * batch_size)
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}\t".format(epsilon, correct, len(testloader)*batch_size, final_acc))

    return final_acc, adv_inputs, adv_labels


def attack_cw(model, testloader, device):
    count = 0
    correct = 0
    adv_inputs = []
    adv_labels = []
    batch_size = testloader.batch_size
    for inputs_batch, labels_batch in testloader:

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
            perturbed_data = cw_l2_attack(model, inputs, labels, device, c=args.c_value)
            new_outputs = model(perturbed_data)
            _, new_preds = torch.max(new_outputs.data, 1)
            correct += (new_preds == labels).sum().item()
            inputs_to_put = perturbed_data.view(-1, 28*28).detach().cpu().numpy()

        count += 1
        if count % (len(testloader) // 10) == 0:
            print("Finished {} / {}".format(count * batch_size, len(testloader) * batch_size))

        adv_inputs.extend(inputs_to_put)
        adv_labels.extend(labels_batch.numpy())

    final_acc = correct / (len(testloader) * batch_size)
    print("Test Accuracy = {} / {} = {}\t".format(correct, len(testloader)*batch_size, final_acc))

    return final_acc, adv_inputs, adv_labels


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    lenet = LeNet()
    lenet.to(device)
    state_dict = load_model(args)
    lenet.load_state_dict(state_dict)
    criterion = nn.CrossEntropyLoss()



    if args.attack_name == 'fgsm':
        adversarial_dir = os.path.join(args.adversarial_dir, args.attack_name)
        if not os.path.exists(adversarial_dir):
            os.mkdir(adversarial_dir)
        _, _, test = get_standard_mnist_dataset(os.path.join(args.data_dir, 'mnist.pkl'), batch_size=1)
        for epsilon in [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
            print("Generating test set for epsilon {}".format(epsilon))
            acc, adv_inputs, adv_labels = attack_fgsm(lenet, device, test, epsilon)
            with open(os.path.join(adversarial_dir, "{}_epsilon_{}.pkl".format(args.attack_name, epsilon)), 'wb') as f:
                pickle.dump([adv_inputs, adv_labels], f)

    elif args.attack_name == 'cw':
        adversarial_dir = os.path.join(args.adversarial_dir, "cw_c_{}".format(args.c_value))
        if not os.path.exists(adversarial_dir):
            os.mkdir(adversarial_dir)
        print("Generating test set for CW attack")
        _, _, test = get_standard_mnist_dataset(os.path.join(args.data_dir, 'mnist.pkl'), batch_size=args.batch_size)
        acc, adv_inputs, adv_labels = attack_cw(lenet, test, device)

        with open(os.path.join(adversarial_dir, "{}.pkl".format(args.attack_name)), 'wb') as f:
            pickle.dump([adv_inputs, adv_labels], f)


