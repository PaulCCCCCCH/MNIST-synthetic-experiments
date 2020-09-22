import torch
import torch.nn as nn
from attacks import fgsm_attack
from data_utils import get_standard_mnist_dataset
from args import *
from models import LeNet, load_model
import pickle
import matplotlib as plt


def attack(model, device, testloader, epsilon, attack_f):
    count = 0
    correct = 0
    adv_inputs = []
    adv_labels = []

    for inputs_batch, labels_batch in testloader:

        inputs = inputs_batch.to(device)
        labels = labels_batch.to(device)

        inputs.requires_grad = True

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        if predicted.item() != labels.item():
            continue

        loss = criterion(outputs, labels)

        model.zero_grad()

        loss.backward()

        input_grad = inputs.grad.data

        perturbed_data = attack_f(inputs, epsilon, input_grad)

        new_outputs = model(perturbed_data)
        _, new_pred = torch.max(new_outputs, 1)
        if new_pred.item() == labels.item():
            correct += 1
        else:
            adv_example = perturbed_data.view(28*28, 1).squeeze().detach().cpu().numpy()
            adv_inputs.append(adv_example, )
            adv_labels.append(labels_batch.numpy())

        count += 1
        if count % (len(testloader) // 10) == 0:
            print("Finished {} / {}".format(count, len(testloader)))

    final_acc = correct / len(testloader)
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}\t".format(epsilon, correct, len(testloader), final_acc))

    return final_acc, adv_inputs, adv_labels


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    lenet = LeNet()
    lenet.to(device)
    state_dict = load_model(args)
    lenet.load_state_dict(state_dict)
    _, _, test = get_standard_mnist_dataset(os.path.join(args.data_dir, 'mnist.pkl'), batch_size=1)
    criterion = nn.CrossEntropyLoss()

    for epsilon in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
        print("Generating test set for epsilon {}".format(epsilon))
        acc, adv_inputs, adv_labels = attack(lenet, device, test, epsilon, fgsm_attack)
        with open(os.path.join(args.adversarial_dir, "{}_epsilon_{}.pkl".format(args.attack_name, epsilon)), 'wb') as f:
            pickle.dump([adv_inputs, adv_labels], f)
