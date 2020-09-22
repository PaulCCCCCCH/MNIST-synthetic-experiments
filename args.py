import os
import torch
import argparse

parser = argparse.ArgumentParser()

# For training and testing
parser.add_argument('model_name', type=str, help='Give model name, this will name logs and checkpoints.')
parser.add_argument('--save_dir', type=str, help='Root directory where all models are saved', default='models')
parser.add_argument('--data_dir', type=str, help='Directory of data file', default='data')
parser.add_argument('--epoch', type=int, help='Max number of epochs to train', default=50)
parser.add_argument('--batch_size', type=int, help='Batch size to use for training', default=64)
parser.add_argument('--learning_rate', type=float, help='Learning rate to use', default=0.001)
parser.add_argument('--momentum', type=float, help='Momentum of SGD algorithm', default=0.8)

# For adversarial examples generation only
parser.add_argument('--adversarial_dir', type=str, help='Place to store adversarial examples', default='adversarial')
parser.add_argument('--attack_name', type=str, help='The attack to be performed', default='fgsm')


args = parser.parse_args()

def get_args():
    return args

"""
save_dir = 'models'
save_file_name = 'mnist_standard_60'

dir_mnist_standard = os.path.join('..', 'Datasets', 'MNISTPerturbed', 'standardMNIST')
file_mnist_standard = 'mnist.pkl'
path_mnist_standard = os.path.join(dir_mnist_standard, file_mnist_standard)

save_path = os.path.join(save_dir, save_file_name)

dir_mnist_perturbed = os.path.join('..', 'Datasets', 'MNISTPerturbed', 'extraTestData')
files_mnist_perturbed = [s for s in os.listdir(dir_mnist_perturbed) if s.endswith('npy')]
paths_mnist_perturbed = [os.path.join(dir_mnist_perturbed, f) for f in files_mnist_perturbed]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 50
learning_rate = 0.001
num_epoch = 60
"""
