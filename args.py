import os
import torch


save_dir = 'models'
save_file_name = 'mnist_two_tasks'

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
num_epoch = 30
