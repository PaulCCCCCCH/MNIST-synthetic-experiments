import os
import argparse
import sys

script_name = sys.argv[0]

parser = argparse.ArgumentParser()

# For training and testing
parser.add_argument('model_name', type=str, help='Give model name, this will name logs and checkpoints.')
parser.add_argument('--data_path', type=str, help='Path to the file with train, dev and test data', default='')
parser.add_argument('--save_dir', type=str, help='Root directory where all models are saved', default='models')
parser.add_argument('--test_only_data_path', type=str, help='Path to the file containing only test samples', default='')
parser.add_argument('--epoch', type=int, help='Max number of epochs to train', default=50)
parser.add_argument('--batch_size', type=int, help='Batch size to use for training', default=50)
parser.add_argument('--learning_rate', type=float, help='Learning rate to use', default=0.001)
parser.add_argument('--momentum', type=float, help='Momentum of SGD algorithm', default=0.8)
parser.add_argument('--first_n_samples', type=int, help='Only use first n samples of the training set', default=None)
parser.add_argument('--new_model_name', type=str, help='If given, will save the trained model as a new one', default='')
parser.add_argument('--use_reg_model', help='Use the model with regularization layers', action='store_true')
parser.add_argument('--patience', type=int, help='Number of epochs w/o improvements before stopping', default=5)

# For adversarial examples generation only
parser.add_argument('--adversarial_dir', type=str, help='Place to store adversarial examples', default='adversarial')
parser.add_argument('--attack_name', type=str, help='The attack to be performed', default='fgsm',
                    choices=['fgsm', 'cw'])
parser.add_argument('--test_data_only', help='Generate adv samples only for test set', action='store_true')
# parser.add_argument('--c_value', type=float, help='C value of the cw attack', default=0.5)


# For training on adversarial data

# For paired training only
parser.add_argument('--paired_data_path', type=str, help='Path to the paired data to be used', default='')

help_str = """
    Select regularization method.
    1: Gradient norm
    2: KL divergence
    3: L2 distance
    4:  ??? 
    5: L1 distance
"""
parser.add_argument('--method', type=int, metavar='M', choices=[0, 1, 2, 3, 5], help=help_str, default=1)

help_str = """
    Select which object to be used as the input to calculate the regularization loss.
    0: Logits of the cnn
    1: Activation value of the second to last layer
"""
parser.add_argument('--reg_object', metavar='T', type=int, choices=[0, 1], help=help_str, default=0)

help_str = "Number of fc layers to use to produce the regularization loss"
parser.add_argument('--reg_layers', type=int, metavar='N', choices=[1, 2], help=help_str, default=1)

help_str = 'Whether to use dropout as regularization'
parser.add_argument('--use_dropout', type=int, metavar='D', choices=[0, 1], help=help_str, default=0)

parser.add_argument('--lam', type=float, metavar='L', help='Coefficient for grad_loss for method 1', default=1.0)
# parser.add_argument('--gamma', type=float, help='Coefficient for regularization loss', default=1.0)

args = parser.parse_args()


def get_args():
    return args


class ARGS:
    # For training and testing
    model_name =            args.model_name
    data_path =             args.data_path
    new_model_name =        args.new_model_name
    save_dir =              os.path.join(args.save_dir, model_name)
    save_path =             os.path.join(save_dir, model_name + '.pt')
    new_save_dir =          os.path.join(args.save_dir, new_model_name)
    new_save_path =         os.path.join(new_save_dir, new_model_name + '.pt')
    epoch =                 args.epoch
    batch_size =            args.batch_size
    learning_rate =         args.learning_rate
    momentum =              args.momentum
    first_n_samples =       args.first_n_samples
    use_reg_model =         args.use_reg_model
    patience =              args.patience

    # For adversarial examples generation only
    attack_name =           args.attack_name
    adversarial_dir =       os.path.join(args.adversarial_dir, attack_name)
    test_data_only =        args.test_data_only

    # For evaluation
    test_only_data_path =   args.test_only_data_path

    # For paired training only
    paired_data_path =      args.paired_data_path
    method =                args.method
    reg_object =            args.reg_object
    reg_layers =            args.reg_layers
    use_dropout =           args.use_dropout
    lam =                   args.lam

    # Modes
    isGeneration =          script_name.startswith('generate_adversarial')
    saveAsNew =             bool(args.new_model_name)
    isPairedTrain =         script_name.startswith('train_paired')

    assert bool(data_path) ^ bool(test_only_data_path), "Require a data_path or a test_only_data_path argument, but not both"
    assert not isPairedTrain or bool(paired_data_path), "Paired training requires a paired_data_path"
    # assert not isPairedTrain or new_model_name, "Paired training requires a new_model_name"

    # Check directories
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if isGeneration:
        if not os.path.exists(args.adversarial_dir):
            os.mkdir(args.adversarial_dir)
        if not os.path.exists(adversarial_dir):
            os.mkdir(adversarial_dir)

    # Pre-processing
    if attack_name == 'fgsm' and isGeneration:
        batch_size = 1

    @staticmethod
    def toString():
        return str(args)

# Validate arguments
"""
valid = True
msg = ""
if args.paired_training:
    if not args.adv_data_path:
        msg += 'Paired training requires adv_data_path. '
        valid = False

assert valid, "Invalid arguments: " + msg
"""

# Original
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
