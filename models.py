import torch.nn as nn
import os
import torch


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(
            # (1, 28, 28) => (6, 28, 28)
            nn.Conv2d(in_channels=1,
                      out_channels=6,
                      kernel_size=5,
                      padding=2),
            nn.ReLU(),
            # (6, 28, 28) => (6, 14, 14)
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            # (6, 14, 14) => (16, 10, 10)
            nn.Conv2d(in_channels=6,
                      out_channels=16,
                      kernel_size=5),
            nn.ReLU(),
            # (16, 10, 10) => (16, 5, 5)
            nn.MaxPool2d(2, 2)
        )

        # A flatten layer here: (16, 5, 5) => (16*5*5)

        self.fc1 = nn.Sequential(
            # (16*5*5) => (120)
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            # (120, 84)
            nn.Linear(120, 84),
            nn.ReLU()
        )

        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def save_model(state_dict: dict, args):
    dir_name = args.save_dir
    model_name = args.model_name
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    save_dir = os.path.join(dir_name, model_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    save_path = os.path.join(save_dir, model_name + '.pt')
    torch.save(state_dict, save_path)


def load_model(args):
    model_name = args.model_name
    dir_name = args.save_dir
    save_path = os.path.join(dir_name, model_name, model_name + '.pt')
    if not os.path.exists(save_path):
        return None
    return torch.load(save_path)
