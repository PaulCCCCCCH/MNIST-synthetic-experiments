import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
from torchvision import transforms
from PIL import Image


# with open(os.path.join('data',  'mnist.pkl'), 'rb') as f:
# with open(os.path.join('adversarial',  'fgsm', 'fgsm_epsilon_0.2.pkl'), 'rb') as f:
with open(os.path.join('adversarial',  'colored', 'colored.pkl'), 'rb') as f:
    s = pickle.load(f, encoding='bytes')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5),
                         std=(0.5, 0.5, 0.5))])

# imgs = [s[2][0][i].reshape(28, 28) for i in range(9)]
n_groups_to_show = 1
show_from = 100
for base in range(show_from, show_from + n_groups_to_show * 9, 9):
    plt.clf()
    for row in range(3):
        for col in range(3):
            plt.subplot(3, 3, row*3+col + 1)
            img = s[0][0][row*3+col+base]
            img = transform(img)
            img = np.swapaxes(img, 0, 1)
            img = np.swapaxes(img, 1, 2)
            print(img)
            plt.imshow(img)
    plt.show()



