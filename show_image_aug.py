import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
from torchvision import transforms
from PIL import Image


# with open(os.path.join('data',  'mnist.pkl'), 'rb') as f:
# with open(os.path.join('adversarial',  'fgsm', 'fgsm_epsilon_0.2.pkl'), 'rb') as f:
with open(os.path.join('adversarial',  'colored', 'colored_partial_aug_noise.pkl'), 'rb') as f:
    s = pickle.load(f, encoding='bytes')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5),
                         std=(0.5, 0.5, 0.5))])

print("Showing first augmented images")
# imgs = [s[2][0][i].reshape(28, 28) for i in range(9)]
n_groups_to_show = 1
n = 6
# for show_from in [100, 10000, 20000, 30000, 35000, 39000]:
# for show_from in [200, 11000, 21000, 31000, 35100, 39100]:
# for show_from in [100, 200, 300, 2000, 5000, 8000]:
for show_from in [100]:
    for base in range(show_from, show_from + n_groups_to_show * n * n, n * n):
        plt.clf()
        for row in range(n):
            for col in range(n):
                plt.subplot(n, n, row*n+col + 1)
                img = s[0][0][row*n+col+base][0]
                img = transform(img)
                img = np.swapaxes(img, 0, 1)
                img = np.swapaxes(img, 1, 2)
                # print(img)
                plt.imshow(img)
        plt.show()


print("Showing second augmented images")
# for show_from in [100, 2000, 5000, 8000]:
for show_from in [100]:
    for base in range(show_from, show_from + n_groups_to_show * n * n, n* n):
        plt.clf()
        for row in range(n):
            for col in range(n):
                plt.subplot(n, n, row*n+col + 1)
                img = s[0][0][row*n+col+base][1]
                img = transform(img)
                img = np.swapaxes(img, 0, 1)
                img = np.swapaxes(img, 1, 2)
                # print(img)
                plt.imshow(img)
        plt.show()

print("Showing test images")
for show_from in [100]:
    # for show_from in [200, 11000, 21000, 31000, 35100, 39100]:
    for base in range(show_from, show_from + n_groups_to_show * n * n, n * n):
        plt.clf()
        for row in range(n):
            for col in range(n):
                plt.subplot(n, n, row*n+col + 1)
                img = s[2][0][row*n+col+base]
                img = transform(img)
                img = np.swapaxes(img, 0, 1)
                img = np.swapaxes(img, 1, 2)
                # print(img)
                plt.imshow(img)
        plt.show()



