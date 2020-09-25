import matplotlib.pyplot as plt
import pickle
import os

show_standard = True


if show_standard:
    with open(os.path.join('data',  'mnist.pkl'), 'rb') as f:
        s = pickle.load(f, encoding='bytes')

    for i in range(5):
        plt.figure()
        plt.imshow(s[2][0][i].reshape(28, 28))
        plt.show()

else:
    with open(os.path.join('adversarial', 'cw', 'cw.pkl'), 'rb') as f:
        s = pickle.load(f)

    for i in range(10, 20):
        plt.figure()
        plt.imshow(s[0][i].reshape(28, 28))
        plt.show()

