import matplotlib.pyplot as plt
import pickle
import os

with open(os.path.join('adversarial', 'fgsm.pkl'), 'rb') as f:
    s = pickle.load(f)

for i in range(5):
    plt.figure()
    plt.imshow(s[0][i].reshape(28, 28))
    plt.show()

