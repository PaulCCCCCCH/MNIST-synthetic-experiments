import matplotlib.pyplot as plt
import pickle
import os

myMap = {'train': 0, 'dev': 1, 'test': 2}
##############################################
### Choose the data set you want to inspect here:
set_to_see = 'train'

### Choose the data file you want to inspect here
with open(os.path.join('data',  'mnist.pkl'), 'rb') as f:
# with open(os.path.join('adversarial',  'fgsm', 'fgsm_epsilon_0.2.pkl'), 'rb') as f:
    s = pickle.load(f, encoding='bytes')
##############################################

# imgs = [s[2][0][i].reshape(28, 28) for i in range(9)]
for row in range(3):
    for col in range(3):
        plt.subplot(3, 3, row*3+col + 1)
        plt.imshow(s[set_to_see][0][row*3+col].reshape(28, 28))
plt.show()