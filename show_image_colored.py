import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
from torchvision import transforms

myMap = {'train': 0, 'dev': 1, 'test': 2}

##############################################
### Choose the data set you want to inspect here:
set_to_see = myMap['train']

### Choose the data file you want to inspect here
with open(os.path.join('adversarial',  'colored', 'colored_partial_test_pure123.pkl'), 'rb') as f:
    s = pickle.load(f, encoding='bytes')
##############################################

transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize(mean=(0.5, 0.5, 0.5),
    #                     std=(0.5, 0.5, 0.5))
    ])

# imgs = [s[2][0][i].reshape(28, 28) for i in range(9)]
n_groups_to_show = 1
# for show_from in [100, 10000, 20000, 30000, 35000, 39000]:
n = 6
# for show_from in [100, 200, 300, 2000, 5000, 8000]:
for show_from in [100, 500]:
# for show_from in [100, 2000, 5000, 8000]:
# for show_from in [200, 11000, 21000, 31000, 35100, 39100]:
    for base in range(show_from, show_from + n_groups_to_show * n, n):
        plt.clf()
        for row in range(n):
            for col in range(n):
                plt.subplot(n, n, row*n+col + 1)
                img = s[set_to_see][0][row*n+col+base]
                img = transform(img)
                img = np.swapaxes(img, 0, 1)
                img = np.swapaxes(img, 1, 2)
                plt.imshow(img)
        plt.show()



