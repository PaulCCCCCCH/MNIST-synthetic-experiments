import logging
import os
import matplotlib.pyplot as plt
import numpy as np


def set_logger(args):
    if args.isGeneration:
        save_dir = args.adversarial_dir
    elif args.saveAsNew:
        save_dir = args.new_save_dir
    else:
        save_dir = args.save_dir

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s: %(message)s',
                        handlers=[logging.FileHandler(os.path.join(save_dir, 'log.txt')),
                                  logging.StreamHandler()])
    """
    logging.basicConfig(level=logging.CRITICAL,
                        format='%(asctime)s: %(message)s',
                        handlers=[logging.FileHandler(os.path.join(save_dir, 'log.txt')),
                                  logging.FileHandler("results.txt"),
                                  logging.StreamHandler()])
    """


# Copied from https://blog.csdn.net/kane7csdn/article/details/83756583
def plot_confusion_matrix(cm, labels_name, title):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    # 归一化
    plt.imshow(cm, interpolation='nearest')    # 在特定的窗口上显示图像
    plt.title(title)    # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, rotation=90)    # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)    # 将标签印在y轴坐标上
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


