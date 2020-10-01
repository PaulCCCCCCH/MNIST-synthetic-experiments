import torch
import torch.nn as nn
import torch.optim as optim


def fgsm_attack(image, epsilon, data_grad):
    """
    Copied from https://pytorch.org/tutorials/beginner/fgsm_tutorial.html
    """
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def cw_l2_attack(model, images, labels, device, targeted=False, c=0.5, kappa=0, max_iter=1000, learning_rate=0.01):
    """
    Adopted from https://github.com/Harry24k/CW-pytorch/blob/master/CW.ipynb
    """
    images = images.to(device)
    labels = labels.to(device)

    # Define f-function
    def f(x):

        outputs = model(x)

        # TODO: The output is already one-hot?
        one_hot_labels = torch.eye(len(outputs[0]))[labels].to(device)

        i, _ = torch.max((1 - one_hot_labels) * outputs, dim=1)
        mask = one_hot_labels.byte() == 1

        j = torch.masked_select(outputs, mask)

        # If targeted, optimize for making the other class most likely
        if targeted:
            return torch.clamp(i - j, min=-kappa)

        # If untargeted, optimize for making the other class most likely
        else:
            return torch.clamp(j - i, min=-kappa)

    w = torch.zeros_like(images, requires_grad=True).to(device)

    optimizer = optim.Adam([w], lr=learning_rate)

    prev = 1e10

    for step in range(max_iter):

        a = 1 / 2 * (nn.Tanh()(w) + 1)

        loss1 = nn.MSELoss(reduction='sum')(a, images)
        loss2 = torch.sum(c * f(a))

        cost = loss1 + loss2

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        # Early Stop when loss does not converge.
        if step % (max_iter // 10) == 0:
            if cost > prev:
                print('Attack Stopped due to CONVERGENCE....')
                return a
            prev = cost

        # print('- Learning Progress : %2.2f %%        ' % ((step + 1) / max_iter * 100), end='\r')

    attack_images = 1 / 2 * (nn.Tanh()(w) + 1)

    return attack_images
