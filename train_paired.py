from models import LeNetWithReg, save_model
import torch
import torch.optim as optim
import torch.nn as nn
from args import *
from data_utils import get_mnist_dataset, get_mnist_dataset_test_only
from models import load_model
import logging
from utils import set_logger
from torch.distributions.categorical import Categorical

# TODO: There are only 40000 training samples in the original training set
if ARGS.first_n_samples is not None:
    ARGS.first_n_samples = max(ARGS.first_n_samples, 40000)

set_logger(ARGS)
logging.info("Called train_paired.py with args:\n" + ARGS.toString())

paired_train, paired_dev, paired_test = get_mnist_dataset(ARGS, paired=True)
if ARGS.test_only_data_path:
    train = get_mnist_dataset_test_only(ARGS)
else:
    train, dev, test = get_mnist_dataset(ARGS)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.info("Using device: " + str(device))

# Initializing model
lenet = LeNetWithReg(ARGS)
lenet.to(device)
logging.info("Model created with structure: \n" + str(lenet))

state_dict = lenet.state_dict()
logging.info("Parameters: " + str(list(state_dict.keys())))
saved_state = load_model(ARGS)
if saved_state is not None:
    logging.info("Loading previous model with name: '{}'".format(ARGS.model_name))
    for k, v in saved_state.items():
        state_dict.update({k: v})
    lenet.load_state_dict(state_dict)
    logging.info("Loaded parameters: " + str(list(saved_state.keys())))
else:
    logging.info("No previous model found with name '{}', training a new one.".format(ARGS.model_name))

lenet.train()

min_loss = float('inf')
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(lenet.cnn_parameters, lr=ARGS.learning_rate, momentum=ARGS.momentum)
streak = 0

for epoch in range(ARGS.epoch):
    epoch_loss = 0
    total = 0
    correct = 0
    # Train
    for a, b in zip(iter(train), iter(paired_train)):
        inputs_batch, labels_batch = a
        inputs_batch_paired, labels_batch_paired = b
        # Format input data
        inputs = inputs_batch.to(device)
        labels = labels_batch.to(device)
        inputs_paired = inputs_batch_paired.to(device)
        labels_paired = labels_batch_paired.to(device)

        # Check for the first batch whether data are actually in pairs
        if epoch == 0 and epoch_loss == 0:
            # print(labels)
            # print(labels_paired)
            assert torch.equal(labels, labels_paired), "Data are not in pairs!"

        # Training step
        optimizer.zero_grad()

        if ARGS.method == 1:  # Gradient norm. Not working. Need research on element-wise grad on PyTorch.
            raise NotImplementedError
            optimizer_reg = optim.SGD(lenet.reg_parameters, lr=ARGS.learning_rate, momentum=ARGS.momentum)

            # Update regularization layers 5 times
            for i in range(5):
                # Calculate loss1 as usual
                lenet.zero_grad()
                logit1, reg1, reg_obj1 = lenet(inputs)
                outputs = logit1
                loss1 = criterion(logit1, labels)

                # Calculate loss2 as the loss from paired training data
                logit2, reg2, reg_obj2 = lenet(inputs_paired)
                loss2 = criterion(logit2, labels_paired)

                # Calculate gradient loss
                """
                reg_obj1.retain_grad()
                reg_obj2.retain_grad()
                reg1.backward(torch.ones_like(reg1))
                grad1 = reg_obj1.grad.data
                lenet.zero_grad()
                reg2.backward(torch.ones_like(reg2))
                grad2 = reg_obj2.grad.data
                lenet.zero_grad()
                """
                reg_obj1.retain_grad()
                reg_obj2.retain_grad()
                grad1 = torch.zeros_like(reg_obj1)
                grad2 = torch.zeros_like(reg_obj2)
                for i in range(reg1.shape[0]):
                    reg1[i].backward(torch.ones_like(reg1[i]), retain_graph=True)
                    grad1[i] = reg_obj1[i].grad
                    reg2[i].backward(torch.ones_like(reg2[i]), retain_graph=True)
                    grad2[i] = reg_obj2[i].grad
                # torch.autograd.grad(reg1, reg_obj1)
                # torch.autograd.grad(reg1, reg_obj1)
                grad_loss = 0.5 * (torch.square(torch.norm(grad1) - 1) + torch.square(torch.norm(grad2) - 1))
                # logging.info(grad1)
                # logging.info(grad2)

                # Calculate regularization loss
                dloss = -(torch.mean(reg1) - torch.mean(reg2))
                reg_loss = dloss + ARGS.lam * grad_loss

                # Update
                reg_loss.backward()
                optimizer_reg.step()
                lenet.zero_grad()

            # Then update cnn layers once
            # Calculate loss1 as usual
            lenet.zero_grad()
            logit1, reg1, reg_obj1 = lenet(inputs)
            outputs = logit1
            loss1 = criterion(logit1, labels)

            # Calculate loss2 as the loss from paired training data
            logit2, reg2, reg_obj2 = lenet(inputs_paired)
            loss2 = criterion(logit2, labels_paired)

            # Calculate total loss
            dloss = -(torch.mean(reg1) - torch.mean(reg2))
            loss = loss1 + loss2 - dloss

            # Update
            loss.backward()
            optimizer.step()
            lenet.zero_grad()

        elif ARGS.method == 2:  # KL Divergence

            # Calculate loss1 as usual
            lenet.zero_grad()
            logit1, _, reg_obj1 = lenet(inputs)
            outputs = logit1
            loss1 = criterion(logit1, labels)

            # Calculate loss2 as the loss from paired training data
            logit2, _, reg_obj2 = lenet(inputs_paired)
            loss2 = criterion(logit2, labels_paired)

            # Calculate loss + KL divergence
            probs1 = torch.nn.functional.softmax(reg_obj1, dim=1)
            distr1 = Categorical(probs1)
            probs2 = torch.nn.functional.softmax(reg_obj2, dim=1)
            distr2 = Categorical(probs2)
            loss = loss1 + loss2 + torch.mean(torch.distributions.kl_divergence(distr1, distr2))

            # Update
            loss.backward()
            optimizer.step()
            lenet.zero_grad()

        elif ARGS.method == 3:  # L2 regularization

            # Calculate loss1 as usual
            lenet.zero_grad()
            logit1, _, reg_obj1 = lenet(inputs)
            outputs = logit1
            loss1 = criterion(logit1, labels)

            # Calculate loss2 as the loss from paired training data
            logit2, _, reg_obj2 = lenet(inputs_paired)
            loss2 = criterion(logit2, labels_paired)

            # Calculate loss + l2 regularization
            # loss = loss1 + loss2 + torch.mean(torch.square(torch.norm(reg_obj1 - reg_obj2)))
            loss = loss1 + loss2 + ARGS.reg * torch.mean(torch.square(torch.norm(reg_obj1 - reg_obj2)))

            # Update
            loss.backward()
            optimizer.step()
            lenet.zero_grad()

        elif ARGS.method == 4:
            raise NotImplementedError

        elif ARGS.method == 5:  # L1 norm

            # Calculate loss1 as usual
            lenet.zero_grad()
            logit1, _, reg_obj1 = lenet(inputs)
            outputs = logit1
            loss1 = criterion(logit1, labels)

            # Calculate loss2 as the loss from paired training data
            logit2, _, reg_obj2 = lenet(inputs_paired)
            loss2 = criterion(logit2, labels_paired)

            # Calculate loss + l2 regularization
            # loss = loss1 + loss2 + torch.mean(torch.square(torch.norm(reg_obj1 - reg_obj2, p=1)))
            loss = loss1 + loss2 + ARGS.reg * torch.mean(torch.square(torch.norm(reg_obj1 - reg_obj2, p=1)))

            # Update
            loss.backward()
            optimizer.step()
            lenet.zero_grad()

        else:  # No regularization method
            lenet.zero_grad()
            logit1, _, reg_obj1 = lenet(inputs)
            outputs = logit1
            loss1 = criterion(logit1, labels)
            # Calculate loss2 as the loss from paired training data
            logit2, _, reg_obj2 = lenet(inputs_paired)
            loss2 = criterion(logit2, labels_paired)
            loss = loss1 + loss2
            # Update
            loss.backward()
            optimizer.step()
            lenet.zero_grad()


        ## TODO: More methods here

        # Calculate statistics
        epoch_loss += loss.data
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    logging.info("Epoch {}, total train loss: {}".format(epoch, epoch_loss))
    logging.info("Epoch {}, train accuracy: {}".format(epoch, correct / total))

    # Skip evaluation on dev set when training on test_only MNIST and save
    if ARGS.test_only_data_path:
        save_model(lenet.state_dict(), ARGS, new=True)

    # Evaluate on paired dev set
    else:
        with torch.no_grad():
            dev_loss = 0
            total = 0
            correct = 0
            for inputs_batch, labels_batch in paired_dev:
                inputs = inputs_batch.to(device)
                labels = labels_batch.to(device)
                outputs, _, _ = lenet(inputs)
                loss = criterion(outputs, labels)
                dev_loss += loss.data
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            logging.info("Epoch {}, total dev loss: {}".format(epoch, dev_loss))
            logging.info("Epoch {}, dev accuracy : {}".format(epoch, correct / total))

            # Save the model if it is the best so far
            if dev_loss < min_loss:
                min_loss = dev_loss
                logging.info("Best model at epoch {}, model saved.".format(epoch))
                save_model(lenet.state_dict(), ARGS, new=True)
                streak = 0
            else:
                streak += 1
                if streak == ARGS.patience:
                    logging.info("No improvements in {} epochs. Stopped early.".format(ARGS.patience))
                    break



