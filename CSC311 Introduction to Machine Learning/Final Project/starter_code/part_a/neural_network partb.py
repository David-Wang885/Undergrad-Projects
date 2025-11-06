from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch
import matplotlib.pyplot as plt


def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """ Return ||W^1|| + ||W^2||.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2)
        h_w_norm = torch.norm(self.h.weight, 2)
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """

        out = torch.sigmoid(self.g(inputs))
        out = self.h(out) / (1 + torch.abs(self.h(out)))
        return out


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    
    # Tell PyTorch you are training the model.
    model.train()

    # Set up array for plotting
    plotting = np.zeros((3, num_epoch))

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    for epoch in range(0, num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            loss = torch.sum((output - target) ** 2.)
            # loss = torch.sum((output - target) ** 2.) + lamb * model.get_weight_norm() / 2
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)

        # Put values into the array
        plotting[0][epoch] = epoch
        plotting[1][epoch] = valid_acc
        plotting[2][epoch] = train_loss

        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))

    # Plot and save
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle('Plots of Accuracy and Train Loss')
    ax1.plot(plotting[0], plotting[1])
    plt.setp(ax1.get_xticklabels(), fontsize=6)

    # share x only
    ax2 = plt.subplot(212, sharex=ax1)
    plt.plot(plotting[0], plotting[2])

    # make these tick labels invisible
    plt.setp(ax2.get_xticklabels(), visible=False)

    # Add axis label
    ax1.set_ylabel('Accuracy of validation data')
    ax2.set_ylabel('Training loss')
    ax2.set_xlabel('Epoch')
    plt.savefig('partb1')
    plt.show()

    # print out the maximum accuracy
    max_acc = np.max(plotting[1])
    print(max_acc)


def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    # Set model hyperparameters.
    n_question = zero_train_matrix.shape[1]
    # k = [10, 50, 100, 200, 500]
    k = 10

    # Set optimization hyperparameters.
    # lr = [0.005, 0.01, 0.015, 0.02, 0.025]
    lr = 0.015
    # lamb = [0.001, 0.01, 0.1, 1.]
    lamb = 0.1
    # epoch_arr = [5, 10, 20, 30, 50]
    num_epoch = 40

    # Loops for tuning the hyperparameters
    # for num in k:
    #     model = AutoEncoder(n_question, num)
    #     train(model, lr, lamb, train_matrix, zero_train_matrix,
    #           valid_data, num_epoch)
    #     print(evaluate(model, zero_train_matrix, test_data))
    # for num in lr:
    #     model = AutoEncoder(n_question, k)
    #     train(model, num, lamb, train_matrix, zero_train_matrix,
    #           valid_data, num_epoch)
    # for num in lamb:
    #     model = AutoEncoder(n_question, k)
    #     train(model, lr, num, train_matrix, zero_train_matrix,
    #           valid_data, num_epoch)

    # After the hyperparameters are tuned, make plot
    model = AutoEncoder(n_question, k)
    train(model, lr, lamb, train_matrix, zero_train_matrix,
          valid_data, num_epoch)
    print(evaluate(model, zero_train_matrix, test_data))


    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
