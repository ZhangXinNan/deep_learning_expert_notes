import os
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from mlp import NeuralNetwork
from LeNet5 import Net
import sys
__dir__ = os.path.abspath(os.path.dirname(__file__))
__root__ = os.path.abspath(os.path.join(__dir__, '..'))
print(__dir__)
print(__root__)
sys.path.append(__root__)
from utility.plot_curve import plot_accuracy, plot_loss


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='MNIST')
    parser.add_argument('--model_name', default='mlp')
    parser.add_argument('--opti_name', default='sgd')
    parser.add_argument('--epoch', default=50, type=int)
    parser.add_argument('--out_dir', default='output')
    return parser.parse_args()


args = get_args()
print(args)
if not os.path.isdir(args.out_dir):
    os.makedirs(args.out_dir)


if args.dataset_name == 'MNIST':
    # Download training data from open datasets.
    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )
    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
elif args.dataset_name == 'FashionMNIST':
    # Download training data from open datasets.
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
else:
    exit()

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break


# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

if args.model_name == 'mlp':
    model = NeuralNetwork().to(device)
else:
    model = Net().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
if args.opti_name == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    train_loss = 0
    train_acc = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        train_acc += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    train_loss /= num_batches
    train_acc /= size
    return train_acc, train_loss


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct, test_loss


train_acc_list = []
train_loss_list = []
val_acc_list = []
val_loss_list = []
for t in range(args.epoch):
    print(f"Epoch {t+1}\n-------------------------------")
    train_acc, train_loss = train(train_dataloader, model, loss_fn, optimizer)
    val_acc, val_loss = test(test_dataloader, model, loss_fn)
    train_acc_list.append(train_acc)
    train_loss_list.append(train_loss)
    val_acc_list.append(val_acc)
    val_loss_list.append(val_loss)
print("Done!")
print(train_acc_list)
print(train_loss_list)
print(val_acc_list)
print(val_loss_list)

prefix = args.dataset_name + '.' + args.model_name + '.' + args.opti_name + '.'
plot_accuracy(train_acc_list, val_acc_list, os.path.join(args.out_dir, prefix + 'acc.png'))
plot_loss(train_loss_list, val_loss_list, os.path.join(args.out_dir, prefix + 'loss.png'))



