import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize, Compose

from tqdm import tqdm

from losses import SoftmaxLoss
from visualize import FeatureVisualizer

best_acc: float = 0.0
best_loss: float = float("inf")
best_path: str = "./ckpt/best.pth"


class MNistNet(nn.Module):

    def __init__(self, input_shape=(1, 28, 28), use_bias=False):
        super().__init__()
        self.conv0 = nn.Conv2d(input_shape[0], 16, 3, 2, 1)     # [14, 14]
        self.conv1 = nn.Conv2d(16, 32, 3, 2, 1)                 # [7, 7]
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)                 # [4, 4]
        self.conv3 = nn.Conv2d(64, 2, 3, 2, 1)                  # [2, 2]
        self.relu = nn.PReLU(2)
        self.pool = nn.AdaptiveAvgPool2d(1)                     # [1, 1]
        self.fc = nn.Conv2d(2, 10, 1, 1, bias=use_bias)

    def forward(self, x):
        x = F.relu(self.conv0(x), inplace=True)
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        x = self.relu(self.conv3(x))
        features = self.pool(x)
        logits = self.fc(features)
        return logits.view(-1, 10), features.view(-1, 2)


def get_args():
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument("-b", "--batch-size", type=int, default=128, metavar="N")
    parser.add_argument("-e", "--num-epochs", type=int, default=1000, metavar="N")
    parser.add_argument("-lr", "--lr", type=float, default=1e-3, metavar="LR")
    parser.add_argument("--weight-decay", type=float, default=1e-6, help="Weight decay for L2 regularization")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA training")
    parser.add_argument("--use-bias", action="store_true", help="Use bias on last fc layer.")
    parser.add_argument("--download", action="store_true", help="Download dataset.")
    parser.add_argument("--step-size", type=int, default=100, help="Learning rate step size")
    parser.add_argument("--gamma", type=float, default=0.7, help="Learning rate gamma")
    parser.add_argument("--eval-epoch", type=int, default=5, help="Evaluation after this epoch")
    parser.add_argument("--log-freq", type=int, default=10, help="Logging frequency")
    parser.add_argument("--load-best", action="store_true", help="Try load best ckpt")
    parser.add_argument("--vis-freq", type=int, default=5, help="Step size to visualize")
    return parser.parse_args()


def main(args):
    if args.no_cuda:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    ################
    # MNIST Dataset
    ################
    transform = [ToTensor(), Normalize([0.1307], [0.3081])]
    transform = Compose(transform)
    inv_transform = [Normalize([0.0], [1 / 0.3081]), Normalize([-0.1307], [1.0])]
    inv_transform = Compose(inv_transform)

    ds_train = MNIST("./data", True, transform, download=args.download)
    ds_valid = MNIST("./data", False, transform, download=args.download)
    if args.no_cuda:
        ds_train = DataLoader(ds_train, args.batch_size, True, drop_last=True)
        ds_valid = DataLoader(ds_valid, args.batch_size, False, drop_last=True)
    else:
        ds_train = DataLoader(ds_train, args.batch_size, True, drop_last=True, num_workers=2, pin_memory=True)
        ds_valid = DataLoader(ds_valid, args.batch_size, False, drop_last=True, num_workers=2, pin_memory=True)

    ################
    # Visualizer
    ################
    vis = FeatureVisualizer(len(ds_train) * args.batch_size, args.batch_size)

    ################
    # Model
    ################
    model = MNistNet(use_bias=args.use_bias)
    model = model.to(device)
    start_epoch = 0
    if args.load_best:
        try:
            global best_acc, best_loss
            state_dict = torch.load(best_path, map_location=device)
            start_epoch = state_dict.pop("epoch", 0)
            best_acc = state_dict.pop("best_acc", 0.0)
            best_loss = state_dict.pop("best_loss", float("inf"))
            model.load_state_dict(state_dict)
            del state_dict
            print("INFO: Load best ckpt:", best_path)
        except Exception:
            pass

    #################
    # Optimizer
    #################
    # optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    schedular = StepLR(optimizer, args.step_size, args.gamma)

    #################
    # Loss Function
    #################
    criterion = SoftmaxLoss().to(device)

    #################
    # Train loop
    #################
    for epoch in range(start_epoch, args.num_epochs):
        train_step(args, epoch, device, model, ds_train, criterion, optimizer, vis)
        if epoch >= args.eval_epoch:
            valid_step(args, epoch, device, model, ds_valid, criterion)
        schedular.step()

        if epoch > args.eval_epoch and epoch % args.vis_freq == 0:
            vis.show(epoch)


def train_step(args, epoch, device, model, dataset, criterion, optimizer, vis):
    model.train()

    total_loss: float = 0.0
    correct: int = 0
    progress_bar = tqdm(dataset, desc=f"Train: {epoch}/{args.num_epochs}")

    vis.reset()
    for i, (images, labels) in enumerate(dataset):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # forward
        logits, feats = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() / len(dataset)
        avg_loss = total_loss * len(dataset) / (i + 1)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accuracy
        preds = torch.argmax(logits.squeeze(), dim=1).squeeze()
        correct += torch.eq(preds, labels).sum().item()
        acc = correct / ((i + 1) * args.batch_size) * 100

        # Log info
        info_str = "loss: {loss:.4f}, acc: {acc:.1f}%".format(loss=avg_loss, acc=acc)
        progress_bar.set_postfix_str(info_str)
        if i % args.log_freq == 0:
            progress_bar.update(args.log_freq)

        # Record Features
        feats = feats.detach().to("cpu").numpy()
        preds = preds.detach().to("cpu").numpy()
        vis.record(i, feats, preds)

    progress_bar.update(len(dataset) - progress_bar.n)


def valid_step(args, epoch, device, model, dataset, criterion):
    model.eval()

    total_loss: float = 0.0
    correct: int = 0
    acc: float = 0.0
    progress_bar = tqdm(dataset, desc=f"Validate: {epoch}/{args.num_epochs}")
    for i, (images, labels) in enumerate(dataset):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # forward
        with torch.no_grad():
            logits, feats = model(images)
            loss = criterion(logits, labels)

        total_loss += loss.item() / len(dataset)
        avg_loss = total_loss * len(dataset) / (i + 1)

        # Accuracy
        preds = torch.argmax(logits.squeeze(), dim=1).squeeze()
        correct += torch.eq(preds, labels).sum().item()
        acc = correct / ((i + 1) * args.batch_size) * 100

        # Log info
        info_str = "| loss: {loss:.4f} | acc: {acc:.1f}%".format(loss=avg_loss, acc=acc)
        progress_bar.set_postfix_str(info_str)
        if i % args.log_freq == 0:
            progress_bar.update(args.log_freq)

    progress_bar.update(len(dataset) - progress_bar.n)

    # Save best model
    global best_acc
    global best_loss
    if acc > best_acc and total_loss < best_loss:
        print("Save best weights:", best_path, flush=True)
        best_acc = acc
        best_loss = total_loss
        state_dict = model.state_dict()
        state_dict.update(epoch=epoch, best_acc=best_acc, best_loss=best_loss)
        torch.save(state_dict, best_path)


if __name__ == '__main__':
    args = get_args()
    main(args)
