import os
import torch

from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize, Compose

from common import get_softmax_args
from nets import MNIST_Net
from losses import SoftmaxLoss
from visualize import FeatureVisualizer


use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")


def main(args):
    ################
    # MNIST Dataset
    ################
    transform = Compose([ToTensor(), Normalize([0.1307], [0.3081])])
    ds_train = MNIST("./data", True, transform, download=args.download)
    ds_valid = MNIST("./data", False, transform, download=args.download)
    kwargs = {"batch_size": args.batch_size, "num_workers": args.workers, "drop_last": True, "pin_memory": use_gpu}
    ds_train = DataLoader(ds_train, shuffle=True, **kwargs)
    ds_valid = DataLoader(ds_valid, shuffle=False, **kwargs)

    ################
    # Model
    ################
    extractor = MNIST_Net(in_channels=1, out_channels=2).to(device)
    classifier = torch.nn.Linear(2, 10, args.use_bias).to(device)

    #################
    # Loss Function
    #################
    criterion = SoftmaxLoss().to(device)

    #################
    # Optimizer
    #################
    optimizer = Adam([{"params": extractor.parameters()},
                      {"params": classifier.parameters()}],
                     lr=args.lr, weight_decay=args.weight_decay)
    schedular = StepLR(optimizer, args.step_size, args.gamma)

    ################
    # Visualizer
    ################
    visualizer = FeatureVisualizer("SoftmaxLoss", len(ds_train) * args.batch_size, args.batch_size)

    #################
    # Train loop
    #################
    model = (extractor, classifier)
    for epoch in range(1, args.num_epochs + 1):
        train_step(epoch, model, ds_train, criterion, optimizer, visualizer, args)
        if epoch >= args.eval_epoch:
            valid_step(epoch, model, ds_valid, criterion, args)
        schedular.step()


def train_step(epoch, model, dataset, criterion, optimizer, visualizer, args):
    model[0].train()
    model[1].train()

    total_loss: float = 0.0
    total_correct: int = 0
    progress_bar = tqdm(dataset, desc=f"Train: {epoch}/{args.num_epochs}")
    for i, (images, labels) in enumerate(dataset):
        if use_gpu:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

        # forward
        feats = model[0](images)   # [N, 2]
        logits = model[1](feats)   # [N, C]
        loss = criterion(logits, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # loss
        total_loss += loss.item() / len(dataset)
        avg_loss = total_loss * len(dataset) / (i + 1)

        # metric
        preds = logits.argmax(dim=1)
        total_correct += torch.eq(preds, labels).sum().item()
        acc = total_correct / ((i + 1) * args.batch_size) * 100

        # Log info
        info_str = "loss: {loss:.4f}, acc: {acc:.1f}%".format(loss=avg_loss, acc=acc)
        progress_bar.set_postfix_str(info_str)
        if (i + 1) % args.log_freq == 0:
            progress_bar.update(args.log_freq)

        # Record Features
        feats = feats.detach().to("cpu").numpy()
        preds = preds.detach().to("cpu").numpy()
        visualizer.record(feats, preds)

    progress_bar.update(len(dataset) - progress_bar.n)
    if epoch % args.vis_freq == 0:
        visualizer.save_fig(epoch)


def valid_step(epoch, model, dataset, criterion, args):
    model[0].eval()
    model[1].eval()

    total_loss: float = 0.0
    total_correct: int = 0
    progress_bar = tqdm(dataset, desc=f"Validate: {epoch}/{args.num_epochs}")
    for i, (images, labels) in enumerate(dataset):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # forward
        with torch.no_grad():
            feats =  model[0](images)
            preds = model[1](feats)
            loss = criterion(preds, labels)

        # matrics
        total_loss += loss.item() / len(dataset)
        avg_loss = total_loss * len(dataset) / (i + 1)
        total_correct += torch.eq(preds.argmax(dim=1), labels).sum().item()
        acc = total_correct / ((i + 1) * args.batch_size) * 100

        # Log info
        info_str = "loss: {loss:.4f}, acc: {acc:.1f}%".format(loss=avg_loss, acc=acc)
        progress_bar.set_postfix_str(info_str)
        if i % args.log_freq == 0:
            progress_bar.update(args.log_freq)

    progress_bar.update(len(dataset) - progress_bar.n)



if __name__ == '__main__':
    args = get_softmax_args()

    os.makedirs("./ckpt", exist_ok=True)
    os.makedirs("tmp", exist_ok=True)
    main(args)