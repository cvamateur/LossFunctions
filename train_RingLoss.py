"""
Train MNIST with Original RingLoss

Structure:
    extractor -> NormLinear -> SoftmaxLoss + RingLoss
"""
import torch
from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize, Compose

from common import get_ring_loss_args, FeatureVisualizer
from nets import MNIST_Net, NormLinear
from losses import SoftmaxLoss, RingLoss

use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")


def main(args):
    ################
    # MNIST Dataset
    ################
    transform = Compose([ToTensor(), Normalize([0.1307], [0.3081])])
    ds_train = MNIST("./data", True, transform, download=args.download)
    ds_valid = MNIST("./data", False, transform, download=args.download)
    kwargs = {"batch_size": args.batch_size, "num_workers": args.num_workers, "drop_last": True, "pin_memory": use_gpu}
    ds_train = DataLoader(ds_train, shuffle=True, **kwargs)
    ds_valid = DataLoader(ds_valid, shuffle=False, **kwargs)

    ################
    # Model
    ################
    extractor = MNIST_Net(in_channels=1, out_channels=2).to(device)
    classifier = NormLinear(2, 10, args.use_bias).to(device)
    model = (extractor, classifier)

    #################
    # Loss Function
    #################
    criterion_softmax = SoftmaxLoss().to(device)
    criterion_ring = RingLoss(args.R, args.loss_weight).to(device)
    criterion = (criterion_softmax, criterion_ring)

    #################
    # Optimizer
    #################
    optimizer = Adam([{"params": extractor.parameters()},
                      {"params": classifier.parameters()},
                      {"params": criterion_ring.parameters()}],
                     lr=args.lr, weight_decay=args.weight_decay)
    schedular = StepLR(optimizer, args.step_size, args.gamma)

    ################
    # Visualizer
    ################
    vis_train = FeatureVisualizer("RingLoss-train", len(ds_train), args.batch_size, dirname="RingLoss")
    vis_valid = FeatureVisualizer("RingLoss-valid", len(ds_valid), args.batch_size, dirname="RingLoss")

    #################
    # Train loop
    #################
    for epoch in range(1, args.num_epochs + 1):
        train_step(epoch, model, ds_train, criterion, optimizer, vis_train, args)
        if epoch >= args.eval_epoch:
            valid_step(epoch, model, ds_valid, criterion, vis_valid, args)
        schedular.step()


def train_step(epoch, model, dataset, criterion, optimizer, visualizer, args):
    model[0].train()
    model[1].train()
    criterion[1].train()

    total_loss: float = 0.0
    total_correct: int = 0
    progress_bar = tqdm(dataset, desc=f"Training  : {epoch}/{args.num_epochs}")
    for i, (images, labels) in enumerate(dataset):
        if use_gpu:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

        # forward
        feats = model[0](images)   # [N, 2]
        logits = model[1](feats)   # [N, C]
        loss_softmax = criterion[0](logits, labels)
        loss_ring = criterion[1](feats)
        loss = loss_softmax + loss_ring

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
        R = criterion[1].R.item()
        info_str = "loss: {:.4f}, acc: {:.1f}%, softmax: {:.4f}, ring: {:.4f}, R: {:.4f}".format(
            avg_loss, acc, loss_softmax, loss_ring, R)
        progress_bar.set_postfix_str(info_str)
        if (i + 1) % args.log_freq == 0:
            progress_bar.update(args.log_freq)

        # Record Features
        feats = feats.detach().to("cpu").numpy()
        preds = preds.detach().to("cpu").numpy()
        if epoch % args.vis_freq == 0:
            visualizer.record(feats, preds)

    progress_bar.update(len(dataset) - progress_bar.n)
    if epoch % args.vis_freq == 0:
        visualizer.save_fig(epoch, init_R=args.R, loss_weight=args.loss_weight)


@torch.no_grad()
def valid_step(epoch, model, dataset, criterion, visualizer, args):
    model[0].eval()
    model[1].eval()
    criterion[1].eval()

    total_loss: float = 0.0
    total_correct: int = 0
    progress_bar = tqdm(dataset, desc=f"Validation: {epoch}/{args.num_epochs}")
    for i, (images, labels) in enumerate(dataset):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # forward
        feats =  model[0](images)
        logits = model[1](feats)
        loss_softmax = criterion[0](logits, labels)
        loss_ring = criterion[1](feats)
        loss = loss_softmax + loss_ring

        # loss
        total_loss += loss.item() / len(dataset)
        avg_loss = total_loss * len(dataset) / (i + 1)

        # metric
        preds = logits.argmax(dim=1)
        total_correct += torch.eq(preds, labels).sum().item()
        acc = total_correct / ((i + 1) * args.batch_size) * 100

        # Log info
        R = criterion[1].R.item()
        info_str = "loss: {:.4f}, acc: {:.1f}%, softmax: {:.4f}, ring: {:.4f}, R: {:.4f}".format(
            avg_loss, acc, loss_softmax, loss_ring, R)
        progress_bar.set_postfix_str(info_str)
        if i % args.log_freq == 0:
            progress_bar.update(args.log_freq)

        # Record Features
        feats = feats.to("cpu").numpy()
        preds = preds.to("cpu").numpy()
        if epoch % args.vis_freq == 0:
            visualizer.record(feats, preds)

    progress_bar.update(len(dataset) - progress_bar.n)
    if epoch % args.vis_freq == 0:
        visualizer.save_fig(epoch, init_R=args.R, loss_weight=args.loss_weight)




if __name__ == '__main__':
    args = get_ring_loss_args()
    main(args)