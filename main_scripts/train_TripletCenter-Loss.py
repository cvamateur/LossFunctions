"""
Train MNIST with Original Softmax Loss

Structure:
    extractor -> nn.Linear -> SoftmaxLoss + CenterLoss
"""
import torch

from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize, Compose

from losses import SoftmaxLoss
from losses.metric import TripletCenterLoss
from common.nets import MNIST_Net
from common.cli_parser import get_tripletcenter_loss_args
from common.visualizer import  FeatureVisualizer


use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")


def main(args):
    ################
    # MNIST Dataset
    ################
    transform = Compose([ToTensor(), Normalize([0.1307], [0.3081])])
    ds_train = MNIST(args.data_root, True, transform, download=args.download)
    ds_valid = MNIST(args.data_root, False, transform, download=args.download)
    kwargs = {"batch_size": args.batch_size, "num_workers": args.num_workers, "drop_last": True, "pin_memory": use_gpu}
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
    softmax_loss_fn = SoftmaxLoss().to(device)
    tripletcenter_loss_fn = TripletCenterLoss(2, 10, args.margin).to(device)
    criterion = (softmax_loss_fn, tripletcenter_loss_fn)

    #################
    # Optimizer
    #################
    optimizer = Adam([{"params": extractor.parameters()},
                      {"params": classifier.parameters()},
                      {"params": tripletcenter_loss_fn.parameters(), "weight_decay": 0}],
                     lr=args.lr, weight_decay=args.weight_decay)
    schedular = StepLR(optimizer, args.step_size, args.gamma)

    ################
    # Visualizer
    ################
    visualizer = FeatureVisualizer("TripletCenterLoss", len(ds_train), len(ds_valid), args.batch_size,
                                   args.eval_epoch, args.vis_freq, args.use_bias, args.dark_theme)

    #################
    # Train loop
    #################
    model = (extractor, classifier)
    for epoch in range(1, args.num_epochs + 1):
        train_step(epoch, model, ds_train, criterion, optimizer, visualizer, args)
        if epoch >= args.eval_epoch:
            valid_step(epoch, model, ds_valid, criterion, visualizer, args)
        visualizer.save_fig(epoch, dpi=args.dpi, m=args.margin, loss_weight=args.loss_weight)
        schedular.step()


def train_step(epoch, model, dataset, criterion, optimizer, visualizer, args):
    model[0].train()
    model[1].train()

    total_loss: float = 0.0
    total_correct: int = 0
    progress_bar = tqdm(dataset, desc=f"Training  : {epoch}/{args.num_epochs}")
    for i, (images, labels) in enumerate(dataset):
        if use_gpu:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

        # forward
        feats = model[0](images)        # [N, 2]
        logits = model[1](feats)        # [N, C]
        softmax_loss = criterion[0](logits, labels)
        tripletcenter_loss = criterion[1](feats, labels) * args.loss_weight
        loss = softmax_loss + tripletcenter_loss

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
        info_str = "loss: {:.4f}, softmax: {:.4f}, triplet-center: {:.4f}, acc: {:.1f}%, centers: {}".format(
            avg_loss, softmax_loss, tripletcenter_loss, acc, criterion[1].centers.detach().cpu()[0].numpy())
        progress_bar.set_postfix_str(info_str)
        if (i + 1) % args.log_freq == 0:
            progress_bar.update(args.log_freq)

        # Record Features
        feats = feats.detach().to("cpu").numpy()
        preds = preds.detach().to("cpu").numpy()
        visualizer.record(epoch, feats, preds)

    progress_bar.update(len(dataset) - progress_bar.n)


@torch.no_grad()
def valid_step(epoch, model, dataset, criterion, visualizer, args):
    model[0].eval()
    model[1].eval()

    total_loss: float = 0.0
    total_correct: int = 0
    progress_bar = tqdm(dataset, desc=f"Validation: {epoch}/{args.num_epochs}")
    for i, (images, labels) in enumerate(dataset):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # forward
        feats = model[0](images)
        logits = model[1](feats)
        softmax_loss = criterion[0](logits, labels)
        tripletcenter_loss = criterion[1](feats, labels) * args.loss_weight
        loss = softmax_loss + tripletcenter_loss

        # loss
        total_loss += loss.item() / len(dataset)
        avg_loss = total_loss * len(dataset) / (i + 1)

        # metric
        preds = logits.argmax(dim=1)
        total_correct += torch.eq(preds, labels).sum().item()
        acc = total_correct / ((i + 1) * args.batch_size) * 100

        # Log info
        info_str = "loss: {:.4f}, softmax: {:.4f}, triplet-center: {:.4f}, acc: {:.1f}% ".format(avg_loss, softmax_loss, tripletcenter_loss, acc)
        progress_bar.set_postfix_str(info_str)
        if i % args.log_freq == 0:
            progress_bar.update(args.log_freq)

        # Record Features
        feats = feats.to("cpu").numpy()
        preds = preds.to("cpu").numpy()
        visualizer.record(epoch, feats, preds)

    progress_bar.update(len(dataset) - progress_bar.n)


if __name__ == '__main__':
    args = get_tripletcenter_loss_args()
    main(args)