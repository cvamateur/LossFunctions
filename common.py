import os
import argparse
import numpy as np
import matplotlib.colors as mc
import matplotlib.pyplot as plt


def get_common_parser(desc: str = "MNIST Training"):
    parser = argparse.ArgumentParser(description=desc)

    # Global configurations
    parser.add_argument("--download", action="store_true", help="Download MNIST dataset")
    parser.add_argument("--dark-theme", action="store_true", help="Pictures background will be black")
    parser.add_argument("--log-freq", type=int, default=50, help="Logging frequency")
    parser.add_argument("--vis-freq", type=int, default=2, help="Step size to visualize")
    parser.add_argument("--eval-epoch", type=int, default=5, help="Evaluation and visualization start at this epoch")
    parser.add_argument("--num-workers", type=int, default=4, help="Number workers for dataset")

    # Model configurations
    parser.add_argument("--use-bias", action="store_true", help="Use bias on last FC layer.")

    # Training configurations
    parser.add_argument("--batch-size", type=int, default=128, metavar="N")
    parser.add_argument("--num-epochs", type=int, default=500, metavar="N")
    parser.add_argument("--lr", type=float, default=1e-3, metavar="LR")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for L2 regularization")
    parser.add_argument("--step-size", type=int, default=100, help="Learning rate step size")
    parser.add_argument("--gamma", type=float, default=0.7, help="Learning rate gamma")
    return parser


def get_softmax_args():
    parser = get_common_parser("MNIST - SoftmaxLoss")
    return parser.parse_args()


def get_l2_softmax_args():
    parser = get_common_parser("MNIST - L2 SoftmaxLoss")
    parser.add_argument("--alpha", type=float, default=24.0, help="Scale factor after feature normalization")
    parser.add_argument("--trainable", action="store_true", help="Alpha is trainable")
    return parser.parse_args()


def get_ring_loss_args():
    parser = get_common_parser("MNIST - SoftmaxLoss + RingLoss")
    parser.add_argument("-R", type=float, default=24.0, help="Initial value of R")
    parser.add_argument("--loss-weight", type=float, default=0.01, help="Loss weight for RingLoss")
    return parser.parse_args()


class FeatureVisualizer(object):
    """
    Feature Visualizer that record train and validation features at
    the same time.
    """
    FEAT_COLORS = list(mc.TABLEAU_COLORS.values())
    TEXT_COLOR: str = "black"

    def __init__(self, name: str,
                 train_batches: int,
                 valid_batches: int,
                 batch_size: int,
                 start_epoch: int = 100,
                 frequency: int = 100,
                 use_bias: bool = False,
                 dark_theme: bool = False):
        if use_bias: name += "_bias"
        self.root = os.path.join("./pics", name)
        self.num_train = train_batches * batch_size
        self.num_valid = valid_batches * batch_size
        self.next_epoch = start_epoch
        self.frequency = frequency
        self._t = train_batches + valid_batches
        self._b = batch_size
        self._i = 0
        self.features = np.empty([self._t * self._b, 2], dtype=np.float32)
        self.labels = np.empty([self._t * self._b], dtype=np.int64)
        os.makedirs(self.root, exist_ok=True)
        print("info: visualizer root directory:", self.root)
        if dark_theme:
            plt.style.use('dark_background')
            self.TEXT_COLOR = "white"

    def record(self, epoch, features, labels):
        if epoch == self.next_epoch:
            i, b = self._i, self._b
            self.features[i * b: (i + 1) * b] = features
            self.labels[i * b:(i + 1) * b] = labels
            self._i = (i + 1) % self._t

    def save_fig(self, epoch: int, **kwargs):
        if epoch == self.next_epoch:
            self.next_epoch += self.frequency
            suffix = ",".join(f"{k}={v}" for k, v in kwargs.items())
            fname = f"epoch={epoch}{',' if suffix else ''}{suffix}"
            dest_path = os.path.join(self.root, f"{fname}.jpg")
            if not self.num_valid:
                fig, ax = plt.subplots(layout="tight", dpi=300)
                ax.set_aspect("equal")
                self._save_fig(ax, self.features, self.labels, split="Training")
            else:
                fig, axes = plt.subplots(1, 2, sharex='all', sharey="all", layout="tight", dpi=100)
                feats_train = self.features[:self.num_train]
                label_train = self.labels[:self.num_train]
                feats_valid = self.features[self.num_train:]
                label_valid = self.labels[self.num_train:]
                self._save_fig(axes[0], feats_train, label_train, split="Training")
                self._save_fig(axes[1], feats_valid, label_valid, split="Validation")
            fig.suptitle(fname, fontsize="medium")
            plt.savefig(dest_path)
            print("Save features:", dest_path)

    def _save_fig(self, ax, feats, labels, **kwargs):
        split = kwargs.pop("split")
        ax.grid(False)
        ax.set_xlabel(split, fontsize="medium")
        ax.tick_params(labelsize="xx-small")
        for i in range(10):
            feats_i = feats[labels == i]
            if len(feats_i) == 0:
                print(f"info: got empty predict of class {i}")
                continue
            center = np.mean(feats_i, axis=0)
            ax.scatter(feats_i[:, 0], feats_i[:, 1], c=self.c(i), marker='.', s=0.6)
            ax.text(center[0], center[1], str(i), c=self.TEXT_COLOR, fontsize="large", fontweight="bold")

    @classmethod
    def c(cls, i: int):
        return cls.FEAT_COLORS[i]

    @classmethod
    def inv_c(cls, i: int):
        rgba = mc.to_rgba_array(cls.FEAT_COLORS[i]) * 255
        return "black" if np.mean(rgba[:3]) >= 128 else "white"

