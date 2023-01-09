import os
import argparse
import numpy as np
import matplotlib.colors as mc
import matplotlib.pyplot as plt


def get_common_parser(desc: str = "MNIST Training"):
    parser = argparse.ArgumentParser(description=desc)

    # Global configurations
    parser.add_argument("--download", action="store_true", help="Download MNIST dataset")
    parser.add_argument("--load-best", action="store_true", help="Try load best ckpt")
    parser.add_argument("--log-freq", type=int, default=50, help="Logging frequency")
    parser.add_argument("--vis-freq", type=int, default=2, help="Step size to visualize")
    parser.add_argument("--eval-epoch", type=int, default=2, help="Evaluation after this epoch")
    parser.add_argument("--num-workers", type=int, default=2, help="Number workers for dataset")
    parser.add_argument("--dark-theme", action="store_true", help="Pictures background will be black")

    # Model configurations
    parser.add_argument("--use-bias", action="store_true", help="Use bias on last FC layer.")

    # Training configurations
    parser.add_argument("--batch-size", type=int, default=128, metavar="N")
    parser.add_argument("--num-epochs", type=int, default=500, metavar="N")
    parser.add_argument("--lr", type=float, default=1e-3, metavar="LR")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for L2 regularization")
    parser.add_argument("--step-size", type=int, default=50, help="Learning rate step size")
    parser.add_argument("--gamma", type=float, default=0.7, help="Learning rate gamma")
    return parser


def get_softmax_args():
    parser = get_common_parser("MNIST - SoftmaxLoss")
    return parser.parse_args()


def get_l2_softmax_args():
    parser = get_common_parser("MNIST - L2 SoftmaxLoss")
    parser.add_argument("--alpha", type=float, default=8.0, help="Scale factor after feature normalization")
    parser.add_argument("--trainable", action="store_true", help="Alpha is trainable")
    return parser.parse_args()


def get_ring_loss_args():
    parser = get_common_parser("MNIST - SoftmaxLoss + RingLoss")
    parser.add_argument("-R", type=float, default=1.0, help="Initial value of R")
    parser.add_argument("--loss-weight", type=float, default=0.01, help="Loss weight for RingLoss")
    return parser.parse_args()


class FeatureVisualizer(object):
    COLORS = list(mc.TABLEAU_COLORS.values())

    def __init__(self, name: str, num_batches: int, batch_size: int = 16, *, dirname: str = ""):
        self.name = name
        self._t = num_batches
        self._b = batch_size
        self._i = 0
        self.features = np.empty([self._t * self._b, 2], dtype=np.float32)
        self.labels = np.empty([self._t * self._b], dtype=np.int64)
        self._root = os.path.join("./pics", dirname)
        os.makedirs(self._root, exist_ok=True)

    def record(self, features, labels):
        i, b = self._i, self._b
        self.features[i * b: (i + 1) * b] = features
        self.labels[i * b:(i + 1) * b] = labels
        self._i = (i + 1) % self._t

    def save_fig(self, epoch: int, **kwargs):
        kwargs.update(epoch=epoch)
        title = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        fig, ax = plt.subplots()
        ax.set_title(title)
        ax.grid(False)
        for i in range(10):
            feats = self.features[self.labels == i]
            center = np.mean(feats, axis=0)
            ax.plot(feats[:, 0], feats[:, 1], '.', c=self.c(i), label=str(i))
            ax.text(center[0], center[1], str(i), c="black", fontsize="large", fontweight="bold")
        dest_path = os.path.join(self._root, f"{self.name}_{title}.jpg")
        plt.savefig(dest_path)
        print("Save features:", dest_path)

    @classmethod
    def c(cls, i: int):
        return cls.COLORS[i]

    @classmethod
    def inv_c(cls, i: int):
        rgba = mc.to_rgba_array(cls.COLORS[i]) * 255
        return "black" if np.mean(rgba[:3]) >= 128 else "white"


class FeatureVisualizerV2(object):
    """
    Feature Visualizer that record train and validation features at
    the same time.
    """
    COLORS = list(mc.TABLEAU_COLORS.values())

    def __init__(self, name: str, batch_size: int, num_train: int, num_valid: int = 0, dark_theme: bool = False):
        self.name = name
        self._root = os.path.join("./pics", name)
        self._t = num_train + num_valid
        self._b = batch_size
        self._i = 0
        self.features = np.empty([self._t * self._b, 2], dtype=np.float32)
        self.labels = np.empty([self._t * self._b], dtype=np.int64)
        self.num_train = num_train * batch_size
        self.num_valid = num_valid * batch_size
        os.makedirs(self._root, exist_ok=True)
        if dark_theme:
            plt.style.use('dark_background')
            self._tc = "white"
        else:
            self._tc = "black"

    def record(self, features, labels):
        i, b = self._i, self._b
        self.features[i * b: (i + 1) * b] = features
        self.labels[i * b:(i + 1) * b] = labels
        self._i = (i + 1) % self._t

    def save_fig(self, epoch: int, **kwargs):
        suffix = ",".join(f"{k}={v}" for k, v in kwargs.items())
        fname = f"Epoch={epoch}{suffix}"
        dest_path = os.path.join(self._root, f"{fname}.jpg")
        if not self.num_valid:
            fig, ax = plt.subplots(layout="tight", dpi=300)
            ax.set_aspect("equal")
            self._save_fig(ax, self.features, self.labels, split="Training")
        else:
            fig, axes = plt.subplots(1, 2, sharex='all', sharey="all", layout="tight", dpi=300)
            feats_train = self.features[:self.num_train]
            feats_valid = self.features[self.num_train:]
            label_train = self.labels[:self.num_train]
            label_valid = self.labels[self.num_train:]
            self._save_fig(axes[0], feats_train, label_train, split="Training")
            self._save_fig(axes[1], feats_valid, label_valid, split="Validation")

        fig.suptitle(fname)
        plt.savefig(dest_path)
        print("Save features:", dest_path)

    def _save_fig(self, ax, feats, labels, **kwargs):
        split = kwargs.pop("split")
        ax.set_xlabel(split)
        ax.grid(False)
        for i in range(10):
            feats_i = feats[labels == i]
            center = np.mean(feats_i, axis=0)
            ax.scatter(feats_i[:, 0], feats_i[:, 1], c=self.c(i), marker='.', s=0.8)
            ax.text(center[0], center[1], str(i), c=self._tc, fontsize="large", fontweight="bold")

    @classmethod
    def c(cls, i: int):
        return cls.COLORS[i]

    @classmethod
    def inv_c(cls, i: int):
        rgba = mc.to_rgba_array(cls.COLORS[i]) * 255
        return "black" if np.mean(rgba[:3]) >= 128 else "white"


