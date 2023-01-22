import os
import numpy as np
import matplotlib.colors as mc
import matplotlib.pyplot as plt



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
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if use_bias: name += "_bias"
        self.root = os.path.join(root_dir, "pics", name)
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
        print("info: visualizer root directory:", os.path.relpath(self.root))
        if dark_theme:
            plt.style.use('dark_background')
            self.TEXT_COLOR = "white"

    def record(self, epoch, features, labels):
        if epoch == self.next_epoch:
            i, b = self._i, self._b
            self.features[i * b: (i + 1) * b] = features
            self.labels[i * b:(i + 1) * b] = labels
            self._i = (i + 1) % self._t

    def save_fig(self, epoch: int, dpi=100, **kwargs):
        if epoch == self.next_epoch:
            self.next_epoch += self.frequency
            suffix = ",".join(f"{k}={v}" for k, v in kwargs.items())
            fname = f"epoch={epoch}{',' if suffix else ''}{suffix}"
            dest_path = os.path.join(self.root, f"{fname}.jpg")
            if not self.num_valid:
                fig, ax = plt.subplots(layout="tight", dpi=dpi)
                ax.set_aspect("equal")
                self._save_fig(ax, self.features, self.labels, split="Training")
            else:
                fig, axes = plt.subplots(1, 2, sharex='all', sharey="all", layout="tight", dpi=dpi)
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

