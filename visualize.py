import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

from torchvision.utils import make_grid

COLORS = ["#ff4500", "#d2691e", "#ff7518", "#32cd32", "#367588",
          "#1e90ff", "#ba55d3", "#674846", "#006400", "#f984e5"]


def visualize_images(images, nrows=4, normalize=False):
    grid_image = make_grid(images, nrows, normalize=normalize)
    grid_image = grid_image.permute(1, 2, 0)
    plt.imshow(grid_image)
    plt.show()


class FeatureVisualizer(object):

    def __init__(self, num_images: int, batch_size: int = 10):
        self.bs = batch_size
        self.features = np.zeros([num_images, 2], dtype=np.float32)
        self.labels = np.zeros([num_images], dtype=np.int64)
        self._i = 0
        shutil.rmtree("./pics")
        os.makedirs("./pics")

    def reset(self):
        self._i = 0
        self.features[:] = 0
        self.labels[:] = 0

    def record(self, i, features, labels):
        self.features[i * self.bs: (i + 1) * self.bs] = features
        self.labels[i * self.bs:(i + 1) * self.bs] = labels

    def show(self, epoch):
        fig, ax = plt.subplots()
        ax.set_title(f"Epoch: {epoch}")
        ax.grid(False)
        for i in range(len(COLORS)):
            mask = (self.labels == i)
            feats = self.features[mask]
            ax.scatter(feats[:, 0], feats[:, 1], c=COLORS[i], linewidths=0)
        path = f"./pics/epoch{epoch}"
        plt.savefig(path)
        print("Save Features:", path)
