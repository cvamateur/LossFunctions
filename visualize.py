import numpy as np
import matplotlib.pyplot as plt

from torchvision.utils import make_grid


def visualize_dataset(images, nrows=4, normalize=False):
    grid_image = make_grid(images, nrows, normalize=normalize)
    grid_image = grid_image.permute(1, 2, 0)
    plt.imshow(grid_image)
    plt.show()
