import numpy as np
from scipy import ndimage
from PIL import Image
import matplotlib.pyplot as plt


def perona_malik_diffusion(image, iterations=30, delta=0.14, kappa=15, verbose=False):
    """
    Anisotropic diffusion of perona and malik.

    Parameters:
        image: Input image of type numpy.ndarray.
        iterations: Number of iterations.
        delta: Controls the time step.
        kappa: Controls conduction.
        verbose: Boolean, True or False.
    """
    # initial condition
    u = image

    dd = np.sqrt(2)

    # 2D finite difference windows
    windows = [
        np.array([[0, 1, 0], [0, -1, 0], [0, 0, 0]], np.float64),
        np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]], np.float64),
        np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]], np.float64),
        np.array([[0, 0, 0], [1, -1, 0], [0, 0, 0]], np.float64),
        np.array([[0, 0, 1], [0, -1, 0], [0, 0, 0]], np.float64),
        np.array([[0, 0, 0], [0, -1, 0], [0, 0, 1]], np.float64),
        np.array([[0, 0, 0], [0, -1, 0], [1, 0, 0]], np.float64),
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], np.float64),
    ]

    for r in range(iterations):
        nabla = [ndimage.convolve(u, w) for w in windows]
        diff = [1. / (1 + (n / kappa) ** 2) for n in nabla]

        terms = [diff[i] * nabla[i] for i in range(4)]
        terms += [(1 / (dd ** 2)) * diff[i] * nabla[i] for i in range(4, 8)]
        u = u + delta * (sum(terms))

    if verbose:
        plt.subplot(1, 2, 1), plt.imshow(image, cmap='gray')
        plt.xticks([]), plt.yticks([])
        plt.xlabel('Original')
        plt.subplot(1, 2, 2), plt.imshow(u, cmap='gray')
        plt.xticks([]), plt.yticks([])
        plt.xlabel('After diffusion')
        plt.show()

    return u


# Usage
if __name__ == "__main__":
    image_file = 'lenna.png'
    im = Image.open(image_file).convert('L')
    im = np.array(im).astype('float64')
    res = perona_malik_diffusion(im, verbose=True)
