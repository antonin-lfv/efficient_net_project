import torch
import numpy as np
from PIL import Image
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = 'mps'


def compute_gradients(img):
    # Définir les filtres de gradient pour les directions x et y
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

    # S'assurer que l'image et les filtres sont sur le même appareil (CPU ou GPU)
    sobel_x = sobel_x.to(img.device)
    sobel_y = sobel_y.to(img.device)

    # Appliquer les filtres pour obtenir les gradients
    grad_x = F.conv2d(img, sobel_x, padding=1).to(device)
    grad_y = F.conv2d(img, sobel_y, padding=1).to(device)

    return grad_x, grad_y


def coherence_enhancing_diffusion_weickert(img, iter_n=100, dt=0.1, k=5, rho=4):
    """
    Applies coherence enhancing diffusion to a grayscale image.

    Parameters:
        img: Input image of type torch.Tensor.
        iter_n: Number of iterations.
        dt: Controls the time step.
        k: Controls conduction.
        rho: Controls the scale of the gaussian.
    """
    # Assurez-vous que img est un tenseur 4D [batch_size, channels, height, width]
    if len(img.shape) == 2:  # Si l'image est de dimension 2 (H, W)
        img = img.unsqueeze(0).unsqueeze(0)  # Ajoutez les dimensions de batch et de canal
    elif len(img.shape) == 3:  # Si l'image est de dimension 3 (C, H, W)
        img = img.unsqueeze(0)  # Ajoutez seulement la dimension de batch

    h, w = img.shape[2], img.shape[3]
    # Copy the image
    u = img.clone().detach().to(device)

    # Pré-calculer la gaussienne pour le tenseur de structure
    kernel_size = 2 * rho + 1  # kernel_size doit être impair
    sigma = rho
    gauss_blur = torchvision.transforms.GaussianBlur(kernel_size, sigma)
    gauss_rho = gauss_blur(img).to(device)

    for _ in range(iter_n):
        # Calculer les gradients
        grad_x, grad_y = compute_gradients(img)

        # Calculer le tenseur de structure
        # S11 = gaussian_filter(grad_x * grad_x, rho) - gauss_rho
        S11 = torchvision.transforms.GaussianBlur(kernel_size, sigma)(grad_x * grad_x) - gauss_rho
        # S12 = gaussian_filter(grad_x * grad_y, rho) - gauss_rho
        S12 = torchvision.transforms.GaussianBlur(kernel_size, sigma)(grad_x * grad_y) - gauss_rho
        # S22 = gaussian_filter(grad_y * grad_y, rho) - gauss_rho
        S22 = torchvision.transforms.GaussianBlur(kernel_size, sigma)(grad_y * grad_y) - gauss_rho

        # Calculer les valeurs propres
        tmp = torch.sqrt((S11 - S22) ** 2 + 4 * S12 ** 2)
        lambda1 = (S11 + S22 + tmp) / 2
        lambda2 = (S11 + S22 - tmp) / 2

        # Calculer la conductance (fonction de Weickert)
        c1 = torch.exp(-(lambda1 ** 2) / k ** 2)
        c2 = torch.exp(-(lambda2 ** 2) / k ** 2)

        # Mettre à jour l'image
        u_t = u + dt * ((c1 * lambda1) + (c2 * lambda2))

        u = u_t.clone().detach().to(device)

    return u


# Load the image
im = Image.open("Transfer_Learning_using_EfficientNet_PyTorch/input/Brain-Tumor-Classification-DataSet/glioma_tumor/gg (2).jpg").convert('L')
img = np.array(im).astype('float32')
img = torch.from_numpy(img)

# Apply coherence enhancing diffusion
result = coherence_enhancing_diffusion_weickert(img)

# result is torch.Size([1, 1, x, y]) but we need torch.Size([x, y])
result = result.squeeze(0).squeeze(0)

# Show the original image and the result
plt.subplot(1, 2, 1)
plt.imshow(img.cpu(), cmap='gray')
plt.title('Original')

plt.subplot(1, 2, 2)
plt.imshow(result.cpu(), cmap='gray')
plt.title('Enhanced')

plt.show()
