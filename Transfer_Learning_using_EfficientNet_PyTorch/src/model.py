import torchvision.models as models
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import transforms

EFFICIENTNET_MODELS = {
    "efficient_net_b0": models.efficientnet_b0,
    "efficient_net_b1": models.efficientnet_b1,
    "efficient_net_b2": models.efficientnet_b2,
}

# device = ('mps' if torch.backends.mps.is_available() else 'cpu')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PeronaMalikLayer(nn.Module):
    def __init__(self):
        super(PeronaMalikLayer, self).__init__()

        # Trainable parameters
        # And display one image before and after the diffusion at each epoch
        self.iterations = nn.Parameter(torch.tensor(30., dtype=torch.float)) # initialize as learnable parameter
        self.delta = nn.Parameter(torch.tensor(0.10))  # initialize as learnable parameter
        self.kappa = nn.Parameter(torch.tensor(15.))  # initialize as learnable parameter

        # Trainable parameters list
        self.trainable_params_list = [self.iterations, self.delta, self.kappa]
        self.trainable_params_name_list = ['iterations', 'delta', 'kappa']

        # 2D finite difference windows
        windows = [
            torch.tensor([[0, 1, 0], [0, -1, 0], [0, 0, 0]], dtype=torch.float32),
            torch.tensor([[0, 0, 0], [0, -1, 0], [0, 1, 0]], dtype=torch.float32),
            torch.tensor([[0, 0, 0], [0, -1, 1], [0, 0, 0]], dtype=torch.float32),
            torch.tensor([[0, 0, 0], [1, -1, 0], [0, 0, 0]], dtype=torch.float32),
            torch.tensor([[0, 0, 1], [0, -1, 0], [0, 0, 0]], dtype=torch.float32),
            torch.tensor([[0, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=torch.float32),
            torch.tensor([[0, 0, 0], [0, -1, 0], [1, 0, 0]], dtype=torch.float32),
            torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=torch.float32),
        ]
        # Non-trainable parameters
        self.windows = nn.ParameterList([nn.Parameter(w, requires_grad=False) for w in windows])

        self.expand_channels = nn.Conv2d(3, 3, kernel_size=(1, 1))

    def forward(self, x):
        # 1. Convertir en niveaux de gris: poids standard pour les canaux R, G, B
        u = 0.299 * x[:, 0, :, :] + 0.587 * x[:, 1, :, :] + 0.114 * x[:, 2, :, :]
        u = u.unsqueeze(1)  # Ajoutez une dimension pour le canal

        dd = torch.sqrt(torch.tensor(2.))

        # 2. Appliquer la diffusion de Perona-Malik
        for r in range(int(self.iterations.item())):
            nabla = [F.conv2d(u, w.unsqueeze(0).unsqueeze(0), padding=1) for w in self.windows]
            diff = [1. / (1 + (n / self.kappa) ** 2) for n in nabla]

            terms = [diff[i] * nabla[i] for i in range(4)]
            terms += [(1 / (dd ** 2)) * diff[i] * nabla[i] for i in range(4, 8)]
            u = u + self.delta * sum(terms)

        # 3. Dupliquez le canal en niveaux de gris pour obtenir une image "pseudo-RGB"
        u_rgb = u.repeat(1, 3, 1, 1)

        self.expand_channels = self.expand_channels.to(x.device)
        u_rgb_expanded = self.expand_channels(u_rgb)
        return u_rgb_expanded


class CoherenceEnhancingDiffusionLayer(nn.Module):
    def __init__(self, iter_n=100):
        super(CoherenceEnhancingDiffusionLayer, self).__init__()

        # Trainable parameters
        # And display one image before and after the diffusion at each epoch
        self.iter_n = iter_n
        self.dt = nn.Parameter(torch.tensor(0.1))
        self.k = nn.Parameter(torch.tensor(5.0))
        self.rho = nn.Parameter(torch.tensor(4.0))

        # Trainable parameters list
        self.trainable_params_list = [self.dt, self.k, self.rho]
        self.trainable_params_name_list = ['dt', 'k', 'rho']

        # Define the gradient filters for x and y directions
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                    dtype=torch.float32).view(1, 1, 3, 3).to(device)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                    dtype=torch.float32).view(1, 1, 3, 3).to(device)

    def compute_gradients(self, img):
        grad_x = F.conv2d(img, self.sobel_x, padding=1)
        grad_y = F.conv2d(img, self.sobel_y, padding=1)
        return grad_x, grad_y

    def forward(self, x):
        # Convert to grayscale if image is RGB
        if x.size(1) == 3:
            u = 0.299 * x[:, 0, :, :] + 0.587 * x[:, 1, :, :] + 0.114 * x[:, 2, :, :]
        else:
            u = x.squeeze(1)

        u = u.unsqueeze(1)  # Add channel dimension

        rho_int = int(self.rho.item())  # Convert rho to an integer
        kernel_size = (2 * rho_int + 1, 2 * rho_int + 1)
        gauss_blur = transforms.GaussianBlur(kernel_size, rho_int)

        gauss_rho = gauss_blur(u)

        for _ in range(self.iter_n):
            grad_x, grad_y = self.compute_gradients(u)

            S11 = gauss_blur(grad_x * grad_x) - gauss_rho
            S12 = gauss_blur(grad_x * grad_y) - gauss_rho
            S22 = gauss_blur(grad_y * grad_y) - gauss_rho

            tmp = torch.sqrt((S11 - S22) ** 2 + 4 * S12 ** 2)
            lambda1 = (S11 + S22 + tmp) / 2
            lambda2 = (S11 + S22 - tmp) / 2

            c1 = torch.exp(-(lambda1 ** 2) / self.k ** 2)
            c2 = torch.exp(-(lambda2 ** 2) / self.k ** 2)

            u = u + self.dt * ((c1 * lambda1) + (c2 * lambda2))

        # Optional: Replicate the grayscale channel to pseudo-RGB if original was RGB
        if x.size(1) == 3:
            u = u.repeat(1, 3, 1, 1)

        return u


def build_model(pretrained=True, fine_tune=True, num_classes=10, diffusion=None, model_name='efficient_net_b0',
                verbose=False):
    """
    Function to build the model.

    Parameters:
        pretrained: Boolean, True or False.
        fine_tune: Boolean, True or False.
        num_classes: Number of classes.
        diffusion: ['no_diffusion', 'perona-malik', 'coherence-enhancing']
        model_name: Name of the model to use among efficientnet family.
        verbose: Boolean, True or False.

    Actions:
        1. Load the pre-trained weights.
        2. Freeze the hidden layers.
        3. Modify the last layer to have num_classes neurons.

    Returns the model.
    """
    # Check if the model_name is valid.
    assert model_name in EFFICIENTNET_MODELS.keys(), f'[ERROR]: Model name should be in {EFFICIENTNET_MODELS.keys()}'
    # if pretrained is True, the model will load pretrained weights.
    if verbose:
        if pretrained:
            print('[INFO]: Loading pre-trained weights')
            # Load the pre-trained weights.
            # (for example, a model already trained on a large dataset like ImageNet).
        else:
            print('[INFO]: Not loading pre-trained weights')
            # No pre-trained weights. The model will be initialized with random weights.

    # Create an instance of the EfficientNet model with or without pre-trained weights.
    model = EFFICIENTNET_MODELS[model_name](weights='DEFAULT' if pretrained else None)

    # If fine_tune is True, then all parameters of the model are trainable during training.
    if fine_tune:
        if verbose:
            print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad = True  # Gradients will be calculated for these parameters during backpropagation.

    # If fine_tune is False, then all parameters of the model will be frozen
    # and will not be updated during training.
    elif not fine_tune:
        if verbose:
            print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False  # Gradient will not be calculated for these parameters during backpropagation.

    if diffusion == 'perona-malik':
        if verbose:
            print('[IMPORTANT]: Using Perona-Malik diffusion')
        # Insert the PeronaMalikLayer in the beginning of the original features.
        # Extract the original features
        original_features = model.features
        # Replace the first layer with the diffusion layer
        model.features = nn.Sequential(
            PeronaMalikLayer(),
            *original_features
        )

    elif diffusion == 'coherence-enhancing':
        if verbose:
            print('[IMPORTANT]: Using Coherence-Enhancing diffusion')
        # Insert the CoherenceEnhancingDiffusionLayer in the beginning of the original features.
        # Replacing the first layer with the diffusion layer.
        original_features = model.features
        model.features = nn.Sequential(
            CoherenceEnhancingDiffusionLayer(),
            *original_features
        )

    else:
        if verbose:
            print('[IMPORTANT]: No diffusion used')

    # Modify the last layer (classification head) to have num_classes output neurons.
    model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)
    return model
