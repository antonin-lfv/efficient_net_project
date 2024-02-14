import torch
import torch.nn as nn
import torch.optim as optim
import os
import json

from tqdm import tqdm

from model import build_model
from datasets import get_datasets, get_data_loaders
from utils import save_model, save_plots
from inference import start_inference

# device = ('mps' if torch.backends.mps.is_available() else 'cpu')
device = torch.device("cuda" if torch.cuda.is_available() else "mps")


def train(model, trainloader, optimizer, criterion):
    """
    Function to train the model.

    Parameters:
        model: The model to train.
        trainloader: The training data loader.
        optimizer: The optimizer to use for training.
        criterion: The loss function.

    Actions:
        1. Sets the model in training mode.
        2. Iterates through each batch of data in the trainloader.
        3. Gets the images and labels from the current batch.
        4. Moves images and labels to the compute device (CPU or GPU).
        5. Zero the gradients before each forward pass.
        6. Compute predicted outputs by the model.
        7. Calculate loss using the model's output and the actual labels.
        8. Add the loss to the running total.
        9. Calculate accuracy for the batch.
        10. Backpropagation: Compute gradients with respect to the loss.
        11. Update the model weights.
        12. Compute the average loss and accuracy for the entire epoch.

    Returns the average loss and accuracy for the entire epoch.
    """
    model.train()  # Sets the model in training mode.

    train_running_loss = 0.0  # Sum of losses during training.
    train_running_correct = 0  # Count of correct predictions.
    counter = 0  # Counter for processed batches.

    # Iterate through each batch of data in the trainloader.
    for i, data in enumerate(trainloader):
        counter += 1  # Increment the batch counter.

        # Get the images and labels from the current batch.
        image, labels = data
        # Move images and labels to the compute device (CPU or GPU).
        image = image.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()  # Zero the gradients before each forward pass.

        # Forward pass: Compute predicted outputs by the model.
        outputs = model(image)

        # Calculate loss using the model's output and the actual labels.
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()  # Add the loss to the running total.

        # Calculate accuracy for the batch.
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()

        # Backpropagation: Compute gradients with respect to the loss.
        loss.backward()

        # Update the model weights.
        optimizer.step()

    # Compute the average loss and accuracy for the entire epoch.
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))

    return epoch_loss, epoch_acc


# Validation function.
def validate(model, testloader, criterion):
    """
    Function to validate the model.

    Parameters:
        model: The model to validate.
        testloader: The validation data loader.
        criterion: The loss function.

    Actions:
        1. Sets the model in evaluation mode.
        2. Iterates through each batch of data in the testloader.
        3. Gets the images and labels from the current batch.
        4. Moves images and labels to the compute device (CPU or GPU).
        5. Forward pass: Compute predicted outputs by the model.
        6. Calculate loss using the model's output and the actual labels.
        7. Add the loss to the running total.
        8. Calculate accuracy for the batch.
        9. Compute the average loss and accuracy for the entire epoch.

    Returns the average loss and accuracy for the entire epoch.
    """
    model.eval()  # Sets the model in evaluation mode.

    valid_running_loss = 0.0  # Sum of losses during validation.
    valid_running_correct = 0  # Count of correct predictions.
    counter = 0  # Counter for processed batches.

    with torch.no_grad():  # No need to calculate gradients during validation.
        # Iterate through each batch of data in the testloader.
        for i, data in enumerate(testloader):
            counter += 1  # Increment the batch counter.

            # Get the images and labels from the current batch.
            image, labels = data
            # Move images and labels to the compute device (CPU or GPU).
            image = image.to(device)
            labels = labels.to(device)

            # Forward pass: Compute predicted outputs by the model.
            outputs = model(image)

            # Calculate loss using the model's output and the actual labels.
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()  # Add the loss to the running total.

            # Calculate accuracy for the batch.
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()

    # Compute the average loss and accuracy for the entire epoch.
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))

    return epoch_loss, epoch_acc


def start_train(model_name, diffusion, epoch_to_inference=100, epochs=100, lr=0.001,
                pretrained=True, fine_tune=True, verbose=False):
    """
    Start the training.

    Parameters:
        model_name: Name of the model to use among efficientnet family. ie: 'efficient_net_b0'
        diffusion: ['no_diffusion', 'perona-malik']
        epoch_to_inference: Number of epochs to train before inference and stop training. (at this runnnig)
        epochs: Number of epochs to train for.
        lr: Learning rate.
        pretrained: Boolean, True or False.
        fine_tune: Boolean, True or False.
        verbose: Boolean, True or False. Display information if True.

    Actions:

    """
    if epochs and epoch_to_inference:
        epochs = min(epochs, epoch_to_inference)
    elif epochs and not epoch_to_inference:
        epoch_to_inference = epochs
    elif not epochs and epoch_to_inference:
        epochs = epoch_to_inference
    elif not epochs and not epoch_to_inference:
        raise ValueError('Please provide either epochs or epoch_to_inference.')

    # Load the training and validation datasets.
    dataset_train, dataset_valid, dataset_classes = get_datasets(pretrained)
    if verbose:
        print(f"[INFO]: Number of training images: {len(dataset_train)}")
        print(f"[INFO]: Number of validation images: {len(dataset_valid)}")
        print(f"[INFO]: Class names: {dataset_classes}\n")
    # Load the training and validation data loaders.
    train_loader, valid_loader = get_data_loaders(dataset_train, dataset_valid)

    # Set the computation device.
    if verbose:
        print(f"\n[PARAMETERS]: Computation device: {device}")
        print(f"[PARAMETERS]: Learning rate: {lr}")
        print(f"[PARAMETERS]: Epochs to train for: {epochs}\n")
        print(f"[PARAMETERS]: Model name: {model_name}")

    # If there is already a model saved, load it from outputs/model_..
    if os.path.exists(f'../outputs/{model_name}/model_pretrained_True'
                      f'{f"_{diffusion}" if diffusion else "_no_diffusion"}.pth'):
        if verbose:
            print('[INFO]: Loading trained model weights to train further...')
        model = build_model(pretrained=True,
                            fine_tune=True,
                            num_classes=len(dataset_classes),
                            diffusion=diffusion,
                            model_name=model_name,
                            verbose=verbose
                            ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        checkpoint = torch.load(f"../outputs/{model_name}/model_pretrained_True_"
                                f"{diffusion if diffusion else 'no_diffusion'}.pth", map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        criterion = checkpoint['loss']

        # Lists to keep track of losses and accuracies from training_stats{f"_{diffusion}" if diffusion else ""}.json.
        stats = json.load(open(f'../outputs/{model_name}/training_stats.json'))
        stats = stats[diffusion if diffusion else 'no_diffusion']
        train_loss = stats['train_loss']
        valid_loss = stats['valid_loss']
        train_acc = stats['train_acc']
        valid_acc = stats['valid_acc']
        epochs_done = stats['epochs']

    else:
        if verbose:
            print('[INFO]: No .pth model found...')
        model = build_model(
            pretrained=pretrained,
            fine_tune=fine_tune,
            num_classes=len(dataset_classes),
            diffusion=diffusion,
            model_name=model_name,
            verbose=verbose
        ).to(device)

        # Optimizer.
        optimizer = optim.Adam(model.parameters(), lr=lr)
        # Loss function.
        criterion = nn.CrossEntropyLoss()

        # Lists to keep track of losses and accuracies.
        train_loss, valid_loss = [], []
        train_acc, valid_acc = [], []
        epochs_done = 0

    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"[INFO]: {total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"[INFO]: {total_trainable_params:,} training parameters.")

    # Start the training.
    for epoch in range(epochs):
        if verbose:
            print(f"\n[INFO]: Epoch {epoch + 1} of {epochs} for this session.")
            print(f"[INFO]: Number of epochs already done : {epochs_done + epoch}")
        train_epoch_loss, train_epoch_acc = train(model, train_loader,
                                                  optimizer, criterion)
        valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader,
                                                     criterion)
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        if verbose:
            print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
            print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
            print('-' * 50)

        # Save the trained model weights.
        save_model(epochs, model, optimizer, criterion, pretrained, diffusion, model_name)

        # Save the loss and accuracy plots.
        save_plots(train_acc, valid_acc, train_loss, valid_loss, pretrained, diffusion, model_name)

        # Save the training stats.
        stats = {
            'epochs': epoch + epochs_done + 1,
            'train_loss': train_loss,
            'valid_loss': valid_loss,
            'train_acc': train_acc,
            'valid_acc': valid_acc
        }

        # Load the existing stats or create a new one.
        if not os.path.exists(f'../outputs/{model_name}/training_stats.json'):
            with open(f'../outputs/{model_name}/training_stats.json', 'w') as fp:
                json.dump({}, fp)

        with open(f'../outputs/{model_name}/training_stats.json', 'r') as fp:
            data = json.load(fp)

        # Update the stats.
        key = diffusion if diffusion else 'no_diffusion'
        data[key] = stats

        # Save the updated stats.
        with open(f'../outputs/{model_name}/training_stats.json', 'w') as fp:
            json.dump(data, fp)

        if epoch_to_inference:
            if epoch == epoch_to_inference - 1:
                # Get the accuracy, precision, recall, f1 score and confusion matrix.
                accuracy, precision, recall, f1, conf_matrix = start_inference(diffusion, model_name)
                with open(f'../outputs/results.txt', 'a') as file:
                    file.write(f'\n\n-- {model_name} / Diffusion: {diffusion if diffusion else "no_diffusion"} --\n')
                    file.write(f'Epochs: {epoch_to_inference + epochs_done}\n')
                    file.write(f'Accuracy: {accuracy}\n')
                    file.write(f'Precision: {precision}\n')
                    file.write(f'Recall: {recall}\n')
                    file.write(f'F1 Score: {f1}\n')
                    file.write('Confusion Matrix:\n')
                    file.write(str(conf_matrix))
                    file.write('\n')
                    # Get the values of the parameters of the first diffusion layer.
                    if diffusion != 'no_diffusion':
                        diffusion_layer = model.features[0]
                        diffusion_trainable_params_list = diffusion_layer.trainable_params_list
                        diffusion_trainable_params_name_list = diffusion_layer.trainable_params_name_list
                        for param, name in zip(diffusion_trainable_params_list, diffusion_trainable_params_name_list):
                            file.write(f'{name}: {param.detach().item()}\n')

                break

    if verbose:
        print('TRAINING COMPLETE\n')


if __name__ == '__main__':
    diffusions = ['no_diffusion', 'perona-malik', 'coherence-enhancing']
    model_name = 'efficient_net_b3'
    for diffusion in diffusions:
        print(f'[INFO]: Starting training for {model_name} with diffusion: {diffusion}')
        start_train(model_name=model_name,
                    diffusion=diffusion,
                    epoch_to_inference=30,
                    epochs=50,
                    pretrained=True,
                    fine_tune=True,
                    verbose=True)
        print('TRAINING COMPLETE\n')
