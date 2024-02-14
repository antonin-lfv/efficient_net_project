import torch
import plotly.graph_objects as go
import os


def save_model(epochs, model, optimizer, criterion, pretrained, diffusion, model_name):
    """
    Function to save the trained model.

    Parameters:
        epochs: Number of epochs.
        model: Trained model.
        optimizer: Optimizer used for training.
        criterion: Loss function.
        pretrained: Boolean, True or False.
        diffusion: None or in ['perona-malik']
        model_name: Name of the model.
    """

    # create the directory if it does not exist.
    os.makedirs(f"../outputs/{model_name}", exist_ok=True)

    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, f"../outputs/{model_name}/model_pretrained_{pretrained}_{diffusion if diffusion else 'no_diffusion'}.pth")


def save_plots(train_acc, valid_acc, train_loss, valid_loss, pretrained, diffusion, model_name):
    """
    Function to save the loss and accuracy plots to disk using Plotly.

    Parameters:
        train_acc: List of training accuracies.
        valid_acc: List of validation accuracies.
        train_loss: List of training losses.
        valid_loss: List of validation losses.
        pretrained: Boolean, True or False.
        diffusion: None or in ['perona-malik']
        model_name: Name of the model.
    """

    # Accuracy plots
    fig_acc = go.Figure()
    fig_acc.add_trace(go.Scatter(y=train_acc, mode='lines', name='train accuracy', line=dict(color='green')))
    fig_acc.add_trace(go.Scatter(y=valid_acc, mode='lines', name='validation accuracy', line=dict(color='blue')))
    fig_acc.update_layout(
        title=f'Accuracy vs Epochs {model_name} - {diffusion if diffusion else "no_diffusion"}',
        xaxis_title='Epochs',
        yaxis_title='Accuracy'
    )
    fig_acc.update_yaxes(range=[-1, 101])
    fig_acc.write_image(f"../outputs/{model_name}/accuracy_pretrained_{pretrained}_{diffusion if diffusion else 'no_diffusion'}.png")

    # Loss plots
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(y=train_loss, mode='lines', name='train loss', line=dict(color='orange')))
    fig_loss.add_trace(go.Scatter(y=valid_loss, mode='lines', name='validation loss', line=dict(color='red')))
    fig_loss.update_layout(
        title=f'Loss vs Epochs {model_name} - {diffusion if diffusion else "no_diffusion"}',
        xaxis_title='Epochs',
        yaxis_title='Loss'
    )
    fig_loss.write_image(f"../outputs/{model_name}/loss_pretrained_{pretrained}_{diffusion if diffusion else 'no_diffusion'}.png")
