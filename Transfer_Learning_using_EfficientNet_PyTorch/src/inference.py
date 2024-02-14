import glob as glob
import os

import cv2
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from model import build_model

# Constants.
DATA_PATH = '../input/test_images'
IMAGE_SIZE = 224
# device = ('mps' if torch.backends.mps.is_available() else 'cpu')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")


def start_inference(diffusion, model_name, show_results=False) -> [float, float, float, float]:
    """
    Parameters:
        diffusion: in ['no_diffusion', 'perona-malik', 'coherence-enhancing']
        model_name: Name of the model to use among efficientnet family. ie: 'efficient_net_b0'
        show_results: Boolean, True or False.

    Returns the accuracy, precision, recall and f1-score.
    """
    # parameters
    class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

    # Init pred and gt lists.
    y_true, y_pred = [], []

    # Load the trained model.
    model = build_model(pretrained=False, fine_tune=False, num_classes=len(class_names),
                        diffusion=diffusion, model_name=model_name)
    checkpoint = torch.load(f"../outputs/{model_name}/"
                            f"model_pretrained_True_{diffusion if diffusion else 'no_diffusion'}.pth",
                            map_location=DEVICE)
    if show_results:
        print(f"Using {model_name} model")
        print('Loading trained model weights...')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)

    # Get all the test image paths in DATA_PATH/classe_name/*.jpg
    all_image_paths = glob.glob(os.path.join(DATA_PATH, '*/*.jpg'))
    # shuffle the list of image paths.
    np.random.shuffle(all_image_paths)
    if show_results:
        print(f"[INFO]: Number of test images: {len(all_image_paths)}")

    # Create the directory to save the predictions if it doesn't exist.
    os.makedirs(f"../outputs/predictions/{model_name}", exist_ok=True)

    # Iterate over all the images and do forward pass.
    for image_path in all_image_paths:
        # Get the ground truth class name from the image path.
        gt_class_name = image_path.split('/')[-2]
        # Read the image and create a copy.
        image = cv2.imread(image_path)
        orig_image = image.copy()

        # Preprocess the image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        image = transform(image)
        image = torch.unsqueeze(image, 0)
        image = image.to(DEVICE)

        # Forward pass throught the image.
        outputs = model(image)
        outputs = outputs.detach().cpu().numpy()
        pred_class_name = class_names[np.argmax(outputs[0])]

        # Append the ground truth and predicted class names to the lists.
        y_true.append(gt_class_name)
        y_pred.append(pred_class_name.lower())

        # Annotate the image with ground truth.
        """cv2.putText(
            orig_image, f"GT: {gt_class_name}",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2, lineType=cv2.LINE_AA
        )"""
        # Annotate the image with prediction.
        """cv2.putText(
            orig_image, f"Pred: {pred_class_name.lower()}",
            (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (100, 100, 225), 2, lineType=cv2.LINE_AA
        )"""
        # cv2.imshow('Result', orig_image)
        cv2.imwrite(f"../outputs/predictions/{model_name}/{gt_class_name}_"
                    f"{diffusion if diffusion else 'no_diffusion'}.jpg", orig_image)

    # Compute the classification metrics.
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_true, y_pred)

    if show_results:
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1-score: {f1}")
        print(f"Confusion matrix: \n{confusion_matrix(y_true, y_pred)}")

    return accuracy, precision, recall, f1, conf_matrix


if __name__ == '__main__':
    start_inference(diffusion='no_diffusion', model_name='efficient_net_b1', show_results=False)
