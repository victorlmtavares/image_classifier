##
# PROGRAMMER: Victor Loyola Maia Tavares
# DATE CREATED: 11/22/2020                                 
# REVISED DATE: 
# PURPOSE: 
# Creating functions to handle:
#   Input args;
#   
#
##
import torch.utils.data
import argparse
from torchvision import datasets, transforms
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import numpy as np
import json

def get_training_input_args():
    """Parses the input arguments for training function.
    """
    parser = argparse.ArgumentParser(description = "Retrieve arguments passed on CLI.")
    parser.add_argument("data_directory", default="flowers", type=str, help="Defines the training data directory.")
    parser.add_argument("--save_dir", default="checkpoint.pth", help="Sets the directory to save checkpoints. Please provide path including filename with extension .pth")
    parser.add_argument("--arch", default="vgg16", type=str, help="Sets the neural network architecture.")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Sets the training optimizer learning rate.")
    parser.add_argument("--hidden_units", default=512, type=int, help="Sets the number of hidden units.")
    parser.add_argument("--epochs", default=20, type=int, help="Sets the number of epochs.")
    parser.add_argument("--gpu", type=str2bool, default=False, help="Sets the network to use GPU for training.")
    parser.add_argument("--checkpoint", type=str, default=False, help="Path to the neural network checkpoint to continue training.")
    return parser.parse_args()

def get_prediction_input_args():
    """Parses the input arguments for prediciton function.
    """
    parser = argparse.ArgumentParser(description = "Retrieve arguments passed on CLI.")
    parser.add_argument("path_to_image", type=str, help="Defines the image path to be evaluated.")
    parser.add_argument("path_to_checkpoint", type=str, help="Defines the path to the neural network checkpoint that is going to be used to evaluate the image.")
    parser.add_argument("--top_k", default=5, type=int, help="Defines the number of most likely cases to be shown.")
    parser.add_argument("--category_names", default="cat_to_name.json", type=str, help="Path to a list of names in JSON format to be used as the categories real names.")
    parser.add_argument("--gpu", type=str2bool, default=False, help="Sets the network to use GPU for inference.")
    return parser.parse_args()

def str2bool(string):
    """Converts a string to a boolean.
    """
    if isinstance(string, bool):
       return string
    if string.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif string.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def prepare_data(data_dir):
    """Prepares the data folder to be used by trainer.
        !IMPORTANT Your folder must have the structure shown below.
    """
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    ##Creating transformations
    train_transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])
    test_transforms = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])
    validation_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])    

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
    validation_dataset = datasets.ImageFolder(valid_dir, transform=validation_transforms)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 64, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 64)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size = 64)
    #Getting number of classes
    number_of_classes = len(train_dataset.classes)
    #Storing data labels
    data_labels = {value : key for (key, value) in train_dataset.class_to_idx.items()}

    return train_loader, test_loader, validation_loader, number_of_classes, data_labels

def prepare_image_for_inference(path_to_image):
    """Prepares the image for inference using transformations.
        Returns a pytorch tensor.
    """
    pil_image = Image.open(path_to_image)   
    tensor = TF.to_tensor(pil_image)
    resized_tensor_image = TF.resize(tensor, 255)
    cropped_tensor_image = TF.center_crop(resized_tensor_image, 224)
    normalized_tensor_image = TF.normalize(cropped_tensor_image,[0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    
    return normalized_tensor_image

def print_prediction_results(probabilities, categories, category_names, image_tensor):
    """Prints the prediction results.
    """
    cat_to_name = {}
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)

    imshow(image_tensor, title="Evaluated Image")

    plt.rcdefaults()
    fig, ax = plt.subplots()

    ax.barh(np.arange(len(categories)), probabilities[0].tolist())
    ax.set_yticks(np.arange(len(categories)))
    ax.set_yticklabels(cat_to_name[category] for category in categories)
    ax.invert_yaxis()
    print("The probabilities for the evaluated image shown on your screen are: ")
    for idx in range(len(categories)):
        print("The probability of being {} is {}%.".format(cat_to_name[categories[idx]], round(probabilities[0].tolist()[idx] * 100, 3)))
    plt.show()
    

def imshow(image, ax=None, title=None):
    """Imshow for Tensor. Created by Udacity"""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = np.array(image).transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax