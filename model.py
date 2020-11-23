##
# PROGRAMMER: Victor Loyola Maia Tavares
# DATE CREATED: 11/22/2020                                 
# REVISED DATE: 
# PURPOSE: 
# Creating functions to handle model functions such as:
#   Creating model;
#   Loading model;
#   Training;
#   Predicting;   
#
##

import torch
from torch import nn 
from torch import optim 
from torchvision import models
from collections import OrderedDict

import util

resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
squeezenet = models.squeezenet1_0(pretrained=True)
vgg16 = models.vgg16(pretrained=True)
densenet = models.densenet161(pretrained=True)
inception = models.inception_v3(pretrained=True)
googlenet = models.googlenet(pretrained=True)
shufflenet = models.shufflenet_v2_x1_0(pretrained=True)
mobilenet = models.mobilenet_v2(pretrained=True)
resnext50_32x4d = models.resnext50_32x4d(pretrained=True)
wide_resnet50_2 = models.wide_resnet50_2(pretrained=True)
mnasnet = models.mnasnet1_0(pretrained=True)

archs_dict = {'resnet18': resnet18, 
        'alexnet': alexnet,
        'squeezenet':squeezenet, 
        'vgg16': vgg16,
        'densenet': densenet,
        'inception':inception,
        'googlenet':googlenet,
        'shufflenet':shufflenet,
        'mobilenet':mobilenet,
        'resnext50_32x4d':resnext50_32x4d,
        'wide_resnet50_2':wide_resnet50_2,
        'mnasnet':mnasnet
        }

def create_model(gpu, arch = 'vgg16', input_size = 25088, hidden_layer_size = 512, output_size = 102):
    """Creates a neural network model.
    """
    if arch in archs_dict:
        model = archs_dict[arch]
    else:
        print("You haven`t inserted a valid architecture. Check the available architectures at https://pytorch.org/docs/stable/torchvision/models.html.")
        return False
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.classifier = nn.Sequential(OrderedDict([
        ('Input', nn.Linear(input_size, hidden_layer_size)),
        ('hidden1', nn.ReLU()),
        ('DropOut1', nn.Dropout(p=0.2)),
        ('layer1', nn.Linear(hidden_layer_size, int(hidden_layer_size/2))),
        ('hidden2', nn.ReLU()),
        ('layer2', nn.Linear(int(hidden_layer_size/2), output_size)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    device = 'cuda' if gpu else 'cpu'
    model.to(device)

    return model

def train(in_args):
    """Starts the training process of the neural network.
    """
    ##Peparing data
    train_loader, test_loader, validation_loader, number_of_classes, data_labels = util.prepare_data(in_args.data_directory)
    #Preparing Model, Criterion and Optimizer
    if(in_args.checkpoint):
        model, data_labels, criterion, optimizer = load_checkpoint(in_args.checkpoint, in_args.gpu, True)
    else:
        model = create_model(gpu = in_args.gpu, arch = in_args.arch,  hidden_layer_size = in_args.hidden_units, output_size=number_of_classes)
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=in_args.learning_rate)
    if(model):
        device = 'cuda' if in_args.gpu else 'cpu'
        running_loss = 0
        for epoch in range(in_args.epochs):
            steps = 0
            for inputs, labels in train_loader:
                steps += 1
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                output = model(inputs)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                if steps % 5 == 0:
                    test_loss = 0
                    accuracy = 0
                    model.eval()

                    with torch.no_grad():
                        for inputs, labels in test_loader:
                            inputs, labels = inputs.to(device), labels.to(device)
                            output = model(inputs)
                            batch_loss = criterion(output, labels)
                            test_loss += batch_loss.item()
                            probability = torch.exp(output)
                            top_p, top_class = probability.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    print("---------")
                    print("Epoch {}/{}. Step {}.".format((epoch+1), in_args.epochs, steps))
                    print("Train loss: {}".format("%.3f" % (running_loss/5)))
                    print("Test loss: {}".format("%.3f" % (test_loss/len(test_loader))))
                    print("Test accuracy: {}".format("%.3f" % (accuracy/len(test_loader))))
                    print("---------")
                    running_loss = 0
                    model.train()

        save_directory = in_args.save_dir
        print("Network training finished.\nPlease wait while progress is being saved to file {}.".format(save_directory))
        checkpoint = {
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            'hidden_layer_size':in_args.hidden_units,
            'output_size':number_of_classes,
            'architecture':in_args.arch,
            'data_labels': data_labels
        }
        torch.save(checkpoint, save_directory)
        print("The neural network will perform a validation test. Please wait.")
        validation_loss = 0
        accuracy = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                output = model(inputs)
                batch_loss = criterion(output, labels)                   
                validation_loss += batch_loss.item()                    
                probability = torch.exp(output)
                top_p, top_class = probability.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        print("Validation loss: {}".format("%.3f" % (validation_loss/len(validation_loader))))
        print("Validation accuracy: {}".format("%.3f" % (accuracy/len(validation_loader))))
        print("Above results show expected performance during inference.")
    
def predict(in_args):
    """Used for inference. Uses the trained network for predictions.
    """
    device = 'cuda' if in_args.gpu else 'cpu'
    model, data_labels = load_checkpoint(in_args.path_to_checkpoint, in_args.gpu)
    image_tensor = util.prepare_image_for_inference(in_args.path_to_image)
    model.eval()
    prediction = model(image_tensor.unsqueeze(0).to(device))
    prob, classes = prediction.topk(in_args.top_k)
    probabilities = torch.exp(prob).cpu().detach().numpy()    
    numpy_classes = classes.cpu().numpy()
    categories = []
    for idx in numpy_classes[0]:
        categories.append(data_labels[idx])
    util.print_prediction_results(probabilities, categories, in_args.category_names, image_tensor)


def load_checkpoint(filepath, gpu, training=False):
    """Loads a neural network checkpoint previously saved.
    """
    checkpoint = torch.load(filepath)

    model = create_model(gpu, arch=checkpoint['architecture'], hidden_layer_size=checkpoint['hidden_layer_size'], output_size=checkpoint['output_size'])
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
        
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])    
    data_labels = checkpoint['data_labels']
    
    if(training):
        return model, data_labels, criterion, optimizer
    return model, data_labels

